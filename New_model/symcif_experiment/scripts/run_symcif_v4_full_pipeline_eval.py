#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.formula import normalize_formula_counts
from symcif_v4.orbit_engine import OrbitEngine

from run_generation_eval import evaluate_mode_with_hard_timeouts


TOP_KS = (1, 5, 20)
TARGET_SGS = (2, 65, 71, 127)
MODE_TO_DIR = {
    "full_current": "symcif_v4_full_eval_current",
    "wa_upper_bound": "symcif_v4_wa_upper_bound",
    "geometry_bottleneck": "symcif_v4_geometry_bottleneck",
}
MODE_DESCRIPTIONS = {
    "full_current": "predicted WA top20 + current retrieved/default geometry",
    "wa_upper_bound": "predicted WA top20 + GT free params/lattice only for exact GT WA hits",
    "geometry_bottleneck": "GT WA + current retrieved/default geometry",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)) or "sample"


def wa_multiset_key_from_rows(rows: list[dict[str, Any]]) -> str:
    items = sorted((str(row["orbit_id"]), str(row["element"])) for row in rows)
    return json.dumps(items, ensure_ascii=True, separators=(",", ":"))


def skeleton_multiset_key_from_rows(rows: list[dict[str, Any]]) -> str:
    items = sorted(str(row["orbit_id"]) for row in rows)
    return json.dumps(items, ensure_ascii=True, separators=(",", ":"))


def record_wa_multiset_key(record: dict[str, Any]) -> str:
    return wa_multiset_key_from_rows(list(record.get("wa_table") or []))


def record_skeleton_multiset_key(record: dict[str, Any]) -> str:
    return skeleton_multiset_key_from_rows(list(record.get("wa_table") or []))


def load_predictions(path: Path, top_k: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["ranked_wa_candidates"] = list(row.get("ranked_wa_candidates") or row.get("candidates") or [])[: int(top_k)]
            out[str(row["sample_id"])] = row
    return out


def lattice_median(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("a", "b", "c", "alpha", "beta", "gamma")
    if not records:
        return {"a": 6.0, "b": 6.0, "c": 6.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    return {key: float(statistics.median(float(r["lattice"][key]) for r in records)) for key in keys}


def build_train_indexes(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[int, dict[str, float]], dict[str, float]]:
    by_wa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_skel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_wa[record_wa_multiset_key(record)].append(record)
        by_skel[record_skeleton_multiset_key(record)].append(record)
        by_sg[int(record["sg"])].append(record)
    med_sg = {sg: lattice_median(items) for sg, items in by_sg.items()}
    return dict(by_wa), dict(by_skel), med_sg, lattice_median(records)


def deterministic_params(engine: OrbitEngine, orbit_id: str, row_index: int) -> dict[str, float]:
    orbit = engine.get_orbit_by_id(str(orbit_id))
    base = {"x": 0.173, "y": 0.271, "z": 0.389}
    out: dict[str, float] = {}
    for symbol in orbit.free_symbols:
        out[str(symbol)] = (base.get(str(symbol), 0.173) + 0.037 * row_index) % 1.0
    return out


def params_from_reference(
    engine: OrbitEngine,
    candidate_rows: list[dict[str, Any]],
    reference: dict[str, Any] | None,
    *,
    match_element: bool,
    source_when_reference: str,
) -> tuple[dict[int, dict[str, float]], str]:
    params: dict[int, dict[str, float]] = {}
    source = "deterministic_free_params"
    remaining: dict[tuple[str, str] | tuple[str], list[dict[str, Any]]] = defaultdict(list)
    if reference is not None:
        for row in reference["wa_table"]:
            key = (str(row["orbit_id"]), str(row["element"])) if match_element else (str(row["orbit_id"]),)
            remaining[key].append(row)
        source = source_when_reference
    for idx, row in enumerate(candidate_rows):
        key = (str(row["orbit_id"]), str(row["element"])) if match_element else (str(row["orbit_id"]),)
        refs = remaining.get(key) or []
        if refs:
            ref = refs.pop(0)
            params[idx] = {str(k): float(v) for k, v in dict(ref.get("free_params") or {}).items()}
        else:
            params[idx] = deterministic_params(engine, str(row["orbit_id"]), idx)
            if reference is not None and not source.endswith("+deterministic_fallback"):
                source = source + "+deterministic_fallback"
    return params, source


def candidate_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in candidate.get("rows", []):
        out.append(
            {
                "element": str(row["element"]),
                "orbit_id": str(row["orbit_id"]),
                "letter": row.get("letter"),
                "multiplicity": row.get("multiplicity"),
                "site_symmetry": row.get("site_symmetry"),
                "enumeration": row.get("enumeration"),
            }
        )
    return out


def candidate_wa_multiset_key(candidate: dict[str, Any]) -> str:
    return wa_multiset_key_from_rows(candidate_rows(candidate))


def candidate_skeleton_multiset_key(candidate: dict[str, Any]) -> str:
    return skeleton_multiset_key_from_rows(candidate_rows(candidate))


def gt_candidate_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_wa_key": record["canonical_wa_key"],
        "canonical_skeleton_key": record["canonical_skeleton_key"],
        "formula_counts": record["formula_counts"],
        "sg": int(record["sg"]),
        "rows": [
            {
                "element": row["element"],
                "orbit_id": row["orbit_id"],
                "letter": row.get("letter"),
                "multiplicity": row.get("multiplicity"),
                "site_symmetry": row.get("site_symmetry"),
                "enumeration": row.get("enumeration"),
            }
            for row in record["wa_table"]
        ],
        "source_labels": ["gt_wa"],
    }


def current_geometry(
    engine: OrbitEngine,
    record: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    train_wa: dict[str, list[dict[str, Any]]],
    train_skel: dict[str, list[dict[str, Any]]],
    median_lattice_by_sg: dict[int, dict[str, float]],
    global_median_lattice: dict[str, float],
) -> tuple[dict[str, Any], dict[int, dict[str, float]], str]:
    reference = None
    match_element = False
    source = "deterministic_free_params"
    same_wa = train_wa.get(candidate_wa_multiset_key(candidate)) or []
    same_skel = train_skel.get(candidate_skeleton_multiset_key(candidate)) or []
    if same_wa:
        reference = same_wa[0]
        match_element = True
        source = "train_same_wa"
    elif same_skel:
        reference = same_skel[0]
        match_element = False
        source = "train_same_skeleton"
    params, source = params_from_reference(
        engine,
        rows,
        reference,
        match_element=match_element,
        source_when_reference=source,
    )
    lattice = (
        dict(reference["lattice"])
        if reference is not None
        else dict(median_lattice_by_sg.get(int(record["sg"]), global_median_lattice))
    )
    if reference is None:
        source = "sg_median_lattice+" + source
    return lattice, params, source


def upper_bound_geometry(
    engine: OrbitEngine,
    record: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, float]], str]:
    if candidate_wa_multiset_key(candidate) == record_wa_multiset_key(record):
        params, source = params_from_reference(
            engine,
            rows,
            record,
            match_element=True,
            source_when_reference="gt_exact_wa_gt_free_params",
        )
    else:
        params, source = params_from_reference(
            engine,
            rows,
            None,
            match_element=True,
            source_when_reference="unused",
        )
        source = "gt_lattice_wrong_wa_deterministic_free_params"
    return dict(record["lattice"]), params, source


def expanded_atom_count_ok(
    engine: OrbitEngine,
    rows: list[dict[str, Any]],
    params: dict[int, dict[str, float]],
    record: dict[str, Any],
) -> tuple[int | None, bool]:
    try:
        count = int(engine.expanded_atom_count(rows, params))
    except Exception:
        return None, False
    target = sum(int(v) for v in normalize_formula_counts(record["formula_counts"]).values())
    return count, count == target


def render_one(
    *,
    mode: str,
    engine: OrbitEngine,
    record: dict[str, Any],
    candidate: dict[str, Any],
    rank: int,
    train_wa: dict[str, list[dict[str, Any]]],
    train_skel: dict[str, list[dict[str, Any]]],
    median_lattice_by_sg: dict[int, dict[str, float]],
    global_median_lattice: dict[str, float],
) -> dict[str, Any]:
    started = time.monotonic()
    rows = candidate_rows(candidate)
    try:
        if mode == "wa_upper_bound":
            lattice, params, geometry_source = upper_bound_geometry(engine, record, candidate, rows)
        else:
            lattice, params, geometry_source = current_geometry(
                engine,
                record,
                candidate,
                rows,
                train_wa,
                train_skel,
                median_lattice_by_sg,
                global_median_lattice,
            )
        expanded_count, atom_count_ok = expanded_atom_count_ok(engine, rows, params, record)
        cif = engine.render_cif_from_wa_table(
            rows,
            lattice=lattice,
            free_params_by_row=params,
            formula_counts=record["formula_counts"],
            sg=int(record["sg"]),
            sg_symbol=str(record.get("sg_symbol") or ""),
            data_name=f"{record['sample_id']}_rank{rank}",
        )
        return {
            "ok": True,
            "cif": cif,
            "render_time_seconds": time.monotonic() - started,
            "geometry_source": geometry_source,
            "atom_count_after_expansion": expanded_count,
            "atom_count_ok": atom_count_ok,
            "canonical_wa_key": candidate.get("canonical_wa_key"),
            "canonical_skeleton_key": candidate.get("canonical_skeleton_key"),
            "source_labels": candidate.get("source_labels") or [],
            "policy_rank": candidate.get("policy_rank"),
            "old_rank": candidate.get("old_rank"),
            "final_rerank_rank": candidate.get("final_rerank_rank"),
            "final_rerank_score": candidate.get("final_rerank_score"),
            "hybrid_score": candidate.get("hybrid_score"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "cif": "",
            "render_time_seconds": time.monotonic() - started,
            "geometry_source": "render_failed",
            "atom_count_after_expansion": None,
            "atom_count_ok": False,
            "canonical_wa_key": candidate.get("canonical_wa_key"),
            "canonical_skeleton_key": candidate.get("canonical_skeleton_key"),
            "source_labels": candidate.get("source_labels") or [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def case_payload(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        out.append(
            {
                "index": index,
                "sample_id": row["sample_id"],
                "source_path": str(row["source_path"]),
                "target_formula": row["formula"],
                "target_sg_number": int(row["sg"]),
                "target_sg_symbol": row.get("sg_symbol"),
            }
        )
    return out


def candidates_for_mode(
    mode: str,
    record: dict[str, Any],
    prediction: dict[str, Any] | None,
    top_k: int,
) -> list[dict[str, Any]]:
    if mode == "geometry_bottleneck":
        candidate = gt_candidate_from_record(record)
        return [candidate for _ in range(int(top_k))]
    if prediction is None:
        return []
    return list(prediction.get("ranked_wa_candidates") or [])[: int(top_k)]


def render_mode_generations(
    *,
    mode: str,
    out_dir: Path,
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    engine: OrbitEngine,
    train_wa: dict[str, list[dict[str, Any]]],
    train_skel: dict[str, list[dict[str, Any]]],
    median_lattice_by_sg: dict[int, dict[str, float]],
    global_median_lattice: dict[str, float],
    top_k: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[tuple[int, int], dict[str, Any]]]:
    cif_dir = out_dir / "generated_cifs"
    gen_dir = out_dir / "generations"
    cif_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[dict[str, Any]]] = {}
    meta_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    generation_lines: list[dict[str, Any]] = []
    top20_lines: list[dict[str, Any]] = []
    eval_top_k = 1 if mode == "geometry_bottleneck" else int(top_k)

    for sample_index, record in enumerate(records):
        prediction = predictions.get(str(record["sample_id"]))
        candidates = candidates_for_mode(mode, record, prediction, top_k)
        sample_dir = cif_dir / safe_filename(str(record["sample_id"]))
        sample_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        grouped[sample_index] = []
        for gen_index in range(eval_top_k):
            rank = gen_index + 1
            candidate = candidates[gen_index] if gen_index < len(candidates) else None
            if candidate is None:
                meta = {
                    "sample_index": sample_index,
                    "sample_id": record["sample_id"],
                    "rank": rank,
                    "gen_index": gen_index,
                    "render_success": False,
                    "error": "missing_candidate",
                    "canonical_wa_key": None,
                    "canonical_skeleton_key": None,
                    "geometry_source": "missing_candidate",
                    "atom_count_after_expansion": None,
                    "atom_count_ok": False,
                }
                gen = {
                    "sample_index": sample_index,
                    "sample_id": record["sample_id"],
                    "gen_index": gen_index,
                    "seed": gen_index,
                    "raw_generation_success": False,
                    "generated_text": "",
                    "generation_time_seconds": 0.0,
                    "error": "missing_candidate",
                }
            else:
                rendered = render_one(
                    mode=mode,
                    engine=engine,
                    record=record,
                    candidate=candidate,
                    rank=rank,
                    train_wa=train_wa,
                    train_skel=train_skel,
                    median_lattice_by_sg=median_lattice_by_sg,
                    global_median_lattice=global_median_lattice,
                )
                cif_path = sample_dir / f"rank_{rank:03d}.cif"
                if rendered["ok"]:
                    cif_path.write_text(str(rendered["cif"]), encoding="utf-8")
                meta = {
                    "sample_index": sample_index,
                    "sample_id": record["sample_id"],
                    "rank": rank,
                    "gen_index": gen_index,
                    "render_success": bool(rendered["ok"]),
                    "cif_path": str(cif_path) if rendered["ok"] else None,
                    "geometry_source": rendered.get("geometry_source"),
                    "atom_count_after_expansion": rendered.get("atom_count_after_expansion"),
                    "atom_count_ok": bool(rendered.get("atom_count_ok")),
                    "canonical_wa_key": rendered.get("canonical_wa_key"),
                    "canonical_skeleton_key": rendered.get("canonical_skeleton_key"),
                    "source_labels": rendered.get("source_labels") or [],
                    "policy_rank": rendered.get("policy_rank"),
                    "old_rank": rendered.get("old_rank"),
                    "final_rerank_rank": rendered.get("final_rerank_rank"),
                    "final_rerank_score": rendered.get("final_rerank_score"),
                    "hybrid_score": rendered.get("hybrid_score"),
                    "render_time_seconds": rendered.get("render_time_seconds"),
                    "error": rendered.get("error"),
                }
                gen = {
                    "sample_index": sample_index,
                    "sample_id": record["sample_id"],
                    "gen_index": gen_index,
                    "seed": gen_index,
                    "raw_generation_success": bool(rendered["ok"]),
                    "generated_text": str(rendered["cif"] or ""),
                    "generation_time_seconds": rendered.get("render_time_seconds"),
                    "error": rendered.get("error"),
                }
            grouped[sample_index].append(gen)
            generation_lines.append(gen)
            meta_by_key[(sample_index, gen_index)] = meta
            entries.append(
                {
                    k: v
                    for k, v in meta.items()
                    if k
                    not in {
                        "sample_index",
                        "sample_id",
                        "gen_index",
                    }
                }
            )
        if mode == "geometry_bottleneck" and entries:
            base_entry = dict(entries[0])
            for duplicate_rank in range(2, int(top_k) + 1):
                duplicate = dict(base_entry)
                duplicate["rank"] = duplicate_rank
                entries.append(duplicate)
        top20_lines.append(
            {
                "sample_index": sample_index,
                "sample_id": record["sample_id"],
                "formula": record["formula"],
                "formula_counts": record["formula_counts"],
                "sg": int(record["sg"]),
                "n_sites": int(record["n_sites"]),
                "num_elements": int(record["num_elements"]),
                "gt_wa_key": record["canonical_wa_key"],
                "gt_skeleton_key": record["canonical_skeleton_key"],
                "predictions": entries,
            }
        )

    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
    write_jsonl(gen_dir / "baseline.jsonl", generation_lines)
    write_jsonl(out_dir / "top20_predictions.jsonl", top20_lines)
    return grouped, meta_by_key


def bond_lengths_reasonable(metric: dict[str, Any]) -> bool:
    value = metric.get("bond_length_score")
    if value is None:
        return False
    try:
        return float(value) >= 1.0
    except Exception:
        return False


def enrich_metrics(
    metrics: list[dict[str, Any]],
    meta_by_key: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for metric in metrics:
        key = (int(metric["sample_index"]), int(metric["gen_index"]))
        meta = meta_by_key.get(key, {})
        row = dict(metric)
        for name, value in meta.items():
            if name in {"sample_index", "sample_id", "gen_index", "error"}:
                continue
            row[name] = value
        row["rank"] = int(metric["gen_index"]) + 1
        row["readable"] = bool(row.get("pymatgen_readable"))
        row["SG_ok"] = bool(row.get("space_group_ok"))
        row["bond_lengths_reasonable"] = bond_lengths_reasonable(row)
        row["strict_valid"] = bool(
            row.get("pymatgen_readable")
            and row.get("formula_ok")
            and row.get("atom_count_ok")
            and row.get("space_group_ok")
            and row.get("bond_lengths_reasonable")
            and row.get("valid")
        )
        if not row.get("error"):
            row["error"] = meta.get("error")
        enriched.append(row)
    enriched.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    return enriched


def expand_geometry_bottleneck_metrics(metrics: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_sample[int(metric["sample_index"])].append(metric)
    for sample_index in sorted(by_sample):
        rows = sorted(by_sample[sample_index], key=lambda r: int(r["gen_index"]))
        if not rows:
            continue
        base = rows[0]
        for gen_index in range(int(top_k)):
            item = dict(base)
            item["gen_index"] = gen_index
            item["rank"] = gen_index + 1
            item["seed"] = gen_index
            expanded.append(item)
    expanded.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    return expanded


def subset_records(records: list[dict[str, Any]], subset: str) -> set[int]:
    selected: set[int] = set()
    for idx, row in enumerate(records):
        if subset == "overall":
            selected.add(idx)
        elif subset == "n_sites>=6" and int(row["n_sites"]) >= 6:
            selected.add(idx)
        elif subset == "num_elements>=4" and int(row["num_elements"]) >= 4:
            selected.add(idx)
        elif subset.startswith("SG=") and int(row["sg"]) == int(subset.split("=", 1)[1]):
            selected.add(idx)
    return selected


def summarize_for_k(
    records: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    sample_indexes: set[int],
    k: int,
) -> dict[str, Any]:
    subset = [m for m in metrics if int(m["sample_index"]) in sample_indexes and int(m["gen_index"]) < int(k)]
    attempts = len(sample_indexes) * int(k)
    out: dict[str, Any] = {"samples": len(sample_indexes), "k": int(k), "num_attempts": attempts}
    bool_fields = {
        "render_success": "render_success",
        "readable": "pymatgen_readable",
        "formula_ok": "formula_ok",
        "atom_count_ok": "atom_count_ok",
        "SG_ok": "space_group_ok",
        "multiplicity_ok": "multiplicity_ok",
        "valid": "valid",
        "bond_lengths_reasonable": "bond_lengths_reasonable",
        "strict_valid": "strict_valid",
        "eval_timeout": "eval_timeout",
    }
    for out_name, field in bool_fields.items():
        out[out_name] = sum(1 for m in subset if m.get(field)) / attempts if attempts else math.nan
    score_sum = sum(float(m["bond_length_score"]) for m in subset if m.get("bond_length_score") is not None)
    out["bond_length_score"] = score_sum / attempts if attempts else math.nan

    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in subset:
        by_sample[int(metric["sample_index"])].append(metric)
    match_count = 0
    strict_any = 0
    best_rms: list[float] = []
    wa_hit = 0
    skeleton_hit = 0
    for sample_index in sample_indexes:
        rows = sorted(by_sample.get(sample_index, []), key=lambda r: int(r["gen_index"]))
        if any(r.get("match_ok") for r in rows):
            match_count += 1
        if any(r.get("strict_valid") for r in rows):
            strict_any += 1
        rms_values = [float(r["rms"]) for r in rows if r.get("match_ok") and r.get("rms") is not None]
        if rms_values:
            best_rms.append(min(rms_values))
        record = records[sample_index]
        gt_wa = record_wa_multiset_key(record)
        gt_skel = record_skeleton_multiset_key(record)
        gt_wa_ordered = str(record["canonical_wa_key"])
        gt_skel_ordered = str(record["canonical_skeleton_key"])
        if any(str(r.get("wa_multiset_key")) == gt_wa for r in rows):
            wa_hit += 1
        if any(str(r.get("skeleton_multiset_key")) == gt_skel for r in rows):
            skeleton_hit += 1
        out.setdefault("_wa_ordered_hits", 0)
        out.setdefault("_skeleton_ordered_hits", 0)
        if any(str(r.get("canonical_wa_key")) == gt_wa_ordered for r in rows):
            out["_wa_ordered_hits"] += 1
        if any(str(r.get("canonical_skeleton_key")) == gt_skel_ordered for r in rows):
            out["_skeleton_ordered_hits"] += 1
    denom = max(1, len(sample_indexes))
    out["match_at_k"] = match_count / denom
    out["strict_valid_any_at_k"] = strict_any / denom
    out["RMSE"] = float(statistics.mean(best_rms)) if best_rms else math.nan
    out["matched_samples_for_RMSE"] = len(best_rms)
    out["wa_hit_at_k"] = wa_hit / denom
    out["skeleton_hit_at_k"] = skeleton_hit / denom
    out["wa_hit_order_sensitive_at_k"] = float(out.pop("_wa_ordered_hits", 0)) / denom
    out["skeleton_hit_order_sensitive_at_k"] = float(out.pop("_skeleton_ordered_hits", 0)) / denom
    return out


def make_summary(
    *,
    mode: str,
    records: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    subsets = ["overall", "n_sites>=6", "num_elements>=4"] + [f"SG={sg}" for sg in TARGET_SGS]
    breakdown_rows: list[dict[str, Any]] = []
    summary_by_k: dict[str, dict[str, Any]] = {}
    for subset in subsets:
        sample_indexes = subset_records(records, subset)
        for k in TOP_KS:
            row = {"mode": mode, "subset": subset, **summarize_for_k(records, metrics, sample_indexes, k)}
            breakdown_rows.append(row)
            if subset == "overall":
                summary_by_k[f"top{k}"] = row
    source_counts = Counter(str(m.get("geometry_source")) for m in metrics if m.get("geometry_source"))
    render_errors = Counter(str(m.get("error")) for m in metrics if m.get("error"))
    return {
        "mode": mode,
        "description": MODE_DESCRIPTIONS[mode],
        "data_root": str(args.data_root),
        "predictions": str(args.predictions),
        "samples": len(records),
        "top_k": int(args.top_k),
        "evaluator": {
            "source": "scripts/run_generation_eval.py:evaluate_mode_with_hard_timeouts",
            "direct_cif_mode": "baseline",
            "bond_timeout_seconds": args.bond_timeout_seconds,
            "valid_timeout_seconds": args.valid_timeout_seconds,
            "match_timeout_seconds": args.match_timeout_seconds,
            "sample_timeout_seconds": args.sample_timeout_seconds,
            "max_match_sites": args.max_match_sites,
            "max_eval_sites": args.max_eval_sites,
            "eval_workers": args.eval_workers,
        },
        "overall": summary_by_k,
        "geometry_source_counts": dict(sorted(source_counts.items())),
        "top_errors": [{"error": k, "count": v} for k, v in render_errors.most_common(20)],
        "artifacts": {
            "summary_json": str(out_dir / "full_eval_summary.json"),
            "breakdown_csv": str(out_dir / "full_eval_breakdown.csv"),
            "metrics_jsonl": str(out_dir / "metrics" / "baseline_per_generation_metrics.jsonl"),
            "generated_cifs": str(out_dir / "generated_cifs"),
            "top20_predictions": str(out_dir / "top20_predictions.jsonl"),
            "failed_cases": str(out_dir / "failed_cases.jsonl"),
        },
    }


def failed_cases(records: list[dict[str, Any]], metrics: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_sample[int(metric["sample_index"])].append(metric)
    out: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        rows = sorted([r for r in by_sample.get(idx, []) if int(r["gen_index"]) < top_k], key=lambda r: int(r["gen_index"]))
        if any(r.get("match_ok") for r in rows):
            continue
        best_rms = [float(r["rms"]) for r in rows if r.get("rms") is not None]
        out.append(
            {
                "sample_index": idx,
                "sample_id": record["sample_id"],
                "formula": record["formula"],
                "sg": int(record["sg"]),
                "n_sites": int(record["n_sites"]),
                "num_elements": int(record["num_elements"]),
                "gt_wa_key": record["canonical_wa_key"],
                "gt_skeleton_key": record["canonical_skeleton_key"],
                "gt_wa_multiset_key": record_wa_multiset_key(record),
                "gt_skeleton_multiset_key": record_skeleton_multiset_key(record),
                "wa_hit_top20": any(str(r.get("wa_multiset_key")) == record_wa_multiset_key(record) for r in rows),
                "skeleton_hit_top20": any(
                    str(r.get("skeleton_multiset_key")) == record_skeleton_multiset_key(record) for r in rows
                ),
                "wa_hit_order_sensitive_top20": any(
                    str(r.get("canonical_wa_key")) == str(record["canonical_wa_key"]) for r in rows
                ),
                "skeleton_hit_order_sensitive_top20": any(
                    str(r.get("canonical_skeleton_key")) == str(record["canonical_skeleton_key"]) for r in rows
                ),
                "render_success_count": sum(1 for r in rows if r.get("render_success")),
                "readable_count": sum(1 for r in rows if r.get("pymatgen_readable")),
                "formula_ok_count": sum(1 for r in rows if r.get("formula_ok")),
                "atom_count_ok_count": sum(1 for r in rows if r.get("atom_count_ok")),
                "SG_ok_count": sum(1 for r in rows if r.get("space_group_ok")),
                "strict_valid_count": sum(1 for r in rows if r.get("strict_valid")),
                "best_rms_if_any": min(best_rms) if best_rms else None,
                "top_errors": list(Counter(str(r.get("error")) for r in rows if r.get("error")).most_common(5)),
                "top_predictions": [
                    {
                        "rank": int(r["gen_index"]) + 1,
                        "canonical_wa_key": r.get("canonical_wa_key"),
                        "canonical_skeleton_key": r.get("canonical_skeleton_key"),
                        "geometry_source": r.get("geometry_source"),
                        "readable": bool(r.get("pymatgen_readable")),
                        "formula_ok": bool(r.get("formula_ok")),
                        "atom_count_ok": bool(r.get("atom_count_ok")),
                        "SG_ok": bool(r.get("space_group_ok")),
                        "bond_length_score": r.get("bond_length_score"),
                        "valid": bool(r.get("valid")),
                        "strict_valid": bool(r.get("strict_valid")),
                    }
                    for r in rows[: min(20, len(rows))]
                ],
            }
        )
    return out


def run_mode(
    *,
    mode: str,
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    train_wa: dict[str, list[dict[str, Any]]],
    train_skel: dict[str, list[dict[str, Any]]],
    median_lattice_by_sg: dict[int, dict[str, float]],
    global_median_lattice: dict[str, float],
    engine: OrbitEngine,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = args.reports_root / MODE_TO_DIR[mode]
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped, meta_by_key = render_mode_generations(
        mode=mode,
        out_dir=out_dir,
        records=records,
        predictions=predictions,
        engine=engine,
        train_wa=train_wa,
        train_skel=train_skel,
        median_lattice_by_sg=median_lattice_by_sg,
        global_median_lattice=global_median_lattice,
        top_k=args.top_k,
    )
    eval_args = SimpleNamespace(
        eval_workers=args.eval_workers,
        bond_timeout_seconds=args.bond_timeout_seconds,
        valid_timeout_seconds=args.valid_timeout_seconds,
        match_timeout_seconds=args.match_timeout_seconds,
        sample_timeout_seconds=args.sample_timeout_seconds,
        max_match_sites=args.max_match_sites,
        max_eval_sites=args.max_eval_sites,
    )
    metrics = evaluate_mode_with_hard_timeouts(
        mode="baseline",
        case_payload=case_payload(records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    metrics = enrich_metrics(metrics, meta_by_key)
    if mode == "geometry_bottleneck":
        metrics = expand_geometry_bottleneck_metrics(metrics, args.top_k)
    metrics_path = out_dir / "metrics" / "baseline_per_generation_metrics.jsonl"
    write_jsonl(metrics_path, metrics)
    summary = make_summary(mode=mode, records=records, metrics=metrics, out_dir=out_dir, args=args)
    if mode == "geometry_bottleneck":
        summary["geometry_bottleneck_eval_note"] = (
            "rank1 GT-WA/current-geometry CIF was evaluated once per sample and copied across top5/top20 "
            "because all ranks are intentionally identical in this bottleneck diagnostic."
        )
    breakdown_rows: list[dict[str, Any]] = []
    for subset in ["overall", "n_sites>=6", "num_elements>=4"] + [f"SG={sg}" for sg in TARGET_SGS]:
        sample_indexes = subset_records(records, subset)
        for k in TOP_KS:
            breakdown_rows.append({"mode": mode, "subset": subset, **summarize_for_k(records, metrics, sample_indexes, k)})
    write_csv(out_dir / "full_eval_breakdown.csv", breakdown_rows)
    write_jsonl(out_dir / "failed_cases.jsonl", failed_cases(records, metrics, args.top_k))
    (out_dir / "full_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SymCIF-v4 end-to-end CIF reconstruction evaluation.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "reports" / "symcif_v4_hybrid_search" / "test_reranked_predictions.jsonl",
    )
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--reports-root", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--modes", nargs="+", default=list(MODE_TO_DIR), choices=list(MODE_TO_DIR))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval-workers", type=int, default=max(1, min(64, os.cpu_count() or 4)))
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--match-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-match-sites", type=int, default=300)
    parser.add_argument("--max-eval-sites", type=int, default=300)
    parser.add_argument("--test-limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_root.name != "structured_symcif_v4_reextracted":
        raise ValueError(f"refusing non-reextracted v4 data root: {args.data_root}")
    records = read_jsonl(args.data_root / "test.jsonl")
    if args.test_limit is not None:
        records = records[: max(0, int(args.test_limit))]
    train_records = read_jsonl(args.data_root / "train.jsonl")
    predictions = load_predictions(args.predictions, args.top_k)
    train_wa, train_skel, median_lattice_by_sg, global_median_lattice = build_train_indexes(train_records)
    engine = OrbitEngine.from_structured_root(args.lookup_json, args.data_root)
    summaries: dict[str, Any] = {}
    for mode in args.modes:
        print(f"[symcif-v4-e2e] running {mode}: {MODE_DESCRIPTIONS[mode]}", flush=True)
        summaries[mode] = run_mode(
            mode=mode,
            records=records,
            predictions=predictions,
            train_wa=train_wa,
            train_skel=train_skel,
            median_lattice_by_sg=median_lattice_by_sg,
            global_median_lattice=global_median_lattice,
            engine=engine,
            args=args,
        )
        print(json.dumps(summaries[mode]["overall"], indent=2, sort_keys=True), flush=True)
    combined_path = args.reports_root / "symcif_v4_full_eval_combined_summary.json"
    combined_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[symcif-v4-e2e] wrote combined summary -> {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
