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

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import evaluate_mode_with_hard_timeouts
from run_symcif_v4_full_pipeline_eval import (
    TOP_KS,
    TARGET_SGS,
    build_train_indexes,
    case_payload,
    candidate_skeleton_multiset_key,
    candidate_rows,
    candidate_wa_multiset_key,
    failed_cases,
    gt_candidate_from_record,
    load_predictions,
    read_jsonl,
    safe_filename,
    subset_records,
    summarize_for_k,
    write_jsonl,
)
from symcif_v4.formula import normalize_formula_counts
from symcif_v4.orbit_engine import OrbitEngine
from train_symcif_v4_geometry_model import GeometryNet, collate_geometry


MODE_DESCRIPTIONS = {
    "gtwa": "GT WA + geometry model/prototype candidates",
    "full": "predicted WA top20 + geometry model/prototype rank1",
}


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


def row_counter(rows: list[dict[str, Any]], include_element: bool) -> Counter[tuple[str, ...]]:
    out: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        if include_element:
            out[(str(row["orbit_id"]), str(row["element"]))] += 1
        else:
            out[(str(row["orbit_id"]),)] += 1
    return out


def formula_l1(a: dict[str, Any], b: dict[str, Any]) -> float:
    aa = normalize_formula_counts(a)
    bb = normalize_formula_counts(b)
    keys = set(aa) | set(bb)
    denom = max(1, sum(aa.values()) + sum(bb.values()))
    return sum(abs(int(aa.get(k, 0)) - int(bb.get(k, 0))) for k in keys) / float(denom)


def formula_exact(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return normalize_formula_counts(a) == normalize_formula_counts(b)


def reference_score(record: dict[str, Any], rows: list[dict[str, Any]], ref: dict[str, Any], tier: str) -> float:
    tier_score = {
        "same_wa": 4000.0,
        "same_skeleton": 3000.0,
        "same_sg": 1000.0,
        "global": 0.0,
    }.get(tier, 0.0)
    target_counts = normalize_formula_counts(record["formula_counts"])
    ref_counts = normalize_formula_counts(ref["formula_counts"])
    target_elements = set(target_counts)
    ref_elements = set(ref_counts)
    element_jaccard = len(target_elements & ref_elements) / max(1, len(target_elements | ref_elements))
    exact_rows = row_counter(rows, include_element=True)
    exact_ref = row_counter(ref["wa_table"], include_element=True)
    orbit_rows = row_counter(rows, include_element=False)
    orbit_ref = row_counter(ref["wa_table"], include_element=False)
    exact_overlap = sum(min(v, exact_ref.get(k, 0)) for k, v in exact_rows.items())
    orbit_overlap = sum(min(v, orbit_ref.get(k, 0)) for k, v in orbit_rows.items())
    target_atoms = sum(target_counts.values())
    ref_atoms = sum(ref_counts.values())
    return (
        tier_score
        + (250.0 if formula_exact(target_counts, ref_counts) else 0.0)
        + 40.0 * element_jaccard
        + 12.0 * exact_overlap
        + 4.0 * orbit_overlap
        - 20.0 * formula_l1(target_counts, ref_counts)
        - 0.6 * abs(int(record.get("n_sites", len(rows))) - int(ref.get("n_sites", len(ref["wa_table"]))))
        - 0.15 * abs(int(target_atoms) - int(ref_atoms))
    )


def build_reference_indexes(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    train_wa, train_skel, median_lattice_by_sg, global_median_lattice = build_train_indexes(train_records)
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        by_sg[int(record["sg"])].append(record)
    return {
        "train_wa": train_wa,
        "train_skel": train_skel,
        "by_sg": dict(by_sg),
        "all": train_records,
        "median_lattice_by_sg": median_lattice_by_sg,
        "global_median_lattice": global_median_lattice,
    }


def ranked_references(
    record: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    indexes: dict[str, Any],
    limit: int,
) -> list[tuple[str, dict[str, Any], float]]:
    pools: list[tuple[str, list[dict[str, Any]]]] = [
        ("same_wa", list(indexes["train_wa"].get(candidate_wa_multiset_key(candidate), []))),
        ("same_skeleton", list(indexes["train_skel"].get(candidate_skeleton_multiset_key(candidate), []))),
        ("same_sg", list(indexes["by_sg"].get(int(record["sg"]), []))),
    ]
    seen: set[str] = set()
    scored: list[tuple[str, dict[str, Any], float]] = []
    for tier, refs in pools:
        for ref in refs:
            sid = str(ref["sample_id"])
            if sid in seen:
                continue
            seen.add(sid)
            scored.append((tier, ref, reference_score(record, rows, ref, tier)))
    if len(scored) < limit:
        for ref in indexes["all"]:
            sid = str(ref["sample_id"])
            if sid in seen:
                continue
            seen.add(sid)
            scored.append(("global", ref, reference_score(record, rows, ref, "global")))
            if len(scored) >= limit * 4:
                break
    scored.sort(key=lambda item: (-item[2], str(item[1]["sample_id"])))
    return scored[:limit]


def deterministic_params(engine: OrbitEngine, orbit_id: str, row_index: int) -> dict[str, float]:
    orbit = engine.get_orbit_by_id(str(orbit_id))
    base = {"x": 0.173, "y": 0.271, "z": 0.389}
    return {str(symbol): (base.get(str(symbol), 0.173) + 0.037 * row_index) % 1.0 for symbol in orbit.free_symbols}


def validate_reference_row_params(
    *,
    engine: OrbitEngine,
    target_row: dict[str, Any],
    reference_row: dict[str, Any],
    allow_element_mismatch: bool = False,
) -> dict[str, float]:
    target_orbit = engine.get_orbit_by_id(str(target_row["orbit_id"]))
    ref_orbit = engine.get_orbit_by_id(str(reference_row["orbit_id"]))
    if not allow_element_mismatch and str(reference_row.get("element")) != str(target_row.get("element")):
        raise ValueError(f"reference element mismatch: {reference_row.get('element')} != {target_row.get('element')}")
    if str(reference_row.get("orbit_id")) != str(target_row.get("orbit_id")):
        raise ValueError(f"reference orbit mismatch: {reference_row.get('orbit_id')} != {target_row.get('orbit_id')}")
    target_mult = int(target_row.get("multiplicity") or target_orbit.multiplicity)
    ref_mult = int(reference_row.get("multiplicity") or ref_orbit.multiplicity)
    if ref_mult != target_mult:
        raise ValueError(f"reference multiplicity mismatch: {ref_mult} != {target_mult}")
    source_params = {str(k): float(v) % 1.0 for k, v in dict(reference_row.get("free_params") or {}).items()}
    expected = {str(symbol) for symbol in target_orbit.free_symbols}
    if set(source_params) != expected:
        raise ValueError(f"reference free-param symbols mismatch: {sorted(source_params)} != {sorted(expected)}")
    return source_params


def validate_render_inputs(
    *,
    engine: OrbitEngine,
    rows: list[dict[str, Any]],
    params: dict[int, dict[str, float]],
    record: dict[str, Any],
) -> None:
    per_element: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
        row_mult = int(row.get("multiplicity") or orbit.multiplicity)
        if row_mult != int(orbit.multiplicity):
            raise ValueError(f"candidate row {idx} multiplicity mismatch: {row_mult} != {orbit.multiplicity}")
        expected_symbols = {str(symbol) for symbol in orbit.free_symbols}
        row_params = {str(k): float(v) % 1.0 for k, v in dict(params.get(idx) or {}).items()}
        if set(row_params) != expected_symbols:
            raise ValueError(f"candidate row {idx} free-param symbols mismatch: {sorted(row_params)} != {sorted(expected_symbols)}")
        per_element[str(row["element"])] += row_mult
    formula_counts = normalize_formula_counts(record["formula_counts"])
    if {str(k): int(v) for k, v in per_element.items()} != {str(k): int(v) for k, v in formula_counts.items()}:
        raise ValueError(f"expanded row formula mismatch: {dict(per_element)} != {formula_counts}")
    expanded_count = int(engine.expanded_atom_count(rows, params))
    target = sum(int(v) for v in formula_counts.values())
    if expanded_count != target:
        raise ValueError(f"expanded atom count mismatch: {expanded_count} != {target}")


def flexible_params_from_reference(
    engine: OrbitEngine,
    candidate_rows_: list[dict[str, Any]],
    reference: dict[str, Any],
    neural_params: dict[int, dict[str, float]] | None = None,
) -> tuple[dict[int, dict[str, float]], int]:
    neural_params = neural_params or {}
    by_exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_orbit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference["wa_table"]:
        by_exact[(str(row["orbit_id"]), str(row["element"]))].append(row)
        by_orbit[str(row["orbit_id"])].append(row)
    used_reference_rows: set[int] = set()

    def pop_unused(pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        while pool:
            row = pool.pop(0)
            marker = id(row)
            if marker in used_reference_rows:
                continue
            used_reference_rows.add(marker)
            return row
        return None

    fallback_count = 0
    params: dict[int, dict[str, float]] = {}
    for idx, row in enumerate(candidate_rows_):
        orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
        ref_row = None
        allow_element_mismatch = False
        exact = by_exact.get((str(row["orbit_id"]), str(row["element"]))) or []
        if exact:
            ref_row = pop_unused(exact)
        if ref_row is None:
            same_orbit = by_orbit.get(str(row["orbit_id"])) or []
            ref_row = pop_unused(same_orbit)
            allow_element_mismatch = ref_row is not None
        row_params: dict[str, float] = {}
        source_params = (
            validate_reference_row_params(
                engine=engine,
                target_row=row,
                reference_row=ref_row,
                allow_element_mismatch=allow_element_mismatch,
            )
            if ref_row is not None
            else {}
        )
        for symbol in orbit.free_symbols:
            if str(symbol) in source_params:
                row_params[str(symbol)] = float(source_params[str(symbol)]) % 1.0
            elif idx in neural_params and str(symbol) in neural_params[idx]:
                row_params[str(symbol)] = float(neural_params[idx][str(symbol)]) % 1.0
                fallback_count += 1
            else:
                row_params[str(symbol)] = deterministic_params(engine, str(row["orbit_id"]), idx).get(str(symbol), 0.173)
                fallback_count += 1
        params[idx] = row_params
    return params, fallback_count


def enrich_candidate_rows(engine: OrbitEngine, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        orbit = engine.get_orbit_by_id(str(item["orbit_id"]))
        item["free_symbols"] = list(orbit.free_symbols)
        item["multiplicity"] = int(item.get("multiplicity") or orbit.multiplicity)
        item["letter"] = item.get("letter") or orbit.letter
        item["site_symmetry"] = item.get("site_symmetry") or orbit.site_symmetry
        item["enumeration"] = item.get("enumeration", orbit.enumeration)
        out.append(item)
    return out


def crystal_system(sg: int) -> str:
    sg = int(sg)
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


def postprocess_lattice(raw: list[float], sg: int) -> dict[str, float]:
    a = min(80.0, max(1.5, math.exp(float(raw[0]))))
    b = min(80.0, max(1.5, math.exp(float(raw[1]))))
    c = min(80.0, max(1.5, math.exp(float(raw[2]))))
    alpha = min(140.0, max(40.0, float(raw[3]) * 180.0))
    beta = min(140.0, max(40.0, float(raw[4]) * 180.0))
    gamma = min(140.0, max(40.0, float(raw[5]) * 180.0))
    system = crystal_system(sg)
    if system == "monoclinic":
        alpha, gamma = 90.0, 90.0
    elif system == "orthorhombic":
        alpha, beta, gamma = 90.0, 90.0, 90.0
    elif system == "tetragonal":
        b = a
        alpha, beta, gamma = 90.0, 90.0, 90.0
    elif system in {"trigonal", "hexagonal"}:
        b = a
        alpha, beta, gamma = 90.0, 90.0, 120.0
    elif system == "cubic":
        b = a
        c = a
        alpha, beta, gamma = 90.0, 90.0, 90.0
    return {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}


class GeometryModelRunner:
    def __init__(self, checkpoint: Path, device: str, engine: OrbitEngine) -> None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.vocabs: dict[str, dict[str, int]] = ckpt["vocabs"]
        self.lattice_mean = torch.tensor(ckpt["lattice_mean"], dtype=torch.float32)
        self.lattice_std = torch.tensor(ckpt["lattice_std"], dtype=torch.float32)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model = GeometryNet({k: len(v) for k, v in self.vocabs.items()})
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.engine = engine
        self.checkpoint = str(checkpoint)
        self.training_config = ckpt.get("config", {})

    @torch.no_grad()
    def predict(self, record: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
        enriched_rows = enrich_candidate_rows(self.engine, rows)
        temp = dict(record)
        temp["wa_table"] = enriched_rows
        raw_batch = collate_geometry(
            [temp],
            vocabs=self.vocabs,
            lattice_mean=self.lattice_mean,
            lattice_std=self.lattice_std,
        )
        batch = {k: v.to(self.device) for k, v in raw_batch.items()}
        lattice_pred, coord_pred = self.model(batch)
        raw_lattice = (lattice_pred[0].detach().cpu() * self.lattice_std + self.lattice_mean).tolist()
        lattice = postprocess_lattice(raw_lattice, int(record["sg"]))
        coords = coord_pred[0].detach().cpu()
        params: dict[int, dict[str, float]] = {}
        for idx, row in enumerate(enriched_rows):
            orbit = self.engine.get_orbit_by_id(str(row["orbit_id"]))
            params[idx] = {str(symbol): float(coords[idx, axis].item()) % 1.0 for axis, symbol in enumerate(("x", "y", "z")) if str(symbol) in orbit.free_symbols}
        return lattice, params


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


def geometry_options(
    *,
    record: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    engine: OrbitEngine,
    model: GeometryModelRunner,
    indexes: dict[str, Any],
    max_options: int,
    include_neural: bool,
    include_prototypes: bool,
) -> list[dict[str, Any]]:
    neural_lattice, neural_params = model.predict(record, rows)
    options: list[dict[str, Any]] = []
    if include_prototypes:
        refs = ranked_references(record, candidate, rows, indexes, limit=max(1, max_options))
        for tier, ref, score in refs:
            params, fallback_count = flexible_params_from_reference(engine, rows, ref, neural_params=neural_params)
            source = f"prototype_{tier}"
            if fallback_count:
                source += "+model_fallback"
            options.append(
                {
                    "lattice": dict(ref["lattice"]),
                    "params": params,
                    "geometry_source": source,
                    "reference_sample_id": ref["sample_id"],
                    "reference_score": float(score),
                }
            )
            if len(options) >= max_options:
                break
    if include_neural and len(options) < max_options:
        options.append(
            {
                "lattice": neural_lattice,
                "params": neural_params,
                "geometry_source": "neural_geometry_model",
                "reference_sample_id": None,
                "reference_score": None,
            }
        )
    if not options:
        options.append(
            {
                "lattice": neural_lattice,
                "params": neural_params,
                "geometry_source": "neural_geometry_model",
                "reference_sample_id": None,
                "reference_score": None,
            }
        )
    return options[:max_options]


def render_one(
    *,
    record: dict[str, Any],
    candidate: dict[str, Any],
    option: dict[str, Any],
    rank: int,
    engine: OrbitEngine,
) -> dict[str, Any]:
    started = time.monotonic()
    rows = candidate_rows(candidate)
    try:
        validate_render_inputs(engine=engine, rows=rows, params=option["params"], record=record)
        expanded_count, atom_count_ok = expanded_atom_count_ok(engine, rows, option["params"], record)
        cif = engine.render_cif_from_wa_table(
            rows,
            lattice=option["lattice"],
            free_params_by_row=option["params"],
            formula_counts=record["formula_counts"],
            sg=int(record["sg"]),
            sg_symbol=str(record.get("sg_symbol") or ""),
            data_name=f"{record['sample_id']}_rank{rank}",
        )
        return {
            "ok": True,
            "cif": cif,
            "render_time_seconds": time.monotonic() - started,
            "geometry_source": option.get("geometry_source"),
            "reference_sample_id": option.get("reference_sample_id"),
            "reference_score": option.get("reference_score"),
            "atom_count_after_expansion": expanded_count,
            "atom_count_ok": atom_count_ok,
            "canonical_wa_key": candidate.get("canonical_wa_key"),
            "canonical_skeleton_key": candidate.get("canonical_skeleton_key"),
            "wa_multiset_key": candidate_wa_multiset_key(candidate),
            "skeleton_multiset_key": candidate_skeleton_multiset_key(candidate),
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
            "geometry_source": f"{option.get('geometry_source')}:render_failed",
            "reference_sample_id": option.get("reference_sample_id"),
            "reference_score": option.get("reference_score"),
            "atom_count_after_expansion": None,
            "atom_count_ok": False,
            "canonical_wa_key": candidate.get("canonical_wa_key"),
            "canonical_skeleton_key": candidate.get("canonical_skeleton_key"),
            "wa_multiset_key": candidate_wa_multiset_key(candidate),
            "skeleton_multiset_key": candidate_skeleton_multiset_key(candidate),
            "source_labels": candidate.get("source_labels") or [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def candidates_for_mode(mode: str, record: dict[str, Any], prediction: dict[str, Any] | None, max_candidates: int) -> list[dict[str, Any]]:
    if mode == "gtwa":
        return [gt_candidate_from_record(record)]
    if prediction is None:
        return []
    return list(prediction.get("ranked_wa_candidates") or [])[: int(max_candidates)]


def first_float(candidate: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = candidate.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def candidate_score(candidate: dict[str, Any], candidate_index: int) -> float:
    value = first_float(candidate, ("final_rerank_score", "hybrid_score", "score", "old_score"))
    if value is not None:
        return value
    rank = first_float(candidate, ("final_rerank_rank", "policy_rank", "old_rank"))
    if rank is not None:
        return -float(rank)
    return -float(candidate_index)


def dedup_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate_wa_multiset_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def rerank_candidates_for_log(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = [(candidate_score(candidate, idx), idx, candidate) for idx, candidate in enumerate(candidates)]
    indexed.sort(key=lambda item: (-item[0], item[1], str(item[2].get("canonical_wa_key"))))
    out: list[dict[str, Any]] = []
    for rank, (score, raw_index, candidate) in enumerate(indexed, start=1):
        row = dict(candidate)
        row["raw_candidate_index"] = int(raw_index)
        row["candidate_log_rank"] = int(rank)
        row["candidate_log_score"] = float(score)
        row["wa_multiset_key"] = candidate_wa_multiset_key(row)
        row["skeleton_multiset_key"] = candidate_skeleton_multiset_key(row)
        row["source_labels"] = list(row.get("source_labels") or [])
        out.append(row)
    return out


def geometry_selection_score(candidate: dict[str, Any], option: dict[str, Any], candidate_index: int, variant_index: int) -> float:
    score = 10.0 * candidate_score(candidate, candidate_index)
    ref_score = option.get("reference_score")
    if ref_score is not None:
        try:
            score += float(ref_score)
        except Exception:
            pass
    else:
        score -= 50.0
    score -= 1e-3 * float(variant_index)
    score -= 1e-4 * float(candidate_index)
    return float(score)


def failed_geometry_option(exc: Exception, *, candidate_index: int | None = None, variant_index: int = 0) -> dict[str, Any]:
    return {
        "lattice": {},
        "params": {},
        "geometry_source": "geometry_options_failed",
        "reference_sample_id": None,
        "reference_score": None,
        "wa_candidate_index": candidate_index,
        "geometry_variant_index": int(variant_index),
        "selection_mode": "error",
        "selection_score": -1.0e12,
        "error": f"{type(exc).__name__}: {exc}",
    }


def render_mode_generations(
    *,
    mode: str,
    out_dir: Path,
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    engine: OrbitEngine,
    model: GeometryModelRunner,
    indexes: dict[str, Any],
    top_k: int,
    include_neural: bool,
    include_prototypes: bool,
    full_wa_candidates: int,
    full_max_variants_per_wa: int,
    full_selection_mode: str,
) -> tuple[dict[int, list[dict[str, Any]]], dict[tuple[int, int], dict[str, Any]]]:
    cif_dir = out_dir / "generated_cifs"
    gen_dir = out_dir / "generations"
    cif_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[dict[str, Any]]] = {}
    meta_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    generation_lines: list[dict[str, Any]] = []
    top20_lines: list[dict[str, Any]] = []
    candidates_raw_lines: list[dict[str, Any]] = []
    candidates_reranked_lines: list[dict[str, Any]] = []
    selected_top20_lines: list[dict[str, Any]] = []

    for sample_index, record in enumerate(records):
        if sample_index and sample_index % 50 == 0:
            print(f"[geometry-eval] rendered {sample_index}/{len(records)} {mode} samples", flush=True)
        prediction = predictions.get(str(record["sample_id"]))
        candidate_read_limit = top_k if mode == "gtwa" else max(int(top_k), int(full_wa_candidates))
        wa_candidates = candidates_for_mode(mode, record, prediction, candidate_read_limit)
        raw_candidate_count = len(list(prediction.get("ranked_wa_candidates") or [])) if prediction is not None else len(wa_candidates)
        deduped_candidates = dedup_candidates(wa_candidates)
        reranked_candidates = rerank_candidates_for_log(deduped_candidates)
        final_candidate_budget = 1 if mode == "gtwa" else max(1, int(full_wa_candidates))
        candidates_raw_lines.append(
            {
                "sample_index": sample_index,
                "sample_id": record["sample_id"],
                "mode": mode,
                "raw_candidate_limit": int(candidate_read_limit),
                "raw_candidate_count": int(raw_candidate_count),
                "loaded_candidate_count": int(len(wa_candidates)),
                "dedup_candidate_count": int(len(deduped_candidates)),
                "truncated_before_rerank": bool(raw_candidate_count > len(wa_candidates)),
                "candidates": wa_candidates,
            }
        )
        candidates_reranked_lines.append(
            {
                "sample_index": sample_index,
                "sample_id": record["sample_id"],
                "mode": mode,
                "raw_candidate_count": int(raw_candidate_count),
                "dedup_candidate_count": int(len(deduped_candidates)),
                "final_candidate_budget": int(final_candidate_budget),
                "candidates": reranked_candidates,
            }
        )
        sample_dir = cif_dir / safe_filename(str(record["sample_id"]))
        sample_dir.mkdir(parents=True, exist_ok=True)
        grouped[sample_index] = []
        entries: list[dict[str, Any]] = []
        output_rank = 0
        if mode == "gtwa":
            candidate = wa_candidates[0]
            rows = candidate_rows(candidate)
            try:
                options = geometry_options(
                    record=record,
                    candidate=candidate,
                    rows=rows,
                    engine=engine,
                    model=model,
                    indexes=indexes,
                    max_options=top_k,
                    include_neural=include_neural,
                    include_prototypes=include_prototypes,
                )
            except Exception as exc:  # noqa: BLE001
                options = [failed_geometry_option(exc)]
            candidate_options = [(candidate, opt) for opt in options]
        else:
            per_candidate_options: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
            for candidate_index, candidate in enumerate(reranked_candidates[: max(1, int(full_wa_candidates))]):
                rows = candidate_rows(candidate)
                try:
                    options = geometry_options(
                        record=record,
                        candidate=candidate,
                        rows=rows,
                        engine=engine,
                        model=model,
                        indexes=indexes,
                        max_options=max(1, int(full_max_variants_per_wa)),
                        include_neural=include_neural,
                        include_prototypes=include_prototypes,
                    )
                except Exception as exc:  # noqa: BLE001
                    options = [failed_geometry_option(exc, candidate_index=candidate_index)]
                per_candidate_options.append((candidate_index, candidate, options))
            scored_options: list[tuple[float, int, int, dict[str, Any], dict[str, Any]]] = []
            round_robin_options: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for variant_index in range(max(1, int(full_max_variants_per_wa))):
                for candidate_index, candidate, options in per_candidate_options:
                    if variant_index < len(options):
                        opt = dict(options[variant_index])
                        opt["geometry_variant_index"] = int(variant_index)
                        opt["wa_candidate_index"] = int(candidate_index)
                        opt["selection_mode"] = str(full_selection_mode)
                        opt["selection_score"] = geometry_selection_score(candidate, opt, candidate_index, variant_index)
                        round_robin_options.append((candidate, opt))
                        scored_options.append((float(opt["selection_score"]), candidate_index, variant_index, candidate, opt))
            if str(full_selection_mode) == "score":
                scored_options.sort(key=lambda item: (-item[0], item[1], item[2], str(item[3].get("canonical_wa_key"))))
                candidate_options = [(candidate, opt) for _score, _ci, _vi, candidate, opt in scored_options[:top_k]]
            else:
                candidate_options = round_robin_options[:top_k]
        for candidate, option in candidate_options[:top_k]:
            output_rank += 1
            gen_index = output_rank - 1
            rendered = render_one(record=record, candidate=candidate, option=option, rank=output_rank, engine=engine)
            cif_path = sample_dir / f"rank_{output_rank:03d}.cif"
            if rendered["ok"]:
                cif_path.write_text(str(rendered["cif"]), encoding="utf-8")
            meta = {
                "sample_index": sample_index,
                "sample_id": record["sample_id"],
                "rank": output_rank,
                "gen_index": gen_index,
                "render_success": bool(rendered["ok"]),
                "cif_path": str(cif_path) if rendered["ok"] else None,
                "geometry_source": rendered.get("geometry_source"),
                "reference_sample_id": rendered.get("reference_sample_id"),
                "reference_score": rendered.get("reference_score"),
                "atom_count_after_expansion": rendered.get("atom_count_after_expansion"),
                "atom_count_ok": bool(rendered.get("atom_count_ok")),
                "canonical_wa_key": rendered.get("canonical_wa_key"),
                "canonical_skeleton_key": rendered.get("canonical_skeleton_key"),
                "wa_multiset_key": rendered.get("wa_multiset_key"),
                "skeleton_multiset_key": rendered.get("skeleton_multiset_key"),
                "source_labels": rendered.get("source_labels") or [],
                "policy_rank": rendered.get("policy_rank"),
                "old_rank": rendered.get("old_rank"),
                "final_rerank_rank": rendered.get("final_rerank_rank"),
                "final_rerank_score": rendered.get("final_rerank_score"),
                "hybrid_score": rendered.get("hybrid_score"),
                "wa_candidate_index": option.get("wa_candidate_index"),
                "geometry_variant_index": option.get("geometry_variant_index"),
                "selection_mode": option.get("selection_mode"),
                "selection_score": option.get("selection_score"),
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
            entries.append({k: v for k, v in meta.items() if k not in {"sample_index", "sample_id", "gen_index"}})
        while output_rank < top_k:
            output_rank += 1
            gen_index = output_rank - 1
            meta = {
                "sample_index": sample_index,
                "sample_id": record["sample_id"],
                "rank": output_rank,
                "gen_index": gen_index,
                "render_success": False,
                "error": "missing_candidate",
                "canonical_wa_key": None,
                "canonical_skeleton_key": None,
                "wa_multiset_key": None,
                "skeleton_multiset_key": None,
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
            grouped[sample_index].append(gen)
            generation_lines.append(gen)
            meta_by_key[(sample_index, gen_index)] = meta
            entries.append({k: v for k, v in meta.items() if k not in {"sample_index", "sample_id", "gen_index"}})
        selected_top20_lines.append(
            {
                "sample_index": sample_index,
                "sample_id": record["sample_id"],
                "mode": mode,
                "raw_candidate_count": int(raw_candidate_count),
                "dedup_candidate_count": int(len(deduped_candidates)),
                "final_candidate_count": int(sum(1 for item in entries if item.get("canonical_wa_key"))),
                "truncated_before_rerank": bool(raw_candidate_count > len(wa_candidates)),
                "selected": entries,
            }
        )
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
                "gt_wa_multiset_key": candidate_wa_multiset_key(gt_candidate_from_record(record)),
                "gt_skeleton_multiset_key": candidate_skeleton_multiset_key(gt_candidate_from_record(record)),
                "raw_candidate_count": int(raw_candidate_count),
                "dedup_candidate_count": int(len(deduped_candidates)),
                "final_candidate_count": int(sum(1 for item in entries if item.get("canonical_wa_key"))),
                "truncated_before_rerank": bool(raw_candidate_count > len(wa_candidates)),
                "predictions": entries,
            }
        )
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
    write_jsonl(gen_dir / "baseline.jsonl", generation_lines)
    write_jsonl(out_dir / "candidates_raw.jsonl", candidates_raw_lines)
    write_jsonl(out_dir / "candidates_reranked.jsonl", candidates_reranked_lines)
    write_jsonl(out_dir / "selected_top20.jsonl", selected_top20_lines)
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


def benchmark_subset_records(records: list[dict[str, Any]], subset: str) -> set[int]:
    sg_counts = Counter(int(row["sg"]) for row in records)
    selected: set[int] = set()
    for idx, row in enumerate(records):
        n_sites = int(row.get("n_sites", 0))
        num_elements = int(row.get("num_elements", 0))
        max_mult = max((int(w.get("multiplicity", 1)) for w in row.get("wa_table", [])), default=1)
        if subset == "overall":
            selected.add(idx)
        elif subset == "n_sites>=6" and n_sites >= 6:
            selected.add(idx)
        elif subset == "n_sites>=12" and n_sites >= 12:
            selected.add(idx)
        elif subset == "n_sites>=20" and n_sites >= 20:
            selected.add(idx)
        elif subset == "num_elements>=4" and num_elements >= 4:
            selected.add(idx)
        elif subset == "rare_sg" and sg_counts[int(row["sg"])] <= 10:
            selected.add(idx)
        elif subset == "high_multiplicity_orbit" and max_mult >= 12:
            selected.add(idx)
        elif subset == "extraction_hard" and not bool(row.get("free_param_reextract_all_success", True)):
            selected.add(idx)
        elif subset.startswith("SG=") and int(row["sg"]) == int(subset.split("=", 1)[1]):
            selected.add(idx)
    return selected


def timeout_attempt_summary(records: list[dict[str, Any]], metrics: list[dict[str, Any]], k: int) -> dict[str, Any]:
    subset = [m for m in metrics if int(m.get("gen_index", 0)) < int(k)]
    attempts = max(1, len(records) * int(k))
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in subset:
        by_sample[int(metric["sample_index"])].append(metric)

    def sample_any(flag: str) -> int:
        return sum(1 for rows in by_sample.values() if any(bool(row.get(flag)) for row in rows))

    return {
        "k": int(k),
        "attempts": attempts,
        "eval_timeout_attempt_rate": sum(bool(m.get("eval_timeout")) for m in subset) / attempts,
        "parse_timeout_attempt_rate": sum(bool(m.get("parse_timeout")) for m in subset) / attempts,
        "sg_timeout_attempt_rate": sum(bool(m.get("sg_timeout")) for m in subset) / attempts,
        "matcher_timeout_attempt_rate": sum(bool(m.get("matcher_timeout")) for m in subset) / attempts,
        "eval_timeout_sample_rate": sample_any("eval_timeout") / max(1, len(records)),
        "parse_timeout_sample_rate": sample_any("parse_timeout") / max(1, len(records)),
        "sg_timeout_sample_rate": sample_any("sg_timeout") / max(1, len(records)),
        "matcher_timeout_sample_rate": sample_any("matcher_timeout") / max(1, len(records)),
    }


def timeout_breakdown_rows(records: list[dict[str, Any]], metrics: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    record_by_index = {idx: record for idx, record in enumerate(records)}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        if int(metric.get("gen_index", 0)) >= int(k):
            continue
        record = record_by_index.get(int(metric["sample_index"]))
        if record is None:
            continue
        groups[("sg", str(int(record["sg"])))].append(metric)
        groups[("n_sites", str(int(record.get("n_sites", 0))))].append(metric)
        groups[("num_elements", str(int(record.get("num_elements", 0))))].append(metric)
    rows: list[dict[str, Any]] = []
    for (group, value), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        attempts = max(1, len(items))
        sample_count = len({int(m["sample_index"]) for m in items})
        rows.append(
            {
                "k": int(k),
                "group": group,
                "value": value,
                "samples": sample_count,
                "attempts": attempts,
                "eval_timeout_attempt_rate": sum(bool(m.get("eval_timeout")) for m in items) / attempts,
                "parse_timeout_attempt_rate": sum(bool(m.get("parse_timeout")) for m in items) / attempts,
                "sg_timeout_attempt_rate": sum(bool(m.get("sg_timeout")) for m in items) / attempts,
                "matcher_timeout_attempt_rate": sum(bool(m.get("matcher_timeout")) for m in items) / attempts,
            }
        )
    return rows


def enrich_metrics(metrics: list[dict[str, Any]], meta_by_key: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
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


def make_summary(
    *,
    mode: str,
    records: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
    model: GeometryModelRunner,
) -> dict[str, Any]:
    breakdown_rows: list[dict[str, Any]] = []
    summary_by_k: dict[str, dict[str, Any]] = {}
    summary_top_ks = [
        int(k.strip())
        for k in str(getattr(args, "summary_top_ks", "1,5,20")).split(",")
        if k.strip()
    ]
    summary_top_ks = [k for k in summary_top_ks if 0 < int(k) <= int(args.top_k)]
    if not summary_top_ks:
        summary_top_ks = [int(args.top_k)]
    subsets = [
        "overall",
        "n_sites>=6",
        "n_sites>=12",
        "n_sites>=20",
        "num_elements>=4",
        "rare_sg",
        "high_multiplicity_orbit",
        "extraction_hard",
    ] + [f"SG={sg}" for sg in TARGET_SGS]
    for subset in subsets:
        sample_indexes = benchmark_subset_records(records, subset)
        for k in summary_top_ks:
            row = {"mode": mode, "subset": subset, **summarize_for_k(records, metrics, sample_indexes, k)}
            breakdown_rows.append(row)
            if subset == "overall":
                summary_by_k[f"top{k}"] = row
    write_csv(out_dir / "full_eval_breakdown.csv", breakdown_rows)
    timeout_summary = [timeout_attempt_summary(records, metrics, k) for k in summary_top_ks]
    timeout_rows: list[dict[str, Any]] = []
    for k in summary_top_ks:
        timeout_rows.extend(timeout_breakdown_rows(records, metrics, k))
    write_csv(out_dir / "timeout_breakdown.csv", timeout_rows)
    source_counts = Counter(str(m.get("geometry_source")) for m in metrics if m.get("geometry_source"))
    render_errors = Counter(str(m.get("error")) for m in metrics if m.get("error"))
    return {
        "mode": mode,
        "description": MODE_DESCRIPTIONS[mode],
        "data_root": str(args.data_root),
        "predictions": str(args.predictions) if args.predictions else None,
        "samples": len(records),
        "split": str(getattr(args, "split", "test")),
        "top_k": int(args.top_k),
        "summary_top_ks": summary_top_ks,
        "checkpoint": model.checkpoint,
        "training_config": model.training_config,
        "include_neural": bool(args.include_neural),
        "include_prototypes": bool(args.include_prototypes),
        "full_wa_candidates": int(args.full_wa_candidates),
        "full_max_variants_per_wa": int(args.full_max_variants_per_wa),
        "full_selection_mode": str(args.full_selection_mode),
        "evaluator": {
            "source": "scripts/run_generation_eval.py:evaluate_mode_with_hard_timeouts",
            "direct_cif_mode": "baseline",
            "bond_timeout_seconds": args.bond_timeout_seconds,
            "parse_timeout_seconds": args.parse_timeout_seconds,
            "sg_timeout_seconds": args.sg_timeout_seconds,
            "valid_timeout_seconds": args.valid_timeout_seconds,
            "match_timeout_seconds": args.match_timeout_seconds,
            "sample_timeout_seconds": args.sample_timeout_seconds,
            "max_match_sites": args.max_match_sites,
            "max_eval_sites": args.max_eval_sites,
            "eval_workers": args.eval_workers,
        },
        "overall": summary_by_k,
        "throughput_timeouts": timeout_summary,
        "geometry_source_counts": dict(sorted(source_counts.items())),
        "top_errors": [{"error": k, "count": v} for k, v in render_errors.most_common(20)],
        "artifacts": {
            "summary_json": str(out_dir / "eval_summary.json"),
            "full_eval_summary_json": str(out_dir / "full_eval_summary.json"),
            "breakdown_csv": str(out_dir / "full_eval_breakdown.csv"),
            "timeout_breakdown_csv": str(out_dir / "timeout_breakdown.csv"),
            "metrics_jsonl": str(out_dir / "metrics" / "baseline_per_generation_metrics.jsonl"),
            "generated_cifs": str(out_dir / "generated_cifs"),
            "candidates_raw": str(out_dir / "candidates_raw.jsonl"),
            "candidates_reranked": str(out_dir / "candidates_reranked.jsonl"),
            "selected_top20": str(out_dir / "selected_top20.jsonl"),
            "top20_predictions": str(out_dir / "top20_predictions.jsonl"),
            "failed_cases": str(out_dir / "failed_cases.jsonl"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SymCIF-v4 geometry model with same evaluator.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_hybrid_search" / "test_reranked_predictions.jsonl")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("gtwa", "full"), required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval-workers", type=int, default=max(1, min(64, os.cpu_count() or 4)))
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--match-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-match-sites", type=int, default=300)
    parser.add_argument("--max-eval-sites", type=int, default=300)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include-neural", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-prototypes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-wa-candidates", type=int, default=20)
    parser.add_argument("--full-max-variants-per-wa", type=int, default=1)
    parser.add_argument("--full-selection-mode", choices=("round_robin", "score"), default="round_robin")
    parser.add_argument("--summary-top-ks", type=str, default="1,5,20")
    parser.add_argument("--description", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.data_root / f"{args.split}.jsonl")
    if args.test_limit is not None:
        records = records[: max(0, int(args.test_limit))]
    train_records = read_jsonl(args.data_root / "train.jsonl")
    prediction_top_k = max(int(args.top_k), int(args.full_wa_candidates))
    predictions = load_predictions(args.predictions, prediction_top_k) if args.mode == "full" else {}
    indexes = build_reference_indexes(train_records)
    engine = OrbitEngine.from_structured_root(args.lookup_json, args.data_root)
    model = GeometryModelRunner(args.checkpoint, args.device, engine)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "samples": len(records),
                "checkpoint": str(args.checkpoint),
                "device": str(model.device),
                "include_neural": args.include_neural,
                "include_prototypes": args.include_prototypes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    grouped, meta_by_key = render_mode_generations(
        mode=args.mode,
        out_dir=args.out_dir,
        records=records,
        predictions=predictions,
        engine=engine,
        model=model,
        indexes=indexes,
        top_k=args.top_k,
        include_neural=args.include_neural,
        include_prototypes=args.include_prototypes,
        full_wa_candidates=args.full_wa_candidates,
        full_max_variants_per_wa=args.full_max_variants_per_wa,
        full_selection_mode=args.full_selection_mode,
    )
    eval_args = SimpleNamespace(
        eval_workers=args.eval_workers,
        bond_timeout_seconds=args.bond_timeout_seconds,
        parse_timeout_seconds=args.parse_timeout_seconds,
        sg_timeout_seconds=args.sg_timeout_seconds,
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
    metrics_path = args.out_dir / "metrics" / "baseline_per_generation_metrics.jsonl"
    write_jsonl(metrics_path, metrics)
    write_jsonl(args.out_dir / "failed_cases.jsonl", failed_cases(records, metrics, args.top_k))
    summary = make_summary(mode=args.mode, records=records, metrics=metrics, out_dir=args.out_dir, args=args, model=model)
    (args.out_dir / "full_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
