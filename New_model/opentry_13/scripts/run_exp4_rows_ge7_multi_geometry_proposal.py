#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp4_rows_ge7_multi_geometry_proposal"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from run_exp2_predicted_skeleton_renderer_site_mapping import (  # noqa: E402
    candidate_rows_for_mapping,
    formula_counts,
    median_lattice_by_sg,
    read_jsonl,
    sample_id,
    write_json,
    write_jsonl,
)
from run_symcif_v4_geometry_model_eval import (  # noqa: E402
    GeometryModelRunner,
    deterministic_params,
    flexible_params_from_reference,
)
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
GEOM_CKPT = SYMCIF_ROOT / "runs" / "symcif_v4_geometry_model_no_oversampling" / "ckpt_best.pt"
EXP3_PROPOSALS = OUT_DIR / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP3_RESULT = RESULT_DIR / "experiment_3_rows_ge7_skeleton_proposer_validation_gate.json"
CRYSTALLM_VAL_LABELS = NEW_MODEL / "opentry_10" / "labels" / "mpts52_val_k50_candidate_labels.jsonl"

MATCHER_CONFIG = {"ltol": 0.3, "stol": 0.5, "angle_tol": 10}
BUDGETS = (1, 5, 20, 50)
REPORT_BUDGETS = (1, 5, 20)

_TARGET_CACHE: dict[str, Structure] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_or_replace_report(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker not in text:
        with REPORT_PATH.open("a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(replacement)
        return
    start = text.index(marker)
    next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
    if next_marker == -1:
        REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement, encoding="utf-8")
    else:
        REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement + text[next_marker:], encoding="utf-8")


def material_id_from_sample_id(sid: str) -> str:
    return str(sid).split("__")[-1]


def formula_l1(a: dict[str, Any], b: dict[str, Any]) -> float:
    aa = normalize_formula_counts(a)
    bb = normalize_formula_counts(b)
    keys = set(aa) | set(bb)
    return sum(abs(int(aa.get(k, 0)) - int(bb.get(k, 0))) for k in keys) / float(max(1, sum(aa.values()) + sum(bb.values())))


def formula_exact(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return normalize_formula_counts(a) == normalize_formula_counts(b)


def row_counter(rows: list[dict[str, Any]], include_element: bool) -> Counter[tuple[str, ...]]:
    out: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        if include_element:
            out[(str(row["orbit_id"]), str(row.get("element")))] += 1
        else:
            out[(str(row["orbit_id"]),)] += 1
    return out


def reference_score(target: dict[str, Any], rows: list[dict[str, Any]], ref: dict[str, Any], tier: str) -> float:
    tier_score = {
        "source": 5000.0,
        "same_skeleton": 3600.0,
        "same_sg_atom_count": 1800.0,
        "same_sg": 1000.0,
        "global": 0.0,
    }.get(tier, 0.0)
    target_counts = normalize_formula_counts(target["formula_counts"])
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
    target_atoms = sum(int(v) for v in target_counts.values())
    ref_atoms = sum(int(v) for v in ref_counts.values())
    return (
        tier_score
        + (350.0 if formula_exact(target_counts, ref_counts) else 0.0)
        + 45.0 * element_jaccard
        + 16.0 * exact_overlap
        + 6.0 * orbit_overlap
        - 35.0 * formula_l1(target_counts, ref_counts)
        - 1.2 * abs(int(target.get("n_sites", len(rows))) - int(ref.get("n_sites", len(ref["wa_table"]))))
        - 0.25 * abs(target_atoms - ref_atoms)
    )


def build_reference_indexes(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_sg_atom_count: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_skeleton: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        sg = int(record["sg"])
        atom_count = int(record.get("atom_count") or sum(int(v) for v in normalize_formula_counts(record["formula_counts"]).values()))
        by_sg[sg].append(record)
        by_sg_atom_count[(sg, atom_count)].append(record)
        by_skeleton[str(record.get("canonical_skeleton_key") or "")].append(record)
    return {
        "all": train_records,
        "by_sg": dict(by_sg),
        "by_sg_atom_count": dict(by_sg_atom_count),
        "by_skeleton": dict(by_skeleton),
    }


def ranked_references(
    *,
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    skeleton_key: str,
    source_record: dict[str, Any] | None,
    indexes: dict[str, Any],
    limit: int,
) -> list[tuple[str, dict[str, Any], float]]:
    target_atoms = sum(int(v) for v in normalize_formula_counts(target["formula_counts"]).values())
    pools: list[tuple[str, list[dict[str, Any]]]] = []
    if source_record is not None:
        pools.append(("source", [source_record]))
    pools.extend(
        [
            ("same_skeleton", list(indexes["by_skeleton"].get(str(skeleton_key), []))),
            ("same_sg_atom_count", list(indexes["by_sg_atom_count"].get((int(target["sg"]), int(target_atoms)), []))),
            ("same_sg", list(indexes["by_sg"].get(int(target["sg"]), []))),
        ]
    )
    seen: set[str] = set()
    scored: list[tuple[str, dict[str, Any], float]] = []
    for tier, refs in pools:
        for ref in refs:
            sid = str(ref["sample_id"])
            if sid in seen:
                continue
            seen.add(sid)
            scored.append((tier, ref, reference_score(target, rows, ref, tier)))
    if len(scored) < limit:
        for ref in indexes["all"]:
            sid = str(ref["sample_id"])
            if sid in seen:
                continue
            seen.add(sid)
            scored.append(("global", ref, reference_score(target, rows, ref, "global")))
            if len(scored) >= limit * 4:
                break
    scored.sort(key=lambda item: (-item[2], str(item[1].get("sample_id"))))
    return scored[:limit]


def deterministic_param_block(engine: OrbitEngine, rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    return {idx: deterministic_params(engine, str(row["orbit_id"]), idx) for idx, row in enumerate(rows)}


def inference_record_for_model(target: dict[str, Any], rows: list[dict[str, Any]], lattice: dict[str, float]) -> dict[str, Any]:
    counts = normalize_formula_counts(target["formula_counts"])
    return {
        "sample_id": target["sample_id"],
        "formula_counts": {str(k): int(v) for k, v in counts.items()},
        "sg": int(target["sg"]),
        "sg_symbol": str(target.get("sg_symbol") or ""),
        "n_sites": int(len(rows)),
        "num_elements": int(len(counts)),
        "lattice": {k: float(lattice[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")},
        "wa_table": rows,
    }


def geometry_options(
    *,
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    skeleton_key: str,
    source_record: dict[str, Any] | None,
    engine: OrbitEngine,
    model: GeometryModelRunner,
    indexes: dict[str, Any],
    median_by_sg: dict[int, dict[str, float]],
    global_median: dict[str, float],
    max_options: int,
) -> list[dict[str, Any]]:
    sg = int(target["sg"])
    median_lattice = dict(median_by_sg.get(sg) or global_median)
    neural_lattice: dict[str, float] | None = None
    neural_params: dict[int, dict[str, float]] | None = None
    try:
        neural_lattice, neural_params = model.predict(inference_record_for_model(target, rows, median_lattice), rows)
    except Exception:
        neural_lattice, neural_params = None, None

    options: list[dict[str, Any]] = []
    refs = ranked_references(
        target=target,
        rows=rows,
        skeleton_key=skeleton_key,
        source_record=source_record,
        indexes=indexes,
        limit=max(2, max_options),
    )
    for tier, ref, score in refs:
        try:
            params, fallback_count = flexible_params_from_reference(engine, rows, ref, neural_params=neural_params)
        except Exception:
            params, fallback_count = deterministic_param_block(engine, rows), len(rows)
        options.append(
            {
                "geometry_source": f"prototype_{tier}" + ("+fallback" if fallback_count else ""),
                "lattice": {k: float(ref["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")},
                "params": params,
                "reference_sample_id": str(ref["sample_id"]),
                "reference_score": float(score),
                "param_fallback_rows": int(fallback_count),
            }
        )
        if len(options) >= max_options:
            return dedup_options(options)[:max_options]

    if neural_lattice is not None and neural_params is not None:
        options.append(
            {
                "geometry_source": "neural_geometry_model",
                "lattice": neural_lattice,
                "params": neural_params,
                "reference_sample_id": None,
                "reference_score": None,
                "param_fallback_rows": 0,
            }
        )
    options.append(
        {
            "geometry_source": "train_sg_median_lattice_deterministic_params",
            "lattice": median_lattice,
            "params": deterministic_param_block(engine, rows),
            "reference_sample_id": None,
            "reference_score": None,
            "param_fallback_rows": len(rows),
        }
    )
    return dedup_options(options)[:max_options]


def dedup_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str | None]] = set()
    out: list[dict[str, Any]] = []
    for option in options:
        key = (str(option.get("geometry_source")), None if option.get("reference_sample_id") is None else str(option.get("reference_sample_id")))
        if key in seen:
            continue
        seen.add(key)
        out.append(option)
    return out


def render_candidate(
    *,
    engine: OrbitEngine,
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    option: dict[str, Any],
    data_name: str,
) -> tuple[str, dict[str, Any]]:
    params = option["params"]
    expanded = int(engine.expanded_atom_count(rows, params))
    counts = formula_counts(target)
    cif = engine.render_cif_from_wa_table(
        rows,
        lattice=option["lattice"],
        free_params_by_row=params,
        formula_counts=counts,
        sg=int(target["sg"]),
        sg_symbol=str(target.get("sg_symbol") or ""),
        data_name=data_name,
    )
    return cif, {
        "atom_count_after_expansion": expanded,
        "exact_cover_retained": expanded == sum(counts.values()),
    }


def structure_counts(structure: Structure) -> dict[str, int]:
    return {str(k): int(round(v)) for k, v in structure.composition.get_el_amt_dict().items()}


def min_pair_distance(structure: Structure) -> float | None:
    if len(structure) < 2:
        return None
    matrix = structure.distance_matrix
    vals = [float(matrix[i, j]) for i in range(len(structure)) for j in range(i + 1, len(structure))]
    return min(vals) if vals else None


def target_structure(path: str) -> Structure:
    if path not in _TARGET_CACHE:
        _TARGET_CACHE[path] = Structure.from_file(path)
    return _TARGET_CACHE[path]


def eval_sample(payload: dict[str, Any]) -> list[dict[str, Any]]:
    expected_counts = {str(k): int(v) for k, v in payload["formula_counts"].items()}
    target_atoms = int(payload["target_atom_count"])
    sg = int(payload["sg"])
    matcher = StructureMatcher(**MATCHER_CONFIG)
    try:
        target = target_structure(str(payload["target_cif_path"]))
    except Exception as exc:  # noqa: BLE001
        target = None
        target_error = f"{type(exc).__name__}: {exc}"
    else:
        target_error = None

    out: list[dict[str, Any]] = []
    for item in payload["candidates"]:
        row = {k: v for k, v in item.items() if k != "cif"}
        cif = str(item.get("cif") or "")
        if not cif:
            row.update(
                {
                    "parse_success": False,
                    "legal_cif": False,
                    "formula_ok": False,
                    "space_group_ok": False,
                    "site_count_ok": False,
                    "valid": False,
                    "collision_flag": None,
                    "min_pair_distance": None,
                    "volume_per_atom": None,
                    "match": False,
                    "rms": None,
                    "eval_error": row.get("render_error") or "empty_cif",
                    "target_error": target_error,
                }
            )
            out.append(row)
            continue
        try:
            generated = Structure.from_str(cif, fmt="cif")
            legal = True
            formula_ok = structure_counts(generated) == expected_counts
            site_count_ok = len(generated) == target_atoms
            try:
                detected_sg = int(SpacegroupAnalyzer(generated, symprec=0.1).get_space_group_number())
            except Exception:
                detected_sg = None
            sg_ok = detected_sg == sg
            min_dist = min_pair_distance(generated)
            collision = bool(min_dist is not None and min_dist < 0.5)
            volume_per_atom = float(generated.volume) / max(1, len(generated))
            volume_per_atom_ok = 2.0 <= volume_per_atom <= 120.0
            valid = bool(legal and formula_ok and site_count_ok and sg_ok and not collision and volume_per_atom_ok)
            match = False
            rms = None
            if target is not None and formula_ok and site_count_ok:
                match = bool(matcher.fit(target, generated))
                if match:
                    try:
                        raw = matcher.get_rms_dist(target, generated)
                        rms = float(raw[0] if isinstance(raw, (list, tuple)) else raw)
                    except Exception:
                        rms = None
            row.update(
                {
                    "parse_success": True,
                    "legal_cif": legal,
                    "formula_ok": formula_ok,
                    "space_group_ok": sg_ok,
                    "detected_sg": detected_sg,
                    "site_count_ok": site_count_ok,
                    "valid": valid,
                    "collision_flag": collision,
                    "too_short_distance": collision,
                    "min_pair_distance": min_dist,
                    "volume_per_atom": volume_per_atom,
                    "volume_per_atom_ok": volume_per_atom_ok,
                    "match": match,
                    "rms": rms,
                    "eval_error": None,
                    "target_error": target_error,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row.update(
                {
                    "parse_success": False,
                    "legal_cif": False,
                    "formula_ok": False,
                    "space_group_ok": False,
                    "site_count_ok": False,
                    "valid": False,
                    "collision_flag": None,
                    "min_pair_distance": None,
                    "volume_per_atom": None,
                    "match": False,
                    "rms": None,
                    "eval_error": f"{type(exc).__name__}: {exc}",
                    "target_error": target_error,
                }
            )
        out.append(row)
    return out


def structural_score(row: dict[str, Any]) -> float:
    score = 0.0
    score += 1200.0 * float(bool(row.get("valid")))
    score += 180.0 * float(bool(row.get("legal_cif")))
    score += 180.0 * float(bool(row.get("formula_ok")))
    score += 180.0 * float(bool(row.get("space_group_ok")))
    score += 160.0 * float(bool(row.get("site_count_ok")))
    score += 160.0 * float(bool(row.get("exact_cover_retained")))
    score += 110.0 * float(not bool(row.get("collision_flag")))
    score += 60.0 * float(bool(row.get("volume_per_atom_ok")))
    vpa = row.get("volume_per_atom")
    if vpa is not None:
        score -= min(80.0, abs(math.log(max(1.0e-6, float(vpa) / 18.0))) * 10.0)
    score += 0.05 * float(row.get("reference_score") or 0.0)
    score -= 0.2 * float(row.get("proposal_rank") or 999)
    score -= 0.02 * float(row.get("geometry_rank") or 999)
    return float(score)


def assign_structural_ranks(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    ranked: list[dict[str, Any]] = []
    for sid, sample_rows in by_sid.items():
        ordered = sorted(
            sample_rows,
            key=lambda r: (
                -structural_score(r),
                int(r.get("proposal_rank") or 999999),
                int(r.get("geometry_rank") or 999999),
                str(r.get("geometry_source") or ""),
                str(r.get("reference_sample_id") or ""),
            ),
        )[:top_k]
        for rank, row in enumerate(ordered, start=1):
            item = dict(row)
            item["rank"] = rank
            item["structural_selection_score"] = structural_score(row)
            ranked.append(item)
    ranked.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    return ranked


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = sorted({str(r["sample_id"]) for r in rows})
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    unique_skeletons = {(str(r["sample_id"]), int(r.get("proposal_rank") or 0), str(r.get("predicted_skeleton_key") or "")) for r in rows}
    out: dict[str, Any] = {
        "samples": len(sample_ids),
        "candidate_records": len(rows),
        "unique_predicted_skeletons": len(unique_skeletons),
        "mean_geometry_proposals_per_skeleton": ratio(len(rows), len(unique_skeletons)),
        "valid_rate": ratio(sum(bool(r.get("valid")) for r in rows), len(rows)),
        "formula_consistency": ratio(sum(bool(r.get("formula_ok")) for r in rows), len(rows)),
        "sg_consistency": ratio(sum(bool(r.get("space_group_ok")) for r in rows), len(rows)),
        "exact_cover_retained": ratio(sum(bool(r.get("exact_cover_retained")) for r in rows), len(rows)),
        "legal_cif_rate": ratio(sum(bool(r.get("legal_cif")) for r in rows), len(rows)),
        "collision_rate": ratio(sum(bool(r.get("collision_flag")) for r in rows), len(rows)),
        "volume_per_atom_ok_rate": ratio(sum(bool(r.get("volume_per_atom_ok")) for r in rows), len(rows)),
    }
    for k in BUDGETS:
        match_hits = valid_any = formula_any = sg_any = exact_any = skeleton_any = skeleton_and_match = 0
        rms_vals: list[float] = []
        for sid in sample_ids:
            top = sorted(by_sid[sid], key=lambda r: int(r.get("rank") or 10**9))[:k]
            match_any = any(bool(r.get("match")) for r in top)
            skel_any = any(bool(r.get("predicted_skeleton_hit")) for r in top)
            match_hits += int(match_any)
            valid_any += int(any(bool(r.get("valid")) for r in top))
            formula_any += int(any(bool(r.get("formula_ok")) for r in top))
            sg_any += int(any(bool(r.get("space_group_ok")) for r in top))
            exact_any += int(any(bool(r.get("exact_cover_retained")) for r in top))
            skeleton_any += int(skel_any)
            skeleton_and_match += int(skel_any and match_any)
            matched_rms = [float(r["rms"]) for r in top if bool(r.get("match")) and r.get("rms") is not None]
            if matched_rms:
                rms_vals.append(min(matched_rms))
        out[f"match@{k}"] = ratio(match_hits, len(sample_ids))
        out[f"RMSE@{k}"] = mean(rms_vals)
        out[f"valid_any@{k}"] = ratio(valid_any, len(sample_ids))
        out[f"formula_ok_any@{k}"] = ratio(formula_any, len(sample_ids))
        out[f"sg_ok_any@{k}"] = ratio(sg_any, len(sample_ids))
        out[f"exact_cover_any@{k}"] = ratio(exact_any, len(sample_ids))
        out[f"skeleton_hit_coverage@{k}"] = ratio(skeleton_any, len(sample_ids))
        out[f"skeleton_to_match_conversion@{k}"] = ratio(skeleton_and_match, skeleton_any)
    return out


def crystallm_val_baseline(val_repr: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows7_by_mid = {material_id_from_sample_id(sid): int(row.get("row_count") or 0) >= 7 for sid, row in val_repr.items()}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with CRYSTALLM_VAL_LABELS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                groups[str(row.get("material_id") or material_id_from_sample_id(str(row["sample_id"])))].append(row)
    out: dict[str, Any] = {"samples": len(groups), "rows_ge7_samples": sum(bool(rows7_by_mid.get(mid)) for mid in groups)}
    for k in BUDGETS:
        all_hits = rows_hits = all_valid = rows_valid = 0
        for mid, sample_rows in groups.items():
            top = [r for r in sample_rows if int(r.get("rank") or int(r.get("gen_index") or 0) + 1) <= k]
            hit = any(bool(r.get("match")) for r in top)
            valid = any(bool(r.get("valid")) for r in top)
            all_hits += int(hit)
            all_valid += int(valid)
            if rows7_by_mid.get(mid):
                rows_hits += int(hit)
                rows_valid += int(valid)
        out[f"match@{k}"] = ratio(all_hits, len(groups))
        out[f"valid_any@{k}"] = ratio(all_valid, len(groups))
        out[f"rows_ge7_match@{k}"] = ratio(rows_hits, out["rows_ge7_samples"])
        out[f"rows_ge7_valid_any@{k}"] = ratio(rows_valid, out["rows_ge7_samples"])
    return out


def pct(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:.3f}%"


def pp(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:+.3f}pp"


def triplet(d: dict[str, Any], prefix: str = "match") -> str:
    return " / ".join(pct(d.get(f"{prefix}@{k}")) for k in REPORT_BUDGETS)


def report_body(result: dict[str, Any]) -> str:
    overall = result["overall"]
    rows7 = result["rows_ge7"]
    baseline = result["baselines"]["crystallm_val_k50"]
    rows7_base = result["baselines"]["crystallm_rows_ge7_reference"]
    gate = result["gate"]
    return f"""## opentry_13 实验 4：rows>=7 multi-geometry proposal

结果文件：`model/New_model/opentry_13/results/experiment_4_rows_ge7_multi_geometry_proposal.json`
候选评估：`model/New_model/opentry_13/artifacts/exp4_rows_ge7_multi_geometry_proposal/evaluated_ranked_candidates.jsonl`

- 为什么做：同一个 predicted exact-cover skeleton 可能对应多个 lattice/free-parameter/site-mapping 解；实验 3 显示单一 learned repair 不能把 skeleton-hit 稳定转成 match，因此这里对每个 skeleton 生成多几何 proposal，并只用 inference-safe structural checks 排序。
- 核心假设：如果 geometry 多解覆盖是真瓶颈，rows>=7 top50 match coverage 和 skeleton-to-match conversion 应明显超过单一 hydrated/prototype 结果，并超过 CrystaLLM K50 rows>=7 top50 `18.431%` 至少 +5pp。
- 数据规模：scope=`{result['data_scale']['scope']}`；samples `{result['data_scale']['samples']}`，rows>=7 samples `{result['data_scale']['rows_ge7_samples']}`；candidate records `{result['data_scale']['candidate_records']}`；unique predicted skeletons `{result['data_scale']['unique_predicted_skeletons']}`；平均 geometry proposals/skeleton `{result['data_scale']['mean_geometry_proposals_per_skeleton']:.3f}`；top output K `{result['data_scale']['top_output_k']}`。
- baseline：validation CrystaLLM K50 overall match@1/5/20 = `{triplet(baseline)}`；exp3 记录的 CrystaLLM rows>=7 top1/top5/top20/top50 = `{pct(rows7_base.get('top1_match'))} / {pct(rows7_base.get('top5_match'))} / {pct(rows7_base.get('top20_match'))} / {pct(rows7_base.get('top50_match'))}`。
- 方法变化：对 exp3 predicted skeleton proposals 进行 composition exact-cover site mapping；每个 skeleton 的几何候选来自 train source prototype、同 skeleton / 同 SG+atom_count / 同 SG train prototype、旧 geometry model 初始化与 SG-median+deterministic fallback。排序只使用 legal CIF、formula、SG、site count、exact-cover、collision、volume/atom 和 train-reference score，不使用 match/RMSD/StructureMatcher label。
- 结果 overall：match@1/5/20 = `{triplet(overall)}`；RMSE@1/5/20 = `{overall.get('RMSE@1')} / {overall.get('RMSE@5')} / {overall.get('RMSE@20')}`；valid `{pct(overall.get('valid_rate'))}`，formula `{pct(overall.get('formula_consistency'))}`，SG `{pct(overall.get('sg_consistency'))}`，exact-cover `{pct(overall.get('exact_cover_retained'))}`，collision `{pct(overall.get('collision_rate'))}`。
- 结果 rows>=7：match@1/5/20/50 = `{pct(rows7.get('match@1'))} / {pct(rows7.get('match@5'))} / {pct(rows7.get('match@20'))} / {pct(rows7.get('match@50'))}`；RMSE@1/5/20/50 = `{rows7.get('RMSE@1')} / {rows7.get('RMSE@5')} / {rows7.get('RMSE@20')} / {rows7.get('RMSE@50')}`；skeleton-hit@50 `{pct(rows7.get('skeleton_hit_coverage@50'))}`；skeleton-to-match conversion@50 `{pct(rows7.get('skeleton_to_match_conversion@50'))}`。
- rows>=7 结构指标：valid `{pct(rows7.get('valid_rate'))}`，formula `{pct(rows7.get('formula_consistency'))}`，SG `{pct(rows7.get('sg_consistency'))}`，exact-cover `{pct(rows7.get('exact_cover_retained'))}`，collision `{pct(rows7.get('collision_rate'))}`，valid_any@50 `{pct(rows7.get('valid_any@50'))}`。
- gate 判定：passed={gate['passed']}；rows>=7 top50 delta vs CrystaLLM K50 = `{pp(gate.get('rows_ge7_top50_delta_vs_crystallm_k50'))}`；rows>=7 match@5 delta = `{pp(gate.get('rows_ge7_match5_delta_vs_crystallm'))}`；rows>=7 match@20 delta = `{pp(gate.get('rows_ge7_match20_delta_vs_crystallm'))}`；overall match@5/20 no-drop={gate.get('overall_match5_20_not_decreased')}；失败原因={gate['failure_reasons']}。
- 可信度：中等。该实验真实 render/parse/SG-detect/StructureMatcher，且排序不看 match；限制是 geometry proposals 仍主要来自 train prototype 和旧 GT-WA-style geometry initializer，没有训练 predicted-skeleton-noise repair。
- 和历史实验关系：实验 2 证明 selected renderer/site mapping 可过结构 gate；实验 3 证明旧 learned repair 不满足 predicted-skeleton-aware 条件且 conversion 崩掉；本实验测试 multi-geometry 是否能单独提高 rows>=7 hydrated match coverage。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=20)
    parser.add_argument("--geometry-options", type=int, default=8)
    parser.add_argument("--top-output-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scope", choices=("all", "rows_ge7"), default="all")
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    structured_train = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    structured_val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    exp3_result = read_json(EXP3_RESULT)
    train_median_by_sg, train_global_median = median_lattice_by_sg(list(structured_train.values()))
    ref_indexes = build_reference_indexes(list(structured_train.values()))
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in list(structured_val.values()) + list(structured_train.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    model = GeometryModelRunner(GEOM_CKPT, str(args.device), engine)

    selected_sids = [sid for sid in sorted(proposals) if sid in val_repr and sid in structured_val]
    if args.scope == "rows_ge7":
        selected_sids = [sid for sid in selected_sids if int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    sample_payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    mapping_failures: Counter[str] = Counter()
    geometry_source_counts: Counter[str] = Counter()
    unique_skeletons: set[tuple[str, int, str]] = set()

    for sample_idx, sid in enumerate(selected_sids, start=1):
        if sample_idx % 200 == 0:
            print(f"[exp4-multigeom] rendered payloads {sample_idx}/{len(selected_sids)}", flush=True)
        target_repr = val_repr[sid]
        target = structured_val[sid]
        target_counts = formula_counts(target)
        target_atoms = sum(target_counts.values())
        candidates: list[dict[str, Any]] = []
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_skeletons)]:
            proposal_rank = int(proposal.get("rank") or 0)
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            if source_repr is None:
                mapping_failures["missing_source_repr"] += 1
                continue
            rows, mapping_rule, mapping_error = candidate_rows_for_mapping(
                engine=engine,
                target_record=target,
                source_repr=source_repr,
                prefer_source_elements=True,
            )
            if rows is None:
                mapping_failures[str(mapping_error or mapping_rule)] += 1
                continue
            skeleton_key = str(proposal.get("skeleton_key") or "")
            unique_skeletons.add((sid, proposal_rank, skeleton_key))
            source_record = structured_train.get(source_id)
            try:
                options = geometry_options(
                    target=target,
                    rows=rows,
                    skeleton_key=skeleton_key,
                    source_record=source_record,
                    engine=engine,
                    model=model,
                    indexes=ref_indexes,
                    median_by_sg=train_median_by_sg,
                    global_median=train_global_median,
                    max_options=int(args.geometry_options),
                )
            except Exception as exc:  # noqa: BLE001
                mapping_failures[f"geometry_options_failed:{type(exc).__name__}"] += 1
                options = [
                    {
                        "geometry_source": "fallback_train_sg_median_deterministic",
                        "lattice": dict(train_median_by_sg.get(int(target["sg"])) or train_global_median),
                        "params": deterministic_param_block(engine, rows),
                        "reference_sample_id": None,
                        "reference_score": None,
                        "param_fallback_rows": len(rows),
                    }
                ]
            for geometry_rank, option in enumerate(options, start=1):
                base = {
                    "sample_id": sid,
                    "material_id": str(target.get("material_id") or material_id_from_sample_id(sid)),
                    "proposal_rank": proposal_rank,
                    "geometry_rank": geometry_rank,
                    "raw_generation_order": len(candidates) + 1,
                    "row_count": int(target_repr.get("row_count") or 0),
                    "sg": int(target["sg"]),
                    "formula_counts": target_counts,
                    "target_atom_count": int(target_atoms),
                    "source_sample_id": source_id,
                    "proposal_source": str(proposal.get("source") or ""),
                    "predicted_skeleton_key": skeleton_key,
                    "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                    "predicted_skeleton_hit": skeleton_key == str(target_repr.get("canonical_skeleton_key") or ""),
                    "candidate_row_count": len(rows),
                    "site_mapping_rule": mapping_rule,
                    "geometry_source": str(option.get("geometry_source") or ""),
                    "reference_sample_id": option.get("reference_sample_id"),
                    "reference_score": option.get("reference_score"),
                    "param_fallback_rows": option.get("param_fallback_rows"),
                }
                try:
                    cif, render_meta = render_candidate(
                        engine=engine,
                        target=target,
                        rows=rows,
                        option=option,
                        data_name=f"{sid}_sk{proposal_rank}_g{geometry_rank}",
                    )
                    row = dict(base)
                    row.update(render_meta)
                    row["render_success"] = True
                    row["render_error"] = None
                    row["cif"] = cif
                except Exception as exc:  # noqa: BLE001
                    row = dict(base)
                    row.update(
                        {
                            "render_success": False,
                            "render_error": f"{type(exc).__name__}: {exc}",
                            "atom_count_after_expansion": None,
                            "exact_cover_retained": False,
                            "cif": "",
                        }
                    )
                geometry_source_counts[str(row.get("geometry_source") or "")] += 1
                candidates.append(row)
                generation_meta.append({k: v for k, v in row.items() if k != "cif"})
        sample_payloads.append(
            {
                "sample_id": sid,
                "target_cif_path": str(target["source_path"]),
                "formula_counts": target_counts,
                "target_atom_count": int(target_atoms),
                "sg": int(target["sg"]),
                "candidates": candidates,
            }
        )

    write_jsonl(ARTIFACT_DIR / "generated_candidate_meta.jsonl", generation_meta)

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in sample_payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 100 == 0:
                print(f"[exp4-multigeom] evaluated samples {i}/{len(futures)}", flush=True)

    ranked = assign_structural_ranks(evaluated, int(args.top_output_k))
    write_jsonl(ARTIFACT_DIR / "evaluated_ranked_candidates.jsonl", ranked)

    rows_ge7_rows = [r for r in ranked if int(r.get("row_count") or 0) >= 7]
    rows_lt7_rows = [r for r in ranked if int(r.get("row_count") or 0) < 7]
    overall = summarize(ranked)
    rows_ge7 = summarize(rows_ge7_rows)
    rows_lt7 = summarize(rows_lt7_rows)
    baseline = crystallm_val_baseline(val_repr)
    rows_ge7_ref = exp3_result["baseline_reference"]["crystallm_k50_rows_ge7"]

    rows_ge7_top50_delta = (rows_ge7.get("match@50") or 0.0) - float(rows_ge7_ref["top50_match"])
    rows_ge7_match5_delta = (rows_ge7.get("match@5") or 0.0) - float(rows_ge7_ref["top5_match"])
    rows_ge7_match20_delta = (rows_ge7.get("match@20") or 0.0) - float(rows_ge7_ref["top20_match"])
    overall_no_drop = bool(
        (overall.get("match@5") or 0.0) >= float(baseline.get("match@5") or 0.0)
        or (overall.get("match@20") or 0.0) >= float(baseline.get("match@20") or 0.0)
    )
    failure_reasons: list[str] = []
    if rows_ge7_top50_delta < 0.05:
        failure_reasons.append(f"rows>=7 top50 delta {pp(rows_ge7_top50_delta)} < +5.000pp")
    if float(rows_ge7.get("skeleton_to_match_conversion@50") or 0.0) < 0.30:
        failure_reasons.append(f"rows>=7 skeleton-to-match conversion@50 {pct(rows_ge7.get('skeleton_to_match_conversion@50'))} < 30.000%")
    if rows_ge7_match5_delta < 0.05:
        failure_reasons.append(f"rows>=7 match@5 delta {pp(rows_ge7_match5_delta)} < +5.000pp")
    if rows_ge7_match20_delta < 0.05:
        failure_reasons.append(f"rows>=7 match@20 delta {pp(rows_ge7_match20_delta)} < +5.000pp")
    if not overall_no_drop:
        failure_reasons.append("overall match@5 and match@20 both decreased vs validation CrystaLLM K50 baseline")
    passed = not failure_reasons

    result = {
        "experiment": "opentry_13_exp4_rows_ge7_multi_geometry_proposal_validation",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "predicted_skeleton_multi_geometry_proposal",
            "inference_inputs": ["composition/formula", "GT-SG", "predicted exact-cover skeleton", "train geometry prototypes"],
            "ranking_features": ["legal_cif", "formula", "SG", "site_count", "exact_cover", "collision", "volume_per_atom", "train_reference_score"],
            "not_used_for_ranking": ["StructureMatcher match", "RMSD", "target CIF geometry", "GT-WA", "GT-skeleton", "test CIF", "RF/HGB/scorer"],
            "geometry_sources": dict(geometry_source_counts),
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "data_scale": {
            "scope": str(args.scope),
            "samples": overall["samples"],
            "rows_ge7_samples": rows_ge7["samples"],
            "rows_lt7_samples": rows_lt7["samples"],
            "candidate_records": len(ranked),
            "unique_predicted_skeletons": len(unique_skeletons),
            "mean_geometry_proposals_per_skeleton": float(ratio(len(ranked), len(unique_skeletons)) or 0.0),
            "top_skeletons": int(args.top_skeletons),
            "geometry_options": int(args.geometry_options),
            "top_output_k": int(args.top_output_k),
        },
        "mapping_failures": dict(mapping_failures),
        "overall": overall,
        "rows_ge7": rows_ge7,
        "rows_lt7": rows_lt7,
        "baselines": {
            "crystallm_val_k50": baseline,
            "crystallm_rows_ge7_reference": rows_ge7_ref,
        },
        "gate": {
            "passed": passed,
            "rows_ge7_top50_delta_vs_crystallm_k50": rows_ge7_top50_delta,
            "rows_ge7_match5_delta_vs_crystallm": rows_ge7_match5_delta,
            "rows_ge7_match20_delta_vs_crystallm": rows_ge7_match20_delta,
            "overall_match5_20_not_decreased": overall_no_drop,
            "failure_reasons": failure_reasons,
            "minimum_standard": {
                "rows_ge7_top50_match": float(rows_ge7_ref["top50_match"]) + 0.05,
                "rows_ge7_skeleton_to_match_conversion": 0.30,
                "rows_ge7_match5_delta": 0.05,
                "rows_ge7_match20_delta": 0.05,
            },
        },
        "decision": {
            "verdict": "pass" if passed else "fail_validation_gate",
            "reason": "Multi-geometry satisfies rows>=7 coverage/conversion gate." if passed else "Multi-geometry did not reach the required rows>=7 top50/conversion and K5/K20 lift gates.",
            "next_step": "Use this as the validation candidate for final ablation." if passed else "Do not run official; build predicted-skeleton-noise geometry training or stronger local collision/geometry optimization before rerunning.",
        },
        "artifacts": {
            "generated_candidate_meta": str(ARTIFACT_DIR / "generated_candidate_meta.jsonl"),
            "evaluated_ranked_candidates": str(ARTIFACT_DIR / "evaluated_ranked_candidates.jsonl"),
        },
        "runtime_seconds": time.time() - started,
    }
    out_path = RESULT_DIR / "experiment_4_rows_ge7_multi_geometry_proposal.json"
    write_json(out_path, result)
    append_or_replace_report("<!-- OPENTRY13_EXP4_ROWS_GE7_MULTI_GEOMETRY_PROPOSAL -->", report_body(result))
    print(json.dumps({"result": str(out_path), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
