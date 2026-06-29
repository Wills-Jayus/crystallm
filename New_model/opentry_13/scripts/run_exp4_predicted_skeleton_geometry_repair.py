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
for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.io.cif import CifParser  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from run_symcif_v4_geometry_model_eval import GeometryModelRunner  # noqa: E402
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp4_predicted_skeleton_geometry_repair"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
GEOM_CKPT = SYMCIF_ROOT / "runs" / "symcif_v4_geometry_model_no_oversampling" / "ckpt_best.pt"
EXP3_PROPOSALS = OUT_DIR / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP3_PER_SAMPLE = OUT_DIR / "artifacts" / "exp3_rows7_skeleton_proposer" / "per_sample_metrics.jsonl"

MATCHER_CONFIG = {"ltol": 0.3, "stol": 0.5, "angle_tol": 10}
BUDGETS = (1, 5, 20)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_report_once(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    if marker in text:
        return
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(marker)
        f.write("\n")
        f.write(body.rstrip())
        f.write("\n")


def sample_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("keys", {}).get("sample_id"))


def formula_counts(record: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in normalize_formula_counts(record["formula_counts"]).items()}


def formula_equal(a: dict[str, int], b: dict[str, int]) -> bool:
    return dict(sorted(a.items())) == dict(sorted(b.items()))


def target_formula_tuple(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in counts.items()))


def assign_elements_to_rows(
    *,
    target_counts: dict[str, int],
    source_rows: list[dict[str, Any]],
    source_counts: dict[str, int] | None = None,
) -> list[str] | None:
    if source_counts is not None and formula_equal(target_counts, source_counts):
        elements = [str(row.get("element")) for row in source_rows]
        if Counter(
            {
                element: sum(int(row.get("multiplicity") or 0) for row in source_rows if str(row.get("element")) == element)
                for element in set(elements)
            }
        ):
            per_element: Counter[str] = Counter()
            for row in source_rows:
                per_element[str(row.get("element"))] += int(row.get("multiplicity") or 0)
            if dict(per_element) == target_counts:
                return elements

    n = len(source_rows)
    mults = [int(row.get("multiplicity") or 0) for row in source_rows]
    order = sorted(range(n), key=lambda i: (-mults[i], str(source_rows[i].get("orbit_id")), i))
    remaining = dict(target_counts)
    assigned: list[str | None] = [None] * n

    def rec(pos: int) -> bool:
        if pos == len(order):
            return all(int(v) == 0 for v in remaining.values())
        idx = order[pos]
        mult = mults[idx]
        preferred = str(source_rows[idx].get("element"))
        elements = sorted(
            remaining,
            key=lambda e: (0 if e == preferred else 1, -remaining[e], e),
        )
        for element in elements:
            if remaining[element] < mult:
                continue
            assigned[idx] = element
            remaining[element] -= mult
            if rec(pos + 1):
                return True
            remaining[element] += mult
            assigned[idx] = None
        return False

    if rec(0):
        return [str(x) for x in assigned]
    return None


def enrich_rows_from_source(
    *,
    engine: OrbitEngine,
    source_record: dict[str, Any],
    target_record: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    source_rows: list[dict[str, Any]] = []
    for row in source_record["skeleton_sequence"]:
        orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
        source_rows.append(
            {
                "orbit_id": str(row["orbit_id"]),
                "letter": str(row.get("letter") or orbit.letter),
                "multiplicity": int(row.get("multiplicity") or orbit.multiplicity),
                "site_symmetry": str(row.get("site_symmetry") or orbit.site_symmetry),
                "free_symbols": list(orbit.free_symbols),
                "enumeration": row.get("enumeration", orbit.enumeration),
                "element": str(row.get("element") or "X"),
            }
        )
    target_counts = formula_counts(target_record)
    source_counts = formula_counts(source_record)
    elements = assign_elements_to_rows(target_counts=target_counts, source_rows=source_rows, source_counts=source_counts)
    if elements is None:
        return None, "exact_cover_assignment_failed"
    out: list[dict[str, Any]] = []
    for row, element in zip(source_rows, elements):
        item = dict(row)
        item["element"] = element
        out.append(item)
    per_element: Counter[str] = Counter()
    for row in out:
        per_element[str(row["element"])] += int(row["multiplicity"])
    if dict(per_element) != target_counts:
        return None, "formula_after_assignment_mismatch"
    return out, None


def render_candidate(
    *,
    engine: OrbitEngine,
    model: GeometryModelRunner,
    target_record: dict[str, Any],
    rows: list[dict[str, Any]],
    sample_id_: str,
    rank: int,
) -> tuple[str | None, dict[str, Any]]:
    started = time.monotonic()
    try:
        lattice, params = model.predict(target_record, rows)
        expanded = int(engine.expanded_atom_count(rows, params))
        target_atoms = sum(formula_counts(target_record).values())
        atom_count_ok = expanded == target_atoms
        cif = engine.render_cif_from_wa_table(
            rows,
            lattice=lattice,
            free_params_by_row=params,
            formula_counts=formula_counts(target_record),
            sg=int(target_record["sg"]),
            sg_symbol=str(target_record.get("sg_symbol") or ""),
            data_name=f"{sample_id_}_predskel_repair_rank{rank}",
        )
        return cif, {
            "render_success": True,
            "repair_geometry_source": "learned_geometry_model",
            "render_time_seconds": time.monotonic() - started,
            "atom_count_after_expansion": expanded,
            "atom_count_ok": atom_count_ok,
            "lattice": lattice,
        }
    except Exception as exc:  # noqa: BLE001
        return None, {
            "render_success": False,
            "repair_geometry_source": "learned_geometry_model",
            "render_time_seconds": time.monotonic() - started,
            "atom_count_after_expansion": None,
            "atom_count_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_structure_from_string(cif: str):
    parser = CifParser.from_string(cif)
    return parser.get_structures(primitive=False)[0]


def parse_structure_from_file(path: str):
    parser = CifParser(str(path))
    return parser.get_structures(primitive=False)[0]


def min_pair_distance(structure) -> float | None:
    if len(structure) < 2:
        return None
    matrix = structure.distance_matrix
    vals: list[float] = []
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            vals.append(float(matrix[i, j]))
    return min(vals) if vals else None


def composition_counts_from_structure(structure) -> dict[str, int]:
    comp = structure.composition.get_el_amt_dict()
    return {str(k): int(round(v)) for k, v in comp.items()}


def eval_one(payload: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in payload.items() if k != "cif"}
    cif = str(payload.get("cif") or "")
    if not cif:
        out.update(
            {
                "parse_success": False,
                "formula_ok": False,
                "space_group_ok": False,
                "valid": False,
                "match": False,
                "rms": None,
                "min_pair_distance": None,
                "collision_flag": None,
                "eval_error": "empty_cif",
            }
        )
        return out
    try:
        target = parse_structure_from_file(str(payload["target_cif_path"]))
        generated = parse_structure_from_string(cif)
        expected_counts = {str(k): int(v) for k, v in payload["formula_counts"].items()}
        formula_ok = composition_counts_from_structure(generated) == expected_counts
        try:
            detected_sg = int(SpacegroupAnalyzer(generated, symprec=0.1).get_space_group_number())
        except Exception:
            detected_sg = None
        space_group_ok = detected_sg == int(payload["sg"])
        min_dist = min_pair_distance(generated)
        collision = bool(min_dist is not None and min_dist < 0.5)
        valid = bool(formula_ok and space_group_ok and not collision)
        matcher = StructureMatcher(**MATCHER_CONFIG)
        match = bool(matcher.fit(target, generated))
        rms = None
        if match:
            try:
                rms_raw = matcher.get_rms_dist(target, generated)
                if isinstance(rms_raw, (tuple, list)):
                    rms = float(rms_raw[0])
                elif rms_raw is not None:
                    rms = float(rms_raw)
            except Exception:
                rms = None
        out.update(
            {
                "parse_success": True,
                "formula_ok": formula_ok,
                "space_group_ok": space_group_ok,
                "detected_sg": detected_sg,
                "valid": valid,
                "match": match,
                "rms": rms,
                "min_pair_distance": min_dist,
                "collision_flag": collision,
                "eval_error": None,
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        out.update(
            {
                "parse_success": False,
                "formula_ok": False,
                "space_group_ok": False,
                "valid": False,
                "match": False,
                "rms": None,
                "min_pair_distance": None,
                "collision_flag": None,
                "eval_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return out


def summarize_sample_rows(rows: list[dict[str, Any]], before_by_sid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample_ids = sorted({str(r["sample_id"]) for r in rows})
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    out: dict[str, Any] = {
        "samples": len(sample_ids),
        "candidate_records": len(rows),
        "render_success_rate": ratio(sum(bool(r.get("render_success")) for r in rows), len(rows)),
        "parse_success_rate": ratio(sum(bool(r.get("parse_success")) for r in rows), len(rows)),
        "formula_consistency_rate": ratio(sum(bool(r.get("formula_ok")) for r in rows), len(rows)),
        "sg_consistency_rate": ratio(sum(bool(r.get("space_group_ok")) for r in rows), len(rows)),
        "exact_cover_retained_rate": ratio(sum(bool(r.get("exact_cover_feasible")) for r in rows), len(rows)),
        "valid_rate": ratio(sum(bool(r.get("valid")) for r in rows), len(rows)),
        "collision_rate": ratio(sum(bool(r.get("collision_flag")) for r in rows), len(rows)),
    }
    for k in BUDGETS:
        before_hits = 0
        after_hits = 0
        conversions = 0
        before_neg = 0
        rms_vals: list[float] = []
        valid_any = formula_any = sg_any = exact_any = 0
        skeleton_any = 0
        skeleton_and_match = 0
        for sid in sample_ids:
            before_hit = bool(before_by_sid.get(sid, {}).get(f"hydrated_match@{k}"))
            top = sorted(by_sid.get(sid, []), key=lambda r: int(r.get("rank") or 999999))[:k]
            after_hit = any(bool(r.get("match")) for r in top)
            before_hits += int(before_hit)
            after_hits += int(after_hit)
            if not before_hit:
                before_neg += 1
                conversions += int(after_hit)
            valid_any += int(any(bool(r.get("valid")) for r in top))
            formula_any += int(any(bool(r.get("formula_ok")) for r in top))
            sg_any += int(any(bool(r.get("space_group_ok")) for r in top))
            exact_any += int(any(bool(r.get("exact_cover_feasible")) for r in top))
            skel = any(bool(r.get("predicted_skeleton_hit")) for r in top)
            skeleton_any += int(skel)
            skeleton_and_match += int(skel and after_hit)
            matched_rms = [float(r["rms"]) for r in top if bool(r.get("match")) and r.get("rms") is not None]
            if matched_rms:
                rms_vals.append(min(matched_rms))
        out[f"before_match@{k}"] = ratio(before_hits, len(sample_ids))
        out[f"after_match@{k}"] = ratio(after_hits, len(sample_ids))
        out[f"delta_match@{k}"] = (out[f"after_match@{k}"] or 0.0) - (out[f"before_match@{k}"] or 0.0)
        out[f"repair_conversion@{k}"] = ratio(conversions, before_neg)
        out[f"converted_samples@{k}"] = conversions
        out[f"before_negative_samples@{k}"] = before_neg
        out[f"RMSE@{k}"] = mean(rms_vals)
        out[f"valid_any@{k}"] = ratio(valid_any, len(sample_ids))
        out[f"formula_ok_any@{k}"] = ratio(formula_any, len(sample_ids))
        out[f"sg_ok_any@{k}"] = ratio(sg_any, len(sample_ids))
        out[f"exact_cover_any@{k}"] = ratio(exact_any, len(sample_ids))
        out[f"skeleton_hit_coverage@{k}"] = ratio(skeleton_any, len(sample_ids))
        out[f"skeleton_to_match_conversion@{k}"] = ratio(skeleton_and_match, skeleton_any)
    return out


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f}pp"


def match_triplet(d: dict[str, Any], prefix: str) -> str:
    return " / ".join(pct(d.get(f"{prefix}@{k}")) for k in (1, 5, 20))


def delta_triplet(d: dict[str, Any]) -> str:
    return " / ".join(pp(d.get(f"delta_match@{k}")) for k in (1, 5, 20))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-ge7-limit", type=int, default=384)
    parser.add_argument("--rows-lt7-limit", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    structured_val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    before_by_sid = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PER_SAMPLE)}

    rows_ge7_sids = [sid for sid, r in val_repr.items() if int(r.get("row_count") or 0) >= 7 and sid in structured_val and sid in proposals]
    rows_lt7_sids = [sid for sid, r in val_repr.items() if int(r.get("row_count") or 0) < 7 and sid in structured_val and sid in proposals]
    selected_sids = rows_ge7_sids[: int(args.rows_ge7_limit)] + rows_lt7_sids[: int(args.rows_lt7_limit)]

    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in structured_val.values()}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    model = GeometryModelRunner(GEOM_CKPT, str(args.device), engine)

    generated_rows: list[dict[str, Any]] = []
    eval_payloads: list[dict[str, Any]] = []
    mapping_failures: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    skeleton_hit_records = 0

    for sid in selected_sids:
        target_record = structured_val[sid]
        target_skel = str(val_repr[sid]["canonical_skeleton_key"])
        target_cif_path = str(target_record["source_path"])
        for rank, proposal in enumerate(proposals[sid].get("proposals", [])[: int(args.top_k)], start=1):
            source_id = str(proposal.get("source_sample_id") or "")
            source_record = train_repr.get(source_id)
            base_meta = {
                "sample_id": sid,
                "material_id": target_record.get("material_id") or sid.split("__")[-1],
                "rank": rank,
                "row_count": int(val_repr[sid].get("row_count") or target_record.get("n_sites") or 0),
                "sg": int(target_record["sg"]),
                "formula_counts": formula_counts(target_record),
                "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                "target_skeleton_key": target_skel,
                "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == target_skel,
                "proposal_source": str(proposal.get("source") or ""),
                "source_sample_id": source_id,
                "target_cif_path": target_cif_path,
            }
            source_counts[base_meta["proposal_source"]] += 1
            skeleton_hit_records += int(bool(base_meta["predicted_skeleton_hit"]))
            if source_record is None:
                mapping_failures["missing_source_record"] += 1
                row = dict(base_meta)
                row.update({"render_success": False, "exact_cover_feasible": False, "error": "missing_source_record", "cif": ""})
                generated_rows.append({k: v for k, v in row.items() if k != "cif"})
                eval_payloads.append(row)
                continue
            candidate_rows, err = enrich_rows_from_source(engine=engine, source_record=source_record, target_record=target_record)
            if candidate_rows is None:
                mapping_failures[str(err)] += 1
                row = dict(base_meta)
                row.update({"render_success": False, "exact_cover_feasible": False, "error": str(err), "cif": ""})
                generated_rows.append({k: v for k, v in row.items() if k != "cif"})
                eval_payloads.append(row)
                continue
            cif, render_meta = render_candidate(
                engine=engine,
                model=model,
                target_record=target_record,
                rows=candidate_rows,
                sample_id_=sid,
                rank=rank,
            )
            row = dict(base_meta)
            row.update(render_meta)
            row["exact_cover_feasible"] = True
            row["site_mapping_mode"] = "composition_exact_cover"
            row["candidate_row_count"] = len(candidate_rows)
            row["cif"] = cif or ""
            generated_rows.append({k: v for k, v in row.items() if k != "cif"})
            eval_payloads.append(row)

    write_jsonl(ARTIFACT_DIR / "rendered_candidate_meta.jsonl", generated_rows)

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_one, payload) for payload in eval_payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.append(fut.result())
            if i % 1000 == 0:
                print(f"[exp4] evaluated {i}/{len(futures)}", flush=True)
    evaluated.sort(key=lambda r: (str(r.get("sample_id")), int(r.get("rank") or 0)))
    write_jsonl(ARTIFACT_DIR / "evaluated_repair_candidates.jsonl", evaluated)

    rows_ge7_eval = [r for r in evaluated if int(r.get("row_count") or 0) >= 7]
    rows_lt7_eval = [r for r in evaluated if int(r.get("row_count") or 0) < 7]
    overall = summarize_sample_rows(evaluated, before_by_sid)
    rows_ge7 = summarize_sample_rows(rows_ge7_eval, before_by_sid)
    rows_lt7 = summarize_sample_rows(rows_lt7_eval, before_by_sid)

    result = {
        "experiment": "opentry_13_exp4_predicted_skeleton_learned_geometry_repair_validation",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val_subset",
        "method": {
            "name": "predicted_exact_cover_skeleton_plus_learned_geometry_repair",
            "inference_inputs": ["composition/formula", "GT-SG", "predicted train-derived skeleton", "composition exact-cover site mapping"],
            "geometry_model": str(GEOM_CKPT),
            "not_used_at_inference": [
                "GT-WA",
                "GT-skeleton",
                "match label",
                "RMSD",
                "StructureMatcher result",
                "official feedback",
                "RF/HGB/rerank",
                "threshold tuning",
            ],
            "repair_components": ["site mapping by exact-cover", "learned lattice prediction", "learned free-parameter prediction", "post-render collision/local geometry audit"],
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "data_scale": {
            "selected_samples": len(selected_sids),
            "selected_rows_ge7_samples": len([sid for sid in selected_sids if int(val_repr[sid].get("row_count") or 0) >= 7]),
            "selected_rows_lt7_samples": len([sid for sid in selected_sids if int(val_repr[sid].get("row_count") or 0) < 7]),
            "evaluated_samples_with_candidates": overall["samples"],
            "evaluated_rows_ge7_samples_with_candidates": rows_ge7["samples"],
            "evaluated_rows_lt7_samples_with_candidates": rows_lt7["samples"],
            "top_k": int(args.top_k),
            "candidate_records": len(evaluated),
            "predicted_skeleton_hit_candidate_records": skeleton_hit_records,
        },
        "mapping": {
            "failure_counts": dict(mapping_failures),
            "proposal_source_counts": dict(source_counts),
        },
        "overall": overall,
        "rows_ge7": rows_ge7,
        "rows_lt7": rows_lt7,
        "decision": {
            "validation_gate_pass": False,
            "reason": "Predicted-skeleton learned repair is evaluated without oracle input, but K5/K20 lift and repair conversion do not meet the +5pp main gate.",
            "next_step": "Improve geometry/local collision repair or train a geometry model directly on predicted skeleton/site-mapping noise before official evaluation.",
        },
        "elapsed_seconds": time.time() - start,
    }
    write_json(RESULT_DIR / "experiment_4_predicted_skeleton_geometry_repair.json", result)

    section = f"""## opentry_13 实验 4：predicted-skeleton learned geometry repair validation

结果文件：`model/New_model/opentry_13/results/experiment_4_predicted_skeleton_geometry_repair.json`  
候选评估：`model/New_model/opentry_13/artifacts/exp4_predicted_skeleton_geometry_repair/evaluated_repair_candidates.jsonl`

- 为什么做：GT-WA learned geometry repair 很强，但不能作为 inference 主结果。本实验把 repair 绑定到 exp3 的 predicted exact-cover skeleton，在固定 composition + GT-SG 条件下检查 lattice/free parameters/site mapping/local geometry 是否能把 skeleton-hit 转成 StructureMatcher match。
- 核心假设：如果当前瓶颈主要是 geometry，则在 predicted skeleton 已命中时，learned geometry repair 应带来 K5/K20 提升和非零 repair conversion；如果 conversion 仍低，说明 site mapping / local geometry / collision 仍未打通。
- 数据规模：validation subset 抽样 `{len(selected_sids)}` 样本，其中 rows>=7 `{result['data_scale']['selected_rows_ge7_samples']}`、rows<7 `{result['data_scale']['selected_rows_lt7_samples']}`；实际有可用 predicted-skeleton 候选并进入评估的样本 overall `{overall['samples']}`、rows>=7 `{rows_ge7['samples']}`；topK `{int(args.top_k)}`；candidate records `{len(evaluated)}`。StructureMatcher worker `{int(args.workers)}`，线程环境 OMP/MKL/OPENBLAS/NUMEXPR=1。
- baseline：before 使用 exp3 同一 predicted skeleton proposal 的 hydrated-existing-eval match；after 使用 learned geometry model 重新渲染后的 StructureMatcher。true official anchor 不参与本 validation repair gate。
- 方法变化：新增 composition exact-cover site mapping + learned lattice/free-parameter prediction；推理期不使用 GT-WA、GT-skeleton、match/RMSD/StructureMatcher label、official feedback、RF/HGB/rerank 或 threshold tuning。
- 结果 overall：before match@1/5/20 = `{match_triplet(overall, 'before_match')}`；after match@1/5/20 = `{match_triplet(overall, 'after_match')}`；delta = `{delta_triplet(overall)}`；repair conversion@1/5/20 = `{match_triplet(overall, 'repair_conversion')}`；valid rate `{pct(overall.get('valid_rate'))}`，formula consistency `{pct(overall.get('formula_consistency_rate'))}`，SG consistency `{pct(overall.get('sg_consistency_rate'))}`，exact-cover retained `{pct(overall.get('exact_cover_retained_rate'))}`。
- 结果 rows>=7：before match@1/5/20 = `{match_triplet(rows_ge7, 'before_match')}`；after match@1/5/20 = `{match_triplet(rows_ge7, 'after_match')}`；delta = `{delta_triplet(rows_ge7)}`；repair conversion@1/5/20 = `{match_triplet(rows_ge7, 'repair_conversion')}`；valid rate `{pct(rows_ge7.get('valid_rate'))}`，formula consistency `{pct(rows_ge7.get('formula_consistency_rate'))}`，SG consistency `{pct(rows_ge7.get('sg_consistency_rate'))}`，exact-cover retained `{pct(rows_ge7.get('exact_cover_retained_rate'))}`；skeleton-to-match conversion@20 `{pct(rows_ge7.get('skeleton_to_match_conversion@20'))}`。
- 可信度：中等。这是真 predicted-skeleton repair，不是 GT-WA oracle；但只是 validation subset，且 site mapping 是 composition exact-cover 的 deterministic mapping，未重新训练专门适配 predicted skeleton 的 geometry model。
- 和历史实验关系：opentry_12 deterministic repair conversion=0，GT-WA learned repair 很强；本实验把 learned repair 接到 predicted skeleton 上，验证 inference 链路仍不足。
- 最终判决：validation gate 未通过，不能进入 official，也不能作为主结果 claim。它保留为主方法候选的失败/诊断实验。
- 下一步：训练/微调 geometry repair 以适配 predicted skeleton 与 exact-cover site mapping 噪声，并加入可微或搜索式 local collision repair；在 validation gate 通过前不做 official。
"""
    append_report_once("<!-- OPENTRY13_EXP4_PREDICTED_SKELETON_GEOMETRY_REPAIR -->", section)
    print(RESULT_DIR / "experiment_4_predicted_skeleton_geometry_repair.json")


if __name__ == "__main__":
    main()
