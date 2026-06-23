#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine, coord_key  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402


_ENGINE: OrbitEngine | None = None
_TRAIN_WA: dict[str, list[dict[str, Any]]] = {}
_TRAIN_SKEL: dict[str, list[dict[str, Any]]] = {}
_MEDIAN_LATTICE_BY_SG: dict[int, dict[str, float]] = {}
_GLOBAL_MEDIAN_LATTICE: dict[str, float] = {}
_TOP_K = 1
_AVOID_COLLISIONS = True


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def lattice_median(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("a", "b", "c", "alpha", "beta", "gamma")
    return {key: float(statistics.median(float(r["lattice"][key]) for r in records)) for key in keys}


def build_train_indexes(records: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[int, dict[str, float]], dict[str, float]]:
    wa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        wa[str(record["canonical_wa_key"])].append(record)
        skel[str(record["canonical_skeleton_key"])].append(record)
        by_sg[int(record["sg"])].append(record)
    return dict(wa), dict(skel), {sg: lattice_median(items) for sg, items in by_sg.items()}, lattice_median(records)


def init_worker(
    lookup_json: str,
    data_root: str,
    top_k: int,
    avoid_collisions: bool,
    train_wa: dict[str, list[dict[str, Any]]],
    train_skel: dict[str, list[dict[str, Any]]],
    median_lattice_by_sg: dict[int, dict[str, float]],
    global_median_lattice: dict[str, float],
) -> None:
    global _ENGINE, _TRAIN_WA, _TRAIN_SKEL, _MEDIAN_LATTICE_BY_SG, _GLOBAL_MEDIAN_LATTICE, _TOP_K, _AVOID_COLLISIONS
    _ENGINE = OrbitEngine.from_structured_root(lookup_json, data_root)
    _TRAIN_WA = train_wa
    _TRAIN_SKEL = train_skel
    _MEDIAN_LATTICE_BY_SG = median_lattice_by_sg
    _GLOBAL_MEDIAN_LATTICE = global_median_lattice
    _TOP_K = int(top_k)
    _AVOID_COLLISIONS = bool(avoid_collisions)


def deterministic_params(orbit: Any, row_index: int, attempt: int) -> dict[str, float]:
    bases = {"x": 0.173, "y": 0.271, "z": 0.389}
    symbol_offset = {"x": 0.137, "y": 0.191, "z": 0.223}
    out: dict[str, float] = {}
    for symbol in orbit.free_symbols:
        sym = str(symbol)
        base = bases.get(sym, 0.173)
        step = symbol_offset.get(sym, 0.157)
        out[sym] = (base + step * (row_index + 1) + 0.071 * attempt + 0.019 * (row_index + 1) * (attempt + 1)) % 1.0
    return out


def expanded_keys(orbit: Any, params: dict[str, float]) -> set[tuple[int, int, int]]:
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    return {coord_key(coord) for coord in _ENGINE.expand_orbit(orbit, params)}


def choose_noncolliding_params(orbit: Any, row_index: int, used: set[tuple[int, int, int]]) -> tuple[dict[str, float], bool]:
    if not _AVOID_COLLISIONS:
        return deterministic_params(orbit, row_index, 0), False
    best_params = deterministic_params(orbit, row_index, 0)
    best_overlap = len(expanded_keys(orbit, best_params) & used)
    for attempt in range(64):
        params = deterministic_params(orbit, row_index, attempt)
        keys = expanded_keys(orbit, params)
        overlap = len(keys & used)
        if overlap == 0:
            return params, False
        if overlap < best_overlap:
            best_overlap = overlap
            best_params = params
    return best_params, bool(best_overlap)


def params_from_reference(candidate_rows: list[dict[str, Any]], reference: dict[str, Any] | None, match_element: bool) -> tuple[dict[int, dict[str, float]], str, bool]:
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    params: dict[int, dict[str, float]] = {}
    source = "collision_aware_deterministic_free_params"
    any_collision_left = False
    remaining: dict[tuple[str, str] | tuple[str], list[dict[str, Any]]] = defaultdict(list)
    if reference is not None:
        for row in reference["wa_table"]:
            key = (str(row["orbit_id"]), str(row["element"])) if match_element else (str(row["orbit_id"]),)
            remaining[key].append(row)
        source = "same_wa_train" if match_element else "same_skeleton_train"
    used: set[tuple[int, int, int]] = set()
    for idx, row in enumerate(candidate_rows):
        orbit = _ENGINE.get_orbit_by_id(str(row["orbit_id"]))
        key = (str(row["orbit_id"]), str(row["element"])) if match_element else (str(row["orbit_id"]),)
        ref_rows = remaining.get(key) or []
        if ref_rows:
            ref = ref_rows.pop(0)
            chosen = dict(ref.get("free_params") or {})
        else:
            chosen, collision_left = choose_noncolliding_params(orbit, idx, used)
            any_collision_left = any_collision_left or collision_left
            if source != "collision_aware_deterministic_free_params":
                source = source + "+collision_aware_fallback"
        params[idx] = chosen
        used.update(expanded_keys(orbit, chosen))
    return params, source, any_collision_left


def render_candidate(record: dict[str, Any], candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    rows = [
        {
            "element": row["element"],
            "orbit_id": row["orbit_id"],
            "letter": row.get("letter"),
            "multiplicity": row.get("multiplicity"),
            "site_symmetry": row.get("site_symmetry"),
            "enumeration": row.get("enumeration"),
        }
        for row in candidate.get("rows", [])
    ]
    reference = None
    match_element = False
    same_wa = _TRAIN_WA.get(str(candidate.get("canonical_wa_key"))) or []
    same_skel = _TRAIN_SKEL.get(str(candidate.get("canonical_skeleton_key"))) or []
    if same_wa:
        reference = same_wa[0]
        match_element = True
    elif same_skel:
        reference = same_skel[0]
        match_element = False
    params, geom_source, collision_left = params_from_reference(rows, reference, match_element)
    lattice = (
        dict(reference["lattice"])
        if reference is not None
        else dict(_MEDIAN_LATTICE_BY_SG.get(int(record["sg"]), _GLOBAL_MEDIAN_LATTICE))
    )
    cif = _ENGINE.render_cif_from_wa_table(
        rows,
        lattice=lattice,
        free_params_by_row=params,
        formula_counts=record["formula_counts"],
        sg=int(record["sg"]),
        sg_symbol=str(record.get("sg_symbol") or ""),
        data_name=f"{record['sample_id']}_rank{rank}",
    )
    metric = validate_cif(cif, record["formula_counts"], int(record["sg"]))
    return {
        "sample_id": record["sample_id"],
        "rank": rank,
        "geometry_source": geom_source,
        "collision_left_after_param_search": bool(collision_left),
        "canonical_wa_key": candidate.get("canonical_wa_key"),
        "canonical_skeleton_key": candidate.get("canonical_skeleton_key"),
        "candidate_score": candidate.get("score"),
        "cif": cif,
        **metric,
    }


def process_payload(payload: tuple[dict[str, Any], dict[str, Any] | None]) -> list[dict[str, Any]]:
    record, prediction = payload
    if prediction is None:
        return []
    out: list[dict[str, Any]] = []
    for idx, candidate in enumerate(prediction.get("ranked_wa_candidates", [])[:_TOP_K], start=1):
        out.append(render_candidate(record, candidate, idx))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "readable": sum(bool(r["readable"]) for r in rows) / max(1, len(rows)),
        "formula_ok": sum(bool(r["formula_ok"]) for r in rows) / max(1, len(rows)),
        "atom_count_ok": sum(bool(r["atom_count_ok"]) for r in rows) / max(1, len(rows)),
        "sg_ok": sum(bool(r["sg_ok"]) for r in rows) / max(1, len(rows)),
        "composition_exact": sum(bool(r["composition_exact"]) for r in rows) / max(1, len(rows)),
        "collision_left": sum(bool(r.get("collision_left_after_param_search")) for r in rows) / max(1, len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render opentry_3 Wyckoff W/A predictions with collision-aware deterministic geometry.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write-cif-files", type=int, default=100)
    parser.add_argument("--no-avoid-collisions", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = out_dir / "rendered_cifs"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.data_root / f"{args.split}.jsonl")
    pred_by_id = {str(pred["sample_id"]): pred for pred in read_jsonl(args.predictions)}
    train_records = read_jsonl(args.data_root / "train.jsonl")
    train_wa, train_skel, median_lattice_by_sg, global_median_lattice = build_train_indexes(train_records)
    payloads = [(record, pred_by_id.get(str(record["sample_id"]))) for record in records if str(record["sample_id"]) in pred_by_id]
    avoid_collisions = not bool(args.no_avoid_collisions)
    if int(args.workers) <= 1:
        init_worker(str(args.lookup_json), str(args.data_root), args.top_k, avoid_collisions, train_wa, train_skel, median_lattice_by_sg, global_median_lattice)
        nested = [process_payload(payload) for payload in payloads]
    else:
        with mp.Pool(
            processes=int(args.workers),
            initializer=init_worker,
            initargs=(str(args.lookup_json), str(args.data_root), args.top_k, avoid_collisions, train_wa, train_skel, median_lattice_by_sg, global_median_lattice),
        ) as pool:
            nested = list(pool.imap_unordered(process_payload, payloads, chunksize=4))
    rows = [item for sub in nested for item in sub]
    rows.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    with (out_dir / "rendered_topk.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if i < int(args.write_cif_files):
                safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["sample_id"]))
                (rendered_dir / f"{safe_id}_rank{row['rank']}.cif").write_text(str(row["cif"]), encoding="utf-8")
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "split": args.split,
        "top_k": int(args.top_k),
        "samples_with_prediction_rows": len(payloads),
        "samples_with_rendered_candidates": len({r["sample_id"] for r in rows}),
        "rendered_rows": len(rows),
        "avoid_collisions": avoid_collisions,
        "overall_rows": summarize(rows),
        "rank1": summarize([r for r in rows if int(r["rank"]) == 1]),
        "rank_le_5": summarize([r for r in rows if int(r["rank"]) <= 5]),
        "rank_le_20": summarize([r for r in rows if int(r["rank"]) <= 20]),
        "rank_le_50": summarize([r for r in rows if int(r["rank"]) <= 50]),
    }
    (out_dir / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
