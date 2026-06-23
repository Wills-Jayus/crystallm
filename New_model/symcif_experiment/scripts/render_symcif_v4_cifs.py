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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine
from symcif_v4.render import render_record_cif
from symcif_v4.validation import validate_cif


_ENGINE: OrbitEngine | None = None
_TRAIN_WA: dict[str, list[dict[str, Any]]] = {}
_TRAIN_SKEL: dict[str, list[dict[str, Any]]] = {}
_MEDIAN_LATTICE_BY_SG: dict[int, dict[str, float]] = {}
_GLOBAL_MEDIAN_LATTICE: dict[str, float] = {}
_MODE = "gt_oracle"
_TOP_K = 1


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
    med_sg = {sg: lattice_median(items) for sg, items in by_sg.items()}
    return dict(wa), dict(skel), med_sg, lattice_median(records)


def init_worker(
    lookup_json: str,
    data_root: str,
    mode: str,
    top_k: int,
    train_wa: dict[str, list[dict[str, Any]]],
    train_skel: dict[str, list[dict[str, Any]]],
    median_lattice_by_sg: dict[int, dict[str, float]],
    global_median_lattice: dict[str, float],
) -> None:
    global _ENGINE, _TRAIN_WA, _TRAIN_SKEL, _MEDIAN_LATTICE_BY_SG, _GLOBAL_MEDIAN_LATTICE, _MODE, _TOP_K
    _ENGINE = OrbitEngine.from_structured_root(lookup_json, data_root)
    _TRAIN_WA = train_wa
    _TRAIN_SKEL = train_skel
    _MEDIAN_LATTICE_BY_SG = median_lattice_by_sg
    _GLOBAL_MEDIAN_LATTICE = global_median_lattice
    _MODE = mode
    _TOP_K = int(top_k)


def deterministic_params(orbit: Any, row_index: int) -> dict[str, float]:
    base = {"x": 0.173, "y": 0.271, "z": 0.389}
    out: dict[str, float] = {}
    for symbol in orbit.free_symbols:
        out[str(symbol)] = (base.get(str(symbol), 0.173) + 0.037 * row_index) % 1.0
    return out


def params_from_reference(candidate_rows: list[dict[str, Any]], reference: dict[str, Any] | None, match_element: bool) -> tuple[dict[int, dict[str, float]], str]:
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    params: dict[int, dict[str, float]] = {}
    source = "deterministic_free_params"
    remaining: dict[tuple[str, str] | tuple[str], list[dict[str, Any]]] = defaultdict(list)
    if reference is not None:
        for row in reference["wa_table"]:
            key = (str(row["orbit_id"]), str(row["element"])) if match_element else (str(row["orbit_id"]),)
            remaining[key].append(row)
        source = "same_wa_train" if match_element else "same_skeleton_train"
    for idx, row in enumerate(candidate_rows):
        orbit = _ENGINE.get_orbit_by_id(str(row["orbit_id"]))
        key = (str(row["orbit_id"]), str(row["element"])) if match_element else (str(row["orbit_id"]),)
        ref_rows = remaining.get(key) or []
        if ref_rows:
            ref = ref_rows.pop(0)
            params[idx] = dict(ref.get("free_params") or {})
        else:
            params[idx] = deterministic_params(orbit, idx)
            if source != "deterministic_free_params":
                source = source + "+deterministic_fallback"
    return params, source


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
    if _MODE == "retrieved_geometry":
        same_wa = _TRAIN_WA.get(str(candidate.get("canonical_wa_key"))) or []
        same_skel = _TRAIN_SKEL.get(str(candidate.get("canonical_skeleton_key"))) or []
        if same_wa:
            reference = same_wa[0]
            match_element = True
        elif same_skel:
            reference = same_skel[0]
            match_element = False
    params, geom_source = params_from_reference(rows, reference, match_element)
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
        "mode": _MODE,
        "geometry_source": geom_source,
        "canonical_wa_key": candidate.get("canonical_wa_key"),
        "canonical_skeleton_key": candidate.get("canonical_skeleton_key"),
        "cif": cif,
        **metric,
    }


def process_payload(payload: tuple[dict[str, Any], dict[str, Any] | None]) -> list[dict[str, Any]]:
    record, prediction = payload
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    if _MODE == "gt_oracle":
        cif = render_record_cif(record, _ENGINE, data_name=str(record["sample_id"]))
        return [
            {
                "sample_id": record["sample_id"],
                "rank": 1,
                "mode": _MODE,
                "geometry_source": "gt_wa_gt_free_params_gt_lattice",
                "canonical_wa_key": record["canonical_wa_key"],
                "canonical_skeleton_key": record["canonical_skeleton_key"],
                "cif": cif,
                **validate_cif(cif, record["formula_counts"], int(record["sg"])),
            }
        ]
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render SymCIF-v4 WA predictions through OrbitEngine.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_streaming_wa" / "test_ranked_wa_predictions.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", default="gt_oracle", choices=["gt_oracle", "retrieved_geometry", "default_safe"])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, min(64, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_streaming_wa" / "render_gt_oracle")
    parser.add_argument("--write-cif-files", type=int, default=50)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = args.out_dir / "rendered_cifs"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.data_root / f"{args.split}.jsonl")
    pred_by_id: dict[str, dict[str, Any]] = {}
    if args.mode != "gt_oracle":
        for pred in read_jsonl(args.predictions):
            pred_by_id[str(pred["sample_id"])] = pred
    train_records = read_jsonl(args.data_root / "train.jsonl")
    train_wa, train_skel, median_lattice_by_sg, global_median_lattice = build_train_indexes(train_records)
    payloads = [(record, pred_by_id.get(str(record["sample_id"]))) for record in records]
    if args.workers <= 1:
        init_worker(str(args.lookup_json), str(args.data_root), args.mode, args.top_k, train_wa, train_skel, median_lattice_by_sg, global_median_lattice)
        nested = [process_payload(payload) for payload in payloads]
    else:
        with mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(str(args.lookup_json), str(args.data_root), args.mode, args.top_k, train_wa, train_skel, median_lattice_by_sg, global_median_lattice),
        ) as pool:
            nested = list(pool.imap_unordered(process_payload, payloads, chunksize=4))
    rows = [item for sub in nested for item in sub]
    rows.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    with (args.out_dir / "rendered_test_topk.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if i < int(args.write_cif_files):
                safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["sample_id"]))
                (rendered_dir / f"{safe_id}_rank{row['rank']}.cif").write_text(str(row["cif"]), encoding="utf-8")
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "mode": args.mode,
        "split": args.split,
        "top_k": args.top_k,
        "samples": len(records),
        "overall_rows": summarize(rows),
        "rank1": summarize([r for r in rows if int(r["rank"]) == 1]),
        "gate3_reference": {
            "gt_oracle_pass": args.mode == "gt_oracle"
            and summarize(rows)["readable"] >= 0.99
            and summarize(rows)["formula_ok"] >= 0.99
            and summarize(rows)["sg_ok"] >= 0.98,
            "retrieved_formula_ok_ge_98": args.mode == "retrieved_geometry" and summarize(rows)["formula_ok"] >= 0.98,
            "retrieved_readable_ge_90": args.mode == "retrieved_geometry" and summarize(rows)["readable"] >= 0.90,
        },
    }
    (args.out_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
