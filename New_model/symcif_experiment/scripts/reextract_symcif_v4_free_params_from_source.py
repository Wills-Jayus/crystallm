#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.canonicalize import canonical_skeleton_key, canonical_wa_key, wa_table_from_structured
from symcif_v4.free_param_extractor import extract_free_params_detailed
from symcif_v4.orbit_engine import OrbitEngine


TOP_FAILED = {"225|24e", "189|3f", "189|3g", "193|6g", "216|24f"}
SOURCE_ROOT = Path(
    "/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/"
    "benchmarks_gt_from_prepare_csv_benchmark_symprec0p1"
)

_ENGINE: OrbitEngine | None = None
_SOURCE_ROOT: Path | None = None
_TOLERANCE: float = 1e-4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def init_worker(lookup_json: str, structured_root: str, source_root: str, tolerance: float) -> None:
    global _ENGINE, _SOURCE_ROOT, _TOLERANCE
    _ENGINE = OrbitEngine.from_structured_root(lookup_json, structured_root)
    _SOURCE_ROOT = Path(source_root)
    _TOLERANCE = float(tolerance)


def source_path_for_sample(sample_id: str, source_root: Path) -> Path | None:
    if "__" not in sample_id:
        return None
    prefix, stem = str(sample_id).split("__", 1)
    path = source_root / prefix / f"{stem}.cif"
    if path.exists():
        return path
    return None


def parse_cif_atom_site_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return rows
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        i += 1
        tags: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip().split()[0])
            i += 1
        if not tags or not any(tag.startswith("_atom_site_") for tag in tags):
            continue
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            if stripped.lower() == "loop_" or stripped.startswith("_") or stripped.lower().startswith("data_"):
                break
            try:
                parts = shlex.split(stripped, posix=True)
            except ValueError:
                parts = stripped.split()
            if len(parts) >= len(tags):
                rows.append({tag: parts[j] for j, tag in enumerate(tags)})
            i += 1
    return rows


def _float_cell(value: Any) -> float | None:
    try:
        return float(str(value).strip().split("(")[0]) % 1.0
    except Exception:
        return None


def atom_row_coord(row: dict[str, str]) -> tuple[float, float, float] | None:
    vals = [
        _float_cell(row.get("_atom_site_fract_x")),
        _float_cell(row.get("_atom_site_fract_y")),
        _float_cell(row.get("_atom_site_fract_z")),
    ]
    if any(v is None for v in vals):
        return None
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def atom_row_element(row: dict[str, str]) -> str:
    return str(row.get("_atom_site_type_symbol") or row.get("_atom_site_label") or "").strip()


def atom_row_multiplicity(row: dict[str, str]) -> int | None:
    value = row.get("_atom_site_symmetry_multiplicity")
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except Exception:
        return None


def _old_source_residual(engine: OrbitEngine, orbit_id: str, old_params: dict[str, float], source_coord: tuple[float, float, float]) -> float:
    orbit = engine.get_orbit_by_id(orbit_id)
    expanded = engine.expand_orbit(orbit, old_params)
    if not expanded:
        return 99.0
    def delta(a: float, b: float) -> float:
        return abs(((a - b + 0.5) % 1.0) - 0.5)
    return min((sum(delta(a, b) ** 2 for a, b in zip(source_coord, coord)) ** 0.5) for coord in expanded)


def _round_params(params: dict[str, Any] | None) -> dict[str, float]:
    params = params or {}
    return {key: round(float(value) % 1.0, 8) for key, value in sorted(params.items())}


def _param_pair_key(old_params: dict[str, Any], new_params: dict[str, Any]) -> str:
    return json.dumps({"old": _round_params(old_params), "new": _round_params(new_params)}, sort_keys=True)


def process_record(payload: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    split, input_index, row = payload
    if _ENGINE is None or _SOURCE_ROOT is None:
        raise RuntimeError("worker not initialized")
    engine = _ENGINE
    source_root = _SOURCE_ROOT
    source_path = source_path_for_sample(str(row["sample_id"]), source_root)
    atom_rows = parse_cif_atom_site_rows(source_path) if source_path else []
    wa_table, old_free_params = wa_table_from_structured(row, engine)
    sorted_assignments = sorted(row["assignment"], key=lambda x: int(x["site_order"]))

    source_candidates_by_key: dict[tuple[str, int], list[tuple[int, dict[str, str], tuple[float, float, float]]]] = defaultdict(list)
    fallback_by_element: dict[str, list[tuple[int, dict[str, str], tuple[float, float, float]]]] = defaultdict(list)
    for source_idx, atom_row in enumerate(atom_rows):
        coord = atom_row_coord(atom_row)
        if coord is None:
            continue
        element = atom_row_element(atom_row)
        multiplicity = atom_row_multiplicity(atom_row)
        fallback_by_element[element].append((source_idx, atom_row, coord))
        if multiplicity is not None:
            source_candidates_by_key[(element, multiplicity)].append((source_idx, atom_row, coord))

    used_source_rows: set[int] = set()
    row_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sample_all_extraction_success = True
    sample_all_expansion_ok = True

    for idx, assign in enumerate(sorted_assignments):
        orbit = engine.get_orbit_by_id(str(wa_table[idx]["orbit_id"]))
        old_params = dict(old_free_params.get(idx) or {})
        element = str(assign["element"])
        declared_multiplicity = int(assign["multiplicity"])
        sg_letter = f"{int(row['sg'])}|{declared_multiplicity}{assign['letter']}"
        candidates = list(source_candidates_by_key.get((element, declared_multiplicity), []))
        candidate_source_kind = "same_element_declared_multiplicity"
        if not candidates:
            candidates = list(fallback_by_element.get(element, []))
            candidate_source_kind = "same_element_fallback"

        options: list[tuple[tuple[Any, ...], int, dict[str, str], tuple[float, float, float], Any]] = []
        for source_idx, atom_row, source_coord in candidates:
            result = extract_free_params_detailed(source_coord, orbit, tolerance=_TOLERANCE)
            if result is None:
                continue
            old_residual = _old_source_residual(engine, str(wa_table[idx]["orbit_id"]), old_params, source_coord)
            score = (
                source_idx in used_source_rows,
                not result.expansion_ok,
                float(result.extraction_residual),
                float(old_residual),
                source_idx,
            )
            options.append((score, source_idx, atom_row, source_coord, result))

        options.sort(key=lambda x: x[0])
        chosen = options[0] if options else None
        if chosen is None:
            params = old_params
            expanded = engine.expand_orbit(orbit, params)
            report = {
                "site_order": int(assign["site_order"]),
                "element": element,
                "sg_letter": sg_letter,
                "orbit_id": orbit.canonical_orbit_id,
                "representative_expr": list(orbit.representative_expr),
                "old_free_params": _round_params(old_params),
                "new_free_params": _round_params(params),
                "source_coord": None,
                "mapped_coord": None,
                "extraction_residual": None,
                "extraction_method": "fallback_old_params_no_valid_source_extraction",
                "source_candidate_kind": candidate_source_kind,
                "source_candidate_count": len(candidates),
                "source_atom_row": None,
                "extraction_success": False,
                "expansion_count_after_reextract": len(expanded),
                "declared_multiplicity": declared_multiplicity,
                "expansion_ok": len(expanded) == declared_multiplicity,
            }
        else:
            _score, source_idx, atom_row, source_coord, result = chosen
            used_source_rows.add(source_idx)
            params = result.free_params
            report = {
                "site_order": int(assign["site_order"]),
                "element": element,
                "sg_letter": sg_letter,
                "orbit_id": orbit.canonical_orbit_id,
                "representative_expr": list(orbit.representative_expr),
                "old_free_params": _round_params(old_params),
                "new_free_params": _round_params(params),
                "source_coord": list(source_coord),
                "mapped_coord": list(result.mapped_coord),
                "representative_coord": list(result.representative_coord),
                "matched_operation": result.matched_operation,
                "extraction_residual": result.extraction_residual,
                "extraction_method": result.extraction_method,
                "source_candidate_kind": candidate_source_kind,
                "source_candidate_count": len(candidates),
                "source_atom_row": atom_row,
                "extraction_success": True,
                "expansion_count_after_reextract": result.expansion_count_after_reextract,
                "declared_multiplicity": declared_multiplicity,
                "expansion_ok": result.expansion_ok,
            }
        if not report["extraction_success"]:
            sample_all_extraction_success = False
        if not report["expansion_ok"]:
            sample_all_expansion_ok = False
        if not (report["extraction_success"] and report["expansion_ok"]):
            failures.append(
                {
                    "split": split,
                    "sample_id": row["sample_id"],
                    "sg": int(row["sg"]),
                    "formula": row["formula"],
                    **report,
                }
            )
        wa_table[idx]["free_params"] = params
        wa_table[idx]["old_free_params"] = _round_params(old_params)
        wa_table[idx]["new_free_params"] = _round_params(params)
        wa_table[idx]["source_coord"] = report["source_coord"]
        wa_table[idx]["mapped_coord"] = report["mapped_coord"]
        wa_table[idx]["extraction_residual"] = report["extraction_residual"]
        wa_table[idx]["extraction_method"] = report["extraction_method"]
        wa_table[idx]["expansion_count_after_reextract"] = report["expansion_count_after_reextract"]
        wa_table[idx]["declared_multiplicity"] = declared_multiplicity
        wa_table[idx]["expansion_ok"] = report["expansion_ok"]
        wa_table[idx]["extraction_success"] = report["extraction_success"]
        row_reports.append(report)

    out = {
        "id": row["sample_id"],
        "sample_id": row["sample_id"],
        "split": split,
        "input_index": input_index,
        "formula": row["formula"],
        "formula_counts": row["formula_counts"],
        "sg": int(row["sg"]),
        "sg_symbol": row["sg_symbol"],
        "wa_table": wa_table,
        "lattice": row["lattice"],
        "canonical_wa_key": canonical_wa_key(wa_table),
        "canonical_skeleton_key": canonical_skeleton_key(wa_table),
        "legacy_skeleton_template_key": row.get("skeleton_template_key"),
        "n_sites": int(row["n_sites"]),
        "num_elements": int(row["num_elements"]),
        "atom_count": int(row["atom_count"]),
        "source_path": None if source_path is None else str(source_path),
        "free_param_reextract_rows": row_reports,
        "free_param_reextract_all_success": sample_all_extraction_success,
        "row_expansion_all_ok": sample_all_expansion_ok,
    }
    return {"split": split, "input_index": input_index, "record": out, "failures": failures, "row_reports": row_reports}


def accumulate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    extraction_success = 0
    expansion_ok = 0
    sample_success = 0
    sample_expansion_ok = 0
    per_key: dict[str, Counter[str]] = defaultdict(Counter)
    old_new_hist: Counter[str] = Counter()
    top_old_new_hist: dict[str, Counter[str]] = {key: Counter() for key in TOP_FAILED}
    method_counter: Counter[str] = Counter()
    for result in results:
        record = result["record"]
        sample_success += int(bool(record["free_param_reextract_all_success"]))
        sample_expansion_ok += int(bool(record["row_expansion_all_ok"]))
        for row_report in result["row_reports"]:
            key = str(row_report["sg_letter"])
            total_rows += 1
            per_key[key]["rows"] += 1
            per_key[key]["extraction_success"] += int(bool(row_report["extraction_success"]))
            per_key[key]["expansion_ok"] += int(bool(row_report["expansion_ok"]))
            method_counter[str(row_report["extraction_method"])] += 1
            extraction_success += int(bool(row_report["extraction_success"]))
            expansion_ok += int(bool(row_report["expansion_ok"]))
            pair = _param_pair_key(row_report["old_free_params"], row_report["new_free_params"])
            old_new_hist[pair] += 1
            if key in top_old_new_hist:
                top_old_new_hist[key][pair] += 1
    per_sg_letter: dict[str, dict[str, Any]] = {}
    for key, counter in sorted(per_key.items()):
        rows = max(1, counter["rows"])
        per_sg_letter[key] = {
            "rows": counter["rows"],
            "extraction_success": counter["extraction_success"],
            "extraction_success_rate": counter["extraction_success"] / rows,
            "expansion_ok": counter["expansion_ok"],
            "expansion_ok_rate": counter["expansion_ok"] / rows,
        }
    return {
        "records": len(results),
        "rows": total_rows,
        "sample_reextract_all_success": sample_success,
        "sample_reextract_all_success_rate": sample_success / max(1, len(results)),
        "sample_row_expansion_all_ok": sample_expansion_ok,
        "sample_row_expansion_all_ok_rate": sample_expansion_ok / max(1, len(results)),
        "row_extraction_success": extraction_success,
        "row_extraction_success_rate": extraction_success / max(1, total_rows),
        "row_expansion_ok": expansion_ok,
        "row_expansion_ok_rate": expansion_ok / max(1, total_rows),
        "extraction_methods": dict(method_counter.most_common()),
        "per_sg_letter": per_sg_letter,
        "top_failed_sg_letters": {key: per_sg_letter.get(key, {}) for key in sorted(TOP_FAILED)},
        "top_old_new_param_pairs": [{"pair": k, "count": v} for k, v in old_new_hist.most_common(100)],
        "top_failed_old_new_param_pairs": {
            key: [{"pair": k, "count": v} for k, v in counter.most_common(25)]
            for key, counter in sorted(top_old_new_hist.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reextract SymCIF-v4 canonical free params from source CIF atom_site rows.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_reextract")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=max(1, min(64, os.cpu_count() or 1)))
    parser.add_argument("--max-records-per-split", type=int, default=None)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[tuple[str, int, dict[str, Any]]] = []
    split_counts: dict[str, int] = {}
    for split in args.splits:
        rows = read_jsonl(args.structured_root / f"{split}.jsonl")
        if args.max_records_per_split is not None:
            rows = rows[: int(args.max_records_per_split)]
        split_counts[split] = len(rows)
        payloads.extend((split, i, row) for i, row in enumerate(rows))

    if args.workers <= 1:
        init_worker(str(args.lookup_json), str(args.structured_root), str(args.source_root), args.tolerance)
        results = [process_record(payload) for payload in payloads]
    else:
        with mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(str(args.lookup_json), str(args.structured_root), str(args.source_root), args.tolerance),
        ) as pool:
            results = list(pool.imap_unordered(process_record, payloads, chunksize=16))

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for result in results:
        by_split[result["split"]].append(result)
        failures.extend(result["failures"])
    for split, items in by_split.items():
        items.sort(key=lambda r: int(r["input_index"]))
        write_jsonl(args.out_root / f"{split}.jsonl", [item["record"] for item in items])

    summary = accumulate_summary(results)
    summary["splits"] = {
        split: {
            **accumulate_summary(items),
            "input_records": split_counts.get(split, 0),
            "written_records": len(items),
        }
        for split, items in sorted(by_split.items())
    }
    summary["config"] = {
        "structured_root": str(args.structured_root),
        "lookup_json": str(args.lookup_json),
        "source_root": str(args.source_root),
        "out_root": str(args.out_root),
        "tolerance": args.tolerance,
        "workers": args.workers,
    }
    (args.out_dir / "free_param_reextract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.out_dir / "free_param_reextract_failures.jsonl", failures)
    print(json.dumps({
        "records": summary["records"],
        "rows": summary["rows"],
        "row_extraction_success_rate": summary["row_extraction_success_rate"],
        "row_expansion_ok_rate": summary["row_expansion_ok_rate"],
        "failures": len(failures),
        "out_root": str(args.out_root),
        "out_dir": str(args.out_dir),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
