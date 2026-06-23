#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sample_id_from_repr(row: dict[str, Any]) -> str:
    keys = row.get("keys") or {}
    return str(keys.get("sample_id") or row.get("sample_id") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a JSONL by train/val Wyckoff repr metadata conditions.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-name", default="filtered.jsonl")
    parser.add_argument("--min-row-count", type=int, default=0)
    parser.add_argument("--min-atom-count", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repr_rows = read_jsonl(args.repr_jsonl)
    meta_by_id = {sample_id_from_repr(row): row for row in repr_rows}
    selected: list[dict[str, Any]] = []
    missing = 0
    failed_condition = 0
    for row in read_jsonl(args.input_jsonl):
        sample_id = str(row.get("sample_id") or "")
        meta = meta_by_id.get(sample_id)
        if meta is None:
            missing += 1
            continue
        if int(meta.get("row_count", 0)) < int(args.min_row_count):
            failed_condition += 1
            continue
        if int(meta.get("atom_count", 0)) < int(args.min_atom_count):
            failed_condition += 1
            continue
        selected.append(row)
        if int(args.max_records) > 0 and len(selected) >= int(args.max_records):
            break

    out_path = out_dir / args.out_name
    write_jsonl(out_path, selected)
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "repr_jsonl": str(args.repr_jsonl),
        "out_jsonl": str(out_path),
        "min_row_count": int(args.min_row_count),
        "min_atom_count": int(args.min_atom_count),
        "max_records": int(args.max_records),
        "selected_records": len(selected),
        "missing_metadata_rows": missing,
        "failed_condition_rows_before_stop": failed_condition,
        "note": "row_count/atom_count are used only for train/val subset construction and reporting, not as model features.",
    }
    write_json(out_dir / "filter_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
