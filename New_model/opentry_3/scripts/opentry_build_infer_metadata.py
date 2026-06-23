#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_wyckoff_sequence_models import ensure_under_opentry, write_json, write_jsonl


def read_source_rows(path: Path, max_records: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            rows.append(
                {
                    "keys": {
                        "sample_id": sample_id,
                        "material_id": row.get("material_id"),
                        "id": row.get("id") or sample_id,
                        "input_index": row.get("input_index"),
                    },
                    "sg": int(row.get("sg", 0) or 0),
                    "atom_count": int(row.get("atom_count", 0) or 0),
                    "formula_counts": dict(row.get("formula_counts") or {}),
                    "split": row.get("split"),
                }
            )
            if max_records is not None and len(rows) >= int(max_records):
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GT-free formula/SG inference metadata from structured split JSONL.")
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = int(args.max_records) if int(args.max_records) > 0 else None
    rows = read_source_rows(args.source_jsonl, max_records)
    out_path = out_dir / f"{args.split_name}.jsonl"
    write_jsonl(out_path, rows)
    summary = {
        "source_jsonl": str(args.source_jsonl),
        "out_jsonl": str(out_path),
        "records": len(rows),
        "split_name": str(args.split_name),
        "field_policy": "Only sample id, material id, SG, atom_count, formula_counts, and split are emitted. Canonical W/A, skeleton, row_count, CIF path, lattice, coordinates, StructureMatcher labels, and target diagnostics are not copied.",
    }
    write_json(out_dir / "build_infer_metadata_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
