#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def formula_string(counts: dict[str, Any]) -> str:
    parts: list[str] = []
    for element in sorted(str(k) for k in counts):
        value = int(counts[element])
        parts.append(f"{element}{value if value != 1 else ''}")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build render-only data-root without test target W/A labels.")
    parser.add_argument("--train-jsonl", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52" / "train.jsonl")
    parser.add_argument("--infer-jsonl", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52_test_infer" / "test.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.train_jsonl, out_dir / "train.jsonl")
    test_rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.infer_jsonl):
        keys = dict(row.get("keys") or {})
        sample_id = str(keys.get("sample_id") or keys.get("id") or "")
        material_id = str(keys.get("material_id") or sample_id)
        counts = {str(k): int(v) for k, v in dict(row.get("formula_counts") or {}).items()}
        test_rows.append(
            {
                "atom_count": int(row["atom_count"]),
                "formula": formula_string(counts),
                "formula_counts": counts,
                "keys": keys,
                "material_id": material_id,
                "sample_id": sample_id,
                "sg": int(row["sg"]),
                "split": "test",
                "wa_table": [],
            }
        )
    write_jsonl(out_dir / "test.jsonl", test_rows)
    summary = {
        "train_jsonl": str(args.train_jsonl),
        "infer_jsonl": str(args.infer_jsonl),
        "out_dir": str(out_dir),
        "train_records": sum(1 for _ in args.train_jsonl.open("r", encoding="utf-8")),
        "test_records": len(test_rows),
        "label_policy": "test.jsonl contains formula/SG/sample ids and empty wa_table only; no test target W/A, row_count, lattice, coordinates, or StructureMatcher labels are copied.",
    }
    (out_dir / "render_infer_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
