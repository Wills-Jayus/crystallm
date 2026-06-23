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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare final-eval-only repr JSONL with keys.sample_id for evaluator compatibility.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    args = parser.parse_args()
    out_jsonl = ensure_under_opentry(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            material_id = str(row.get("material_id") or sample_id)
            keys = dict(row.get("keys") or {})
            keys.setdefault("sample_id", sample_id)
            keys.setdefault("material_id", material_id)
            keys.setdefault("id", str(row.get("id") or sample_id))
            row["keys"] = keys
            if "row_count" not in row:
                row["row_count"] = len(list(row.get("wa_table") or []))
            if "complex_flag" not in row:
                row["complex_flag"] = bool(int(row.get("row_count") or 0) >= 7 or int(row.get("atom_count") or 0) >= 12)
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(out_jsonl),
        "records": count,
        "label_policy": "Final evaluation only. Contains test canonical W/A, skeleton, row_count, and GT-derived diagnostics for reporting metrics; never used for training, sorting, filtering, or config selection.",
    }
    summary_path = out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
