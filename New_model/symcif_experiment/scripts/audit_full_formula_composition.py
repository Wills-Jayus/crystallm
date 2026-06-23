#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.formula import normalize_formula_counts, parse_formula_counts, reduced_formula_from_counts, z_from_counts
from train_skeleton_template_ranker import read_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit full conventional formula composition usage.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "composition_exact_v1")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    split_counts: Counter[str] = Counter()
    parse_fail = 0
    count_mismatch = 0
    z_match = 0
    examples_parse_fail: list[dict[str, Any]] = []
    examples_count_mismatch: list[dict[str, Any]] = []
    examples_z_mismatch: list[dict[str, Any]] = []
    reduced_examples: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for row in read_jsonl(args.structured_root / f"{split}.jsonl"):
            total += 1
            split_counts[split] += 1
            try:
                parsed = parse_formula_counts(row["formula"])
            except Exception as exc:
                parse_fail += 1
                if len(examples_parse_fail) < 10:
                    examples_parse_fail.append({"sample_id": row.get("sample_id"), "formula": row.get("formula"), "error": str(exc)})
                parsed = {}
            structured = normalize_formula_counts(row["formula_counts"])
            if parsed != structured:
                count_mismatch += 1
                if len(examples_count_mismatch) < 20:
                    examples_count_mismatch.append(
                        {
                            "sample_id": row.get("sample_id"),
                            "formula": row.get("formula"),
                            "parsed_formula_counts": parsed,
                            "structured_formula_counts": structured,
                        }
                    )
            z = z_from_counts(structured)
            reduced = reduced_formula_from_counts(structured)
            if int(row.get("z", -1)) == z:
                z_match += 1
            elif len(examples_z_mismatch) < 20:
                examples_z_mismatch.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "formula": row.get("formula"),
                        "structured_z": row.get("z"),
                        "z_from_counts": z,
                        "reduced_formula_for_logging_only": reduced,
                    }
                )
            if len(reduced_examples) < 10:
                reduced_examples.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "formula": row.get("formula"),
                        "target_counts_full_conventional": structured,
                        "reduced_formula_for_logging_only": reduced,
                        "z_from_counts_for_logging_only": z,
                    }
                )
    report = {
        "total_records": total,
        "split_counts": dict(split_counts),
        "parsed_equals_structured_rate": (total - count_mismatch) / max(1, total),
        "z_consistency_rate": z_match / max(1, total),
        "num_formula_parse_fail": parse_fail,
        "num_formula_count_mismatch": count_mismatch,
        "examples_formula_parse_fail": examples_parse_fail,
        "examples_formula_count_mismatch": examples_count_mismatch,
        "examples_z_mismatch": examples_z_mismatch,
        "reduced_formula_examples_logging_only": reduced_examples,
        "hard_rule": "exact-cover target_counts are always full conventional formula_counts; reduced_formula and z are logging/statistics only.",
    }
    (args.out_dir / "formula_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
