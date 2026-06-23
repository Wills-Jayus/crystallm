#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from analyze_mp20_k5_dev import read_jsonl, write_fix_predictions, write_json


def limit_raw_candidates(rows: list[dict[str, Any]], raw_limit: int | None) -> list[dict[str, Any]]:
    if raw_limit is None or raw_limit <= 0:
        return rows
    limited: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        candidates = list(copied.get("candidates") or copied.get("ranked_wa_candidates") or [])
        copied["candidates"] = candidates[:raw_limit]
        copied.pop("ranked_wa_candidates", None)
        limited.append(copied)
    return limited


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen train-prior hybrid WA top5 predictions for a structured SymCIF split."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--raw-limit", type=int, default=100)
    args = parser.parse_args()

    train_records = read_jsonl(args.data_root / "train.jsonl")
    split_records = read_jsonl(args.data_root / f"{args.split}.jsonl")
    raw_predictions = limit_raw_candidates(read_jsonl(args.raw_predictions), args.raw_limit)

    split_ids = {str(row["sample_id"]) for row in split_records}
    raw_ids = {str(row["sample_id"]) for row in raw_predictions}
    missing = sorted(split_ids - raw_ids)
    extra = sorted(raw_ids - split_ids)
    if missing:
        raise SystemExit(f"raw predictions missing {len(missing)} split samples; first={missing[:5]}")

    audit = write_fix_predictions(train_records, split_records, raw_predictions, args.out_path)
    audit.update(
        {
            "split": args.split,
            "data_root": str(args.data_root),
            "raw_predictions": str(args.raw_predictions),
            "raw_limit": args.raw_limit,
            "extra_raw_prediction_rows": len(extra),
        }
    )
    write_json(args.out_path.with_name("test_hybrid_prior_build_audit.json"), audit)


if __name__ == "__main__":
    main()
