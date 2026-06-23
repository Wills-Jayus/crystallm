#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_generation_eval as eval_utils  # noqa: E402


JOBS = [
    (
        PROJECT_ROOT / "eval_runs" / "baseline_reeval_same_as_v4_20260522",
        "baseline",
        "baseline_same_as_v4",
    ),
    (
        PROJECT_ROOT / "eval_runs" / "baseline_minprompt_reeval_same_as_v4_20260522",
        "baseline_minprompt",
        "baseline_minprompt_same_as_v4",
    ),
]


def main() -> None:
    for out_dir, metric_mode, report_mode in JOBS:
        metrics_path = out_dir / "metrics" / f"{metric_mode}_per_generation_metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.open(encoding="utf-8")]
        rows = []
        for n in (1, 5, 20):
            row = {"mode": report_mode}
            row.update(eval_utils.aggregate_metrics(metrics, n=n, total_cases=500))
            rows.append(row)

        (out_dir / "summary_with_n5.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (out_dir / "summary_with_n5.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(out_dir)
        print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
