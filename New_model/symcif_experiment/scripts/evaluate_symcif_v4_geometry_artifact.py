#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import evaluate_mode_with_hard_timeouts, load_generation_records  # noqa: E402
from run_symcif_v4_geometry_model_eval import (  # noqa: E402
    enrich_metrics,
    failed_cases,
    make_summary,
    read_jsonl,
    write_jsonl,
)
from run_symcif_v4_geometry_model_eval import case_payload as geometry_case_payload  # noqa: E402


def load_selected_meta(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    meta_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_index = int(row["sample_index"])
        for item in row.get("selected", []):
            rank = int(item.get("rank", 0))
            if rank <= 0:
                continue
            meta_by_key[(sample_index, rank - 1)] = dict(item)
    return meta_by_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a rendered SymCIF-v4 geometry artifact without rerendering CIFs.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval-workers", type=int, default=32)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--match-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-match-sites", type=int, default=300)
    parser.add_argument("--max-eval-sites", type=int, default=300)
    parser.add_argument("--full-wa-candidates", type=int, default=20)
    parser.add_argument("--full-max-variants-per-wa", type=int, default=1)
    parser.add_argument("--full-selection-mode", choices=("round_robin", "score"), default="round_robin")
    parser.add_argument("--include-neural", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-prototypes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.data_root / "test.jsonl")
    generation_path = args.out_dir / "generations" / "baseline.jsonl"
    selected_path = args.out_dir / "selected_top20.jsonl"
    if not generation_path.exists():
        raise FileNotFoundError(generation_path)
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)

    grouped = load_generation_records(generation_path)
    eval_args = SimpleNamespace(
        eval_workers=int(args.eval_workers),
        bond_timeout_seconds=float(args.bond_timeout_seconds),
        parse_timeout_seconds=float(args.parse_timeout_seconds),
        sg_timeout_seconds=float(args.sg_timeout_seconds),
        valid_timeout_seconds=float(args.valid_timeout_seconds),
        match_timeout_seconds=float(args.match_timeout_seconds),
        sample_timeout_seconds=float(args.sample_timeout_seconds),
        max_match_sites=int(args.max_match_sites),
        max_eval_sites=int(args.max_eval_sites),
    )
    metrics = evaluate_mode_with_hard_timeouts(
        mode="baseline",
        case_payload=geometry_case_payload(records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    metrics = enrich_metrics(metrics, load_selected_meta(selected_path))
    write_jsonl(args.out_dir / "metrics" / "baseline_per_generation_metrics.jsonl", metrics)
    write_jsonl(args.out_dir / "failed_cases.jsonl", failed_cases(records, metrics, args.top_k))

    fake_model = SimpleNamespace(
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        training_config={},
    )
    summary = make_summary(mode="full", records=records, metrics=metrics, out_dir=args.out_dir, args=args, model=fake_model)
    (args.out_dir / "full_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
