#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import evaluate_mode_with_hard_timeouts  # noqa: E402
from run_symcif_v4_geometry_model_eval import (  # noqa: E402
    enrich_metrics,
    failed_cases,
    make_summary,
    read_jsonl,
    write_jsonl,
)
from run_symcif_v4_full_pipeline_eval import case_payload  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_grouped_generations(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        grouped[int(row["sample_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["gen_index"]))
    return dict(grouped)


def load_meta(top20_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    meta_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in load_jsonl(top20_path):
        sample_index = int(row["sample_index"])
        sample_id = str(row["sample_id"])
        for pred in row.get("predictions", []):
            rank = int(pred.get("rank", 0))
            if rank <= 0:
                continue
            gen_index = rank - 1
            meta = dict(pred)
            meta["sample_index"] = sample_index
            meta["sample_id"] = sample_id
            meta["gen_index"] = gen_index
            meta_by_key[(sample_index, gen_index)] = meta
    return meta_by_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume SymCIF-v4 evaluation from rendered geometry outputs.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "gtwa"), default="full")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval-workers", type=int, default=64)
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
    parser.add_argument("--description", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.data_root / "test.jsonl")
    generations_path = args.out_dir / "generations" / "baseline.jsonl"
    top20_path = args.out_dir / "top20_predictions.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)
    if not top20_path.exists():
        raise FileNotFoundError(top20_path)

    grouped = load_grouped_generations(generations_path)
    meta_by_key = load_meta(top20_path)
    eval_args = SimpleNamespace(
        eval_workers=args.eval_workers,
        bond_timeout_seconds=args.bond_timeout_seconds,
        parse_timeout_seconds=args.parse_timeout_seconds,
        sg_timeout_seconds=args.sg_timeout_seconds,
        valid_timeout_seconds=args.valid_timeout_seconds,
        match_timeout_seconds=args.match_timeout_seconds,
        sample_timeout_seconds=args.sample_timeout_seconds,
        max_match_sites=args.max_match_sites,
        max_eval_sites=args.max_eval_sites,
    )
    metrics = evaluate_mode_with_hard_timeouts(
        mode="baseline",
        case_payload=case_payload(records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    metrics = enrich_metrics(metrics, meta_by_key)
    write_jsonl(args.out_dir / "metrics" / "baseline_per_generation_metrics.jsonl", metrics)
    write_jsonl(args.out_dir / "failed_cases.jsonl", failed_cases(records, metrics, args.top_k))

    summary_args = SimpleNamespace(
        data_root=args.data_root,
        predictions=top20_path,
        top_k=args.top_k,
        full_wa_candidates=args.full_wa_candidates,
        full_max_variants_per_wa=args.full_max_variants_per_wa,
        full_selection_mode=args.full_selection_mode,
        bond_timeout_seconds=args.bond_timeout_seconds,
        parse_timeout_seconds=args.parse_timeout_seconds,
        sg_timeout_seconds=args.sg_timeout_seconds,
        valid_timeout_seconds=args.valid_timeout_seconds,
        match_timeout_seconds=args.match_timeout_seconds,
        sample_timeout_seconds=args.sample_timeout_seconds,
        max_match_sites=args.max_match_sites,
        max_eval_sites=args.max_eval_sites,
        eval_workers=args.eval_workers,
        include_neural=True,
        include_prototypes=True,
    )
    model_stub = SimpleNamespace(checkpoint=str(args.checkpoint), training_config={})
    summary = make_summary(mode=args.mode, records=records, metrics=metrics, out_dir=args.out_dir, args=summary_args, model=model_stub)
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
