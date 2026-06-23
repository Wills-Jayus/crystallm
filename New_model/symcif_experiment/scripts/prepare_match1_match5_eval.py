#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def subset_jsonl(src: Path, dst: Path, *, n: int, mode: str) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            rec: dict[str, Any] = json.loads(line)
            if int(rec["gen_index"]) >= n:
                continue
            rec["mode"] = mode
            out.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare first-n generation files for match@1/match@5 evaluation.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_runs" / "symcif_v2_constrained_v2_match1_match5",
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument(
        "--baseline-src",
        type=Path,
        default=PROJECT_ROOT
        / "eval_runs"
        / "generation_eval_input_ablation_t1_topk10_n20_20260520"
        / "generations"
        / "baseline_minprompt.jsonl",
    )
    parser.add_argument(
        "--old-src",
        type=Path,
        default=PROJECT_ROOT
        / "eval_runs"
        / "symcif_v2_constrained_eval_t1_topk10_n20"
        / "generations"
        / "symcif_v2_constrained.jsonl",
    )
    args = parser.parse_args()
    generation_dir = args.out_dir / "generations"
    baseline_count = subset_jsonl(
        args.baseline_src,
        generation_dir / "baseline_minprompt.jsonl",
        n=args.n,
        mode="baseline_minprompt",
    )
    old_count = subset_jsonl(
        args.old_src,
        generation_dir / "symcif_v2_constrained_old.jsonl",
        n=args.n,
        mode="symcif_v2_constrained_old",
    )
    metadata = {
        "n": args.n,
        "baseline_src": str(args.baseline_src),
        "old_src": str(args.old_src),
        "baseline_records": baseline_count,
        "old_records": old_count,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prepare_match1_match5_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[prepare] baseline_minprompt records: {baseline_count}")
    print(f"[prepare] symcif_v2_constrained_old records: {old_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

