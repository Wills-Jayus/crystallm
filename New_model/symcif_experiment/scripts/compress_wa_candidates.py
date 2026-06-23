#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.scorer import BaselineScorer, rank_candidates_baseline
from symcif_v4.wa_table import gt_wa_key
from train_skeleton_template_ranker import read_jsonl


def trim_line(line: dict[str, Any], sample: dict[str, Any], scorer: BaselineScorer, keep: int) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = line["wa_candidates"]
    ranked = rank_candidates_baseline(scorer, sample, candidates)
    target = gt_wa_key(sample)
    selected = ranked[:keep]
    selected_keys = {c["wa_key"] for c in selected}
    gt_candidate = next((c for c in candidates if c["wa_key"] == target), None)
    gt_preserved = False
    if gt_candidate is not None and target not in selected_keys:
        gt_item = dict(gt_candidate)
        gt_item["score"] = scorer.score(sample, gt_candidate)
        gt_item["rank_source"] = "baseline_gt_preserved"
        selected.append(gt_item)
        gt_preserved = True
    return (
        {**{k: line[k] for k in line if k != "wa_candidates"}, "wa_candidates": selected},
        {
            "sample_id": sample["sample_id"],
            "input_candidates": len(candidates),
            "kept_candidates": len(selected),
            "gt_in_input": gt_candidate is not None,
            "gt_in_kept": any(c["wa_key"] == target for c in selected),
            "gt_preserved_extra": gt_preserved,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress large WA candidate files using baseline preselection.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--candidate-dir", type=Path, default=PROJECT_ROOT / "reports" / "composition_exact_v1")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "composition_exact_v1_trimmed")
    parser.add_argument("--keep", type=int, default=200)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.structured_root / "train.jsonl")
    scorer = BaselineScorer.from_rows(train_rows)
    summary: dict[str, Any] = {}
    for split in args.splits:
        rows = read_jsonl(args.structured_root / f"{split}.jsonl")
        by_id = {row["sample_id"]: row for row in rows}
        inp = args.candidate_dir / f"{split}_wa_candidates.jsonl"
        out = args.out_dir / f"{split}_wa_candidates.jsonl"
        stats = []
        with inp.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
            for i, raw in enumerate(f_in):
                if not raw.strip():
                    continue
                line = json.loads(raw)
                trimmed, stat = trim_line(line, by_id[line["sample_id"]], scorer, args.keep)
                stats.append(stat)
                f_out.write(json.dumps(trimmed, ensure_ascii=False, sort_keys=True) + "\n")
                if i and i % 250 == 0:
                    print(f"[compress:{split}] {i}/{len(rows)}", flush=True)
        summary[split] = {
            "samples": len(stats),
            "keep": args.keep,
            "input_candidates_mean": sum(s["input_candidates"] for s in stats) / max(1, len(stats)),
            "kept_candidates_mean": sum(s["kept_candidates"] for s in stats) / max(1, len(stats)),
            "gt_in_input_rate": sum(bool(s["gt_in_input"]) for s in stats) / max(1, len(stats)),
            "gt_in_kept_rate": sum(bool(s["gt_in_kept"]) for s in stats) / max(1, len(stats)),
            "gt_preserved_extra": sum(bool(s["gt_preserved_extra"]) for s in stats),
        }
    (args.out_dir / "compression_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
