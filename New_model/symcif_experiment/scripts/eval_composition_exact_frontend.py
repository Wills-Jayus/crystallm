#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.wa_table import gt_skeleton_key, gt_wa_key
from train_skeleton_template_ranker import read_jsonl


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scorer_breakdown(structured_root: Path, predictions: Path, split: str, group_key: str) -> list[dict[str, object]]:
    row_by_id = {row["sample_id"]: row for row in read_jsonl(structured_root / f"{split}.jsonl")}
    groups: dict[object, list[dict[str, object]]] = defaultdict(list)
    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pred = json.loads(line)
            row = row_by_id[pred["sample_id"]]
            target_skel = gt_skeleton_key(row)
            target_wa = gt_wa_key(row)
            ranked = pred.get("ranked_wa_candidates") or []
            item = {
                "skeleton_top1": any(c["skeleton_key"] == target_skel for c in ranked[:1]),
                "skeleton_top5": any(c["skeleton_key"] == target_skel for c in ranked[:5]),
                "skeleton_top20": any(c["skeleton_key"] == target_skel for c in ranked[:20]),
                "wa_top1": any(c["wa_key"] == target_wa for c in ranked[:1]),
                "wa_top5": any(c["wa_key"] == target_wa for c in ranked[:5]),
                "wa_top20": any(c["wa_key"] == target_wa for c in ranked[:20]),
                "candidate_nonempty": bool(ranked),
            }
            groups[row[group_key]].append(item)
    out: list[dict[str, object]] = []
    for value, items in sorted(groups.items(), key=lambda kv: kv[0]):
        row: dict[str, object] = {group_key: value, "samples": len(items)}
        for key in ("candidate_nonempty", "skeleton_top1", "skeleton_top5", "skeleton_top20", "wa_top1", "wa_top5", "wa_top20"):
            row[key] = sum(bool(x[key]) for x in items) / max(1, len(items))
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect composition-exact frontend evaluation artifacts.")
    parser.add_argument("--structured-root", type=Path, required=False)
    parser.add_argument("--candidate-dir", type=Path, default=Path("reports/composition_exact_v1"))
    parser.add_argument("--scorer-dir", type=Path, default=Path("runs/wa_table_scorer_v1"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/composition_exact_v1"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "split": args.split,
        "candidate_summary": None,
        "wa_scorer_summary": None,
    }
    cand = args.candidate_dir / "candidate_generation_summary.json"
    if cand.exists():
        out["candidate_summary"] = json.loads(cand.read_text(encoding="utf-8")).get("splits", {}).get(args.split)
    scorer = args.out_dir / "wa_scorer_eval_summary.json"
    if scorer.exists():
        out["wa_scorer_summary"] = json.loads(scorer.read_text(encoding="utf-8"))
    if args.structured_root is not None:
        structured_root = Path(args.structured_root)
        predictions = args.out_dir / "test_wa_predictions.jsonl"
        if predictions.exists():
            write_csv(
                args.out_dir / "wa_scorer_breakdown_per_sg_neural.csv",
                scorer_breakdown(structured_root, predictions, args.split, "sg"),
            )
            write_csv(
                args.out_dir / "wa_scorer_breakdown_per_nsites_neural.csv",
                scorer_breakdown(structured_root, predictions, args.split, "n_sites"),
            )
    (args.out_dir / "frontend_eval_summary.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
