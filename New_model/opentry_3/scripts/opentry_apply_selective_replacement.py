#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from opentry_selective_replacement_selector import read_jsonl, select_rows, strip_labels
from opentry_train_geometry_compat_selector import ensure_under_opentry, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a frozen selective replacement selector without label-based tuning.")
    parser.add_argument("--scored-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="compat_score")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--anchor-count", type=int, required=True)
    parser.add_argument("--max-per-wa", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.scored_candidates)
    ranked = select_rows(
        rows,
        score_field=str(args.score_field),
        threshold=float(args.threshold),
        anchor_count=int(args.anchor_count),
        top_k=int(args.top_k),
        max_per_wa=int(args.max_per_wa),
    )
    write_jsonl(out_dir / "reranked_features.jsonl", ranked)
    write_jsonl(out_dir / "rendered_topk_selective.jsonl", strip_labels(ranked))
    summary: dict[str, Any] = {
        "scored_candidates": str(args.scored_candidates),
        "score_field": str(args.score_field),
        "threshold": float(args.threshold),
        "anchor_count": int(args.anchor_count),
        "max_per_wa": int(args.max_per_wa),
        "top_k": int(args.top_k),
        "input_rows": len(rows),
        "output_rows": len(ranked),
        "samples": len({str(row.get("sample_id")) for row in ranked}),
        "selection_note": "Apply-only frozen selector. No metric labels or StructureMatcher values are required; rendered output strips label/target diagnostic fields.",
    }
    write_json(out_dir / "selective_apply_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
