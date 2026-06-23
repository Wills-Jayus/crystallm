#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from opentry_train_geometry_compat_selector import ensure_under_opentry, grouped, metrics_from_ranked, write_json, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_by_wa_group(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped(rows).items()):
        by_wa: dict[str, list[dict[str, Any]]] = {}
        for row in items:
            by_wa.setdefault(str(row.get("canonical_wa_key") or ""), []).append(row)
        group_payloads: list[tuple[int, float, str, dict[str, Any]]] = []
        for wa_key, group in by_wa.items():
            best = sorted(
                group,
                key=lambda row: (
                    -float(row.get("compat_score", 0.0)),
                    int(row.get("original_rank") or row.get("rank") or 10**9),
                ),
            )[0]
            first_rank = min(int(row.get("original_rank") or row.get("rank") or 10**9) for row in group)
            best_self = max(float(row.get("self_score", 0.0)) for row in group)
            group_payloads.append((first_rank, -best_self, wa_key, best))
        group_payloads.sort(key=lambda item: (item[0], item[1], item[2]))
        for rank, (_, _, _, row) in enumerate(group_payloads[: int(top_k)], start=1):
            item = dict(row)
            item["rank"] = rank
            item["compat_group_mode"] = "wa_order_preserve_best_geometry"
            out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a compatibility model only inside W/A groups, preserving W/A order/diversity.")
    parser.add_argument("--scored-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.scored_candidates)
    ranked = select_by_wa_group(rows, top_k=int(args.top_k))
    write_jsonl(out_dir / "val_reranked_features.jsonl", ranked)
    rendered = [{k: v for k, v in row.items() if not k.startswith("label_")} for row in ranked]
    write_jsonl(out_dir / "rendered_topk_compat_group.jsonl", rendered)
    metrics = metrics_from_ranked(ranked)
    summary = {
        "mode": "wa_order_preserve_best_geometry",
        "input": str(args.scored_candidates),
        "top_k": int(args.top_k),
        "metrics": {k: v for k, v in metrics.items() if k != "per_sample"},
    }
    write_json(out_dir / "compat_group_selector_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
