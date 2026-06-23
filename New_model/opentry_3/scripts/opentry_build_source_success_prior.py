#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing path outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = ensure_under_opentry(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a train-only source success prior from rendered-candidate labels.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--min-source-count", type=int, default=2)
    parser.add_argument("--max-abs-penalty", type=float, default=2.0)
    args = parser.parse_args()

    rows = read_jsonl(args.train_features)
    out_json = ensure_under_opentry(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    source_counts: dict[str, int] = defaultdict(int)
    source_pos: dict[str, int] = defaultdict(int)
    source_rows7_counts: dict[str, int] = defaultdict(int)
    source_rows7_pos: dict[str, int] = defaultdict(int)
    total = 0
    positives = 0
    rows7_total = 0
    rows7_pos = 0
    for row in rows:
        sid = str(row.get("source_sample_id") or "")
        if not sid:
            continue
        y = int(bool(row.get("label_match")))
        source_counts[sid] += 1
        source_pos[sid] += y
        total += 1
        positives += y
        if int(row.get("target_row_count", 0)) >= 7:
            source_rows7_counts[sid] += 1
            source_rows7_pos[sid] += y
            rows7_total += 1
            rows7_pos += y

    if total <= 0:
        raise SystemExit("No rows with source_sample_id found.")
    global_rate = positives / max(1, total)
    rows7_rate = rows7_pos / max(1, rows7_total)

    def penalty(pos: int, count: int, base_rate: float) -> float:
        alpha = float(args.alpha)
        smoothed = (float(pos) + alpha * float(base_rate)) / max(1.0, float(count) + alpha)
        # Lower penalty means a better source. Use a bounded negative log odds
        # relative to the global train source success rate.
        raw = -math.log(max(1e-6, smoothed) / max(1e-6, float(base_rate)))
        cap = float(args.max_abs_penalty)
        return float(max(-cap, min(cap, raw)))

    source_penalty: dict[str, float] = {}
    source_rate: dict[str, float] = {}
    for sid, count in source_counts.items():
        if count < int(args.min_source_count):
            continue
        pos = source_pos[sid]
        source_rate[sid] = float(pos / max(1, count))
        source_penalty[sid] = penalty(pos, count, global_rate)

    rows7_source_penalty: dict[str, float] = {}
    rows7_source_rate: dict[str, float] = {}
    for sid, count in source_rows7_counts.items():
        if count < int(args.min_source_count):
            continue
        pos = source_rows7_pos[sid]
        rows7_source_rate[sid] = float(pos / max(1, count))
        rows7_source_penalty[sid] = penalty(pos, count, rows7_rate if rows7_total > 0 else global_rate)

    payload = {
        "source": "train_rendered_candidate_labels",
        "train_features": str(ensure_under_opentry(args.train_features)),
        "alpha": float(args.alpha),
        "min_source_count": int(args.min_source_count),
        "max_abs_penalty": float(args.max_abs_penalty),
        "global": {
            "rows": int(total),
            "positives": int(positives),
            "positive_rate": float(global_rate),
            "sources_total": int(len(source_counts)),
            "sources_kept": int(len(source_penalty)),
        },
        "rows_ge_7": {
            "rows": int(rows7_total),
            "positives": int(rows7_pos),
            "positive_rate": float(rows7_rate),
            "sources_total": int(len(source_rows7_counts)),
            "sources_kept": int(len(rows7_source_penalty)),
        },
        "default_penalty": 0.0,
        "source_penalty": source_penalty,
        "source_rate": source_rate,
        "rows_ge_7_source_penalty": rows7_source_penalty,
        "rows_ge_7_source_rate": rows7_source_rate,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["global", "rows_ge_7"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
