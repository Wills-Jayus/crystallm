#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_records is not None and len(rows) >= int(max_records):
                break
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def candidate_key(cand: dict[str, Any]) -> str:
    key = str(cand.get("wa_key") or "")
    if key:
        return key
    pairs = cand.get("pairs") or []
    return "|".join(f"{str(pair[1])}:{str(pair[0])}" for pair in pairs if isinstance(pair, (list, tuple)) and len(pair) == 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Priority-merge exact DP W/A candidate files without labels.")
    parser.add_argument("--candidate-files", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--per-source-limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = None if int(args.max_records) <= 0 else int(args.max_records)
    per_source_limit = None if int(args.per_source_limit) <= 0 else int(args.per_source_limit)
    sources = []
    for path in args.candidate_files:
        rows = read_jsonl(path, max_records=max_records)
        sources.append({str(row.get("sample_id")): row for row in rows})
    sample_ids: list[str] = []
    seen_samples: set[str] = set()
    for source in sources:
        for sample_id in source:
            if sample_id in seen_samples:
                continue
            seen_samples.add(sample_id)
            sample_ids.append(sample_id)

    out_rows: list[dict[str, Any]] = []
    input_candidates = 0
    output_candidates = 0
    for sample_id in sample_ids:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        stats: list[dict[str, Any]] = []
        for source_idx, source in enumerate(sources):
            row = source.get(sample_id) or {}
            cands = list(row.get("candidates") or [])
            if per_source_limit is not None:
                cands = cands[:per_source_limit]
            added = 0
            for cand in cands:
                input_candidates += 1
                key = candidate_key(cand)
                if not key or key in seen:
                    continue
                seen.add(key)
                out = dict(cand)
                out["source"] = f"priority_merge_source{source_idx}:{out.get('source') or 'candidate'}"
                merged.append(out)
                added += 1
                if len(merged) >= int(args.top_k):
                    break
            stats.append({"source_index": source_idx, "input": len(cands), "added": added})
            if len(merged) >= int(args.top_k):
                break
        output_candidates += len(merged)
        out_rows.append({"sample_id": sample_id, "candidates": merged[: int(args.top_k)], "stats": stats})

    summary = {
        "candidate_files": [str(x) for x in args.candidate_files],
        "top_k": int(args.top_k),
        "per_source_limit": per_source_limit,
        "samples": len(out_rows),
        "input_candidates_seen": input_candidates,
        "output_candidates": output_candidates,
        "candidate_nonempty_rate": sum(int(bool(row["candidates"])) for row in out_rows) / max(1, len(out_rows)),
        "out_candidates": str(out_dir / "merged_dp_candidates.jsonl"),
    }
    write_jsonl(out_dir / "merged_dp_candidates.jsonl", out_rows)
    write_json(out_dir / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
