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


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("canonical_wa_key") or candidate.get("raw_dp_wa_key") or "")


def merge_round_robin(
    source_rows: list[dict[str, dict[str, Any]]],
    sample_ids: list[str],
    *,
    top_k: int,
    per_source_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    source_added = [0 for _ in source_rows]
    for sample_id in sample_ids:
        source_candidates: list[list[dict[str, Any]]] = []
        for source in source_rows:
            cands = list((source.get(sample_id) or {}).get("ranked_wa_candidates") or [])
            if per_source_limit is not None:
                cands = cands[:per_source_limit]
            source_candidates.append(cands)
            total_input += len(cands)

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        rank_by_source = [0 for _ in source_candidates]
        max_len = max((len(cands) for cands in source_candidates), default=0)
        for rank in range(max_len):
            for source_idx, cands in enumerate(source_candidates):
                if rank >= len(cands):
                    continue
                cand = cands[rank]
                key = candidate_key(cand)
                if not key or key in seen:
                    continue
                seen.add(key)
                out = dict(cand)
                out["merge_source_index"] = source_idx
                out["merge_rank_in_source"] = rank
                out["source"] = f"balanced_source{source_idx}:{out.get('source') or 'canonical_prediction'}"
                merged.append(out)
                source_added[source_idx] += 1
                rank_by_source[source_idx] += 1
                if len(merged) >= int(top_k):
                    break
            if len(merged) >= int(top_k):
                break
        total_output += len(merged)
        out_rows.append(
            {
                "sample_id": sample_id,
                "ranked_wa_candidates": merged,
                "merge_stats": [
                    {
                        "source_index": idx,
                        "input": len(source_candidates[idx]),
                        "added": rank_by_source[idx],
                    }
                    for idx in range(len(source_candidates))
                ],
            }
        )
    summary = {
        "mode": "round_robin",
        "samples": len(out_rows),
        "top_k": int(top_k),
        "per_source_limit": per_source_limit,
        "input_candidates_seen": total_input,
        "output_candidates": total_output,
        "source_added": source_added,
        "candidate_nonempty_rate": sum(int(bool(row["ranked_wa_candidates"])) for row in out_rows) / max(1, len(out_rows)),
        "mean_output_candidates": total_output / max(1, len(out_rows)),
    }
    return out_rows, summary


def merge_priority(
    source_rows: list[dict[str, dict[str, Any]]],
    sample_ids: list[str],
    *,
    top_k: int,
    per_source_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    source_added = [0 for _ in source_rows]
    for sample_id in sample_ids:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        stats: list[dict[str, int]] = []
        for source_idx, source in enumerate(source_rows):
            cands = list((source.get(sample_id) or {}).get("ranked_wa_candidates") or [])
            if per_source_limit is not None:
                cands = cands[:per_source_limit]
            total_input += len(cands)
            added = 0
            for rank, cand in enumerate(cands):
                key = candidate_key(cand)
                if not key or key in seen:
                    continue
                seen.add(key)
                out = dict(cand)
                out["merge_source_index"] = source_idx
                out["merge_rank_in_source"] = rank
                out["source"] = f"canonical_priority_source{source_idx}:{out.get('source') or 'canonical_prediction'}"
                merged.append(out)
                source_added[source_idx] += 1
                added += 1
                if len(merged) >= int(top_k):
                    break
            stats.append({"source_index": source_idx, "input": len(cands), "added": added})
            if len(merged) >= int(top_k):
                break
        total_output += len(merged)
        out_rows.append({"sample_id": sample_id, "ranked_wa_candidates": merged, "merge_stats": stats})
    summary = {
        "mode": "priority",
        "samples": len(out_rows),
        "top_k": int(top_k),
        "per_source_limit": per_source_limit,
        "input_candidates_seen": total_input,
        "output_candidates": total_output,
        "source_added": source_added,
        "candidate_nonempty_rate": sum(int(bool(row["ranked_wa_candidates"])) for row in out_rows) / max(1, len(out_rows)),
        "mean_output_candidates": total_output / max(1, len(out_rows)),
    }
    return out_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Source-balanced merge for canonical renderer W/A predictions.")
    parser.add_argument("--prediction-files", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("round_robin", "priority"), default="round_robin")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--per-source-limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = None if int(args.max_records) <= 0 else int(args.max_records)
    per_source_limit = None if int(args.per_source_limit) <= 0 else int(args.per_source_limit)

    source_maps: list[dict[str, dict[str, Any]]] = []
    sample_ids: list[str] = []
    seen_samples: set[str] = set()
    for path in args.prediction_files:
        rows = read_jsonl(path, max_records=max_records)
        source_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = str(row.get("sample_id"))
            source_map[sample_id] = row
            if sample_id not in seen_samples:
                seen_samples.add(sample_id)
                sample_ids.append(sample_id)
        source_maps.append(source_map)

    merge_fn = merge_round_robin if args.mode == "round_robin" else merge_priority
    out_rows, summary = merge_fn(source_maps, sample_ids, top_k=int(args.top_k), per_source_limit=per_source_limit)
    summary.update(
        {
            "prediction_files": [str(path) for path in args.prediction_files],
            "out_predictions": str(out_dir / "renderer_predictions.jsonl"),
        }
    )
    write_jsonl(out_dir / "renderer_predictions.jsonl", out_rows)
    write_json(out_dir / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
