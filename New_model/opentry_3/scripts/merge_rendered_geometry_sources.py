#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
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


def group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["sample_id"])].append(row)
    for items in out.values():
        items.sort(key=lambda row: int(row["rank"]))
    return dict(out)


def is_usable(row: dict[str, Any]) -> bool:
    return bool(row.get("readable")) and bool(row.get("composition_exact")) and bool(row.get("cif"))


def add_candidate(
    merged: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    seen_hashes: set[str],
    per_wa_count: Counter[str],
    max_per_wa: int,
    merge_source: str,
    top_k: int,
) -> bool:
    if len(merged) >= int(top_k) or not is_usable(row):
        return False
    cif = str(row.get("cif") or "")
    if cif in seen_hashes:
        return False
    wa_key = str(row.get("canonical_wa_key") or "")
    if max_per_wa > 0 and per_wa_count[wa_key] >= int(max_per_wa):
        return False
    out = dict(row)
    out["original_rank"] = int(row["rank"])
    out["merge_source"] = merge_source
    out["rank"] = len(merged) + 1
    merged.append(out)
    seen_hashes.add(cif)
    per_wa_count[wa_key] += 1
    return True


def merge_one(
    sample_id: str,
    primary_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    *,
    top_k: int,
    max_primary_per_wa: int,
    max_final_per_wa: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    per_wa_count: Counter[str] = Counter()
    # First pass: keep e08 geometry, but prevent a single W/A key from filling K.
    for row in primary_rows:
        add_candidate(
            merged,
            row,
            seen_hashes=seen_hashes,
            per_wa_count=per_wa_count,
            max_per_wa=max_primary_per_wa,
            merge_source="primary_e08",
            top_k=top_k,
        )
    # Second pass: fill new W/A keys from collision-aware deterministic fallback.
    for row in fallback_rows:
        wa_key = str(row.get("canonical_wa_key") or "")
        if per_wa_count[wa_key] > 0:
            continue
        add_candidate(
            merged,
            row,
            seen_hashes=seen_hashes,
            per_wa_count=per_wa_count,
            max_per_wa=max_final_per_wa,
            merge_source="fallback_collision_aware_new_wa",
            top_k=top_k,
        )
    # Final pass: if still short, allow additional fallback geometry variants.
    for row in fallback_rows:
        add_candidate(
            merged,
            row,
            seen_hashes=seen_hashes,
            per_wa_count=per_wa_count,
            max_per_wa=max_final_per_wa,
            merge_source="fallback_collision_aware_fill",
            top_k=top_k,
        )
    for row in merged:
        row["sample_id"] = sample_id
    return merged


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sorted({str(row["sample_id"]) for row in rows})
    by_id = group(rows)
    return {
        "samples_with_candidates": len(samples),
        "rendered_rows": len(rows),
        "candidate_count_mean": sum(len(v) for v in by_id.values()) / max(1, len(by_id)),
        "unique_wa_mean": sum(len({str(row.get("canonical_wa_key") or "") for row in v}) for v in by_id.values()) / max(1, len(by_id)),
        "readable": sum(bool(row.get("readable")) for row in rows) / max(1, len(rows)),
        "composition_exact": sum(bool(row.get("composition_exact")) for row in rows) / max(1, len(rows)),
        "sg_ok": sum(bool(row.get("sg_ok")) for row in rows) / max(1, len(rows)),
        "merge_source_counts": dict(Counter(str(row.get("merge_source")) for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge e08 geometry rows with collision-aware W/A-diverse fallback rows.")
    parser.add_argument("--primary-rendered", type=Path, required=True)
    parser.add_argument("--fallback-rendered", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-primary-per-wa", type=int, default=2)
    parser.add_argument("--max-final-per-wa", type=int, default=2)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    primary = group(read_jsonl(args.primary_rendered))
    fallback = group(read_jsonl(args.fallback_rendered))
    sample_ids = sorted(set(primary) | set(fallback))
    merged_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        merged_rows.extend(
            merge_one(
                sample_id,
                primary.get(sample_id, []),
                fallback.get(sample_id, []),
                top_k=int(args.top_k),
                max_primary_per_wa=int(args.max_primary_per_wa),
                max_final_per_wa=int(args.max_final_per_wa),
            )
        )
    merged_rows.sort(key=lambda row: (str(row["sample_id"]), int(row["rank"])))
    summary = {
        "primary_rendered": str(args.primary_rendered),
        "fallback_rendered": str(args.fallback_rendered),
        "top_k": int(args.top_k),
        "max_primary_per_wa": int(args.max_primary_per_wa),
        "max_final_per_wa": int(args.max_final_per_wa),
        **summarize(merged_rows),
    }
    write_jsonl(out_dir / "rendered_topk.jsonl", merged_rows)
    write_json(out_dir / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
