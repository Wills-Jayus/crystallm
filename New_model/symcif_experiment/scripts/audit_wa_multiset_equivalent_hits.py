#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOP_KS = (20, 100, 200, 700)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def wa_multiset_key(rows: list[dict[str, Any]]) -> str:
    items = sorted((str(row["orbit_id"]), str(row["element"])) for row in rows)
    return json.dumps(items, ensure_ascii=True, separators=(",", ":"))


def skeleton_multiset_key(rows: list[dict[str, Any]]) -> str:
    items = sorted(str(row["orbit_id"]) for row in rows)
    return json.dumps(items, ensure_ascii=True, separators=(",", ":"))


def record_index(data_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(data_root / "test.jsonl"):
        out[str(record["sample_id"])] = record
    return out


def hit_at(candidates: list[dict[str, Any]], target: str, key: str, k: int) -> bool:
    return any(str(candidate.get(key)) == target for candidate in candidates[:k])


def multiset_hit_at(candidates: list[dict[str, Any]], target: str, kind: str, k: int) -> bool:
    for candidate in candidates[:k]:
        rows = list(candidate.get("rows") or [])
        if kind == "wa" and wa_multiset_key(rows) == target:
            return True
        if kind == "skeleton" and skeleton_multiset_key(rows) == target:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records = record_index(args.data_root)
    per_sample: list[dict[str, Any]] = []
    source_labels = Counter()
    candidate_counts: list[int] = []
    for row in read_jsonl(args.candidates_jsonl):
        sample_id = str(row["sample_id"])
        record = records.get(sample_id)
        if record is None:
            continue
        candidates = list(row.get("candidates") or row.get("ranked_wa_candidates") or [])
        candidate_counts.append(len(candidates))
        for candidate in candidates:
            labels = list(candidate.get("source_labels") or [])
            source_labels.update(labels or ["unlabeled"])
        gt_wa_multiset = wa_multiset_key(list(record.get("wa_table") or []))
        gt_skel_multiset = skeleton_multiset_key(list(record.get("wa_table") or []))
        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "candidate_count": len(candidates),
            "gt_wa_key_ordered": record.get("canonical_wa_key"),
            "gt_skeleton_key_ordered": record.get("canonical_skeleton_key"),
            "gt_wa_multiset_key": gt_wa_multiset,
            "gt_skeleton_multiset_key": gt_skel_multiset,
        }
        for k in TOP_KS:
            sample[f"ordered_exact_wa_hit_top{k}"] = hit_at(candidates, str(record.get("canonical_wa_key")), "canonical_wa_key", k)
            sample[f"ordered_skeleton_hit_top{k}"] = hit_at(candidates, str(record.get("canonical_skeleton_key")), "canonical_skeleton_key", k)
            sample[f"wa_multiset_equiv_hit_top{k}"] = multiset_hit_at(candidates, gt_wa_multiset, "wa", k)
            sample[f"skeleton_multiset_equiv_hit_top{k}"] = multiset_hit_at(candidates, gt_skel_multiset, "skeleton", k)
            sample[f"order_only_wa_miss_top{k}"] = (
                sample[f"wa_multiset_equiv_hit_top{k}"] and not sample[f"ordered_exact_wa_hit_top{k}"]
            )
        per_sample.append(sample)

    denom = max(1, len(per_sample))
    summary: dict[str, Any] = {
        "samples": len(per_sample),
        "candidate_count_mean": sum(candidate_counts) / max(1, len(candidate_counts)),
        "candidate_count_p50": sorted(candidate_counts)[len(candidate_counts) // 2] if candidate_counts else 0,
        "candidate_count_p90": sorted(candidate_counts)[int(0.9 * (len(candidate_counts) - 1))] if candidate_counts else 0,
        "source_labels": dict(source_labels),
    }
    for k in TOP_KS:
        summary[f"ordered_exact_wa_hit_top{k}"] = sum(bool(s[f"ordered_exact_wa_hit_top{k}"]) for s in per_sample) / denom
        summary[f"wa_multiset_equiv_hit_top{k}"] = sum(bool(s[f"wa_multiset_equiv_hit_top{k}"]) for s in per_sample) / denom
        summary[f"ordered_skeleton_hit_top{k}"] = sum(bool(s[f"ordered_skeleton_hit_top{k}"]) for s in per_sample) / denom
        summary[f"skeleton_multiset_equiv_hit_top{k}"] = sum(bool(s[f"skeleton_multiset_equiv_hit_top{k}"]) for s in per_sample) / denom
        summary[f"order_only_wa_miss_top{k}"] = sum(bool(s[f"order_only_wa_miss_top{k}"]) for s in per_sample) / denom

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "wa_multiset_equiv_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.out_dir / "wa_multiset_equiv_per_sample.jsonl", per_sample)


if __name__ == "__main__":
    main()
