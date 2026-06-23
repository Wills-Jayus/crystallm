#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["sample_id"])].append(row)
    return dict(out)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower() or "pool"


def parse_input(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise SystemExit(f"--input must be NAME:PATH, got {spec!r}")
    name, raw_path = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"Empty input name in {spec!r}")
    return name, Path(raw_path)


def cif_hash(row: dict[str, Any]) -> str | None:
    cif = row.get("cif")
    if not isinstance(cif, str) or not cif:
        return None
    return hashlib.sha1(cif.encode("utf-8")).hexdigest()


def strip_labels(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("label_")}


def summarize_labels(rows: list[dict[str, Any]], budgets: tuple[int, ...] = (1, 5, 20, 50)) -> dict[str, Any]:
    by_id = grouped(rows)
    per_sample = []
    for sample_id, items in sorted(by_id.items()):
        items = sorted(items, key=lambda row: int(row.get("rank", 10**9)))
        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "target_row_count": int(items[0].get("target_row_count", 0)),
            "atom_count": int(items[0].get("atom_count", 0)),
        }
        for k in budgets:
            top = items[:k]
            sample[f"match@{k}"] = any(bool(row.get("label_match")) for row in top)
            sample[f"W/A@{k}"] = any(bool(row.get("candidate_wa_hit")) for row in top)
            sample[f"skeleton@{k}"] = any(bool(row.get("candidate_skeleton_hit")) for row in top)
            sample[f"unique_wa@{k}"] = len(set(str(row.get("canonical_wa_key") or "") for row in top))
        sample["has_positive_any"] = any(bool(row.get("label_match")) for row in items)
        per_sample.append(sample)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        denom = max(1, len(items))
        payload: dict[str, Any] = {"samples": len(items)}
        for k in budgets:
            payload[f"match@{k}"] = sum(int(row[f"match@{k}"]) for row in items) / denom
            payload[f"W/A@{k}"] = sum(int(row[f"W/A@{k}"]) for row in items) / denom
            payload[f"skeleton@{k}"] = sum(int(row[f"skeleton@{k}"]) for row in items) / denom
            payload[f"unique_wa_mean@{k}"] = sum(int(row[f"unique_wa@{k}"]) for row in items) / denom
        payload["positive_any_all_candidates"] = sum(int(row["has_positive_any"]) for row in items) / denom
        return payload

    return {
        "full": summarize(per_sample),
        "rows_ge_7": summarize([row for row in per_sample if int(row["target_row_count"]) >= 7]),
        "atoms_ge_12": summarize([row for row in per_sample if int(row["atom_count"]) >= 12]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple opentry_3 rendered-candidate feature tables with pool metadata.")
    parser.add_argument("--input", action="append", required=True, help="Repeated NAME:PATH feature table input.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-name", default="merged_candidate_features.jsonl")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--dedupe-cif", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict[str, Any]] = []
    input_summaries: list[dict[str, Any]] = []
    seen_by_sample: dict[str, set[str]] = defaultdict(set)
    for pool_index, spec in enumerate(args.input):
        name, path = parse_input(spec)
        rows = read_jsonl(path)
        safe = safe_name(name)
        kept = 0
        dropped = 0
        for row_index, row in enumerate(rows):
            item = dict(row)
            sample_id = str(item.get("sample_id"))
            digest = cif_hash(item)
            if args.dedupe_cif and digest is not None:
                if digest in seen_by_sample[sample_id]:
                    dropped += 1
                    continue
                seen_by_sample[sample_id].add(digest)
            old_uid = str(item.get("candidate_uid") or f"rank{item.get('rank', row_index + 1)}")
            old_rank = int(item.get("original_rank") or item.get("rank") or row_index + 1)
            item["original_rank"] = old_rank
            item["proposal_pool"] = str(name)
            item["proposal_pool_id"] = int(pool_index)
            item["proposal_pool_rank"] = old_rank
            item["proposal_pool_input_index"] = int(row_index)
            item[f"proposal_pool_is_{safe}"] = 1
            item["candidate_uid"] = f"{sample_id}::{safe}::{old_uid}"
            merged.append(item)
            kept += 1
        input_summaries.append(
            {
                "name": name,
                "path": str(path),
                "rows": len(rows),
                "kept_rows": kept,
                "dropped_duplicate_cif_rows": dropped,
                "positive_rate": sum(int(bool(row.get("label_match"))) for row in rows) / max(1, len(rows)),
            }
        )

    ranked: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped(merged).items()):
        ordered = sorted(
            items,
            key=lambda row: (
                int(row.get("proposal_pool_id", 10**9)),
                int(row.get("proposal_pool_rank", row.get("rank", 10**9))),
                int(row.get("proposal_pool_input_index", 10**9)),
            ),
        )
        for new_rank, row in enumerate(ordered, start=1):
            item = dict(row)
            item["rank"] = new_rank
            ranked.append(item)

    pool_counts = Counter(str(row.get("proposal_pool")) for row in ranked)
    sample_counts = {sample_id: len(items) for sample_id, items in grouped(ranked).items()}
    feature_path = out_dir / args.out_name
    write_jsonl(feature_path, ranked)
    pool_order_topk = []
    for sample_id, items in sorted(grouped(ranked).items()):
        for rank, row in enumerate(items[: int(args.top_k)], start=1):
            item = dict(row)
            item["rank"] = rank
            pool_order_topk.append(item)
    write_jsonl(out_dir / "rendered_topk_pool_order.jsonl", [strip_labels(row) for row in pool_order_topk])

    first_pool_name = parse_input(args.input[0])[0]
    first_pool_samples = grouped([row for row in ranked if str(row.get("proposal_pool")) == first_pool_name])
    merged_samples = grouped(ranked)
    secondary_new_positive = 0
    for sample_id, items in merged_samples.items():
        first_items = first_pool_samples.get(sample_id, [])
        first_has = any(bool(row.get("label_match")) for row in first_items)
        merged_has = any(bool(row.get("label_match")) for row in items)
        if merged_has and not first_has:
            secondary_new_positive += 1

    summary = {
        "inputs": input_summaries,
        "out_features": str(feature_path),
        "rows": len(ranked),
        "samples": len(sample_counts),
        "rows_per_sample": {
            "min": min(sample_counts.values()) if sample_counts else 0,
            "max": max(sample_counts.values()) if sample_counts else 0,
            "mean": sum(sample_counts.values()) / max(1, len(sample_counts)),
        },
        "pool_counts": dict(pool_counts),
        "dedupe_cif": bool(args.dedupe_cif),
        "pool_order_label_metrics": summarize_labels(pool_order_topk),
        "all_candidate_label_ceiling": summarize_labels(ranked),
        "secondary_new_positive_samples_vs_first_pool": secondary_new_positive,
        "note": "all_candidate_label_ceiling is a validation/train diagnostic only; it is not an allowed inference ranking.",
    }
    write_json(out_dir / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
