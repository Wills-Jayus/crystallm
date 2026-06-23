#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["sample_id"])].append(row)
    for items in by_id.values():
        items.sort(key=lambda row: int(row.get("rank", 10**9)))
    return dict(by_id)


def add_candidate(
    out: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    source_name: str,
    seen_cifs: set[str],
    top_k: int,
) -> bool:
    if len(out) >= int(top_k):
        return False
    cif = str(row.get("cif") or "")
    if cif in seen_cifs:
        return False
    seen_cifs.add(cif)
    item = dict(row)
    item["merge_source"] = source_name
    item["source_rank"] = int(row.get("rank", 10**9))
    out.append(item)
    return True


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    denom = max(1, len(rows))
    by_source: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[str(row.get("merge_source") or "unknown")] += 1
    return {
        "samples": len(sample_ids),
        "rows": len(rows),
        "by_source": dict(sorted(by_source.items())),
        "readable": sum(bool(row.get("readable")) for row in rows) / denom,
        "composition_exact": sum(bool(row.get("composition_exact")) for row in rows) / denom,
        "sg_ok": sum(bool(row.get("sg_ok")) for row in rows) / denom,
        "unique_wa_mean": sum(
            len({str(row.get("canonical_wa_key") or "") for row in rows if str(row["sample_id"]) == sample_id})
            for sample_id in sample_ids
        )
        / max(1, len(sample_ids)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge rendered CIF lists with a GT-free prefix-tail policy.")
    parser.add_argument("--prefix-jsonl", type=Path, required=True, help="Early-rank list, usually larger-pool self-score.")
    parser.add_argument("--tail-jsonl", type=Path, required=True, help="Tail-preserving list, usually K50-pool self-score.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_by_id = grouped(read_jsonl(args.prefix_jsonl))
    tail_by_id = grouped(read_jsonl(args.tail_jsonl))
    sample_ids = sorted(set(prefix_by_id) | set(tail_by_id))
    merged: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        out: list[dict[str, Any]] = []
        seen_cifs: set[str] = set()
        prefix_rows = prefix_by_id.get(sample_id, [])
        tail_rows = tail_by_id.get(sample_id, [])
        for row in prefix_rows[: int(args.prefix_k)]:
            add_candidate(out, row, source_name="prefix", seen_cifs=seen_cifs, top_k=int(args.top_k))
        for row in tail_rows:
            add_candidate(out, row, source_name="tail", seen_cifs=seen_cifs, top_k=int(args.top_k))
            if len(out) >= int(args.top_k):
                break
        for row in prefix_rows[int(args.prefix_k) :]:
            add_candidate(out, row, source_name="prefix_fill", seen_cifs=seen_cifs, top_k=int(args.top_k))
            if len(out) >= int(args.top_k):
                break
        for rank, row in enumerate(out, start=1):
            item = dict(row)
            item["rank"] = rank
            merged.append(item)
        per_sample.append(
            {
                "sample_id": sample_id,
                "prefix_input_count": len(prefix_rows),
                "tail_input_count": len(tail_rows),
                "output_count": len(out),
                "prefix_used": sum(1 for row in out if str(row.get("merge_source")) == "prefix"),
                "tail_used": sum(1 for row in out if str(row.get("merge_source")) == "tail"),
                "prefix_fill_used": sum(1 for row in out if str(row.get("merge_source")) == "prefix_fill"),
                "unique_wa": len({str(row.get("canonical_wa_key") or "") for row in out}),
            }
        )

    merged.sort(key=lambda row: (str(row["sample_id"]), int(row["rank"])))
    write_jsonl(out_dir / "rendered_topk_prefix_tail.jsonl", merged)
    write_jsonl(out_dir / "prefix_tail_per_sample.jsonl", per_sample)
    summary = {
        "prefix_jsonl": str(args.prefix_jsonl),
        "tail_jsonl": str(args.tail_jsonl),
        "prefix_k": int(args.prefix_k),
        "top_k": int(args.top_k),
        "summary": summarize(merged),
        "output_count_mean": sum(int(row["output_count"]) for row in per_sample) / max(1, len(per_sample)),
        "unique_wa_mean": sum(int(row["unique_wa"]) for row in per_sample) / max(1, len(per_sample)),
    }
    write_json(out_dir / "prefix_tail_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
