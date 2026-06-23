#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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


def atom_bucket(atom_count: int) -> str:
    if atom_count <= 8:
        return "atom_le_8"
    if atom_count <= 11:
        return "atom_9_11"
    if atom_count <= 23:
        return "atom_12_23"
    return "atom_ge_24"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, int] = {}
    for row in rows:
        buckets[atom_bucket(int(row["atom_count"]))] = buckets.get(atom_bucket(int(row["atom_count"])), 0) + 1
    row_counts = [int(row.get("row_count", 0)) for row in rows]
    atom_counts = [int(row.get("atom_count", 0)) for row in rows]
    return {
        "records": len(rows),
        "atom_buckets": buckets,
        "rows_ge_7": sum(int(x >= 7) for x in row_counts),
        "atoms_ge_12": sum(int(x >= 12) for x in atom_counts),
        "complex_flag": sum(int(bool(row.get("complex_flag"))) for row in rows),
        "unique_sg": len({int(row["sg"]) for row in rows}),
        "row_count_mean": None if not row_counts else sum(row_counts) / len(row_counts),
        "atom_count_mean": None if not atom_counts else sum(atom_counts) / len(atom_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a train-only atom-bucket balanced Wyckoff representation subset.")
    parser.add_argument("--input-train", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52" / "train.jsonl")
    parser.add_argument("--input-val", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52" / "val.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=256)
    parser.add_argument("--val-records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.input_train)
    val_rows = read_jsonl(args.input_val)
    rng = random.Random(int(args.seed))

    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        by_bucket.setdefault(atom_bucket(int(row["atom_count"])), []).append(row)

    selected: list[dict[str, Any]] = []
    bucket_summary: dict[str, dict[str, Any]] = {}
    for bucket in ("atom_le_8", "atom_9_11", "atom_12_23", "atom_ge_24"):
        rows = list(by_bucket.get(bucket, []))
        rng.shuffle(rows)
        take = rows[: int(args.per_bucket)]
        selected.extend(take)
        bucket_summary[bucket] = {
            "available": len(rows),
            "selected": len(take),
        }

    selected.sort(key=lambda row: str(row["keys"]["sample_id"]))
    selected_ids = {str(row["keys"]["sample_id"]) for row in selected}
    if len(selected_ids) != len(selected):
        raise SystemExit("Selected subset contains duplicate sample ids.")

    val_take = val_rows[: max(0, int(args.val_records))]
    write_jsonl(out_dir / "train.jsonl", selected)
    write_jsonl(out_dir / "val.jsonl", val_take)
    summary = {
        "input_train": str(args.input_train),
        "input_val": str(args.input_val),
        "selection_policy": "train-only atom_count buckets; row_count is reported only, not used for selection or model input",
        "per_bucket": int(args.per_bucket),
        "seed": int(args.seed),
        "bucket_selection": bucket_summary,
        "train_summary": summarize(selected),
        "val_records_copied_for_interface": len(val_take),
        "no_leakage_note": "Uses train split for selected training subset; val copy is only for script interface compatibility.",
    }
    write_json(out_dir / "selection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
