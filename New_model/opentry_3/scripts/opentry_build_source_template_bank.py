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
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atom_bucket(atom_count: int) -> str:
    n = int(atom_count)
    if n <= 4:
        return "le4"
    if n <= 8:
        return "5_8"
    if n <= 12:
        return "9_12"
    if n <= 20:
        return "13_20"
    if n <= 40:
        return "21_40"
    return "gt40"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a train-only source/template bank from positive rendered-candidate labels.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-positive-count", type=int, default=1)
    parser.add_argument("--max-entries", type=int, default=20000)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.train_features)
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    positives = 0
    rows_ge_7_positives = 0
    for row in rows:
        if str(row.get("split") or "train") != "train":
            raise SystemExit(f"Expected train-only features, found split={row.get('split')!r}")
        if not bool(row.get("label_match")):
            continue
        sid = str(row.get("source_sample_id") or "")
        if not sid:
            continue
        positives += 1
        sg = int(row.get("sg") or 0)
        row_count = int(row.get("target_row_count") or 0)
        atom_count = int(row.get("atom_count") or 0)
        if row_count >= 7:
            rows_ge_7_positives += 1
        skeleton_key = str(row.get("canonical_skeleton_key") or "")
        wa_key = str(row.get("canonical_wa_key") or "")
        key = (sg, row_count, atom_bucket(atom_count), skeleton_key, wa_key, sid)
        item = groups.setdefault(
            key,
            {
                "sg": sg,
                "target_row_count": row_count,
                "target_atom_bucket": atom_bucket(atom_count),
                "canonical_skeleton_key": skeleton_key,
                "canonical_wa_key": wa_key,
                "source_sample_id": sid,
                "positive_count": 0,
                "rows_ge_7_positive_count": 0,
                "rank_sum": 0.0,
                "rmsd_sum": 0.0,
                "rmsd_count": 0,
                "min_rmsd": None,
            },
        )
        item["positive_count"] += 1
        if row_count >= 7:
            item["rows_ge_7_positive_count"] += 1
        item["rank_sum"] += float(row.get("original_rank") or row.get("rank") or 99)
        if row.get("label_rmsd") is not None:
            rms = float(row["label_rmsd"])
            item["rmsd_sum"] += rms
            item["rmsd_count"] += 1
            item["min_rmsd"] = rms if item["min_rmsd"] is None else min(float(item["min_rmsd"]), rms)

    entries: list[dict[str, Any]] = []
    for item in groups.values():
        pos = int(item["positive_count"])
        if pos < int(args.min_positive_count):
            continue
        mean_rank = float(item["rank_sum"]) / max(1, pos)
        mean_rmsd = None if int(item["rmsd_count"]) <= 0 else float(item["rmsd_sum"]) / float(item["rmsd_count"])
        rows7 = int(item["rows_ge_7_positive_count"])
        score = math.log1p(pos) + 0.4 * math.log1p(rows7) - 0.025 * mean_rank
        if mean_rmsd is not None:
            score -= 0.25 * min(1.0, max(0.0, float(mean_rmsd)))
        entry = {
            "source_sample_id": item["source_sample_id"],
            "sg": int(item["sg"]),
            "target_row_count": int(item["target_row_count"]),
            "target_atom_bucket": str(item["target_atom_bucket"]),
            "canonical_skeleton_key": str(item["canonical_skeleton_key"]),
            "canonical_wa_key": str(item["canonical_wa_key"]),
            "positive_count": pos,
            "rows_ge_7_positive_count": rows7,
            "mean_original_rank": mean_rank,
            "mean_rmsd": mean_rmsd,
            "min_rmsd": item["min_rmsd"],
            "score": float(score),
        }
        entries.append(entry)

    entries.sort(key=lambda item: (float(item["score"]), int(item["positive_count"])), reverse=True)
    if int(args.max_entries) > 0:
        entries = entries[: int(args.max_entries)]

    payload = {
        "source": str(args.train_features),
        "split": "train",
        "min_positive_count": int(args.min_positive_count),
        "max_entries": int(args.max_entries),
        "input_rows": len(rows),
        "positive_rows": int(positives),
        "rows_ge_7_positive_rows": int(rows_ge_7_positives),
        "entries": entries,
    }
    (out_dir / "source_template_bank.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "input_rows": len(rows),
        "positive_rows": int(positives),
        "rows_ge_7_positive_rows": int(rows_ge_7_positives),
        "entries": len(entries),
        "unique_sources": len({str(item["source_sample_id"]) for item in entries}),
        "unique_wa_keys": len({str(item["canonical_wa_key"]) for item in entries}),
        "unique_skeleton_keys": len({str(item["canonical_skeleton_key"]) for item in entries}),
        "top_score": None if not entries else float(entries[0]["score"]),
    }
    (out_dir / "source_template_bank_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
