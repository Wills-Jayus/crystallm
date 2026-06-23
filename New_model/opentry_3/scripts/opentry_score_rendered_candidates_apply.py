#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from opentry_build_geometry_compat_features import crystal_system, parse_wa_metadata
from opentry_train_geometry_compat_selector import ensure_under_opentry, read_jsonl, score_rows, write_json, write_jsonl


def load_gt_free_repr_metadata(path: Path | None, max_records: int | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        keys = row.get("keys") or {}
        sample_id = str(keys.get("sample_id") or row.get("sample_id") or "")
        if not sample_id:
            continue
        formula_counts = row.get("formula_counts") or {}
        out[sample_id] = {
            "material_id": keys.get("material_id"),
            "sg": int(row.get("sg", 0) or 0),
            "atom_count": int(row.get("atom_count", 0) or 0),
            "formula_element_count": len(formula_counts),
        }
        if max_records is not None and len(out) >= int(max_records):
            break
    return out


def enrich_rows(rows: list[dict[str, Any]], repr_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        sample_id = str(item.get("sample_id") or "")
        meta = repr_meta.get(sample_id) or {}
        for key in ("material_id", "sg", "atom_count", "formula_element_count"):
            if key not in item and key in meta:
                item[key] = meta[key]
        try:
            sg = int(item.get("sg") or item.get("detected_sg") or 0)
        except Exception:
            sg = 0
        if "crystal_system" not in item:
            item["crystal_system"] = crystal_system(sg)
        item.update(parse_wa_metadata(str(item.get("canonical_wa_key") or "")))
        if "candidate_uid" not in item:
            rank = int(item.get("rank") or item.get("original_rank") or 0)
            item["candidate_uid"] = f"{sample_id}::rank{rank}"
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a frozen GT-free compatibility model to rendered candidates.")
    parser.add_argument("--rendered-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = int(args.max_records) if int(args.max_records) > 0 else None
    rows = read_jsonl(args.rendered_jsonl)
    repr_meta = load_gt_free_repr_metadata(args.repr_jsonl, max_records)
    if max_records is not None and repr_meta:
        allowed = set(repr_meta)
        rows = [row for row in rows if str(row.get("sample_id")) in allowed]
    enriched = enrich_rows(rows, repr_meta)
    model = joblib.load(args.model)
    scored = score_rows(model, enriched)
    write_jsonl(out_dir / "scored_candidates.jsonl", scored)
    summary = {
        "rendered_jsonl": str(args.rendered_jsonl),
        "model": str(args.model),
        "repr_jsonl": None if args.repr_jsonl is None else str(args.repr_jsonl),
        "max_records": 0 if max_records is None else int(max_records),
        "input_rows": len(rows),
        "scored_rows": len(scored),
        "samples": len({str(row.get("sample_id")) for row in scored}),
        "feature_note": "GT-free apply path: repr metadata contributes only formula-derived atom/element counts and GT-SG; target W/A, row_count, labels, and StructureMatcher values are not read into output features.",
    }
    write_json(out_dir / "score_apply_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
