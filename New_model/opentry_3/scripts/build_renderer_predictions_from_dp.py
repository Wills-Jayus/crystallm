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


def build_orbit_metadata(train_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for record in train_records:
        for row in record.get("wa_sequence") or []:
            orbit_id = str(row["orbit_id"])
            meta.setdefault(
                orbit_id,
                {
                    "orbit_id": orbit_id,
                    "letter": row.get("letter"),
                    "multiplicity": row.get("multiplicity"),
                    "site_symmetry": row.get("site_symmetry"),
                    "enumeration": row.get("enumeration"),
                    "free_symbols": list(row.get("free_param_names") or row.get("free_symbols") or []),
                },
            )
    return meta


def has_duplicate_fixed_orbit(rows: list[dict[str, Any]], orbit_meta: dict[str, dict[str, Any]]) -> bool:
    counts: dict[str, int] = {}
    for row in rows:
        orbit_id = str(row["orbit_id"])
        meta = orbit_meta.get(orbit_id) or {}
        free_symbols = list(meta.get("free_symbols") or [])
        if free_symbols:
            continue
        counts[orbit_id] = counts.get(orbit_id, 0) + 1
        if counts[orbit_id] > 1:
            return True
    return False


def canonical_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    enum = row.get("enumeration")
    enum_key = "" if enum is None else str(enum)
    return (
        int(row.get("multiplicity", 0)),
        str(row.get("letter") or ""),
        enum_key,
        str(row.get("site_symmetry") or ""),
        str(row.get("element") or ""),
        str(row.get("orbit_id") or ""),
    )


def canonical_keys_from_rows(rows: list[dict[str, Any]]) -> tuple[str, str]:
    ordered = sorted(rows, key=canonical_sort_key)
    skeleton_key = "|".join(str(row["orbit_id"]) for row in ordered)
    wa_key = "|".join(f"{row['orbit_id']}:{row['element']}" for row in ordered)
    return skeleton_key, wa_key


def pairs_to_rows(pairs: list[list[Any]] | list[tuple[Any, Any]], orbit_meta: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        element = str(pair[0])
        orbit_id = str(pair[1])
        meta = orbit_meta.get(orbit_id)
        if meta is None:
            missing.append(orbit_id)
            row = {"element": element, "orbit_id": orbit_id}
        else:
            row = {
                "element": element,
                "orbit_id": orbit_id,
                "letter": meta.get("letter"),
                "multiplicity": meta.get("multiplicity"),
                "site_symmetry": meta.get("site_symmetry"),
                "enumeration": meta.get("enumeration"),
            }
        rows.append(row)
    return rows, missing


def convert_candidates(
    dp_rows: list[dict[str, Any]],
    orbit_meta: dict[str, dict[str, Any]],
    *,
    top_k: int,
    drop_duplicate_fixed_orbits: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    samples_with_candidates = 0
    dropped_duplicate_fixed = 0
    missing_orbits: dict[str, int] = {}
    for source in dp_rows:
        sample_id = str(source["sample_id"])
        converted: list[dict[str, Any]] = []
        seen: set[str] = set()
        for cand in source.get("candidates") or []:
            total_in += 1
            wa_key = str(cand.get("wa_key") or "")
            if not wa_key:
                continue
            rows, missing = pairs_to_rows(cand.get("pairs") or [], orbit_meta)
            for oid in missing:
                missing_orbits[oid] = missing_orbits.get(oid, 0) + 1
            if missing or not rows:
                continue
            if drop_duplicate_fixed_orbits and has_duplicate_fixed_orbit(rows, orbit_meta):
                dropped_duplicate_fixed += 1
                continue
            canonical_skeleton_key, canonical_wa_key = canonical_keys_from_rows(rows)
            if canonical_wa_key in seen:
                continue
            seen.add(canonical_wa_key)
            converted.append(
                {
                    "canonical_wa_key": canonical_wa_key,
                    "canonical_skeleton_key": canonical_skeleton_key,
                    "raw_dp_wa_key": wa_key,
                    "raw_dp_skeleton_key": str(cand.get("skeleton_key") or ""),
                    "rows": rows,
                    "score": float(cand.get("score") or 0.0),
                    "source": str(cand.get("source") or "assignment_dp_prior"),
                    "composition_exact": bool(cand.get("composition_exact", True)),
                }
            )
            if len(converted) >= int(top_k):
                break
        if converted:
            samples_with_candidates += 1
        total_out += len(converted)
        predictions.append({"sample_id": sample_id, "ranked_wa_candidates": converted})
    summary = {
        "samples": len(predictions),
        "samples_with_candidates": samples_with_candidates,
        "candidate_nonempty_rate": samples_with_candidates / max(1, len(predictions)),
        "input_candidates_seen": total_in,
        "output_candidates": total_out,
        "top_k": int(top_k),
        "missing_orbit_ids": len(missing_orbits),
        "missing_orbit_references": sum(missing_orbits.values()),
        "missing_orbit_examples": sorted(missing_orbits)[:20],
        "dropped_duplicate_fixed_orbit_candidates": dropped_duplicate_fixed,
        "drop_duplicate_fixed_orbits": bool(drop_duplicate_fixed_orbits),
    }
    return predictions, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert opentry_3 DP W/A candidates to SymCIF-v4 renderer predictions.")
    parser.add_argument("--train-repr", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52" / "train.jsonl")
    parser.add_argument("--dp-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--allow-duplicate-fixed-orbits", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    max_records = None if int(args.max_records) <= 0 else int(args.max_records)
    train_records = read_jsonl(args.train_repr)
    dp_rows = read_jsonl(args.dp_candidates, max_records=max_records)
    orbit_meta = build_orbit_metadata(train_records)
    predictions, summary = convert_candidates(
        dp_rows,
        orbit_meta,
        top_k=int(args.top_k),
        drop_duplicate_fixed_orbits=not bool(args.allow_duplicate_fixed_orbits),
    )
    summary.update(
        {
            "train_repr": str(args.train_repr),
            "dp_candidates": str(args.dp_candidates),
            "orbit_metadata_source": "train_repr_only",
            "train_orbits": len(orbit_meta),
            "out_predictions": str(out_dir / "renderer_predictions.jsonl"),
        }
    )
    write_jsonl(out_dir / "renderer_predictions.jsonl", predictions)
    write_json(out_dir / "renderer_prediction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
