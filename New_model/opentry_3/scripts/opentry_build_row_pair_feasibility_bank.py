#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    return [v2.geometry_training_record(row) for row in read_jsonl(path)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_wa_key(wa_key: str) -> list[tuple[str, str]]:
    chunks = str(wa_key or "").split("|setting=")
    out: list[tuple[str, str]] = []
    for idx, chunk in enumerate(chunks):
        text = chunk if idx == 0 else "setting=" + chunk
        if ":" not in text:
            continue
        orbit_id, element = text.rsplit(":", 1)
        out.append((orbit_id, element))
    return out


def rows_from_wa_key(engine: OrbitEngine, wa_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit_id, element in split_wa_key(wa_key):
        orbit = engine.get_orbit_by_id(str(orbit_id))
        rows.append(
            {
                "element": str(element),
                "orbit_id": orbit.canonical_orbit_id,
                "multiplicity": int(orbit.multiplicity),
                "letter": orbit.letter,
                "enumeration": orbit.enumeration,
                "site_symmetry": orbit.site_symmetry,
                "free_symbols": list(orbit.free_symbols),
            }
        )
    return v2.canonical_rows({"wa_table": rows})


def symbols_key(row: dict[str, Any]) -> str:
    return ",".join(sorted(str(sym) for sym in row.get("free_symbols") or []))


def row_match_cost(target_row: dict[str, Any], source_row: dict[str, Any]) -> float:
    cost = 0.0
    if str(target_row.get("orbit_id")) != str(source_row.get("orbit_id")):
        cost += 1.50
    if str(target_row.get("element")) != str(source_row.get("element")):
        cost += 0.70
    if symbols_key(target_row) != symbols_key(source_row):
        cost += 0.50
    if int(target_row.get("multiplicity", 1)) != int(source_row.get("multiplicity", 1)):
        cost += 0.30
    if str(target_row.get("site_symmetry")) != str(source_row.get("site_symmetry")):
        cost += 0.10
    return float(cost)


def greedy_row_pairs(target_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    unused = set(range(len(source_rows)))
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for target_row in target_rows:
        best_idx = None
        best_cost = float("inf")
        for source_idx in unused:
            cost = row_match_cost(target_row, source_rows[source_idx])
            if cost < best_cost:
                best_idx = source_idx
                best_cost = cost
        if best_idx is None:
            continue
        unused.remove(best_idx)
        pairs.append((target_row, source_rows[best_idx]))
    return pairs


def row_pair_keys(target_row: dict[str, Any], source_row: dict[str, Any]) -> list[tuple[Any, ...]]:
    toid = str(target_row.get("orbit_id"))
    telem = str(target_row.get("element"))
    tsym = symbols_key(target_row)
    soid = str(source_row.get("orbit_id"))
    selem = str(source_row.get("element"))
    ssym = symbols_key(source_row)
    return [
        ("target_source_orbit_element_symbols", toid, telem, tsym, soid, selem, ssym),
        ("target_source_orbit_symbols", toid, tsym, soid, ssym),
        ("target_orbit_source_orbit", toid, soid),
        ("compat_flags", int(toid == soid), int(telem == selem), int(tsym == ssym)),
        ("target_orbit", toid),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train-only row-pair source-context feasibility bank.")
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--structured-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-name", default="row_pair_feasibility_bank.json")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--require-wa-hit", action="store_true")
    parser.add_argument("--min-key-total", type=int, default=6)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrbitEngine(args.lookup_json)
    records = load_records(args.structured_root / "train.jsonl")
    record_by_id = {str(record.get("sample_id")): record for record in records}
    feature_rows = read_jsonl(args.features_jsonl, max_rows=int(args.max_rows))

    counts: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
    used = 0
    missing_source = 0
    skipped = 0
    positives = 0
    negatives = 0
    row_pairs = 0
    for row in feature_rows:
        if int(row.get("target_row_count", 0)) < int(args.min_target_row_count):
            continue
        if bool(args.require_wa_hit) and not bool(row.get("candidate_wa_hit")):
            continue
        source = record_by_id.get(str(row.get("source_sample_id") or ""))
        if source is None:
            missing_source += 1
            continue
        try:
            target_rows = rows_from_wa_key(engine, str(row.get("canonical_wa_key") or ""))
        except Exception:
            skipped += 1
            continue
        source_rows = v2.canonical_rows(source)
        if not target_rows or not source_rows:
            skipped += 1
            continue
        label = 1 if bool(row.get("label_match")) else 0
        positives += int(label)
        negatives += int(not label)
        used += 1
        for target_row, source_row in greedy_row_pairs(target_rows, source_rows):
            row_pairs += 1
            for key in row_pair_keys(target_row, source_row):
                counts[key][label] += 1

    entries: list[dict[str, Any]] = []
    for key, (neg, pos) in sorted(counts.items(), key=lambda item: str(item[0])):
        total = int(pos + neg)
        if total < int(args.min_key_total):
            continue
        score = math.log((pos + 1.0) / (neg + 1.0))
        entries.append({"key": list(key), "positive": int(pos), "negative": int(neg), "total": total, "score": float(score)})
    summary = {
        "features_jsonl": str(args.features_jsonl),
        "input_rows": len(feature_rows),
        "min_target_row_count": int(args.min_target_row_count),
        "require_wa_hit": bool(args.require_wa_hit),
        "used_candidates": int(used),
        "missing_source_candidates": int(missing_source),
        "skipped_candidates": int(skipped),
        "positive_candidates": int(positives),
        "negative_candidates": int(negatives),
        "row_pairs": int(row_pairs),
        "keys_raw": len(counts),
        "keys_kept": len(entries),
        "note": "Train-only row-pair feasibility from StructureMatcher labels; no val/test labels included.",
    }
    write_json(out_dir / args.out_name, {"row_pair_feasibility": entries, "summary": summary})
    write_json(out_dir / "row_pair_feasibility_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
