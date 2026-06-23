#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
LABEL_RE = re.compile(r"^([A-Z][a-z]?)(\d+)_0$")


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


def parse_first_row_coords(cif: str) -> dict[int, dict[str, float]]:
    coords: dict[int, dict[str, float]] = {}
    in_loop = False
    for line in str(cif or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("loop_"):
            in_loop = False
            continue
        if text.startswith("_atom_site_type_symbol"):
            in_loop = True
            continue
        if not in_loop or text.startswith("_"):
            continue
        parts = text.split()
        if len(parts) < 7:
            continue
        match = LABEL_RE.match(parts[1])
        if match is None:
            continue
        row_idx = int(match.group(2))
        if row_idx in coords:
            continue
        try:
            coords[row_idx] = {
                "x": float(parts[3]) % 1.0,
                "y": float(parts[4]) % 1.0,
                "z": float(parts[5]) % 1.0,
            }
        except Exception:
            continue
    return coords


def param_keys(sg: int, orbit_id: str, element: str, sym: str) -> list[tuple[Any, ...]]:
    return [
        ("sg_orbit_element_sym", int(sg), str(orbit_id), str(element), str(sym)),
        ("sg_orbit_sym", int(sg), str(orbit_id), str(sym)),
        ("orbit_element_sym", str(orbit_id), str(element), str(sym)),
        ("orbit_sym", str(orbit_id), str(sym)),
        ("sg_sym", int(sg), str(sym)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a train-only row-level free-param feasibility bank from positives vs hard negatives.")
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-name", default="row_feasibility_bank.json")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--require-wa-hit", action="store_true")
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--min-total-per-bin", type=int, default=2)
    parser.add_argument("--min-key-total", type=int, default=8)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.features_jsonl, max_rows=int(args.max_rows))
    bins = max(4, int(args.bins))
    counts: dict[tuple[Any, ...], list[list[int]]] = defaultdict(lambda: [[0, 0] for _ in range(bins)])
    used_candidates = 0
    skipped = 0
    row_values = 0
    positives = 0
    negatives = 0
    for row in rows:
        if int(row.get("target_row_count", 0)) < int(args.min_target_row_count):
            continue
        if bool(args.require_wa_hit) and not bool(row.get("candidate_wa_hit")):
            continue
        wa_rows = split_wa_key(str(row.get("canonical_wa_key") or ""))
        coords = parse_first_row_coords(str(row.get("cif") or ""))
        if not wa_rows or not coords:
            skipped += 1
            continue
        label = 1 if bool(row.get("label_match")) else 0
        positives += int(label)
        negatives += int(not label)
        used_candidates += 1
        sg = int(row.get("sg") or 0)
        for row_idx, (orbit_id, element) in enumerate(wa_rows):
            coord = coords.get(int(row_idx))
            if coord is None:
                continue
            for sym in ("x", "y", "z"):
                if sym not in coord:
                    continue
                value = float(coord[sym]) % 1.0
                bidx = min(bins - 1, max(0, int(math.floor(value * bins))))
                for key in param_keys(sg, orbit_id, element, sym):
                    counts[key][bidx][label] += 1
                row_values += 1

    entries: list[dict[str, Any]] = []
    for key, bin_counts in sorted(counts.items(), key=lambda item: str(item[0])):
        key_total = sum(pos + neg for neg, pos in bin_counts)
        if key_total < int(args.min_key_total):
            continue
        values: list[dict[str, Any]] = []
        for idx, (neg, pos) in enumerate(bin_counts):
            total = int(pos + neg)
            if total < int(args.min_total_per_bin):
                continue
            score = math.log((pos + 1.0) / (neg + 1.0))
            values.append(
                {
                    "center": (idx + 0.5) / float(bins),
                    "bin": idx,
                    "positive": int(pos),
                    "negative": int(neg),
                    "total": total,
                    "score": float(score),
                }
            )
        if not values:
            continue
        values.sort(key=lambda item: (float(item["score"]), int(item["positive"]), int(item["total"])), reverse=True)
        entries.append({"key": list(key), "total": int(key_total), "values": values[:12]})

    summary = {
        "features_jsonl": str(args.features_jsonl),
        "input_rows": len(rows),
        "min_target_row_count": int(args.min_target_row_count),
        "require_wa_hit": bool(args.require_wa_hit),
        "bins": bins,
        "used_candidates": int(used_candidates),
        "skipped_candidates": int(skipped),
        "positive_candidates": int(positives),
        "negative_candidates": int(negatives),
        "row_values": int(row_values),
        "keys_raw": len(counts),
        "keys_kept": len(entries),
        "note": "Train-only row free-param feasibility from StructureMatcher labels; no val/test labels included.",
    }
    payload = {"row_feasibility_param_bins": entries, "summary": summary}
    write_json(out_dir / args.out_name, payload)
    write_json(out_dir / "row_feasibility_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
