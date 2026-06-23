#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
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


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train-only geometry bundle bank from positive rendered candidates with params/lattice metadata.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--rendered-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=20000)
    parser.add_argument("--require-wa-hit", action="store_true")
    parser.add_argument("--min-target-row-count", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrbitEngine(args.lookup_json)
    rendered_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    rendered_rows = read_jsonl(args.rendered_jsonl)
    for row in rendered_rows:
        rendered_by_key[(str(row.get("sample_id") or ""), int(row.get("rank") or 0))] = row

    entries: list[dict[str, Any]] = []
    input_rows = 0
    positive_rows = 0
    skipped_no_render = 0
    skipped_no_metadata = 0
    skipped_wa = 0
    rows_ge_7 = 0
    for feature in read_jsonl(args.train_features):
        input_rows += 1
        if str(feature.get("split") or "train") != "train":
            raise SystemExit(f"Expected train features only, got split={feature.get('split')!r}")
        if not bool(feature.get("label_match")):
            continue
        if bool(args.require_wa_hit) and not bool(feature.get("candidate_wa_hit")):
            continue
        target_row_count = int(feature.get("target_row_count") or feature.get("candidate_row_count") or 0)
        if int(args.min_target_row_count) > 0 and target_row_count < int(args.min_target_row_count):
            continue
        key = (str(feature.get("sample_id") or ""), int(feature.get("rank") or 0))
        rendered = rendered_by_key.get(key)
        if rendered is None:
            skipped_no_render += 1
            continue
        params = rendered.get("geometry_params")
        lattice = rendered.get("geometry_lattice")
        if not isinstance(params, dict) or not isinstance(lattice, dict):
            skipped_no_metadata += 1
            continue
        wa_key = str(feature.get("canonical_wa_key") or rendered.get("canonical_wa_key") or "")
        try:
            canonical_rows = rows_from_wa_key(engine, wa_key)
        except Exception:
            skipped_wa += 1
            continue
        atom_count = int(feature.get("atom_count") or 0)
        sg = int(feature.get("sg") or 0)
        skel_key = str(feature.get("canonical_skeleton_key") or rendered.get("canonical_skeleton_key") or "")
        row_params: list[dict[str, Any]] = []
        for idx, row in enumerate(canonical_rows):
            row_params.append(
                {
                    "row_index": int(idx),
                    "orbit_id": str(row.get("orbit_id")),
                    "element": str(row.get("element")),
                    "multiplicity": int(row.get("multiplicity", 1)),
                    "site_symmetry": str(row.get("site_symmetry")),
                    "free_symbols": [str(sym) for sym in (row.get("free_symbols") or [])],
                    "params": {str(k): float(v) % 1.0 for k, v in dict(params.get(str(idx)) or {}).items()},
                }
            )
        positive_rows += 1
        rows_ge_7 += int(target_row_count >= 7)
        rank = int(feature.get("rank") or rendered.get("rank") or 99)
        rms = None if feature.get("label_rmsd") is None else float(feature.get("label_rmsd"))
        score = 1.0 + 0.35 * float(target_row_count >= 7) - 0.015 * float(rank)
        if rms is not None:
            score -= 0.35 * min(1.0, max(0.0, rms))
        entry = {
            "entry_id": f"{feature.get('sample_id')}::rank{rank}",
            "source_candidate_uid": str(feature.get("candidate_uid") or ""),
            "sample_id": str(feature.get("sample_id") or ""),
            "source_sample_id": str(feature.get("source_sample_id") or rendered.get("source_sample_id") or ""),
            "sg": sg,
            "atom_count": atom_count,
            "target_row_count": target_row_count,
            "canonical_wa_key": wa_key,
            "canonical_skeleton_key": skel_key,
            "rank": rank,
            "label_rmsd": rms,
            "score": float(score),
            "lattice": {str(k): float(v) for k, v in dict(lattice).items()},
            "row_params": row_params,
            "keys": [
                ["wa", sg, wa_key],
                ["skeleton", sg, skel_key],
                ["sg_row_atom", sg, target_row_count, atom_bucket(atom_count)],
                ["sg_atom", sg, atom_bucket(atom_count)],
                ["sg", sg],
            ],
        }
        entries.append(entry)

    entries.sort(key=lambda item: (float(item.get("score", 0.0)), int(item.get("target_row_count", 0))), reverse=True)
    if int(args.max_entries) > 0:
        entries = entries[: int(args.max_entries)]
    summary = {
        "train_features": str(args.train_features),
        "rendered_jsonl": str(args.rendered_jsonl),
        "input_rows": int(input_rows),
        "positive_rows_seen": int(positive_rows),
        "rows_ge_7_positive_rows_seen": int(rows_ge_7),
        "entries": len(entries),
        "unique_wa_keys": len({str(item["canonical_wa_key"]) for item in entries}),
        "unique_skeleton_keys": len({str(item["canonical_skeleton_key"]) for item in entries}),
        "unique_sg": len({int(item["sg"]) for item in entries}),
        "skipped_no_render": int(skipped_no_render),
        "skipped_no_metadata": int(skipped_no_metadata),
        "skipped_wa": int(skipped_wa),
        "require_wa_hit": bool(args.require_wa_hit),
        "min_target_row_count": int(args.min_target_row_count),
        "note": "Train-only StructureMatcher-positive geometry bundles; no val/test labels included.",
    }
    write_json(out_dir / "geometry_bundle_bank.json", {"geometry_bundle_entries": entries, "summary": summary})
    write_json(out_dir / "geometry_bundle_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
