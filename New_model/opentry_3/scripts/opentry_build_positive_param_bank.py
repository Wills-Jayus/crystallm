#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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
import run_mp20_geometry_breakthrough as gb  # noqa: E402


LABEL_RE = re.compile(r"^([A-Z][a-z]?)(\d+)_0$")


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path, max_records: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if int(max_records) > 0 and len(rows) >= int(max_records):
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
        if not text:
            continue
        if ":" not in text:
            continue
        orbit_id, element = text.rsplit(":", 1)
        out.append((orbit_id, element))
    return out


def symbols_key(symbols: list[str] | tuple[str, ...]) -> str:
    return ",".join(sorted(str(sym) for sym in symbols))


def param_keys(sg: int, orbit_id: str, element: str, symbols: str, sym: str) -> list[tuple[Any, ...]]:
    return [
        ("sg_orbit_element_symbols_sym", int(sg), str(orbit_id), str(element), str(symbols), str(sym)),
        ("sg_orbit_symbols_sym", int(sg), str(orbit_id), str(symbols), str(sym)),
        ("orbit_element_symbols_sym", str(orbit_id), str(element), str(symbols), str(sym)),
        ("orbit_symbols_sym", str(orbit_id), str(symbols), str(sym)),
        ("orbit_sym", str(orbit_id), str(sym)),
    ]


def parse_first_row_coords(cif: str) -> dict[int, dict[str, float]]:
    coords: dict[int, dict[str, float]] = {}
    lines = str(cif or "").splitlines()
    in_loop = False
    for line in lines:
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
        label = parts[1]
        match = LABEL_RE.match(label)
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


def atom_bucket(atom_count: int) -> str:
    if int(atom_count) <= 8:
        return "atom_le_8"
    if int(atom_count) <= 11:
        return "atom_9_11"
    if int(atom_count) <= 23:
        return "atom_12_23"
    return "atom_ge_24"


def crystal_system(sg: int) -> str:
    sg = int(sg)
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train-positive free-parameter/VPA bank from labeled rendered CIF candidates.")
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-name", default="positive_param_bank.json")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-rmsd", type=float, default=0.0)
    parser.add_argument("--min-target-row-count", type=int, default=0)
    parser.add_argument("--rows-ge-7-weight", type=int, default=3)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrbitEngine(args.lookup_json)
    rows = read_jsonl(args.features_jsonl, max_records=int(args.max_rows))

    param_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    vpa_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    scanned_positive = 0
    extracted_positive = 0
    row_param_values = 0
    for row in rows:
        if not bool(row.get("label_match")):
            continue
        if int(row.get("target_row_count", 0)) < int(args.min_target_row_count):
            continue
        if row.get("label_rmsd") is not None and float(row["label_rmsd"]) < float(args.min_rmsd):
            continue
        scanned_positive += 1
        sg = int(row.get("sg") or 0)
        wa_rows = split_wa_key(str(row.get("canonical_wa_key") or ""))
        first_coords = parse_first_row_coords(str(row.get("cif") or ""))
        if not wa_rows or not first_coords:
            continue
        weight = int(args.rows_ge_7_weight) if int(row.get("target_row_count", 0)) >= 7 else 1
        for row_idx, (orbit_id, element) in enumerate(wa_rows):
            coord = first_coords.get(row_idx)
            if coord is None:
                continue
            try:
                orbit = engine.get_orbit_by_id(str(orbit_id))
            except Exception:
                continue
            syms = [str(sym) for sym in getattr(orbit, "free_symbols", []) if str(sym) in coord]
            if not syms:
                continue
            skey = symbols_key(syms)
            for sym in syms:
                value = float(coord[sym]) % 1.0
                for key in param_keys(sg, orbit.canonical_orbit_id, element, skey, sym):
                    param_values[key].extend([value] * max(1, weight))
                row_param_values += 1
        vpa = row.get("self_volume_per_atom")
        if vpa is not None:
            try:
                vpa_f = float(vpa)
            except Exception:
                vpa_f = 0.0
            if vpa_f > 0.0:
                ab = atom_bucket(int(row.get("atom_count", 0)))
                for key in (
                    ("sg_atom_bucket", sg, ab),
                    ("crystal_atom_bucket", crystal_system(sg), ab),
                    ("atom_bucket", ab),
                    ("global",),
                ):
                    vpa_values[key].extend([vpa_f] * max(1, weight))
        extracted_positive += 1

    payload = {
        "positive_param_values": [
            {"key": list(key), "values": values[:5000], "count": len(values)}
            for key, values in sorted(param_values.items(), key=lambda item: str(item[0]))
        ],
        "positive_vpa_values": [
            {"key": list(key), "values": values[:5000], "count": len(values)}
            for key, values in sorted(vpa_values.items(), key=lambda item: str(item[0]))
        ],
        "summary": {
            "features_jsonl": str(args.features_jsonl),
            "input_rows": len(rows),
            "positive_rows_scanned": scanned_positive,
            "positive_rows_extracted": extracted_positive,
            "param_buckets": len(param_values),
            "vpa_buckets": len(vpa_values),
            "row_param_values": row_param_values,
            "min_target_row_count": int(args.min_target_row_count),
            "rows_ge_7_weight": int(args.rows_ge_7_weight),
            "note": "Built only from train rendered candidates with StructureMatcher-positive labels; no val/test labels are included.",
        },
    }
    write_json(out_dir / args.out_name, payload)
    write_json(out_dir / "positive_param_bank_summary.json", payload["summary"])
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
