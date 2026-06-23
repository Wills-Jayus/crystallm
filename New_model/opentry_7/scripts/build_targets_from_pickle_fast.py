#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREFIXES = {
    "mp20": "mp_20",
    "mpts52": "mpts_52",
}


def under_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_7: {resolved}")
    return resolved


def extract_data_name(cif: str) -> str:
    match = re.search(r"(?im)^\s*data_([^\s]+)", cif or "")
    return match.group(1).strip() if match else "UNKNOWN"


def extract_formula_text(cif: str, data_name: str) -> str:
    for key in ("_chemical_formula_structural", "_chemical_formula_sum"):
        for line in cif.splitlines():
            stripped = line.strip()
            if stripped.startswith(key):
                return stripped[len(key):].strip().strip("'\"")
    return data_name


def count_atom_site_rows(cif: str) -> int:
    in_atom_loop = False
    saw_atom_header = False
    count = 0
    for raw in cif.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "loop_":
            if in_atom_loop and saw_atom_header and count:
                break
            in_atom_loop = False
            saw_atom_header = False
            continue
        if stripped.startswith("_atom_site_"):
            in_atom_loop = True
            saw_atom_header = True
            continue
        if stripped.startswith("_") or stripped.startswith("data_"):
            if in_atom_loop and saw_atom_header:
                break
            continue
        if in_atom_loop and saw_atom_header:
            count += 1
    return count


def sample_aliases(prefix: str, split: str, material_id: str) -> list[str]:
    aliases = [
        f"{prefix}_{split}_orig__{material_id}",
        f"{prefix}_{split}__{material_id}",
        material_id,
    ]
    out = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            out.append(alias)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build opentry_7 target cache from official CrystaLLM pickle.")
    parser.add_argument("--dataset", choices=sorted(PREFIXES), required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prefix = PREFIXES[args.dataset]
    out = under_root(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(args.input, "rb") as f:
        records = pickle.load(f)

    with out.open("w", encoding="utf-8") as f:
        for material_id, cif in records:
            material_id = str(material_id)
            data_name = extract_data_name(cif)
            row_count = count_atom_site_rows(cif)
            row = {
                "dataset": prefix,
                "split": args.split,
                "sample_id": f"{prefix}_{args.split}_orig__{material_id}",
                "sample_aliases": sample_aliases(prefix, args.split, material_id),
                "material_id": material_id,
                "csv_index": None,
                "cif": cif,
                "data_name": data_name,
                "formula": extract_formula_text(cif, data_name),
                "n_sites": row_count,
                "row_count": row_count,
                "row_count_method": "raw_atom_site_rows_from_official_pickle",
                "sg_number": None,
                "sg_symbol": "P 1",
                "analyzer_error": "fast pickle target cache: Structure parser and SpacegroupAnalyzer skipped",
            }
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    print(f"wrote {len(records)} targets to {out}")


if __name__ == "__main__":
    main()
