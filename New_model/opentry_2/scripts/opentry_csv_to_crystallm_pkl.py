#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2")


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    root = OPENTRY_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to write outside opentry_2: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CrystaLLM benchmark CSV rows to gzipped pickle pairs.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows after start-index.")
    args = parser.parse_args()

    out_path = ensure_under_opentry(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    missing = {"material_id", "cif"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    start = max(0, int(args.start_index))
    end = None if int(args.limit) <= 0 else start + int(args.limit)
    rows: list[tuple[str, str]] = []
    for _, row in df.iloc[start:end].iterrows():
        material_id = str(row["material_id"])
        cif = str(row["cif"])
        if material_id and material_id != "nan" and cif and cif != "nan":
            rows.append((material_id, cif))
    with gzip.open(out_path, "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    print({"input_csv": str(args.input_csv), "out": str(out_path), "rows": len(rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
