#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise ValueError(f"refusing to write outside opentry_2: {resolved}")
    return resolved


def material_id_from_name(path: Path, prefix: str) -> str:
    stem = path.stem
    if prefix and stem.startswith(prefix):
        return stem[len(prefix) :]
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an opentry-local benchmark CSV and GT CIF dir from prepared CIF files.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    out_dir = ensure_under_opentry(Path(args.out_dir))
    gt_dir = ensure_under_opentry(out_dir / "gt_cifs")
    csv_path = ensure_under_opentry(out_dir / "subset.csv")
    manifest_path = ensure_under_opentry(out_dir / "manifest.json")
    gt_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(source_dir.glob("*.cif"))
    start = max(0, int(args.start_index))
    stop = None if int(args.limit) <= 0 else start + int(args.limit)
    selected = paths[start:stop]
    rows = []
    for src in selected:
        mid = material_id_from_name(src, str(args.prefix))
        cif = src.read_text(encoding="utf-8", errors="ignore")
        dst = gt_dir / f"{mid}.cif"
        if not dst.exists():
            if args.copy_mode == "hardlink":
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        rows.append({"material_id": mid, "cif": cif, "source": str(src)})

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["material_id", "cif"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"material_id": row["material_id"], "cif": row["cif"]})

    manifest = {
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "gt_dir": str(gt_dir),
        "csv": str(csv_path),
        "prefix": str(args.prefix),
        "start_index": int(args.start_index),
        "limit": int(args.limit),
        "rows": len(rows),
        "copy_mode": str(args.copy_mode),
        "first_ids": [row["material_id"] for row in rows[:10]],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
