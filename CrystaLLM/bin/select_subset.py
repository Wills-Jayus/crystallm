#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List


def _list_files(in_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        it = in_dir.rglob(pattern)
    else:
        it = in_dir.glob(pattern)
    return sorted([p for p in it if p.is_file()])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select N files from a directory (optionally random) and copy them to an output directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-dir", required=True, help="Input directory to read files from.")
    p.add_argument("--out-dir", required=True, help="Output directory to copy selected files into.")
    p.add_argument("--num", type=int, default=100, help="Number of files to select.")
    p.add_argument("--glob", default="*.cif", help="Filename glob pattern to match.")
    p.add_argument("--recursive", action="store_true", help="Match files recursively under --in-dir.")
    p.add_argument("--random", action="store_true", help="Randomly select files (otherwise sorted order).")
    p.add_argument("--seed", type=int, default=0, help="Random seed used when --random is set.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    pattern = str(args.glob)
    n = int(args.num)
    recursive = bool(args.recursive)

    if not in_dir.exists():
        raise SystemExit(f"--in-dir not found: {in_dir}")
    if not in_dir.is_dir():
        raise SystemExit(f"--in-dir is not a directory: {in_dir}")
    if n <= 0:
        raise SystemExit("--num must be > 0")

    files = _list_files(in_dir, pattern, recursive)
    if not files:
        raise SystemExit(f"No files matched under {in_dir} (glob={pattern!r}, recursive={recursive}).")

    if args.random:
        rng = random.Random(int(args.seed))
        rng.shuffle(files)

    if len(files) < n:
        raise SystemExit(f"Only matched {len(files)} files but --num={n} was requested.")

    selected = files[:n]
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in selected:
        rel = src.relative_to(in_dir)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and not args.overwrite:
            raise SystemExit(f"Destination already exists (use --overwrite): {dest}")

        shutil.copy2(src, dest)
        copied += 1

    print(f"[select_subset] selected={len(selected)} copied={copied} -> {out_dir}")


if __name__ == "__main__":
    main()
