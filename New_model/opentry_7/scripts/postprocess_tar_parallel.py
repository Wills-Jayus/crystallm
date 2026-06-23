#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
import tarfile
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
CRYSTALLM_ROOT = WORKSPACE / "model/scp_task/CrystaLLM"
if not CRYSTALLM_ROOT.exists():
    CRYSTALLM_ROOT = WORKSPACE / "model/CrystaLLM"
if str(CRYSTALLM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM_ROOT))

from crystallm import (  # noqa: E402
    extract_space_group_symbol,
    remove_atom_props_block,
    replace_symmetry_operators,
)


def under_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_7: {resolved}")
    return resolved


def postprocess(cif: str, fname: str) -> str:
    try:
        space_group_symbol = extract_space_group_symbol(cif)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif = replace_symmetry_operators(cif, space_group_symbol)
        cif = remove_atom_props_block(cif)
    except Exception:  # noqa: BLE001
        cif = "# WARNING: CrystaLLM could not post-process this file properly!\n" + cif
    return cif


def iter_cifs(path: Path) -> Iterable[tuple[str, str]]:
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".cif"):
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            yield member.name, fileobj.read().decode("utf-8", errors="replace")


def worker(row: tuple[str, str]) -> tuple[str, bytes]:
    name, cif = row
    processed = postprocess(cif, name).encode("utf-8")
    return name, processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process a CIF tar.gz with official CrystaLLM logic in parallel.")
    parser.add_argument("inp", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    inp = args.inp.expanduser().resolve()
    out = under_root(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(args.workers))

    rows = iter_cifs(inp)
    with tarfile.open(out, "w:gz") as out_tar:
        if workers == 1:
            result_iter = map(worker, rows)
            for name, payload in result_iter:
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                out_tar.addfile(info, io.BytesIO(payload))
        else:
            with Pool(processes=workers) as pool:
                for name, payload in pool.imap(worker, rows, chunksize=64):
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    out_tar.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    main()
