#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import random
import sys
import tarfile
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

CRYSTALLM_ROOT = Path("/data/users/xsw/autodlmini/model/scp_task/CrystaLLM").resolve()
OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2").resolve()
if str(CRYSTALLM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM_ROOT))

from crystallm import (  # noqa: E402
    CIFTokenizer,
    add_atomic_props_block,
    extract_formula_units,
    replace_data_formula_with_nonreduced_formula,
    round_numbers,
    semisymmetrize_cif,
)


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise ValueError(f"refusing to write outside opentry_2: {resolved}")
    return resolved


def load_pkl(path: Path) -> Any:
    with gzip.open(path.expanduser(), "rb") as f:
        return pickle.load(f)


def dump_pkl(path: Path, obj: Any) -> None:
    path = ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def cmd_preprocess(args: argparse.Namespace) -> int:
    rows: List[Tuple[str, str]] = list(load_pkl(Path(args.input)))
    out: List[Tuple[str, str]] = []
    failed = 0
    for i, (mid, cif) in enumerate(rows, start=1):
        try:
            if extract_formula_units(cif) == 0:
                raise ValueError("formula_units=0")
            cif2 = replace_data_formula_with_nonreduced_formula(cif)
            cif2 = semisymmetrize_cif(cif2)
            cif2 = add_atomic_props_block(cif2, bool(args.oxi))
            cif2 = round_numbers(cif2, decimal_places=int(args.decimal_places))
            out.append((mid, cif2))
        except Exception:
            failed += 1
        if i % int(args.progress_interval) == 0:
            print(f"[serial-preprocess] {i}/{len(rows)} ok={len(out)} failed={failed}", flush=True)
    dump_pkl(Path(args.out), out)
    print({"input_rows": len(rows), "output_rows": len(out), "failed": failed, "out": str(Path(args.out).resolve())}, flush=True)
    return 0


def clean_cif_lines(cif: str) -> str:
    lines = []
    for line in cif.split("\n"):
        s = line.strip()
        if s and not s.startswith("#") and "pymatgen" not in s:
            lines.append(s)
    lines.append("\n")
    return "\n".join(lines)


def tokenize_rows(rows: List[Tuple[str, str]], *, seed: int, shuffle: bool, progress_interval: int) -> List[str]:
    rows = list(rows)
    if shuffle:
        random.Random(int(seed)).shuffle(rows)
    tokenizer = CIFTokenizer()
    tokens: List[str] = []
    for i, (_mid, cif) in enumerate(rows, start=1):
        tokens.extend(tokenizer.tokenize_cif(clean_cif_lines(cif)))
        if i % int(progress_interval) == 0:
            print(f"[serial-tokenize] {i}/{len(rows)} rows tokens={len(tokens)}", flush=True)
    return tokens


def cmd_tokenize(args: argparse.Namespace) -> int:
    out_dir = ensure_under_opentry(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows: List[Tuple[str, str]] = list(load_pkl(Path(args.train_fname)))
    val_rows: List[Tuple[str, str]] = list(load_pkl(Path(args.val_fname))) if str(args.val_fname or "") else []
    train_tokens = tokenize_rows(train_rows, seed=int(args.seed), shuffle=True, progress_interval=int(args.progress_interval))
    val_tokens = tokenize_rows(val_rows, seed=int(args.seed), shuffle=False, progress_interval=int(args.progress_interval)) if val_rows else []

    tokenizer = CIFTokenizer()
    train_ids = np.array(tokenizer.encode(train_tokens), dtype=np.uint16)
    train_ids.tofile(out_dir / "train.bin")
    if val_rows:
        val_ids = np.array(tokenizer.encode(val_tokens), dtype=np.uint16)
        val_ids.tofile(out_dir / "val.bin")
    meta = {"vocab_size": len(tokenizer.token_to_id), "itos": tokenizer.id_to_token, "stoi": tokenizer.token_to_id}
    with (out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    subdir_name = os.path.basename(os.path.normpath(str(out_dir)))
    tar_path = out_dir / f"{subdir_name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for filename in ["train.bin", "val.bin", "meta.pkl"]:
            p = out_dir / filename
            if p.exists():
                tar.add(p, arcname=os.path.join(subdir_name, filename))
    print(
        {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_tokens": int(train_ids.shape[0]),
            "val_tokens": int(val_ids.shape[0]) if val_rows else 0,
            "vocab_size": len(tokenizer.token_to_id),
            "out_dir": str(out_dir),
            "tar": str(tar_path),
        },
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-process CrystaLLM CIF preprocess/tokenize helpers for opentry_2.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--input", required=True)
    p_pre.add_argument("--out", required=True)
    p_pre.add_argument("--oxi", action="store_true")
    p_pre.add_argument("--decimal-places", type=int, default=4)
    p_pre.add_argument("--progress-interval", type=int, default=250)
    p_pre.set_defaults(func=cmd_preprocess)

    p_tok = sub.add_parser("tokenize")
    p_tok.add_argument("--train-fname", required=True)
    p_tok.add_argument("--val-fname", default="")
    p_tok.add_argument("--out-dir", required=True)
    p_tok.add_argument("--seed", type=int, default=1337)
    p_tok.add_argument("--progress-interval", type=int, default=1000)
    p_tok.set_defaults(func=cmd_tokenize)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
