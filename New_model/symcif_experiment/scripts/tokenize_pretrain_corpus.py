#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
if str(CRYSTALLM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM_ROOT))

from crystallm import CIFTokenizer  # type: ignore


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#") and "pymatgen" not in stripped:
            lines.append(line.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def tokenize_file(path: Path, tokenizer: CIFTokenizer) -> list[int]:
    text = clean_text(path.read_text(encoding="utf-8"))
    tokens = tokenizer.tokenize_cif(text)
    unk_count = tokens.count("<unk>")
    if unk_count:
        print(f"[warn] {path}: {unk_count} <unk> tokens")
    return tokenizer.encode(tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize A/B/C txt corpora with the extended CIFTokenizer.")
    parser.add_argument("corpus_dir", type=Path, help="Directory containing train.txt and val.txt.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    tok = CIFTokenizer()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = np.array(tokenize_file(args.corpus_dir / "train.txt", tok), dtype=np.uint16)
    val_ids = np.array(tokenize_file(args.corpus_dir / "val.txt", tok), dtype=np.uint16)
    train_ids.tofile(args.out_dir / "train.bin")
    val_ids.tofile(args.out_dir / "val.bin")
    meta = {
        "vocab_size": len(tok.token_to_id),
        "itos": tok.id_to_token,
        "stoi": tok.token_to_id,
    }
    with open(args.out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    subdir_name = args.out_dir.name
    with tarfile.open(args.out_dir / f"{subdir_name}.tar.gz", "w:gz") as tar:
        for name in ("train.bin", "val.bin", "meta.pkl"):
            tar.add(args.out_dir / name, arcname=f"{subdir_name}/{name}")

    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens: {len(val_ids):,}")
    print(f"vocab size: {len(tok.token_to_id):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
