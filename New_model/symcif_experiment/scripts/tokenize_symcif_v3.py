#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystallm import CIFTokenizer  # type: ignore  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v3.parse import parse_symcif_v3_text  # noqa: E402


def split_concat_records(path: Path) -> list[str]:
    records: list[str] = []
    cur: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("data_") and cur:
            records.append("\n".join(cur).rstrip() + "\n")
            cur = [line]
        else:
            cur.append(line)
    if cur:
        records.append("\n".join(cur).rstrip() + "\n")
    return records


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#") and "pymatgen" not in stripped:
            lines.append(line.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def tokenize_file(path: Path, tokenizer: CIFTokenizer) -> tuple[np.ndarray, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tokens = tokenizer.tokenize_cif(clean_text(text))
    ids = tokenizer.encode(tokens)
    counts = Counter(tokens)
    starts: list[int] = []
    offset = 0
    for record in split_concat_records(path):
        record_tokens = tokenizer.tokenize_cif(clean_text(record))
        starts.append(offset)
        offset += len(record_tokens)
    return np.array(ids, dtype=np.uint16), {
        "path": str(path),
        "records": len(starts),
        "tokens": len(tokens),
        "unknown_tokens": int(counts.get("<unk>", 0)),
        "unk_rate": float(counts.get("<unk>", 0) / len(tokens)) if tokens else 0.0,
        "starts": starts,
    }


def symbol_report(corpus_dir: Path, lookup: WyckoffLookup) -> dict[str, Any]:
    element_counts: Counter[str] = Counter()
    letter_counts: Counter[str] = Counter()
    skeleton_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        path = corpus_dir / f"{split}.txt"
        if not path.exists():
            continue
        for text in split_concat_records(path):
            rec = parse_symcif_v3_text(text, lookup)
            skel = []
            assign = []
            for site in rec.sites:
                element_counts[site.element] += 1
                letter_counts[site.letter] += 1
                skel.append(f"{site.multiplicity}{site.letter}")
                assign.append(f"{site.element}:{site.multiplicity}{site.letter}")
            skeleton_counts["|".join(sorted(skel))] += 1
            assignment_counts["|".join(sorted(assign))] += 1
    return {
        "elements_observed": dict(sorted(element_counts.items())),
        "wyckoff_letters_observed": dict(sorted(letter_counts.items())),
        "unique_skeletons": len(skeleton_counts),
        "unique_assignments": len(assignment_counts),
        "top_skeletons": dict(skeleton_counts.most_common(20)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize SymCIF-v3 staged corpus.")
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "symcif_v3")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "tokens_symcif_v3")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    args = parser.parse_args()

    tokenizer = CIFTokenizer()
    lookup = WyckoffLookup.from_json(args.lookup_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    split_reports: dict[str, Any] = {}
    written_bins: list[str] = []
    for split in ("train", "val", "test"):
        path = args.corpus_dir / f"{split}.txt"
        ids, rep = tokenize_file(path, tokenizer)
        ids.tofile(args.out_dir / f"{split}.bin")
        written_bins.append(f"{split}.bin")
        with (args.out_dir / ("starts.pkl" if split == "train" else f"starts_{split}.pkl")).open("wb") as f:
            pickle.dump(rep.pop("starts"), f)
        rep["encoded_ids"] = int(len(ids))
        split_reports[split] = rep
        print(f"{split} tokens: {len(ids):,}; unk={rep['unknown_tokens']}", flush=True)

    meta = {
        "vocab_size": len(tokenizer.token_to_id),
        "itos": tokenizer.id_to_token,
        "stoi": tokenizer.token_to_id,
    }
    with (args.out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    total_tokens = sum(int(item["tokens"]) for item in split_reports.values())
    total_unknown = sum(int(item["unknown_tokens"]) for item in split_reports.values())
    report = {
        "tokenizer": "CrystaLLM.CIFTokenizer",
        "vocab_size": len(tokenizer.token_to_id),
        "splits": split_reports,
        "total_tokens": total_tokens,
        "unknown_tokens": total_unknown,
        "unk_rate": float(total_unknown / total_tokens) if total_tokens else 0.0,
        **symbol_report(args.corpus_dir, lookup),
    }
    (args.reports_dir / "tokenizer_symcif_v3_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tokenizer_artifact = {
        "tokenizer": "CrystaLLM.CIFTokenizer",
        "vocab_size": len(tokenizer.token_to_id),
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token,
        "added_tokens": [],
        "note": "SymCIF-v3 reuses existing SymCIF field tokens and changes only the staged generation order.",
    }
    (args.artifacts_dir / "tokenizer_symcif_v3.json").write_text(
        json.dumps(tokenizer_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with tarfile.open(args.out_dir / "tokens_symcif_v3.tar.gz", "w:gz") as tar:
        for name in [*written_bins, "meta.pkl", "starts.pkl", "starts_val.pkl", "starts_test.pkl"]:
            path = args.out_dir / name
            if path.exists():
                tar.add(path, arcname=f"tokens_symcif_v3/{name}")
    print(f"vocab size: {len(tokenizer.token_to_id):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

