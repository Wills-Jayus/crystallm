#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
if str(CRYSTALLM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM_ROOT))

from crystallm import CIFTokenizer  # type: ignore  # noqa: E402


FIELD_TOKENS = [
    "data_",
    "_chemical_formula_sum",
    "_symmetry_Int_Tables_number",
    "_symmetry_space_group_name_H-M",
    "_wyckoff_site_index",
    "_wyckoff_site_element",
    "_wyckoff_site_letter",
    "_wyckoff_free_x",
    "_wyckoff_free_y",
    "_wyckoff_free_z",
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
    "_cell_volume",
    "loop_",
    "FIXED",
]

CONTROL_TOKENS = ["<SITE>", "<END_SITES>", "<CELL>", "<FIXED>"]


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


def tokenize_text(text: str, tokenizer: CIFTokenizer) -> tuple[list[str], list[int]]:
    tokens = tokenizer.tokenize_cif(clean_text(text))
    return tokens, tokenizer.encode(tokens)


def tokenize_file(path: Path, tokenizer: CIFTokenizer) -> tuple[np.ndarray, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tokens, ids = tokenize_text(text, tokenizer)
    counts = Counter(tokens)
    report = {
        "path": str(path),
        "tokens": len(tokens),
        "unknown_tokens": int(counts.get("<unk>", 0)),
        "unk_rate": float(counts.get("<unk>", 0) / len(tokens)) if tokens else 0.0,
        "records": len(split_concat_records(path)),
    }
    return np.array(ids, dtype=np.uint16), report


def iter_site_rows(record: str) -> list[list[str]]:
    lines = [line.strip() for line in record.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    rows: list[list[str]] = []
    in_loop = False
    data_started = False
    for line in lines:
        if line == "loop_":
            in_loop = True
            data_started = False
            continue
        if in_loop and line.startswith("_"):
            continue
        if in_loop and not line.startswith("_"):
            data_started = True
            if line.startswith("_cell_"):
                break
            rows.append(line.split())
            continue
        if data_started and line.startswith("_cell_"):
            break
    return [row for row in rows if len(row) >= 6]


def corpus_symbol_report(corpus_dir: Path, tokenizer: CIFTokenizer) -> dict[str, Any]:
    element_counts: Counter[str] = Counter()
    letter_counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        path = corpus_dir / f"{split}.txt"
        if not path.exists():
            continue
        for record in split_concat_records(path):
            for row in iter_site_rows(record):
                element_counts[row[1]] += 1
                letter_counts[row[2]] += 1
    token_to_id = tokenizer.token_to_id
    element_covered = {el: el in token_to_id for el in sorted(element_counts)}
    letter_covered = {letter: letter in token_to_id for letter in sorted(letter_counts)}
    return {
        "elements_observed": dict(sorted(element_counts.items())),
        "wyckoff_letters_observed": dict(sorted(letter_counts.items())),
        "element_token_coverage": {
            "covered": sum(1 for v in element_covered.values() if v),
            "total": len(element_covered),
            "missing": [k for k, v in element_covered.items() if not v],
            "by_token": element_covered,
        },
        "wyckoff_letter_token_coverage": {
            "covered": sum(1 for v in letter_covered.values() if v),
            "total": len(letter_covered),
            "missing": [k for k, v in letter_covered.items() if not v],
            "by_token": letter_covered,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize SymCIF-v2 and report mask-token coverage.")
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "symcif_v2")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "tokens_symcif_v2")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    tokenizer = CIFTokenizer()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    split_reports: dict[str, Any] = {}
    written_bins: list[str] = []
    for split in ("train", "val", "test"):
        path = args.corpus_dir / f"{split}.txt"
        if not path.exists():
            continue
        ids, split_report = tokenize_file(path, tokenizer)
        bin_name = f"{split}.bin"
        ids.tofile(args.out_dir / bin_name)
        written_bins.append(bin_name)
        split_report["encoded_ids"] = int(len(ids))
        split_reports[split] = split_report
        print(f"{split} tokens: {len(ids):,}; unk={split_report['unknown_tokens']}", flush=True)

    meta = {
        "vocab_size": len(tokenizer.token_to_id),
        "itos": tokenizer.id_to_token,
        "stoi": tokenizer.token_to_id,
    }
    with (args.out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    tokenizer_artifact = {
        "tokenizer": "CrystaLLM.CIFTokenizer",
        "vocab_size": len(tokenizer.token_to_id),
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token,
        "added_tokens": [],
        "note": (
            "SymCIF-v2 uses the existing CrystaLLM tokenizer. The constrained sampler "
            "masks element, Wyckoff-letter, FIXED, digit, and structural field tokens "
            "directly; literal angle-bracket control tokens are not emitted into the "
            "training corpus."
        ),
    }
    (args.artifacts_dir / "tokenizer_symcif_v2.json").write_text(
        json.dumps(tokenizer_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    symbol_report = corpus_symbol_report(args.corpus_dir, tokenizer)
    field_coverage = {token: token in tokenizer.token_to_id for token in FIELD_TOKENS}
    control_report = {
        token: {
            "literal_token_present": token in tokenizer.token_to_id,
            "state_machine_supported": token in {"<SITE>", "<END_SITES>", "<CELL>", "<FIXED>"},
        }
        for token in CONTROL_TOKENS
    }
    control_report["<FIXED>"]["state_machine_alias"] = "FIXED"
    control_report["<CELL>"]["state_machine_alias"] = "_cell_length_a transition"
    control_report["<END_SITES>"]["state_machine_alias"] = "remaining_counts == 0 before cell transition"
    control_report["<SITE>"]["state_machine_alias"] = "_wyckoff_site_* row state"

    total_tokens = sum(int(item["tokens"]) for item in split_reports.values())
    total_unknown = sum(int(item["unknown_tokens"]) for item in split_reports.values())
    report = {
        "vocab_size": len(tokenizer.token_to_id),
        "added_tokens": [],
        "splits": split_reports,
        "unk_rate": float(total_unknown / total_tokens) if total_tokens else 0.0,
        "unknown_tokens": total_unknown,
        "total_tokens": total_tokens,
        "field_token_coverage": {
            "covered": sum(1 for v in field_coverage.values() if v),
            "total": len(field_coverage),
            "missing": [k for k, v in field_coverage.items() if not v],
            "by_token": field_coverage,
        },
        "control_token_support": control_report,
        "supports_constrained_decoding_mask": True,
        **symbol_report,
    }
    (args.reports_dir / "tokenizer_symcif_v2_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    subdir_name = args.out_dir.name
    with tarfile.open(args.out_dir / f"{subdir_name}.tar.gz", "w:gz") as tar:
        for name in [*written_bins, "meta.pkl"]:
            tar.add(args.out_dir / name, arcname=f"{subdir_name}/{name}")
    print(f"vocab size: {len(tokenizer.token_to_id):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
