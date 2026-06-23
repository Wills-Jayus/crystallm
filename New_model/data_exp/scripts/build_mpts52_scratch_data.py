#!/usr/bin/env python3
"""Build clean MPTS-52 scratch-training corpora for data-format comparison.

The script writes only under model/New_model/data_exp. It reads:
- CrystaLLM resources/benchmarks/mpts_52/{train,val}.csv for the original CIF
  baseline format.
- symcif_experiment/data/structured_symcif_v4_mpts52/{train,val}.jsonl for the
  modified SymCIF-v4 structured text format.

No test split is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/data/users/xsw/autodlmini")
ORIG_ROOT = WORKSPACE / "model/CrystaLLM/resources/benchmarks/mpts_52"
SYMCIF_ROOT = WORKSPACE / "model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52"
OUT_ROOT = WORKSPACE / "model/New_model/data_exp"


@dataclass
class SplitStats:
    samples: int = 0
    bytes: int = 0
    min_doc_bytes: int = 0
    max_doc_bytes: int = 0
    mean_doc_bytes: float = 0.0
    rows_ge_7: int = 0
    row_count_min: int = 0
    row_count_max: int = 0
    row_count_mean: float = 0.0
    unique_formula: int = 0
    unique_sg: int = 0
    unique_wa: int = 0
    sha256: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_cif_text(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or "pymatgen" in line:
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def read_original_cif_split(split: str) -> list[dict[str, Any]]:
    path = ORIG_ROOT / f"{split}.csv"
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            material_id = str(row.get("material_id") or row.get("id") or len(rows))
            cif = clean_cif_text(str(row["cif"]))
            rows.append(
                {
                    "sample_id": f"mpts_52_{split}__{material_id}",
                    "material_id": material_id,
                    "text": cif,
                }
            )
    return rows


def fmt_num(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "0.0000"


def formula_sum(formula_counts: dict[str, Any]) -> str:
    parts = []
    for element in sorted(formula_counts):
        count = int(round(float(formula_counts[element])))
        parts.append(f"{element}{count}")
    return " ".join(parts)


def data_name(record: dict[str, Any]) -> str:
    formula = re.sub(r"[^A-Za-z0-9]", "", str(record.get("formula") or formula_sum(record.get("formula_counts") or {})))
    return f"data_{formula or record.get('sample_id', 'mpts52')}"


def render_symcif_v4_text(record: dict[str, Any]) -> str:
    """Render a compact, deterministic staged SymCIF-v4 text document."""
    lattice = record.get("lattice") or {}
    rows = list(record.get("wa_table") or [])
    lines: list[str] = [
        data_name(record),
        f"# sample_id: {record.get('sample_id') or record.get('id')}",
        f"_chemical_formula_sum '{formula_sum(record.get('formula_counts') or {})}'",
        f"_symmetry_Int_Tables_number {int(record.get('sg') or 1)}",
        f"_symmetry_space_group_name_H-M '{record.get('sg_symbol') or 'P1'}'",
        f"_cell_formula_units_Z {int(round(float(record.get('formula_units_Z', record.get('z', 1) or 1))))}",
        "",
        "loop_",
        "_wyckoff_site_index",
        "_wyckoff_site_element",
        "_wyckoff_site_multiplicity",
        "_wyckoff_site_letter",
        "_wyckoff_site_symmetry",
        "_wyckoff_site_enumeration",
        "_wyckoff_free_x",
        "_wyckoff_free_y",
        "_wyckoff_free_z",
    ]
    for idx, row in enumerate(rows, start=1):
        free = row.get("free_params") or {}
        values = []
        for axis in ("x", "y", "z"):
            if axis in free:
                values.append(fmt_num(free[axis]))
            else:
                values.append("FIXED")
        site_sym = str(row.get("site_symmetry") or "UNKNOWN").replace(" ", "_")
        enum = int(row.get("enumeration") or 0)
        lines.append(
            f"{idx} {row.get('element')} {int(row.get('multiplicity') or 1)} "
            f"{row.get('letter')} {site_sym} {enum} {values[0]} {values[1]} {values[2]}"
        )
    lines.extend(
        [
            "",
            f"_cell_length_a {fmt_num(lattice.get('a'))}",
            f"_cell_length_b {fmt_num(lattice.get('b'))}",
            f"_cell_length_c {fmt_num(lattice.get('c'))}",
            f"_cell_angle_alpha {fmt_num(lattice.get('alpha'))}",
            f"_cell_angle_beta {fmt_num(lattice.get('beta'))}",
            f"_cell_angle_gamma {fmt_num(lattice.get('gamma'))}",
            f"_cell_volume {fmt_num(lattice.get('volume'))}",
            "",
        ]
    )
    return "\n".join(lines)


def read_symcif_split(split: str) -> list[dict[str, Any]]:
    path = SYMCIF_ROOT / f"{split}.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "sample_id": obj.get("sample_id") or obj.get("id"),
                    "material_id": obj.get("material_id"),
                    "text": render_symcif_v4_text(obj),
                    "row_count": int(obj.get("n_sites") or len(obj.get("wa_table") or [])),
                    "formula": obj.get("formula"),
                    "sg": int(obj.get("sg") or 0),
                    "wa_key": obj.get("canonical_wa_key"),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_text_corpus(path: Path, rows: list[dict[str, Any]]) -> list[int]:
    starts: list[int] = []
    offset = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            text = str(row["text"]).rstrip() + "\n\n"
            starts.append(offset)
            f.write(text)
            offset += len(text.encode("utf-8"))
    return starts


def encode_utf8_bytes(text_path: Path, out_dir: Path, split: str, starts: list[int]) -> dict[str, Any]:
    raw = text_path.read_bytes()
    ids = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    ids.tofile(out_dir / f"{split}.bin")
    with (out_dir / ("starts.pkl" if split == "train" else "starts_val.pkl")).open("wb") as f:
        pickle.dump(starts, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "tokens": int(ids.size),
        "bytes": int(len(raw)),
        "sha256": sha256_file(text_path),
        "starts": len(starts),
    }


def write_meta(out_dir: Path) -> None:
    meta = {
        "vocab_size": 256,
        "itos": {i: bytes([i]).decode("latin1") for i in range(256)},
        "stoi": {bytes([i]).decode("latin1"): i for i in range(256)},
        "tokenizer": "utf8_byte_level_uint8_promoted_to_uint16",
    }
    with (out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)


def split_stats(rows: list[dict[str, Any]], text_path: Path) -> SplitStats:
    lengths = [len(str(r["text"]).encode("utf-8")) for r in rows]
    row_counts = [int(r.get("row_count") or 0) for r in rows if "row_count" in r]
    formulas = {str(r.get("formula")) for r in rows if r.get("formula")}
    sgs = {int(r.get("sg")) for r in rows if r.get("sg") is not None}
    wa = {str(r.get("wa_key")) for r in rows if r.get("wa_key")}
    return SplitStats(
        samples=len(rows),
        bytes=sum(lengths),
        min_doc_bytes=min(lengths) if lengths else 0,
        max_doc_bytes=max(lengths) if lengths else 0,
        mean_doc_bytes=float(sum(lengths) / len(lengths)) if lengths else 0.0,
        rows_ge_7=sum(1 for x in row_counts if x >= 7),
        row_count_min=min(row_counts) if row_counts else 0,
        row_count_max=max(row_counts) if row_counts else 0,
        row_count_mean=float(sum(row_counts) / len(row_counts)) if row_counts else 0.0,
        unique_formula=len(formulas),
        unique_sg=len(sgs),
        unique_wa=len(wa),
        sha256=sha256_file(text_path),
    )


def build_one(name: str, split_rows: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    root = OUT_ROOT / "data" / name
    token_root = OUT_ROOT / "data" / f"tokens_{name}"
    root.mkdir(parents=True, exist_ok=True)
    token_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    shuffled_train = list(split_rows["train"])
    rng.shuffle(shuffled_train)
    rows_by_split = {"train": shuffled_train, "val": split_rows["val"]}

    summary: dict[str, Any] = {
        "name": name,
        "seed": seed,
        "source_policy": "MPTS-52 train/val only; no test split read",
        "text_root": str(root),
        "token_root": str(token_root),
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        write_jsonl(root / f"{split}.jsonl", rows)
        starts = write_text_corpus(root / f"{split}.txt", rows)
        enc = encode_utf8_bytes(root / f"{split}.txt", token_root, split, starts)
        stats = asdict(split_stats(rows, root / f"{split}.txt"))
        stats.update(enc)
        summary["splits"][split] = stats
    write_meta(token_root)
    summary["meta"] = {"vocab_size": 256, "tokenizer": "utf8_byte_level"}
    (root / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (token_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    original = {split: read_original_cif_split(split) for split in ("train", "val")}
    symcif = {split: read_symcif_split(split) for split in ("train", "val")}
    summary = {
        "experiment": "MPTS-52 scratch data-format comparison",
        "no_test_read": True,
        "original_cif": build_one("mpts52_orig_cif_byte", original, args.seed),
        "symcif_v4": build_one("mpts52_symcif_v4_byte", symcif, args.seed),
    }
    out = OUT_ROOT / "data" / "data_build_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
