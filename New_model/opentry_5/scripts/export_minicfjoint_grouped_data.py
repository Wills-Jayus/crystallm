#!/usr/bin/env python3
"""Export opentry_5 grouped canonical JSONL to MiniCFJoint-compatible roots."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def under_root(path: Path) -> Path:
    path = path.resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"refusing to write outside opentry_5: {path}")
    return path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def convert(row: dict, split: str) -> dict:
    out = {
        "sample_id": row["sample_id"],
        "id": row["sample_id"],
        "material_id": row.get("material_id"),
        "dataset": row.get("dataset"),
        "split": split,
        "formula": row.get("formula"),
        "formula_counts": row.get("formula_counts") or {},
        "atom_count": row.get("atom_count") or sum((row.get("formula_counts") or {}).values()),
        "n_sites": row.get("row_count") or len(row.get("wa_table") or []),
        "num_elements": row.get("num_elements") or len(row.get("formula_counts") or {}),
        "sg": row.get("sg"),
        "sg_symbol": row.get("sg_symbol"),
        "lattice": row.get("lattice"),
        "wa_table": row.get("wa_table") or [],
        "canonical_skeleton_key": row.get("canonical_skeleton_key"),
        "canonical_wa_key": row.get("canonical_wa_key"),
        "source_path": str(ROOT / row["canonical_cif_path"]) if row.get("canonical_cif_path") else row.get("source_path"),
        "opentry5_split": row.get("opentry5_split"),
        "grouped_dev_fold": row.get("grouped_dev_fold"),
        "test_access": "none",
    }
    for wa in out["wa_table"]:
        wa.setdefault("sg", out["sg"])
        wa.setdefault("free_params", wa.get("free_params") or {})
        wa.setdefault("free_symbols", wa.get("free_symbols") or [])
    return out


def load_valid(path: Path, split: str, limit: int = 0) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if row.get("canonical_cif_excluded_reason"):
            continue
        if not row.get("canonical_cif_path"):
            continue
        rows.append(convert(row, split))
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> None:
    train = load_valid(ROOT / "data/canonical_train/train_core.jsonl", "train")
    folds = {
        "fold_a": load_valid(ROOT / "data/canonical_dev/fold_a.jsonl", "val"),
        "fold_b": load_valid(ROOT / "data/canonical_dev/fold_b.jsonl", "val"),
        "dev_model": load_valid(ROOT / "data/canonical_dev/dev_model.jsonl", "val"),
        "dev_gate": load_valid(ROOT / "data/canonical_dev/dev_gate.jsonl", "val"),
    }
    summary = {"train": len(train), "folds": {k: len(v) for k, v in folds.items()}, "test_access": "none"}
    for name, val in folds.items():
        out = ROOT / "data/minicfjoint_grouped" / name
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "val.jsonl", val)
    (ROOT / "data/minicfjoint_grouped/summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
