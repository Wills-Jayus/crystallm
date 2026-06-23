#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_render_wyckoff_cifs_e07e08 as rb  # noqa: E402
import opentry_train_source_residual_geometry as train_resid  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def record_row_count(record: dict[str, Any]) -> int:
    return int(record.get("n_sites", len(record.get("wa_table") or [])))


def record_atom_count(record: dict[str, Any]) -> int:
    return int(rb.gb.atom_count(record))


def select_records(
    records: list[dict[str, Any]],
    start: int,
    max_records: int,
    *,
    min_row_count: int = 0,
    min_atom_count: int = 0,
) -> list[dict[str, Any]]:
    if int(min_row_count) > 0:
        records = [record for record in records if record_row_count(record) >= int(min_row_count)]
    if int(min_atom_count) > 0:
        records = [record for record in records if record_atom_count(record) >= int(min_atom_count)]
    start = max(0, int(start))
    selected = records[start:]
    if int(max_records) > 0:
        selected = selected[: int(max_records)]
    return selected


def build_one_example(
    record: dict[str, Any],
    *,
    selector: rb.OpentryGeometrySelector,
    split: str,
    source_pool_k: int,
    exclude_self: bool,
    vpa_strength: float,
    complex_weight: float,
) -> dict[str, Any] | None:
    exclude = str(record.get("sample_id")) if exclude_self else None
    source = train_resid.select_source(
        selector,
        record,
        source_pool_k=int(source_pool_k),
        exclude_sample_id=exclude,
    )
    if source is None:
        return None
    params, align_cost = selector.source_aligned_params(record, source, 0, chemical=True)
    lattice = selector.vpa_calibrated_lattice(source, record, strength=float(vpa_strength))
    return {
        "split": split,
        "record": record,
        "base_params": params,
        "base_lattice": lattice,
        "source_sample_id": str(source.get("sample_id")),
        "source_distance": float(rb.gb.row_condition_distance(record, source)),
        "align_cost": float(align_cost),
        "sample_weight": train_resid.complex_sample_weight(record, float(complex_weight)),
        "source_pool_k": int(source_pool_k),
        "vpa_strength": float(vpa_strength),
    }


def write_examples(
    path: Path,
    records: list[dict[str, Any]],
    *,
    selector: rb.OpentryGeometrySelector,
    split: str,
    source_pool_k: int,
    exclude_self: bool,
    vpa_strength: float,
    complex_weight: float,
    progress_every: int,
) -> dict[str, Any]:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_ids: set[str] = set()
    weights: list[float] = []
    row_counts: list[int] = []
    atom_counts: list[int] = []
    complex_count = 0
    skipped = 0
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            example = build_one_example(
                record,
                selector=selector,
                split=split,
                source_pool_k=int(source_pool_k),
                exclude_self=bool(exclude_self),
                vpa_strength=float(vpa_strength),
                complex_weight=float(complex_weight),
            )
            if example is None:
                skipped += 1
            else:
                f.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                source_ids.add(str(example["source_sample_id"]))
                weights.append(float(example["sample_weight"]))
                row_counts.append(record_row_count(record))
                atom_counts.append(record_atom_count(record))
                if train_resid.complex_sample_weight(record, float(complex_weight)) > 1.0:
                    complex_count += 1
            if int(progress_every) > 0 and idx % int(progress_every) == 0:
                print(json.dumps({"split": split, "seen": idx, "written": written, "skipped": skipped}, sort_keys=True), flush=True)
    return {
        "path": str(path),
        "input_records": len(records),
        "examples": written,
        "skipped_no_source": skipped,
        "unique_source_ids": len(source_ids),
        "mean_weight": float(mean(weights)) if weights else None,
        "max_weight": float(max(weights)) if weights else None,
        "complex_examples": complex_count,
        "rows_ge7_examples": int(sum(1 for value in row_counts if value >= 7)),
        "atoms_ge12_examples": int(sum(1 for value in atom_counts if value >= 12)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache train/val source-pair examples for source-conditioned residual geometry.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selector-max-train-records", type=int, default=0)
    parser.add_argument("--start-train-index", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--min-train-row-count", type=int, default=0)
    parser.add_argument("--min-train-atom-count", type=int, default=0)
    parser.add_argument("--start-val-index", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=1024)
    parser.add_argument("--min-val-row-count", type=int, default=0)
    parser.add_argument("--min-val-atom-count", type=int, default=0)
    parser.add_argument("--source-pool-k", type=int, default=24)
    parser.add_argument("--vpa-strength", type=float, default=0.5)
    parser.add_argument("--complex-weight", type=float, default=3.0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_examples.jsonl"
    val_out = out_dir / "val_examples.jsonl"
    if not args.overwrite and (train_out.exists() or val_out.exists()):
        raise SystemExit(f"Refusing to overwrite existing examples in {out_dir}; pass --overwrite")

    all_train = rb.load_records(args.data_root / "train.jsonl")
    all_val = rb.load_records(args.data_root / "val.jsonl")
    selector_records = all_train
    if int(args.selector_max_train_records) > 0:
        selector_records = selector_records[: int(args.selector_max_train_records)]
    train_targets = select_records(
        all_train,
        int(args.start_train_index),
        int(args.max_train_records),
        min_row_count=int(args.min_train_row_count),
        min_atom_count=int(args.min_train_atom_count),
    )
    val_targets = select_records(
        all_val,
        int(args.start_val_index),
        int(args.max_val_records),
        min_row_count=int(args.min_val_row_count),
        min_atom_count=int(args.min_val_atom_count),
    )
    selector = train_resid.build_selector(selector_records)

    train_summary = write_examples(
        train_out,
        train_targets,
        selector=selector,
        split="train",
        source_pool_k=int(args.source_pool_k),
        exclude_self=True,
        vpa_strength=float(args.vpa_strength),
        complex_weight=float(args.complex_weight),
        progress_every=int(args.progress_every),
    )
    val_summary = write_examples(
        val_out,
        val_targets,
        selector=selector,
        split="val",
        source_pool_k=int(args.source_pool_k),
        exclude_self=False,
        vpa_strength=float(args.vpa_strength),
        complex_weight=1.0,
        progress_every=int(args.progress_every),
    )
    summary = {
        "config": jsonable_args(args),
        "selector_train_records": len(selector_records),
        "all_train_records": len(all_train),
        "all_val_records": len(all_val),
        "selected_train_records": len(train_targets),
        "selected_val_records": len(val_targets),
        "train": train_summary,
        "val": val_summary,
        "leakage_guard": {
            "selector_split": "train",
            "train_targets": "train",
            "val_targets": "val",
            "test_records_used": 0,
            "structurematcher_labels_used": False,
        },
    }
    write_json(out_dir / "source_residual_examples_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
