#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = ensure_under_opentry(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}


def choose_candidates(group: dict[str, Any], *, include_rank0: bool, transfer_top_n: int) -> list[tuple[dict[str, Any], str]]:
    candidates = list(group.get("candidates") or [])
    selected: list[tuple[dict[str, Any], str]] = []
    if include_rank0 and candidates:
        rank0 = next((cand for cand in candidates if int(cand.get("source_rank", -1)) == 0), candidates[0])
        selected.append((rank0, "rank0"))
    transfer_sorted = sorted(
        candidates,
        key=lambda cand: (
            float(cand.get("combined_error", 1e9)),
            float(cand.get("lattice_error", 1e9)),
            int(cand.get("source_rank", 999999)),
        ),
    )
    for cand in transfer_sorted[: max(0, int(transfer_top_n))]:
        selected.append((cand, "transfer_top"))
    out: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for cand, kind in selected:
        sid = str(cand.get("source_sample_id"))
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append((cand, kind))
    return out


def complex_flags(record: dict[str, Any]) -> dict[str, bool]:
    row_count = int(record.get("n_sites", len(record.get("wa_table") or [])))
    atom_count = int(rb.gb.atom_count(record))
    num_elements = int(record.get("num_elements", len(record.get("formula_counts") or {})))
    return {
        "rows_ge_7": row_count >= 7,
        "atoms_ge_12": atom_count >= 12,
        "num_elements_ge_4": num_elements >= 4,
    }


def make_example(
    *,
    group: dict[str, Any],
    record: dict[str, Any],
    source: dict[str, Any],
    candidate: dict[str, Any],
    choice_kind: str,
    selector: rb.OpentryGeometrySelector,
    split: str,
    vpa_strength: float,
    complex_weight: float,
    rank0_weight: float,
    transfer_weight: float,
) -> dict[str, Any]:
    params, align_cost = selector.source_aligned_params(record, source, 0, chemical=True)
    lattice = selector.vpa_calibrated_lattice(source, record, strength=float(vpa_strength))
    base_weight = train_resid.complex_sample_weight(record, float(complex_weight)) if split == "train" else 1.0
    choice_weight = float(rank0_weight) if choice_kind == "rank0" else float(transfer_weight)
    return {
        "split": split,
        "record": record,
        "base_params": params,
        "base_lattice": lattice,
        "source_sample_id": str(source.get("sample_id")),
        "source_rank": int(candidate.get("source_rank", 0)),
        "source_choice_kind": str(choice_kind),
        "source_distance": float(rb.gb.row_condition_distance(record, source)),
        "align_cost": float(align_cost),
        "transfer_lattice_error": float(candidate.get("lattice_error", 0.0)),
        "transfer_coord_error": float(candidate.get("coord_error", 0.0)),
        "transfer_combined_error": float(candidate.get("combined_error", 0.0)),
        "sample_weight": float(base_weight) * max(float(choice_weight), 0.0),
        "source_pool_k": int(group.get("source_pool_k", len(group.get("candidates") or []))),
        "vpa_strength": float(vpa_strength),
    }


def build_split_examples(
    *,
    groups: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    selector: rb.OpentryGeometrySelector,
    split: str,
    include_rank0: bool,
    transfer_top_n: int,
    vpa_strength: float,
    complex_weight: float,
    rank0_weight: float,
    transfer_weight: float,
    max_groups: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    missing_records = 0
    missing_sources = 0
    groups_with_examples = 0
    per_group_counts: list[int] = []
    choice_counts: Counter[str] = Counter()
    rows_ge_7 = 0
    atoms_ge_12 = 0
    transfer_errors: list[float] = []
    selected_groups = groups[: max(0, int(max_groups))] if int(max_groups) > 0 else groups
    for group in selected_groups:
        sample_id = str(group.get("sample_id"))
        record = records_by_id.get(sample_id)
        if record is None:
            missing_records += 1
            continue
        count_before = len(examples)
        for candidate, choice_kind in choose_candidates(group, include_rank0=include_rank0, transfer_top_n=int(transfer_top_n)):
            source_id = str(candidate.get("source_sample_id"))
            source = source_by_id.get(source_id)
            if source is None:
                missing_sources += 1
                continue
            example = make_example(
                group=group,
                record=record,
                source=source,
                candidate=candidate,
                choice_kind=choice_kind,
                selector=selector,
                split=split,
                vpa_strength=float(vpa_strength),
                complex_weight=float(complex_weight),
                rank0_weight=float(rank0_weight),
                transfer_weight=float(transfer_weight),
            )
            if float(example["sample_weight"]) <= 0.0:
                continue
            examples.append(example)
            choice_counts[str(choice_kind)] += 1
            transfer_errors.append(float(example["transfer_combined_error"]))
        written = len(examples) - count_before
        if written:
            groups_with_examples += 1
            per_group_counts.append(written)
            flags = complex_flags(record)
            rows_ge_7 += int(flags["rows_ge_7"])
            atoms_ge_12 += int(flags["atoms_ge_12"])
    weights = [float(ex["sample_weight"]) for ex in examples]
    return examples, {
        "groups_input": len(selected_groups),
        "groups_with_examples": groups_with_examples,
        "examples": len(examples),
        "examples_per_group_mean": float(mean(per_group_counts)) if per_group_counts else None,
        "choice_counts": dict(sorted(choice_counts.items())),
        "missing_records": missing_records,
        "missing_sources": missing_sources,
        "rows_ge_7_groups": rows_ge_7,
        "atoms_ge_12_groups": atoms_ge_12,
        "mean_sample_weight": float(mean(weights)) if weights else None,
        "max_sample_weight": float(max(weights)) if weights else None,
        "mean_transfer_combined_error": float(mean(transfer_errors)) if transfer_errors else None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build multi-source residual geometry examples from source-mode transfer labels.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--source-mode-examples-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-rank0", action="store_true")
    parser.add_argument("--transfer-top-n", type=int, default=2)
    parser.add_argument("--max-train-groups", type=int, default=0)
    parser.add_argument("--max-val-groups", type=int, default=0)
    parser.add_argument("--vpa-strength", type=float, default=0.5)
    parser.add_argument("--complex-weight", type=float, default=3.0)
    parser.add_argument("--rank0-weight", type=float, default=1.0)
    parser.add_argument("--transfer-weight", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    examples_dir = ensure_under_opentry(args.source_mode_examples_dir)
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_examples.jsonl"
    val_out = out_dir / "val_examples.jsonl"
    if not args.overwrite and (train_out.exists() or val_out.exists()):
        raise SystemExit(f"Refusing to overwrite existing examples in {out_dir}; pass --overwrite")

    all_train = rb.load_records(args.data_root / "train.jsonl")
    all_val = rb.load_records(args.data_root / "val.jsonl")
    train_groups = read_jsonl(examples_dir / "train_source_mode_examples.jsonl")
    val_groups = read_jsonl(examples_dir / "val_source_mode_examples.jsonl")
    selector = train_resid.build_selector(all_train)
    train_by_id = {str(record.get("sample_id")): record for record in all_train}
    val_by_id = {str(record.get("sample_id")): record for record in all_val}
    source_by_id = {str(record.get("sample_id")): record for record in all_train}

    train_examples, train_summary = build_split_examples(
        groups=train_groups,
        records_by_id=train_by_id,
        source_by_id=source_by_id,
        selector=selector,
        split="train",
        include_rank0=bool(args.include_rank0),
        transfer_top_n=int(args.transfer_top_n),
        vpa_strength=float(args.vpa_strength),
        complex_weight=float(args.complex_weight),
        rank0_weight=float(args.rank0_weight),
        transfer_weight=float(args.transfer_weight),
        max_groups=int(args.max_train_groups),
    )
    val_examples, val_summary = build_split_examples(
        groups=val_groups,
        records_by_id=val_by_id,
        source_by_id=source_by_id,
        selector=selector,
        split="val",
        include_rank0=bool(args.include_rank0),
        transfer_top_n=int(args.transfer_top_n),
        vpa_strength=float(args.vpa_strength),
        complex_weight=1.0,
        rank0_weight=float(args.rank0_weight),
        transfer_weight=float(args.transfer_weight),
        max_groups=int(args.max_val_groups),
    )
    write_jsonl(train_out, train_examples)
    write_jsonl(val_out, val_examples)
    summary = {
        "config": jsonable_args(args),
        "all_train_records": len(all_train),
        "all_val_records": len(all_val),
        "source_mode_train_groups": len(train_groups),
        "source_mode_val_groups": len(val_groups),
        "train": {**train_summary, "path": str(train_out)},
        "val": {**val_summary, "path": str(val_out)},
        "leakage_guard": {
            "selector_split": "train",
            "source_records_split": "train",
            "train_targets": "train",
            "val_targets": "val",
            "test_records_used": 0,
            "structurematcher_labels_used": False,
            "label_kind": "lattice/free-param transfer error from train/val source-mode examples, not match/rms",
        },
    }
    write_json(out_dir / "source_mixture_residual_examples_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
