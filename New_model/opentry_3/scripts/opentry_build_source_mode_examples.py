#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def select_records(records: list[dict[str, Any]], start: int, max_records: int) -> list[dict[str, Any]]:
    selected = records[max(0, int(start)) :]
    if int(max_records) > 0:
        selected = selected[: int(max_records)]
    return selected


def coord_error(params: dict[int, dict[str, float]], record: dict[str, Any]) -> float:
    rows = rb.v2.canonical_rows(record)
    total = 0.0
    denom = 0.0
    for row_idx, row in enumerate(rows):
        base_vals, _ = train_resid.base_param_values(params, row_idx, row)
        target_vals, target_mask = train_resid.row_param_target(row)
        for value, target, mask in zip(base_vals, target_vals, target_mask):
            if float(mask) <= 0.0:
                continue
            diff = abs((float(value) % 1.0) - (float(target) % 1.0))
            diff = min(diff, 1.0 - diff)
            total += diff * diff
            denom += 1.0
    return float(total / max(1.0, denom))


def lattice_error(lattice: dict[str, Any], record: dict[str, Any], mean_t: torch.Tensor, std_t: torch.Tensor) -> float:
    base = train_resid.normalize_lattice(lattice, mean_t, std_t)
    target = train_resid.normalize_lattice(dict(record["lattice"]), mean_t, std_t)
    return float((base - target).square().mean().item())


def source_vpa(record: dict[str, Any]) -> float | None:
    return rb.record_volume_per_atom(record)


def numeric_source_features(
    *,
    selector: rb.OpentryGeometrySelector,
    record: dict[str, Any],
    source: dict[str, Any],
    source_rank: int,
    align_cost: float,
    source_pool_k: int,
    base_lattice: dict[str, Any],
) -> dict[str, float]:
    target_atom_count = max(1, rb.gb.atom_count(record))
    source_atom_count = max(1, rb.gb.atom_count(source))
    base_volume = rb.cell_volume({k: float(base_lattice[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")})
    base_vpa = float(base_volume) / float(target_atom_count)
    svpa = source_vpa(source)
    expected_vpa = selector.expected_vpa(record)
    return {
        "source_rank_norm": float(source_rank) / max(1.0, float(source_pool_k - 1)),
        "candidate_score_rank_prior": -float(source_rank),
        "chem_distance": float(rb.chem_distance(record, source)),
        "row_condition_chem_distance": float(selector.row_condition_chem_distance(record, source)),
        "align_cost": float(align_cost),
        "source_quality_penalty": float(selector.source_quality_penalty(source)),
        "source_param_prior_penalty": float(selector.source_param_prior_penalty(record, source, chemical=True)),
        "formula_l1": float(rb.gb.formula_l1(rb.gb.formula_frac(record), rb.gb.formula_frac(source))),
        "atom_count_abs_diff_norm": abs(float(target_atom_count - source_atom_count)) / 300.0,
        "same_sg": 1.0 if int(record["sg"]) == int(source["sg"]) else 0.0,
        "same_skeleton_key": 1.0 if str(record.get("canonical_skeleton_key")) == str(source.get("canonical_skeleton_key")) else 0.0,
        "same_wa_key": 1.0 if str(record.get("canonical_wa_key")) == str(source.get("canonical_wa_key")) else 0.0,
        "source_vpa_norm": 0.0 if svpa is None else min(5.0, float(svpa) / 20.0),
        "expected_vpa_norm": 0.0 if expected_vpa is None else min(5.0, float(expected_vpa) / 20.0),
        "base_vpa_norm": min(5.0, float(base_vpa) / 20.0),
    }


def build_group(
    record: dict[str, Any],
    *,
    selector: rb.OpentryGeometrySelector,
    split: str,
    source_pool_k: int,
    exclude_self: bool,
    vpa_strength: float,
    coord_weight: float,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
) -> dict[str, Any] | None:
    source_query_k = int(source_pool_k) + (1 if exclude_self else 0)
    sources = train_resid.fast_chem_quality_candidates(selector, record, source_query_k)
    if exclude_self:
        sample_id = str(record.get("sample_id"))
        sources = [source for source in sources if str(source.get("sample_id")) != sample_id]
    sources = sources[: int(source_pool_k)]
    if not sources:
        return None
    candidates: list[dict[str, Any]] = []
    for source_rank, source in enumerate(sources):
        params, align_cost = selector.source_aligned_params(record, source, 0, chemical=True)
        lattice = selector.vpa_calibrated_lattice(source, record, strength=float(vpa_strength))
        lat_err = lattice_error(lattice, record, mean_t, std_t)
        coord_err = coord_error(params, record)
        combined = float(lat_err + float(coord_weight) * coord_err)
        candidates.append(
            {
                "source_sample_id": str(source.get("sample_id")),
                "source_rank": int(source_rank),
                "lattice_error": float(lat_err),
                "coord_error": float(coord_err),
                "combined_error": combined,
                "features": numeric_source_features(
                    selector=selector,
                    record=record,
                    source=source,
                    source_rank=source_rank,
                    align_cost=align_cost,
                    source_pool_k=int(source_pool_k),
                    base_lattice=lattice,
                ),
            }
        )
    best_index = min(range(len(candidates)), key=lambda idx: (float(candidates[idx]["combined_error"]), int(candidates[idx]["source_rank"])))
    heuristic_error = float(candidates[0]["combined_error"])
    best_error = float(candidates[best_index]["combined_error"])
    return {
        "split": split,
        "sample_id": str(record.get("sample_id")),
        "material_id": str(record.get("material_id")),
        "sg": int(record["sg"]),
        "atom_count": int(rb.gb.atom_count(record)),
        "num_elements": int(record.get("num_elements", len(record.get("formula_counts") or {}))),
        "row_count_label": int(len(record.get("wa_table") or [])),
        "complex_flags": {
            "rows_ge_7": int(len(record.get("wa_table") or [])) >= 7,
            "atoms_ge_12": int(rb.gb.atom_count(record)) >= 12,
            "num_elements_ge_4": int(record.get("num_elements", len(record.get("formula_counts") or {}))) >= 4,
        },
        "canonical_skeleton_key": str(record.get("canonical_skeleton_key")),
        "canonical_wa_key": str(record.get("canonical_wa_key")),
        "source_pool_k": int(source_pool_k),
        "vpa_strength": float(vpa_strength),
        "coord_weight": float(coord_weight),
        "candidates": candidates,
        "best_source_index": int(best_index),
        "best_source_rank": int(candidates[best_index]["source_rank"]),
        "heuristic_error": heuristic_error,
        "best_error": best_error,
        "oracle_gain": float(heuristic_error - best_error),
        "heuristic_is_best": int(best_index) == 0,
    }


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {"groups": 0}
    rows_ge7 = [group for group in groups if group["complex_flags"]["rows_ge_7"]]
    atoms_ge12 = [group for group in groups if group["complex_flags"]["atoms_ge_12"]]

    def sub(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"groups": 0}
        return {
            "groups": len(items),
            "candidate_count_mean": float(mean([len(group["candidates"]) for group in items])),
            "heuristic_is_best_rate": float(mean([1.0 if group["heuristic_is_best"] else 0.0 for group in items])),
            "best_source_rank_mean": float(mean([float(group["best_source_rank"]) for group in items])),
            "heuristic_error_mean": float(mean([float(group["heuristic_error"]) for group in items])),
            "best_error_mean": float(mean([float(group["best_error"]) for group in items])),
            "oracle_gain_mean": float(mean([float(group["oracle_gain"]) for group in items])),
            "oracle_gain_positive_rate": float(mean([1.0 if float(group["oracle_gain"]) > 1e-8 else 0.0 for group in items])),
        }

    return {
        "full": sub(groups),
        "rows_ge_7": sub(rows_ge7),
        "atoms_ge_12": sub(atoms_ge12),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train/val source-mode selection labels from train-source geometry transfer errors.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selector-max-train-records", type=int, default=0)
    parser.add_argument("--start-train-index", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--start-val-index", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=1024)
    parser.add_argument("--source-pool-k", type=int, default=8)
    parser.add_argument("--vpa-strength", type=float, default=0.5)
    parser.add_argument("--coord-weight", type=float, default=4.0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_source_mode_examples.jsonl"
    val_out = out_dir / "val_source_mode_examples.jsonl"
    if not args.overwrite and (train_out.exists() or val_out.exists()):
        raise SystemExit(f"Refusing to overwrite existing examples in {out_dir}; pass --overwrite")

    all_train = rb.load_records(args.data_root / "train.jsonl")
    all_val = rb.load_records(args.data_root / "val.jsonl")
    selector_records = all_train[: int(args.selector_max_train_records)] if int(args.selector_max_train_records) > 0 else all_train
    selector = train_resid.build_selector(selector_records)
    lattice_mean, lattice_std = train_resid.lattice_stats(all_train)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    train_targets = select_records(all_train, int(args.start_train_index), int(args.max_train_records))
    val_targets = select_records(all_val, int(args.start_val_index), int(args.max_val_records))

    split_groups: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split, targets, exclude_self in (("train", train_targets, True), ("val", val_targets, False)):
        groups = split_groups[split]
        for idx, record in enumerate(targets, start=1):
            group = build_group(
                record,
                selector=selector,
                split=split,
                source_pool_k=int(args.source_pool_k),
                exclude_self=exclude_self,
                vpa_strength=float(args.vpa_strength),
                coord_weight=float(args.coord_weight),
                mean_t=mean_t,
                std_t=std_t,
            )
            if group is not None:
                groups.append(group)
            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                print(json.dumps({"split": split, "seen": idx, "groups": len(groups)}, sort_keys=True), flush=True)

    write_jsonl(train_out, split_groups["train"])
    write_jsonl(val_out, split_groups["val"])
    summary = {
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "selector_train_records": len(selector_records),
        "all_train_records": len(all_train),
        "all_val_records": len(all_val),
        "train": summarize(split_groups["train"]),
        "val": summarize(split_groups["val"]),
        "outputs": {"train": str(train_out), "val": str(val_out)},
        "leakage_guard": {
            "selector_split": "train",
            "train_targets": "train",
            "val_targets": "val",
            "test_records_used": 0,
            "structurematcher_labels_used": False,
            "label_kind": "train/val lattice+free-param transfer error, not match/rms",
        },
    }
    write_json(out_dir / "source_mode_examples_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
