#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


RUNTIME_FEATURE_NAMES = [
    "align_cost",
    "atom_count_abs_diff_norm",
    "base_vpa_norm",
    "candidate_score_rank_prior",
    "chem_distance",
    "expected_vpa_norm",
    "formula_l1",
    "row_condition_chem_distance",
    "same_sg",
    "same_skeleton_key",
    "same_wa_key",
    "source_param_prior_penalty",
    "source_quality_penalty",
    "source_rank_norm",
    "source_vpa_norm",
]


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing path outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = ensure_under_opentry(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    resolved = ensure_under_opentry(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = ensure_under_opentry(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def bool01(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def row_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(finite(row.get("rank"), 10**9)),
        int(finite(row.get("original_rank"), 10**9)),
        str(row.get("source_sample_id") or ""),
    )


def candidate_error(row: dict[str, Any]) -> float:
    if bool(row.get("label_match")):
        rms = row.get("label_rmsd")
        return finite(rms, 0.0)
    penalty = 10.0
    penalty += 5.0 * (1.0 - bool01(row.get("readable")))
    penalty += 3.0 * (1.0 - bool01(row.get("composition_exact")))
    penalty += 2.0 * (1.0 - bool01(row.get("sg_ok")))
    min_dist = finite(row.get("self_min_distance"), 1.0)
    penalty += max(0.0, 1.0 - min_dist)
    penalty += min(2.0, abs(finite(row.get("geometry_distance"), 0.0)) / 5.0)
    penalty += 0.01 * finite(row.get("rank"), 0.0)
    return float(penalty)


def make_runtime_features(row: dict[str, Any], source_rank: int, source_count: int) -> dict[str, float]:
    src_vpa = 0.0
    if finite(row.get("src_atom_count"), 0.0) > 0.0 and finite(row.get("self_volume"), 0.0) > 0.0:
        src_vpa = min(5.0, finite(row.get("self_volume"), 0.0) / max(1.0, finite(row.get("src_atom_count"), 1.0)) / 20.0)
    base_vpa = min(5.0, finite(row.get("self_volume_per_atom"), 0.0) / 20.0)
    source_rank_norm = float(source_rank) / max(1.0, float(source_count - 1))
    return {
        "align_cost": finite(row.get("geometry_distance"), 0.0),
        "atom_count_abs_diff_norm": min(5.0, finite(row.get("src_atom_count_absdiff"), 0.0) / 300.0),
        "base_vpa_norm": base_vpa,
        "candidate_score_rank_prior": -float(source_rank),
        "chem_distance": finite(row.get("src_formula_l1_frac"), finite(row.get("geometry_distance"), 0.0)),
        "expected_vpa_norm": base_vpa,
        "formula_l1": finite(row.get("src_formula_l1_frac"), 0.0),
        "row_condition_chem_distance": finite(row.get("geometry_distance"), 0.0),
        "same_sg": bool01(row.get("src_same_sg")),
        "same_skeleton_key": bool01(row.get("src_skeleton_eq_candidate")),
        "same_wa_key": bool01(row.get("src_wa_eq_candidate")),
        "source_param_prior_penalty": 0.0,
        "source_quality_penalty": 0.0,
        "source_rank_norm": source_rank_norm,
        "source_vpa_norm": src_vpa,
    }


def best_per_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=row_sort_key):
        source_id = str(row.get("source_sample_id") or "")
        if not source_id:
            continue
        old = by_source.get(source_id)
        if old is None or (candidate_error(row), row_sort_key(row)) < (candidate_error(old), row_sort_key(old)):
            by_source[source_id] = row
    return sorted(by_source.values(), key=row_sort_key)


def build_groups(
    rows: list[dict[str, Any]],
    *,
    split: str,
    max_groups: int,
    min_sources: int,
    require_positive: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        wa_key = str(row.get("canonical_wa_key") or "")
        if not sample_id or not wa_key:
            continue
        grouped[(sample_id, wa_key)].append(row)

    out: list[dict[str, Any]] = []
    for (_sample_id, _wa_key), items in sorted(grouped.items()):
        source_rows = best_per_source(items)
        if len(source_rows) < int(min_sources):
            continue
        has_positive = any(bool(row.get("label_match")) for row in source_rows)
        if require_positive and not has_positive:
            continue
        source_count = len(source_rows)
        candidates: list[dict[str, Any]] = []
        for source_rank, row in enumerate(source_rows):
            err = candidate_error(row)
            candidates.append(
                {
                    "source_sample_id": str(row.get("source_sample_id")),
                    "source_rank": int(source_rank),
                    "features": make_runtime_features(row, source_rank, source_count),
                    "combined_error": float(err),
                    "label_match": bool(row.get("label_match")),
                    "label_rmsd": row.get("label_rmsd"),
                    "readable": bool(row.get("readable")),
                    "composition_exact": bool(row.get("composition_exact")),
                    "sg_ok": bool(row.get("sg_ok")),
                    "rank": int(finite(row.get("rank"), source_rank + 1)),
                }
            )
        best_idx = min(range(len(candidates)), key=lambda idx: (float(candidates[idx]["combined_error"]), int(candidates[idx]["source_rank"])))
        first = source_rows[0]
        group = {
            "split": split,
            "sample_id": str(first.get("sample_id")),
            "material_id": str(first.get("material_id")),
            "sg": int(finite(first.get("sg"), 0.0)),
            "atom_count": int(finite(first.get("atom_count"), 0.0)),
            "num_elements": int(finite(first.get("formula_element_count"), 0.0)),
            "row_count_label": int(finite(first.get("target_row_count"), finite(first.get("candidate_row_count"), 0.0))),
            "complex_flags": {
                "rows_ge_7": bool(first.get("candidate_rows_ge_7")) or finite(first.get("target_row_count"), 0.0) >= 7.0,
                "atoms_ge_12": finite(first.get("atom_count"), 0.0) >= 12.0,
                "num_elements_ge_4": finite(first.get("formula_element_count"), 0.0) >= 4.0,
            },
            "canonical_skeleton_key": str(first.get("canonical_skeleton_key") or ""),
            "canonical_wa_key": str(first.get("canonical_wa_key") or ""),
            "source_pool_k": int(source_count),
            "vpa_strength": None,
            "label_kind": "StructureMatcher match/rms from train/val rendered candidates; no test labels",
            "candidates": candidates,
            "best_source_index": int(best_idx),
            "best_source_rank": int(candidates[best_idx]["source_rank"]),
            "heuristic_error": float(candidates[0]["combined_error"]),
            "best_error": float(candidates[best_idx]["combined_error"]),
            "oracle_gain": float(candidates[0]["combined_error"] - candidates[best_idx]["combined_error"]),
            "heuristic_is_best": int(best_idx) == 0,
            "positive_any": bool(has_positive),
        }
        out.append(group)
        if int(max_groups) > 0 and len(out) >= int(max_groups):
            break
    return out


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {"groups": 0}

    def sub(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"groups": 0}
        all_candidates = [cand for group in items for cand in group.get("candidates") or []]
        positives = [cand for cand in all_candidates if bool(cand.get("label_match"))]
        return {
            "groups": len(items),
            "positive_any_rate": float(mean([1.0 if group.get("positive_any") else 0.0 for group in items])),
            "candidate_count_mean": float(mean([len(group.get("candidates") or []) for group in items])),
            "candidate_positive_rate": float(len(positives) / max(1, len(all_candidates))),
            "heuristic_is_best_rate": float(mean([1.0 if group.get("heuristic_is_best") else 0.0 for group in items])),
            "best_source_rank_mean": float(mean([finite(group.get("best_source_rank")) for group in items])),
            "oracle_gain_mean": float(mean([finite(group.get("oracle_gain")) for group in items])),
            "oracle_gain_positive_rate": float(mean([1.0 if finite(group.get("oracle_gain")) > 1e-8 else 0.0 for group in items])),
        }

    rows_ge7 = [group for group in groups if bool(dict(group.get("complex_flags") or {}).get("rows_ge_7"))]
    atoms_ge12 = [group for group in groups if bool(dict(group.get("complex_flags") or {}).get("atoms_ge_12"))]
    return {"full": sub(groups), "rows_ge_7": sub(rows_ge7), "atoms_ge_12": sub(atoms_ge12)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source-mode examples from train/val rendered-candidate StructureMatcher labels.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-train-groups", type=int, default=0)
    parser.add_argument("--max-val-groups", type=int, default=0)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--require-positive", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_features)
    val_rows = read_jsonl(args.val_features)
    train_groups = build_groups(
        train_rows,
        split="train",
        max_groups=int(args.max_train_groups),
        min_sources=int(args.min_sources),
        require_positive=bool(args.require_positive),
    )
    val_groups = build_groups(
        val_rows,
        split="val",
        max_groups=int(args.max_val_groups),
        min_sources=int(args.min_sources),
        require_positive=bool(args.require_positive),
    )
    write_jsonl(out_dir / "train_source_mode_examples.jsonl", train_groups)
    write_jsonl(out_dir / "val_source_mode_examples.jsonl", val_groups)
    summary = {
        "config": jsonable_args(args),
        "runtime_feature_names": RUNTIME_FEATURE_NAMES,
        "train_input_rows": len(train_rows),
        "val_input_rows": len(val_rows),
        "train": summarize(train_groups),
        "val": summarize(val_groups),
        "outputs": {
            "train": str(out_dir / "train_source_mode_examples.jsonl"),
            "val": str(out_dir / "val_source_mode_examples.jsonl"),
        },
        "leakage_guard": {
            "train_labels": "StructureMatcher label_match/label_rmsd from train rendered candidate features only",
            "val_labels": "StructureMatcher labels from val rendered candidate features, used for model selection only",
            "test_records_used": 0,
            "feature_policy": "Only renderer-runtime numeric source features are emitted; target keys/row_count and candidate hit flags are labels/reporting only.",
        },
    }
    write_json(out_dir / "match_source_mode_examples_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
