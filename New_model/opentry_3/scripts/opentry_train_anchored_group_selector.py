#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opentry_train_dynamic_compat_selector import feature_dict, grouped
from opentry_train_geometry_compat_selector import ensure_under_opentry, metrics_from_ranked, write_json, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def row_weight(row: dict[str, Any], positive: bool) -> float:
    weight = 5.0 if positive else 1.0
    if int(row.get("atom_count", 0)) >= 12:
        weight *= 1.25
    # target_row_count is a train/val label-side diagnostic; it is never included in features.
    if int(row.get("target_row_count", 0)) >= 7:
        weight *= 2.5
    if bool(row.get("target_complex_flag")):
        weight *= 1.15
    return float(weight)


def group_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row.get("rank") or row.get("original_rank") or 10**9))
    first = ordered[0]
    out = dict(feature_dict(first))
    self_scores = [safe_float(row.get("self_score")) for row in rows]
    geom = [safe_float(row.get("geometry_distance")) for row in rows]
    min_d = [safe_float(row.get("self_min_distance")) for row in rows]
    vpa = [safe_float(row.get("self_volume_per_atom")) for row in rows]
    comp = [1.0 if row.get("composition_exact") else 0.0 for row in rows]
    sg_ok = [1.0 if row.get("sg_ok") else 0.0 for row in rows]
    out.update(
        {
            "group_size": float(len(rows)),
            "group_first_rank": float(min(int(row.get("rank") or 10**9) for row in rows)),
            "group_best_self_score": max(self_scores) if self_scores else 0.0,
            "group_mean_self_score": sum(self_scores) / max(1, len(self_scores)),
            "group_min_geometry_distance": min(geom) if geom else 0.0,
            "group_mean_geometry_distance": sum(geom) / max(1, len(geom)),
            "group_max_min_distance": max(min_d) if min_d else 0.0,
            "group_mean_min_distance": sum(min_d) / max(1, len(min_d)),
            "group_mean_vpa": sum(vpa) / max(1, len(vpa)),
            "group_composition_exact_rate": sum(comp) / max(1, len(comp)),
            "group_sg_ok_rate": sum(sg_ok) / max(1, len(sg_ok)),
        }
    )
    return out


def build_groups(rows: list[dict[str, Any]], *, group_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample_id, items in grouped(rows).items():
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in items:
            if group_key == "wa_source":
                key = f"{row.get('canonical_wa_key') or ''}||{row.get('source_sample_id') or ''}"
            elif group_key == "wa":
                key = str(row.get("canonical_wa_key") or "")
            else:
                key = f"{row.get('canonical_skeleton_key') or ''}||{row.get('source_sample_id') or ''}"
            buckets.setdefault(key, []).append(row)
        for key, group_rows in buckets.items():
            ordered = sorted(group_rows, key=lambda row: int(row.get("rank") or row.get("original_rank") or 10**9))
            label = any(bool(row.get("label_match")) for row in ordered)
            out.append(
                {
                    "sample_id": sample_id,
                    "group_key": key,
                    "wa_key": str(ordered[0].get("canonical_wa_key") or ""),
                    "rows": ordered,
                    "features": group_features(ordered),
                    "label": label,
                    "weight": row_weight(ordered[0], label),
                    "first_rank": int(ordered[0].get("rank") or 10**9),
                }
            )
    return out


def train_model(groups: list[dict[str, Any]], *, model_type: str, seed: int) -> Pipeline:
    x = [group["features"] for group in groups]
    y = [int(bool(group["label"])) for group in groups]
    weights = [float(group["weight"]) for group in groups]
    if model_type == "gbdt":
        model: Pipeline = Pipeline(
            [
                ("vec", DictVectorizer(sparse=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.035,
                        max_leaf_nodes=15,
                        min_samples_leaf=10,
                        l2_regularization=0.12,
                        random_state=seed,
                    ),
                ),
            ]
        )
    else:
        model = Pipeline(
            [
                ("vec", DictVectorizer(sparse=True)),
                ("scale", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        C=0.25,
                        max_iter=500,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        )
    model.fit(x, y, clf__sample_weight=weights)
    return model


def score_groups(model: Pipeline, groups: list[dict[str, Any]]) -> None:
    if not groups:
        return
    scores = model.predict_proba([group["features"] for group in groups])[:, 1]
    for group, score in zip(groups, scores):
        group["score"] = float(score)


def best_row_for_group(group: dict[str, Any]) -> dict[str, Any]:
    return sorted(
        group["rows"],
        key=lambda row: (
            -safe_float(row.get("self_score")),
            int(row.get("rank") or row.get("original_rank") or 10**9),
        ),
    )[0]


def select_anchored(
    val_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    *,
    top_k: int,
    anchor_count: int,
    max_per_wa: int,
) -> list[dict[str, Any]]:
    by_sample_groups: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        by_sample_groups.setdefault(str(group["sample_id"]), []).append(group)
    out: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped(val_rows).items()):
        original = sorted(items, key=lambda row: int(row.get("rank") or row.get("original_rank") or 10**9))
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        wa_counts: dict[str, int] = {}
        for row in original[: max(0, int(anchor_count))]:
            item = dict(row)
            selected.append(item)
            selected_ids.add(str(row.get("candidate_uid") or id(row)))
            wa = str(row.get("canonical_wa_key") or "")
            wa_counts[wa] = wa_counts.get(wa, 0) + 1
            if len(selected) >= int(top_k):
                break
        groups_sorted = sorted(
            by_sample_groups.get(str(sample_id), []),
            key=lambda group: (-safe_float(group.get("score")), int(group.get("first_rank", 10**9))),
        )
        for group in groups_sorted:
            row = best_row_for_group(group)
            uid = str(row.get("candidate_uid") or id(row))
            if uid in selected_ids:
                continue
            wa = str(row.get("canonical_wa_key") or "")
            if wa_counts.get(wa, 0) >= int(max_per_wa):
                continue
            item = dict(row)
            item["anchored_group_score"] = safe_float(group.get("score"))
            selected.append(item)
            selected_ids.add(uid)
            wa_counts[wa] = wa_counts.get(wa, 0) + 1
            if len(selected) >= int(top_k):
                break
        if len(selected) < int(top_k):
            for row in original:
                uid = str(row.get("candidate_uid") or id(row))
                if uid in selected_ids:
                    continue
                item = dict(row)
                selected.append(item)
                selected_ids.add(uid)
                if len(selected) >= int(top_k):
                    break
        for rank, row in enumerate(selected[: int(top_k)], start=1):
            item = dict(row)
            item["rank"] = rank
            item["anchored_group_mode"] = "anchored_group"
            out.append(item)
    return out


def strip_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if not k.startswith("label_")} for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an anchored group-level compatibility selector for rendered candidates.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["gbdt", "logreg"], default="gbdt")
    parser.add_argument("--group-key", choices=["wa", "wa_source", "skeleton_source"], default="wa_source")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--anchor-count", type=int, default=3)
    parser.add_argument("--max-per-wa", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_features)
    val_rows = read_jsonl(args.val_features)
    train_groups = build_groups(train_rows, group_key=str(args.group_key))
    val_groups = build_groups(val_rows, group_key=str(args.group_key))
    model = train_model(train_groups, model_type=str(args.model_type), seed=int(args.seed))
    score_groups(model, val_groups)
    ranked = select_anchored(
        val_rows,
        val_groups,
        top_k=int(args.top_k),
        anchor_count=int(args.anchor_count),
        max_per_wa=int(args.max_per_wa),
    )
    write_jsonl(out_dir / "val_reranked_features.jsonl", ranked)
    write_jsonl(out_dir / "rendered_topk_anchored.jsonl", strip_labels(ranked))
    joblib.dump(model, out_dir / "anchored_group_model.joblib")
    metrics = metrics_from_ranked(ranked)
    summary = {
        "anchor_count": int(args.anchor_count),
        "group_key": str(args.group_key),
        "max_per_wa": int(args.max_per_wa),
        "model_type": str(args.model_type),
        "top_k": int(args.top_k),
        "train_features": str(args.train_features),
        "val_features": str(args.val_features),
        "fit": {
            "train_groups": len(train_groups),
            "train_group_positive_rate": float(sum(int(group["label"]) for group in train_groups) / max(1, len(train_groups))),
            "val_groups": len(val_groups),
            "val_group_positive_rate": float(sum(int(group["label"]) for group in val_groups) / max(1, len(val_groups))),
        },
        "feature_policy": {
            "note": "Uses opentry_train_dynamic_compat_selector.feature_dict; label, target, sample/source ids, keys and CIF text are excluded from features.",
        },
        "metrics": {k: v for k, v in metrics.items() if k != "per_sample"},
    }
    write_json(out_dir / "anchored_group_selector_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
