#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opentry_train_geometry_compat_selector import ensure_under_opentry, grouped, metrics_from_ranked, write_json, write_jsonl


BLOCK_KEYS = {
    "label_match",
    "label_rmsd",
    "label_error",
    "candidate_wa_hit",
    "candidate_skeleton_hit",
    "target_wa_key",
    "target_skeleton_key",
    "target_row_count",
    "target_complex_flag",
    "sample_id",
    "material_id",
    "candidate_uid",
    "split",
    "cif",
    "cif_path",
    "source_sample_id",
    "canonical_wa_key",
    "canonical_skeleton_key",
    "raw_dp_wa_key",
    "raw_dp_skeleton_key",
    "error",
    "self_parse_error",
}

CATEGORICAL_KEYS = {
    "crystal_system",
    "geometry_lattice_mode",
    "geometry_mode",
    "geometry_param_variant_mode",
    "geometry_source",
    "proposal_pool",
    "src_crystal_system",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_blocked_key(key: str) -> bool:
    if key in BLOCK_KEYS:
        return True
    if key.startswith("label_") or key.startswith("target_"):
        return True
    if key.endswith("_key") or key.endswith("_uid"):
        return True
    return False


def feature_dict(row: dict[str, Any]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for key, value in row.items():
        if is_blocked_key(str(key)):
            continue
        if value is None:
            out[f"{key}__missing"] = 1.0
            continue
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
            continue
        if isinstance(value, (int, float)):
            value_f = float(value)
            if math.isfinite(value_f):
                out[key] = value_f
            else:
                out[f"{key}__missing"] = 1.0
            continue
        if isinstance(value, str) and key in CATEGORICAL_KEYS:
            out[key] = value
    out["has_error"] = 1.0 if row.get("error") else 0.0
    out["has_self_parse_error"] = 1.0 if row.get("self_parse_error") else 0.0
    return out


def row_weight(row: dict[str, Any], y: int) -> float:
    weight = 4.0 if y else 1.0
    if int(row.get("atom_count", 0)) >= 12:
        weight *= 1.25
    if int(row.get("target_row_count", 0)) >= 7:
        weight *= 2.0
    if bool(row.get("target_complex_flag")):
        weight *= 1.15
    return float(weight)


def score_pipeline(model: Pipeline, rows: list[dict[str, Any]]) -> np.ndarray:
    x = [feature_dict(row) for row in rows]
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.decision_function(x), dtype=float)


def train_gbdt(train_rows: list[dict[str, Any]], *, seed: int) -> Pipeline:
    x_train = [feature_dict(row) for row in train_rows]
    y_train = [int(bool(row.get("label_match"))) for row in train_rows]
    weights = [row_weight(row, y) for row, y in zip(train_rows, y_train)]
    model: Pipeline = Pipeline(
        [
            ("vec", DictVectorizer(sparse=False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.04,
                    max_leaf_nodes=19,
                    min_samples_leaf=14,
                    l2_regularization=0.08,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train, clf__sample_weight=weights)
    return model


def train_pairwise(train_rows: list[dict[str, Any]], *, seed: int, max_pairs_per_sample: int) -> dict[str, Any]:
    rng = random.Random(seed)
    vectorizer = DictVectorizer(sparse=True)
    x_all = vectorizer.fit_transform([feature_dict(row) for row in train_rows])
    scaler = StandardScaler(with_mean=False)
    x_all = scaler.fit_transform(x_all)
    row_index = {id(row): idx for idx, row in enumerate(train_rows)}
    diffs = []
    labels: list[int] = []
    weights: list[float] = []
    samples_with_pairs = 0
    for items in grouped(train_rows).values():
        positives = [row for row in items if bool(row.get("label_match"))]
        negatives = [row for row in items if not bool(row.get("label_match"))]
        if not positives or not negatives:
            continue
        samples_with_pairs += 1
        pairs = []
        for pos in sorted(positives, key=lambda row: int(row.get("rank", 10**9)))[:8]:
            for neg in sorted(negatives, key=lambda row: int(row.get("rank", 10**9)))[:16]:
                pairs.append((pos, neg))
        rng.shuffle(pairs)
        for pos, neg in pairs[: int(max_pairs_per_sample)]:
            pos_i = row_index[id(pos)]
            neg_i = row_index[id(neg)]
            diff = x_all[pos_i] - x_all[neg_i]
            diffs.append(diff)
            labels.append(1)
            weights.append(row_weight(pos, 1))
            diffs.append(-diff)
            labels.append(0)
            weights.append(row_weight(pos, 1))
    if not diffs:
        raise SystemExit("No pairwise training data available.")
    x_pair = sparse.vstack(diffs, format="csr")
    clf = LogisticRegression(C=0.3, fit_intercept=False, max_iter=500, solver="liblinear", random_state=seed)
    clf.fit(x_pair, labels, sample_weight=weights)
    return {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "clf": clf,
        "fit": {"pair_rows": int(x_pair.shape[0]), "pair_features": int(x_pair.shape[1]), "samples_with_pairs": samples_with_pairs},
    }


def score_pairwise(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    x = model["vectorizer"].transform([feature_dict(row) for row in rows])
    x = model["scaler"].transform(x)
    return np.asarray(x @ model["clf"].coef_.reshape(-1), dtype=float).reshape(-1)


def rerank(rows: list[dict[str, Any]], scores: np.ndarray, *, top_k: int, max_per_wa: int, score_key: str) -> list[dict[str, Any]]:
    scored = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item[score_key] = float(score)
        scored.append(item)
    out: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped(scored).items()):
        ordered = sorted(
            items,
            key=lambda row: (
                -float(row.get(score_key, 0.0)),
                int(row.get("original_rank") or row.get("rank") or 10**9),
            ),
        )
        selected: list[dict[str, Any]] = []
        wa_counts: dict[str, int] = {}
        for row in ordered:
            wa_key = str(row.get("canonical_wa_key") or "")
            if wa_counts.get(wa_key, 0) >= int(max_per_wa):
                continue
            selected.append(row)
            wa_counts[wa_key] = wa_counts.get(wa_key, 0) + 1
            if len(selected) >= int(top_k):
                break
        if len(selected) < int(top_k):
            selected_ids = {id(row) for row in selected}
            for row in ordered:
                if id(row) in selected_ids:
                    continue
                selected.append(row)
                if len(selected) >= int(top_k):
                    break
        for rank, row in enumerate(selected[: int(top_k)], start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    return out


def strip_labels(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("label_")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train dynamic GT-free compatibility selectors over enriched rendered-candidate features.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["gbdt", "pairwise"], required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-per-wa", type=int, default=2)
    parser.add_argument("--max-pairs-per-sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_features)
    val_rows = read_jsonl(args.val_features)
    y_train = [int(bool(row.get("label_match"))) for row in train_rows]
    y_val = [int(bool(row.get("label_match"))) for row in val_rows]
    if args.model_type == "gbdt":
        model = train_gbdt(train_rows, seed=int(args.seed))
        scores = score_pipeline(model, val_rows)
        joblib.dump(model, out_dir / "dynamic_gbdt.joblib")
        fit_info = {"train_rows": len(train_rows), "train_positive_rate": sum(y_train) / max(1, len(y_train))}
    else:
        model = train_pairwise(train_rows, seed=int(args.seed), max_pairs_per_sample=int(args.max_pairs_per_sample))
        scores = score_pairwise(model, val_rows)
        joblib.dump(model, out_dir / "dynamic_pairwise.joblib")
        fit_info = model["fit"] | {"train_rows": len(train_rows), "train_positive_rate": sum(y_train) / max(1, len(y_train))}
    ranked = rerank(val_rows, scores, top_k=int(args.top_k), max_per_wa=int(args.max_per_wa), score_key=f"{args.model_type}_score")
    scored_rows = []
    for row, score in zip(val_rows, scores):
        item = dict(row)
        item[f"{args.model_type}_score"] = float(score)
        scored_rows.append(item)
    write_jsonl(out_dir / "val_scored_candidates.jsonl", scored_rows)
    write_jsonl(out_dir / "val_reranked_features.jsonl", ranked)
    write_jsonl(out_dir / "rendered_topk_dynamic.jsonl", [strip_labels(row) for row in ranked])
    metrics = metrics_from_ranked(ranked)
    score_metrics: dict[str, float] = {}
    if len(set(y_val)) > 1:
        score_metrics = {
            "roc_auc": float(roc_auc_score(y_val, scores)),
            "average_precision": float(average_precision_score(y_val, scores)),
        }
    summary = {
        "model_type": str(args.model_type),
        "top_k": int(args.top_k),
        "max_per_wa": int(args.max_per_wa),
        "fit": fit_info,
        "candidate_score_metrics": score_metrics,
        "feature_policy": {
            "blocked_keys": sorted(BLOCK_KEYS),
            "categorical_keys": sorted(CATEGORICAL_KEYS),
            "note": "Dynamic features exclude labels, target row_count/keys, target flags, raw source IDs, raw CIF, sample/material IDs, and candidate hit flags.",
        },
        "label_metrics": {k: v for k, v in metrics.items() if k != "per_sample"},
    }
    write_json(out_dir / "dynamic_selector_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
