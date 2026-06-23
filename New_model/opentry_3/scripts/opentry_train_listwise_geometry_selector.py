#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opentry_train_geometry_compat_selector import (
    BOOLEAN_KEYS,
    CATEGORICAL_KEYS,
    NUMERIC_KEYS,
    ensure_under_opentry,
    feature_dict,
    grouped,
    metrics_from_ranked,
    write_json,
    write_jsonl,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_weight(row: dict[str, Any]) -> float:
    weight = 1.0
    if int(row.get("atom_count", 0)) >= 12:
        weight *= 1.25
    if int(row.get("target_row_count", 0)) >= 7:
        weight *= 2.0
    return weight


def train_pairwise_ranker(
    train_rows: list[dict[str, Any]],
    *,
    max_pairs_per_sample: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    vectorizer = DictVectorizer(sparse=True)
    x_all = vectorizer.fit_transform([feature_dict(row) for row in train_rows])
    scaler = StandardScaler(with_mean=False)
    x_all = scaler.fit_transform(x_all)
    row_index = {id(row): idx for idx, row in enumerate(train_rows)}
    diff_rows = []
    labels: list[int] = []
    weights: list[float] = []
    samples_with_pairs = 0
    for items in grouped(train_rows).values():
        positives = [row for row in items if bool(row.get("label_match"))]
        negatives = [row for row in items if not bool(row.get("label_match"))]
        if not positives or not negatives:
            continue
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        positives = sorted(positives, key=lambda row: int(row.get("rank", 10**9)))
        negatives = sorted(negatives, key=lambda row: int(row.get("rank", 10**9)))
        for pos in positives:
            for neg in negatives:
                pairs.append((pos, neg))
        rng.shuffle(pairs)
        samples_with_pairs += 1
        for pos, neg in pairs[: int(max_pairs_per_sample)]:
            pos_i = row_index[id(pos)]
            neg_i = row_index[id(neg)]
            diff_rows.append(x_all[pos_i] - x_all[neg_i])
            labels.append(1)
            weights.append(row_weight(pos))
            diff_rows.append(x_all[neg_i] - x_all[pos_i])
            labels.append(0)
            weights.append(row_weight(pos))
    if not diff_rows:
        raise SystemExit("No positive/negative pairs available for pairwise ranker.")
    x_pair = sparse.vstack(diff_rows)
    clf = LogisticRegression(
        C=0.5,
        penalty="l2",
        fit_intercept=False,
        solver="liblinear",
        max_iter=500,
        random_state=seed,
    )
    clf.fit(x_pair, np.asarray(labels), sample_weight=np.asarray(weights))
    return {
        "model_type": "pairwise_logistic",
        "vectorizer": vectorizer,
        "scaler": scaler,
        "clf": clf,
        "fit_summary": {
            "pair_rows": int(len(labels)),
            "samples_with_pairs": int(samples_with_pairs),
            "positive_pair_rate": float(sum(labels) / max(1, len(labels))),
        },
    }


def score_pairwise(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x = model["vectorizer"].transform([feature_dict(row) for row in rows])
    x = model["scaler"].transform(x)
    scores = x @ model["clf"].coef_.reshape(-1)
    out: list[dict[str, Any]] = []
    for row, score in zip(rows, np.asarray(scores).reshape(-1)):
        item = dict(row)
        item["listwise_score"] = float(score)
        out.append(item)
    return out


def select_pairwise_mmr(rows: list[dict[str, Any]], *, top_k: int, max_per_wa: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped(rows).items()):
        ordered = sorted(
            items,
            key=lambda row: (-float(row.get("listwise_score", 0.0)), int(row.get("rank", 10**9))),
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
            seen = {id(row) for row in selected}
            for row in ordered:
                if id(row) in seen:
                    continue
                selected.append(row)
                if len(selected) >= int(top_k):
                    break
        for rank, row in enumerate(selected[: int(top_k)], start=1):
            item = dict(row)
            item["rank"] = rank
            item["listwise_mode"] = "pairwise_mmr"
            out.append(item)
    return out


def group_features(group: list[dict[str, Any]]) -> dict[str, Any]:
    first = min(group, key=lambda row: int(row.get("rank", 10**9)))

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

    self_scores = [safe_float(row.get("self_score", 0.0)) for row in group]
    distances = [safe_float(row.get("geometry_distance", 0.0)) for row in group]
    min_d = [safe_float(row.get("self_min_distance", 0.0)) for row in group]
    vpa = [safe_float(row.get("self_volume_per_atom", 0.0)) for row in group]
    base = feature_dict(first)
    base.update(
        {
            "group_size": float(len(group)),
            "group_first_rank": float(min(int(row.get("rank", 10**9)) for row in group)),
            "group_best_self_score": max(self_scores) if self_scores else 0.0,
            "group_mean_self_score": sum(self_scores) / max(1, len(self_scores)),
            "group_min_geometry_distance": min(distances) if distances else 0.0,
            "group_mean_geometry_distance": sum(distances) / max(1, len(distances)),
            "group_max_min_distance": max(min_d) if min_d else 0.0,
            "group_mean_vpa": sum(vpa) / max(1, len(vpa)),
        }
    )
    return base


def build_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample_id, items in grouped(rows).items():
        by_wa: dict[str, list[dict[str, Any]]] = {}
        for row in items:
            by_wa.setdefault(str(row.get("canonical_wa_key") or ""), []).append(row)
        for wa_key, group in by_wa.items():
            first = min(group, key=lambda row: int(row.get("rank", 10**9)))
            out.append(
                {
                    "sample_id": sample_id,
                    "canonical_wa_key": wa_key,
                    "rows": group,
                    "features": group_features(group),
                    "label": any(bool(row.get("label_match")) for row in group),
                    "weight": row_weight(first) * (4.0 if any(bool(row.get("label_match")) for row in group) else 1.0),
                }
            )
    return out


def train_group_model(train_rows: list[dict[str, Any]], *, seed: int) -> Pipeline:
    groups = build_group_rows(train_rows)
    x = [row["features"] for row in groups]
    y = [int(bool(row["label"])) for row in groups]
    weights = [float(row["weight"]) for row in groups]
    model = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    C=0.35,
                    class_weight=None,
                    max_iter=400,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x, y, clf__sample_weight=weights)
    return model


def select_group_bundle(rows: list[dict[str, Any]], model: Pipeline, *, top_k: int, bundle_size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    group_rows = build_group_rows(rows)
    if group_rows:
        group_scores = model.predict_proba([group["features"] for group in group_rows])[:, 1]
        for group, score in zip(group_rows, group_scores):
            group["group_score"] = float(score)
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for group in group_rows:
        by_sample.setdefault(str(group["sample_id"]), []).append(group)
    for sample_id, groups in sorted(by_sample.items()):
        groups.sort(
            key=lambda group: (
                -float(group["group_score"]),
                min(int(row.get("rank", 10**9)) for row in group["rows"]),
            )
        )
        selected: list[dict[str, Any]] = []
        for group in groups:
            candidates = sorted(
                group["rows"],
                key=lambda row: (
                    -float(row.get("self_score", 0.0)),
                    int(row.get("rank", 10**9)),
                ),
            )
            for row in candidates[: int(bundle_size)]:
                item = dict(row)
                item["group_score"] = float(group["group_score"])
                item["listwise_mode"] = "group_bundle"
                selected.append(item)
                if len(selected) >= int(top_k):
                    break
            if len(selected) >= int(top_k):
                break
        for rank, row in enumerate(selected[: int(top_k)], start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    return out


def strip_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if not k.startswith("label_")} for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train group/listwise GT-free geometry selectors and apply them to validation candidates.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["pairwise_mmr", "group_bundle"], required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-pairs-per-sample", type=int, default=40)
    parser.add_argument("--max-per-wa", type=int, default=2)
    parser.add_argument("--bundle-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_features)
    val_rows = read_jsonl(args.val_features)
    if args.mode == "pairwise_mmr":
        model = train_pairwise_ranker(train_rows, max_pairs_per_sample=int(args.max_pairs_per_sample), seed=int(args.seed))
        scored = score_pairwise(model, val_rows)
        ranked = select_pairwise_mmr(scored, top_k=int(args.top_k), max_per_wa=int(args.max_per_wa))
        joblib.dump(model, out_dir / "pairwise_ranker.joblib")
        fit_summary = model["fit_summary"]
    else:
        model = train_group_model(train_rows, seed=int(args.seed))
        ranked = select_group_bundle(val_rows, model, top_k=int(args.top_k), bundle_size=int(args.bundle_size))
        joblib.dump(model, out_dir / "group_model.joblib")
        train_groups = build_group_rows(train_rows)
        fit_summary = {
            "train_groups": len(train_groups),
            "train_group_positive_rate": sum(int(row["label"]) for row in train_groups) / max(1, len(train_groups)),
        }
    write_jsonl(out_dir / "val_reranked_features.jsonl", ranked)
    write_jsonl(out_dir / "rendered_topk_listwise.jsonl", strip_labels(ranked))
    metrics = metrics_from_ranked(ranked)
    summary = {
        "mode": args.mode,
        "train_features": str(args.train_features),
        "val_features": str(args.val_features),
        "top_k": int(args.top_k),
        "feature_policy": {
            "base_numeric": sorted(NUMERIC_KEYS),
            "base_boolean": sorted(BOOLEAN_KEYS),
            "base_categorical": sorted(CATEGORICAL_KEYS),
            "note": "Only GT-free feature_dict inputs are used for scoring; labels and target keys are excluded by imported feature policy.",
        },
        "fit": fit_summary,
        "metrics": {k: v for k, v in metrics.items() if k != "per_sample"},
    }
    write_json(out_dir / "listwise_selector_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
