#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


BLOCKED_FEATURE_KEYS = {
    "label_match",
    "label_rmsd",
    "label_error",
    "target_wa_key",
    "target_skeleton_key",
    "target_row_count",
    "target_complex_flag",
    "candidate_wa_hit",
    "candidate_skeleton_hit",
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
    "self_parse_error",
    "error",
}

CATEGORICAL_KEYS = {
    "crystal_system",
    "geometry_lattice_mode",
    "geometry_mode",
    "geometry_param_variant_mode",
    "geometry_source",
}

NUMERIC_KEYS = {
    "rank",
    "original_rank",
    "candidate_score",
    "geometry_distance",
    "geometry_rank",
    "source_rank",
    "source_choice_rank",
    "source_mode_score",
    "source_mode_margin",
    "source_mode_raw_best_idx",
    "source_mode_rank0_score",
    "source_mode_choice_rank",
    "source_mode_mixture_size",
    "self_min_distance",
    "self_volume",
    "self_volume_per_atom",
    "self_parsed_sites",
    "self_score",
    "self_train_volume_prior_score",
    "detected_sg",
    "atom_count_after_expansion",
    "sg",
    "atom_count",
    "formula_element_count",
    "candidate_row_count",
    "candidate_unique_orbit_count",
    "candidate_duplicate_orbit_count",
    "candidate_unique_element_count",
    "candidate_max_multiplicity",
    "candidate_mean_multiplicity",
    "source_pair_source_sg",
    "source_pair_source_atom_count",
    "source_pair_source_formula_element_count",
    "source_pair_source_row_count",
    "source_pair_candidate_row_count",
    "source_pair_candidate_unique_orbit_count",
    "source_pair_candidate_duplicate_orbit_count",
    "source_pair_candidate_unique_element_count",
    "source_pair_source_free_param_rows",
    "source_pair_source_free_param_total",
    "source_pair_formula_l1",
    "source_pair_formula_element_jaccard",
    "source_pair_candidate_source_element_jaccard",
    "source_pair_atom_count_ratio_log",
    "source_pair_row_count_delta",
    "source_pair_abs_row_count_delta",
    "source_pair_common_orbit_count",
    "source_pair_common_orbit_frac_candidate",
    "source_pair_common_orbit_frac_source",
    "source_pair_common_wa_pair_count",
    "source_pair_common_wa_frac_candidate",
    "source_pair_common_wa_frac_source",
}

BOOLEAN_KEYS = {
    "readable",
    "formula_ok",
    "atom_count_ok",
    "composition_exact",
    "sg_ok",
    "source_mode_overrode_to_rank0",
    "source_pair_has_source",
    "source_pair_source_sg_same",
    "source_pair_source_complex_flag",
    "source_pair_same_canonical_skeleton",
    "source_pair_same_canonical_wa",
}


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_prefix_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if int(max_rows) <= 0 or len(rows) <= int(max_rows):
        return rows
    positives = [row for row in rows if bool(row.get("label_match"))]
    negatives = [row for row in rows if not bool(row.get("label_match"))]
    pos_take = min(len(positives), max(1, int(max_rows) // 3))
    neg_take = max(0, int(max_rows) - pos_take)
    selected = positives[:pos_take] + negatives[:neg_take]
    selected.sort(key=lambda row: (str(row.get("sample_id")), int(row.get("rank", 10**9))))
    return selected


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


def feature_dict(row: dict[str, Any]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for key in NUMERIC_KEYS:
        if key in BLOCKED_FEATURE_KEYS:
            continue
        value = row.get(key)
        if value is None:
            out[f"{key}__missing"] = 1.0
            out[key] = 0.0
            continue
        try:
            value_f = float(value)
            if not math.isfinite(value_f):
                value_f = 0.0
                out[f"{key}__missing"] = 1.0
            out[key] = value_f
        except Exception:
            out[f"{key}__missing"] = 1.0
            out[key] = 0.0
    for key in BOOLEAN_KEYS:
        if key in BLOCKED_FEATURE_KEYS:
            continue
        out[key] = 1.0 if bool(row.get(key)) else 0.0
    for key in CATEGORICAL_KEYS:
        if key in BLOCKED_FEATURE_KEYS:
            continue
        value = row.get(key)
        if value is not None:
            out[key] = str(value)
    return out


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["sample_id"]), []).append(row)
    for items in by_id.values():
        items.sort(key=lambda row: int(row.get("rank", 10**9)))
    return by_id


def score_rows(model: Pipeline, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [feature_dict(row) for row in rows]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    else:
        scores = model.decision_function(features)
    out: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["compat_score"] = float(score)
        out.append(item)
    return out


def rerank(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped(rows).items()):
        selected = sorted(
            items,
            key=lambda row: (
                -float(row.get("compat_score", 0.0)),
                int(row.get("original_rank") or row.get("rank") or 10**9),
            ),
        )[: int(top_k)]
        for rank, row in enumerate(selected, start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    return out


def metrics_from_ranked(rows: list[dict[str, Any]], budgets: tuple[int, ...] = (1, 5, 20, 50)) -> dict[str, Any]:
    by_id = grouped(rows)
    per_sample: list[dict[str, Any]] = []
    for sample_id, items in by_id.items():
        sample: dict[str, Any] = {
            "sample_id": sample_id,
            "target_row_count": int(items[0].get("target_row_count", 0)),
            "atom_count": int(items[0].get("atom_count", 0)),
            "target_complex_flag": bool(items[0].get("target_complex_flag")),
        }
        for k in budgets:
            top = items[:k]
            matched = [row for row in top if bool(row.get("label_match"))]
            sample[f"match@{k}"] = bool(matched)
            sample[f"rmsd@{k}"] = None if not matched else min(float(row["label_rmsd"]) for row in matched if row.get("label_rmsd") is not None)
            sample[f"W/A@{k}"] = any(bool(row.get("candidate_wa_hit")) for row in top)
            sample[f"skeleton@{k}"] = any(bool(row.get("candidate_skeleton_hit")) for row in top)
            sample[f"unique_wa@{k}"] = len(set(str(row.get("canonical_wa_key") or "") for row in top))
        per_sample.append(sample)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        denom = max(1, len(items))
        payload: dict[str, Any] = {"samples": len(items)}
        for k in budgets:
            payload[f"match@{k}"] = sum(int(row[f"match@{k}"]) for row in items) / denom
            rms = [row[f"rmsd@{k}"] for row in items if row[f"rmsd@{k}"] is not None]
            payload[f"RMSE@{k}"] = None if not rms else float(sum(float(x) for x in rms) / len(rms))
            payload[f"W/A@{k}"] = sum(int(row[f"W/A@{k}"]) for row in items) / denom
            payload[f"skeleton@{k}"] = sum(int(row[f"skeleton@{k}"]) for row in items) / denom
            payload[f"unique_wa_mean@{k}"] = sum(int(row[f"unique_wa@{k}"]) for row in items) / denom
        wa_hit_50 = [row for row in items if bool(row["W/A@50"])]
        payload["wa_hit_match_fail_rate@50"] = None if not wa_hit_50 else sum(int(not row["match@50"]) for row in wa_hit_50) / len(wa_hit_50)
        return payload

    return {
        "full": summarize(per_sample),
        "rows_ge_7": summarize([row for row in per_sample if int(row["target_row_count"]) >= 7]),
        "atoms_ge_12": summarize([row for row in per_sample if int(row["atom_count"]) >= 12]),
        "complex_flag": summarize([row for row in per_sample if bool(row["target_complex_flag"])]),
        "per_sample": per_sample,
    }


def baseline_by_current_rank(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped(rows).items()):
        selected = sorted(items, key=lambda row: int(row.get("rank", 10**9)))[: int(top_k)]
        for rank, row in enumerate(selected, start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/apply a GT-free rendered-CIF geometry compatibility selector.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["gbdt", "mlp", "sgd"], required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--atom12-weight", type=float, default=1.25)
    parser.add_argument("--candidate-row7-weight", type=float, default=1.25)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows_all = read_jsonl(args.train_features)
    val_rows_all = read_jsonl(args.val_features)
    train_rows = stratified_prefix_rows(train_rows_all, int(args.max_train_rows))
    val_rows = stratified_prefix_rows(val_rows_all, int(args.max_val_rows))
    print(
        f"[compat] train rows {len(train_rows)}/{len(train_rows_all)}; val rows {len(val_rows)}/{len(val_rows_all)}",
        flush=True,
    )
    x_train = [feature_dict(row) for row in train_rows]
    y_train = [int(bool(row["label_match"])) for row in train_rows]
    sample_weight = []
    for row, y in zip(train_rows, y_train):
        weight = float(args.positive_weight) if y else 1.0
        if int(row.get("atom_count", 0)) >= 12:
            weight *= float(args.atom12_weight)
        if int(row.get("candidate_row_count", 0)) >= 7:
            weight *= float(args.candidate_row7_weight)
        sample_weight.append(weight)

    if args.model_type == "gbdt":
        estimator = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=16,
            l2_regularization=0.05,
            random_state=20260614,
        )
        model: Pipeline = Pipeline([("vec", DictVectorizer(sparse=False)), ("clf", estimator)])
        model.fit(x_train, y_train, clf__sample_weight=sample_weight)
    else:
        if args.model_type == "sgd":
            estimator = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=2e-5,
                l1_ratio=0.05,
                max_iter=2000,
                tol=1e-4,
                class_weight=None,
                random_state=20260614,
            )
            model = Pipeline([("vec", DictVectorizer(sparse=True)), ("clf", estimator)])
            model.fit(x_train, y_train, clf__sample_weight=sample_weight)
        else:
            estimator = MLPClassifier(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                alpha=2e-4,
                learning_rate_init=8e-4,
                max_iter=350,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=20260614,
            )
            model = Pipeline([("vec", DictVectorizer(sparse=False)), ("scale", StandardScaler()), ("clf", estimator)])
            model.fit(x_train, y_train)

    scored_val = score_rows(model, val_rows)
    ranked_val = rerank(scored_val, top_k=int(args.top_k))
    baseline_val = baseline_by_current_rank(val_rows, top_k=int(args.top_k))
    y_val = [int(bool(row["label_match"])) for row in val_rows]
    val_scores = [float(row["compat_score"]) for row in scored_val]
    clf_metrics: dict[str, Any] = {
        "train_rows": len(train_rows),
        "train_rows_available": len(train_rows_all),
        "train_positive_rate": sum(y_train) / max(1, len(y_train)),
        "val_rows": len(val_rows),
        "val_rows_available": len(val_rows_all),
        "val_positive_rate": sum(y_val) / max(1, len(y_val)),
        "max_train_rows": int(args.max_train_rows),
        "max_val_rows": int(args.max_val_rows),
    }
    if len(set(y_val)) > 1:
        clf_metrics["val_roc_auc"] = float(roc_auc_score(y_val, val_scores))
        clf_metrics["val_average_precision"] = float(average_precision_score(y_val, val_scores))

    model_metrics = metrics_from_ranked(ranked_val)
    baseline_metrics = metrics_from_ranked(baseline_val)
    write_jsonl(out_dir / "val_scored_candidates.jsonl", scored_val)
    write_jsonl(out_dir / "val_reranked_features.jsonl", ranked_val)
    rendered_rows = []
    for row in ranked_val:
        rendered = {k: v for k, v in row.items() if not k.startswith("label_")}
        rendered_rows.append(rendered)
    write_jsonl(out_dir / "rendered_topk_compat.jsonl", rendered_rows)
    joblib.dump(model, out_dir / "compat_model.joblib")
    summary = {
        "model_type": str(args.model_type),
        "feature_policy": {
            "blocked": sorted(BLOCKED_FEATURE_KEYS),
            "numeric": sorted(NUMERIC_KEYS),
            "boolean": sorted(BOOLEAN_KEYS),
            "categorical": sorted(CATEGORICAL_KEYS),
            "note": "target row_count/complex flags and canonical target keys are never used as model features; candidate row count is derived from predicted W/A.",
        },
        "fit": clf_metrics,
        "baseline_current_order": {k: v for k, v in baseline_metrics.items() if k != "per_sample"},
        "compat_reranked": {k: v for k, v in model_metrics.items() if k != "per_sample"},
    }
    write_json(out_dir / "compat_selector_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
