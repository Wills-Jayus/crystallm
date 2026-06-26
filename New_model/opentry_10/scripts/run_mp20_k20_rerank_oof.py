#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "features/mp20_val_k50_candidate_features.jsonl"
DEFAULT_LABELS = ROOT / "labels/mp20_val_k50_candidate_labels.jsonl"
DEFAULT_TRUE_TAR = ROOT / "generations/crystallm_gt_sg_val_anchor_symprec0p1/mp20_val_data_atomtype_gt_sg_symprec0p1_k100/tars/true.tar.gz"


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def atom_site_rows(cif: str) -> int:
    in_atom_loop = False
    headers_seen = False
    count = 0
    for line in cif.splitlines():
        stripped = line.strip()
        if stripped == "loop_":
            if in_atom_loop and headers_seen:
                break
            in_atom_loop = False
            headers_seen = False
            continue
        if stripped.startswith("_atom_site_"):
            in_atom_loop = True
            headers_seen = True
            continue
        if in_atom_loop and headers_seen:
            if not stripped or stripped.startswith("_") or stripped.startswith("#"):
                break
            count += 1
    return count


def read_target_rows(true_tar: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    with tarfile.open(true_tar, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".cif"):
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            cif = handle.read().decode("utf-8", errors="replace")
            rows[Path(member.name).stem] = atom_site_rows(cif)
    return rows


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "cell_length_a",
        "cell_length_b",
        "cell_length_c",
        "cell_angle_alpha",
        "cell_angle_beta",
        "cell_angle_gamma",
        "cell_volume",
        "atom_site_rows",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    lengths = ["cell_length_a", "cell_length_b", "cell_length_c"]
    if all(col in df.columns for col in lengths):
        max_len = df[lengths].max(axis=1)
        min_len = df[lengths].min(axis=1)
        df["lattice_length_ratio"] = max_len / min_len.replace(0, np.nan)
        df["lattice_length_mean"] = df[lengths].mean(axis=1)
    angles = ["cell_angle_alpha", "cell_angle_beta", "cell_angle_gamma"]
    if all(col in df.columns for col in angles):
        df["angle_abs_deviation_sum"] = (df[angles] - 90.0).abs().sum(axis=1)
    if "cell_volume" in df.columns and "atom_site_rows" in df.columns:
        df["candidate_volume_per_atom"] = df["cell_volume"] / df["atom_site_rows"].replace(0, np.nan)
    df["rank_inverse"] = 1.0 / pd.to_numeric(df["rank"], errors="coerce").replace(0, np.nan)
    return df


def load_table(features_path: Path, labels_path: Path, true_tar: Path, *, max_rank: int, allow_partial: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = read_jsonl(features_path)
    labels = read_jsonl(labels_path)
    label_cols = [
        "sample_id",
        "material_id",
        "rank",
        "match",
        "rmsd",
        "label_status",
        "parse_ok",
        "valid",
        "sensible",
    ]
    labels = labels[[col for col in label_cols if col in labels.columns]]
    df = features.merge(labels, on=["sample_id", "material_id", "rank"], how="inner", validate="one_to_one")
    df = df[pd.to_numeric(df["rank"], errors="coerce").between(1, int(max_rank))].copy()
    df = add_derived_features(df)

    target_rows = read_target_rows(true_tar)
    df["target_atom_site_rows"] = df["material_id"].map(target_rows)
    df["target_rows_ge7"] = df["target_atom_site_rows"].fillna(0).astype(int) >= 7
    df["match"] = df["match"].fillna(False).astype(bool)
    df["rmsd"] = pd.to_numeric(df["rmsd"], errors="coerce")

    expected_materials = len(target_rows)
    expected_rows = expected_materials * int(max_rank)
    counts = df.groupby("sample_id")["rank"].nunique()
    complete_samples = sorted(counts[counts == int(max_rank)].index.astype(str))
    incomplete_samples = sorted(counts[counts != int(max_rank)].index.astype(str))
    missing_samples = sorted(set(str(k) for k in target_rows) - set(df["material_id"].astype(str)))
    coverage_complete = len(df) == expected_rows and len(complete_samples) == expected_materials
    if not coverage_complete and not allow_partial:
        raise SystemExit(
            f"label/feature coverage incomplete for K{max_rank}: rows={len(df)} expected={expected_rows}; "
            "rerun with --allow-partial only for sanity diagnostics"
        )

    df = df[df["sample_id"].astype(str).isin(complete_samples)].copy()
    summary = {
        "features": str(features_path.resolve()),
        "labels": str(labels_path.resolve()),
        "true_tar": str(true_tar.resolve()),
        "input_rows_after_join": int(len(df)),
        "expected_rows": int(expected_rows),
        "target_materials": int(expected_materials),
        "complete_samples": int(len(complete_samples)),
        "incomplete_samples": int(len(incomplete_samples)),
        "missing_samples": int(len(missing_samples)),
        "first_incomplete_samples": incomplete_samples[:20],
        "first_missing_material_ids": missing_samples[:20],
        "coverage_complete": bool(coverage_complete),
    }
    return df, summary


def metric_rows(df: pd.DataFrame, score_col: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        if score_col is None:
            ordered = group.sort_values(["rank"], ascending=[True])
        else:
            ordered = group.sort_values([score_col, "rank"], ascending=[False, True])
        material_id = str(ordered["material_id"].iloc[0])
        rows_ge7 = bool(ordered["target_rows_ge7"].iloc[0])
        record: dict[str, Any] = {
            "sample_id": str(sample_id),
            "material_id": material_id,
            "rows_ge7": rows_ge7,
        }
        for budget in [1, 5, 20]:
            top = ordered.head(budget)
            matched = top[top["match"]]
            record[f"hit@{budget}"] = bool(len(matched) > 0)
            if len(matched) > 0:
                record[f"rmsd@{budget}"] = float(matched["rmsd"].min())
            else:
                record[f"rmsd@{budget}"] = None
        rows.append(record)
    return rows


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows7 = [r for r in rows if r["rows_ge7"]]
    out: dict[str, Any] = {"samples": len(rows), "rows_ge7_samples": len(rows7)}
    for budget in [1, 5, 20]:
        hits = [r for r in rows if r[f"hit@{budget}"]]
        rms = [float(r[f"rmsd@{budget}"]) for r in hits if r[f"rmsd@{budget}"] is not None]
        hits7 = [r for r in rows7 if r[f"hit@{budget}"]]
        rms7 = [float(r[f"rmsd@{budget}"]) for r in hits7 if r[f"rmsd@{budget}"] is not None]
        out[f"match@{budget}"] = None if not rows else float(len(hits) / len(rows))
        out[f"rmsd@{budget}"] = None if not rms else float(np.mean(rms))
        out[f"rows>=7_match@{budget}"] = None if not rows7 else float(len(hits7) / len(rows7))
        out[f"rows>=7_rmsd@{budget}"] = None if not rms7 else float(np.mean(rms7))
    return out


def bootstrap_delta(base_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], *, n: int, seed: int) -> dict[str, Any]:
    if n <= 0:
        return {}
    base_by_id = {r["sample_id"]: r for r in base_rows}
    pred_by_id = {r["sample_id"]: r for r in pred_rows}
    ids = sorted(set(base_by_id) & set(pred_by_id))
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for budget in [1, 5, 20]:
        deltas = []
        for _ in range(int(n)):
            sample = rng.choice(ids, size=len(ids), replace=True)
            base_hits = np.array([bool(base_by_id[i][f"hit@{budget}"]) for i in sample], dtype=float)
            pred_hits = np.array([bool(pred_by_id[i][f"hit@{budget}"]) for i in sample], dtype=float)
            deltas.append(float(pred_hits.mean() - base_hits.mean()))
        lo, hi = np.quantile(deltas, [0.025, 0.975])
        out[f"match@{budget}_delta_ci95"] = [float(lo), float(hi)]
    return out


def make_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_candidates = [
        "rank",
        "gen_index",
        "generation_config_rank_start",
        "temperature",
        "top_k",
        "seed",
        "normalized_token_logprob",
        "cif_num_chars",
        "cif_num_lines",
        "atom_site_rows",
        "declared_sg_number",
        "cell_length_a",
        "cell_length_b",
        "cell_length_c",
        "cell_angle_alpha",
        "cell_angle_beta",
        "cell_angle_gamma",
        "cell_volume",
        "cell_formula_units_Z",
        "lattice_length_ratio",
        "lattice_length_mean",
        "angle_abs_deviation_sum",
        "candidate_volume_per_atom",
        "rank_inverse",
    ]
    categorical_candidates = [
        "rank_source",
        "generation_config_name",
        "logprob_available",
        "declared_sg_symbol",
        "label_status",
        "parse_ok",
        "valid",
        "sensible",
    ]
    numeric = [col for col in numeric_candidates if col in df.columns]
    categorical = [col for col in categorical_candidates if col in df.columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric, categorical


@dataclass(frozen=True)
class ModelSpec:
    name: str
    seed: int
    estimator: Any


def model_specs(names: list[str], seeds: list[int]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name in names:
        for seed in seeds:
            if name == "logistic":
                estimator = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced", random_state=seed)
            elif name == "hgb":
                estimator = HistGradientBoostingClassifier(
                    max_iter=160,
                    learning_rate=0.05,
                    l2_regularization=0.05,
                    random_state=seed,
                )
            elif name == "rf":
                estimator = RandomForestClassifier(
                    n_estimators=240,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed,
                )
            else:
                raise ValueError(f"unknown model: {name}")
            specs.append(ModelSpec(name=name, seed=seed, estimator=estimator))
    return specs


def run_oof(df: pd.DataFrame, spec: ModelSpec, n_splits: int) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    preprocessor, numeric, categorical = make_preprocessor(df)
    pipe = Pipeline([("features", preprocessor), ("model", spec.estimator)])
    groups = df["sample_id"].astype(str).to_numpy()
    y = df["match"].astype(int).to_numpy()
    X = df[numeric + categorical].copy()

    pred = df[["sample_id", "material_id", "rank", "match", "rmsd", "target_rows_ge7"]].copy()
    pred["score"] = np.nan
    pred["fold"] = -1

    fold_summaries: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=int(n_splits))
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups=groups)):
        model = pipe
        model.fit(X.iloc[train_idx], y[train_idx])
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X.iloc[valid_idx])[:, 1]
        else:
            raw = model.decision_function(X.iloc[valid_idx])
            score = 1.0 / (1.0 + np.exp(-raw))
        pred.loc[pred.index[valid_idx], "score"] = score
        pred.loc[pred.index[valid_idx], "fold"] = fold
        train_groups = len(set(groups[train_idx]))
        valid_groups = len(set(groups[valid_idx]))
        fold_summaries.append(
            {
                "fold": fold,
                "train_samples": int(train_groups),
                "valid_samples": int(valid_groups),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "train_positive_rate": float(y[train_idx].mean()),
                "valid_positive_rate": float(y[valid_idx].mean()),
            }
        )
    return pred, fold_summaries, {"numeric_features": numeric, "categorical_features": categorical}


def write_predictions(path: Path, pred: pd.DataFrame, model_name: str, seed: int) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = pred.copy()
    rows["model"] = model_name
    rows["seed"] = int(seed)
    rows.to_json(path, orient="records", lines=True)


def make_report(result: dict[str, Any]) -> str:
    dataset_label = result.get("dataset_label", "MP-20")
    lines = [
        f"# {dataset_label} K20 Rerank-Only OOF Search",
        "",
        f"Created: {result['created_at']}",
        "",
        "This search only reorders the original CrystaLLM GT-SG K20 candidates. The candidate set is unchanged, so match@20 should remain unchanged apart from label incompleteness diagnostics.",
        "",
        "## Input",
        "",
        f"- Complete samples used: {result['input_summary']['complete_samples']}",
        f"- Coverage complete: {result['input_summary']['coverage_complete']}",
        f"- Allow partial: {result['allow_partial']}",
        "",
        "## Baseline",
        "",
        "| metric | value | rows>=7 |",
        "| --- | ---: | ---: |",
    ]
    baseline = result["baseline_metrics"]
    for budget in [1, 5, 20]:
        lines.append(
            f"| match@{budget} | {100.0 * baseline[f'match@{budget}']:.3f}% | "
            f"{100.0 * baseline[f'rows>=7_match@{budget}']:.3f}% |"
        )
    lines.extend(["", "## Models", "", "| model | seed | match@1 | delta@1 | match@5 | delta@5 | match@20 | delta@20 | CI95 delta@1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in result["model_results"]:
        metrics = row["metrics"]
        deltas = row["deltas"]
        ci = row.get("bootstrap", {}).get("match@1_delta_ci95")
        ci_text = "NA" if ci is None else f"[{100.0 * ci[0]:.3f}, {100.0 * ci[1]:.3f}] pp"
        lines.append(
            f"| {row['model']} | {row['seed']} | "
            f"{100.0 * metrics['match@1']:.3f}% | {100.0 * deltas['match@1']:.3f} pp | "
            f"{100.0 * metrics['match@5']:.3f}% | {100.0 * deltas['match@5']:.3f} pp | "
            f"{100.0 * metrics['match@20']:.3f}% | {100.0 * deltas['match@20']:.3f} pp | {ci_text} |"
        )
    best = result.get("best_by_match1_delta")
    if best is not None:
        lines.extend(
            [
                "",
                "## Current Best",
                "",
                f"- Model: {best['model']} seed={best['seed']}",
                f"- match@1 delta: {100.0 * best['deltas']['match@1']:.3f} pp",
                f"- match@5 delta: {100.0 * best['deltas']['match@5']:.3f} pp",
                f"- match@20 delta: {100.0 * best['deltas']['match@20']:.3f} pp",
            ]
        )
    lines.extend(
        [
            "",
            "Formal freezing requires full label coverage and a selected hyperparameter set before official test generation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation OOF rerank-only search over original K20 candidates.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--true-tar", default=str(DEFAULT_TRUE_TAR))
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", default="logistic,hgb,rf")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "metrics/rerank_oof_results.json"))
    parser.add_argument("--report", default=str(ROOT / "reports/rerank_model_search.md"))
    parser.add_argument("--predictions-dir", default=str(ROOT / "features/rerank_oof_predictions"))
    parser.add_argument("--prediction-prefix", default="mp20_val_k20")
    parser.add_argument("--dataset-label", default="MP-20")
    parser.add_argument("--task-name", default="mp20_validation_k20_rerank_only_oof")
    args = parser.parse_args()

    names = [name.strip() for name in args.models.split(",") if name.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    df, input_summary = load_table(
        Path(args.features),
        Path(args.labels),
        Path(args.true_tar),
        max_rank=int(args.max_rank),
        allow_partial=bool(args.allow_partial),
    )
    if df["sample_id"].nunique() < int(args.folds):
        raise SystemExit(f"not enough complete samples for {args.folds} folds")

    baseline_rows = metric_rows(df, score_col=None)
    baseline_metrics = summarize_metrics(baseline_rows)

    model_results: list[dict[str, Any]] = []
    pred_dir = under_root(Path(args.predictions_dir))
    pred_dir.mkdir(parents=True, exist_ok=True)
    for spec in model_specs(names, seeds):
        pred, folds, feature_info = run_oof(df, spec, int(args.folds))
        pred_rows = metric_rows(pred, score_col="score")
        metrics = summarize_metrics(pred_rows)
        deltas = {
            key: (metrics[key] - baseline_metrics[key]) if metrics.get(key) is not None and baseline_metrics.get(key) is not None else None
            for key in [
                "match@1",
                "match@5",
                "match@20",
                "rows>=7_match@1",
                "rows>=7_match@5",
                "rows>=7_match@20",
            ]
        }
        boot = bootstrap_delta(baseline_rows, pred_rows, n=int(args.bootstrap), seed=1000 + spec.seed)
        pred_path = pred_dir / f"{args.prediction_prefix}_{spec.name}_seed{spec.seed}_oof_predictions.jsonl"
        write_predictions(pred_path, pred, spec.name, spec.seed)
        model_results.append(
            {
                "model": spec.name,
                "seed": int(spec.seed),
                "metrics": metrics,
                "deltas": deltas,
                "bootstrap": boot,
                "folds": folds,
                "feature_info": feature_info,
                "predictions": str(pred_path.resolve()),
            }
        )

    best = max(model_results, key=lambda r: (r["deltas"].get("match@1") or -999.0, r["deltas"].get("match@5") or -999.0))
    result = {
        "created_at": now_iso(),
        "task": str(args.task_name),
        "dataset_label": str(args.dataset_label),
        "allow_partial": bool(args.allow_partial),
        "input_summary": input_summary,
        "baseline_metrics": baseline_metrics,
        "model_results": model_results,
        "best_by_match1_delta": best,
        "note": "OOF groups are sample_id. This script reorders only the original K20 candidate set.",
    }
    write_json(Path(args.out), result)
    write_text(Path(args.report), make_report(result))


if __name__ == "__main__":
    main()
