#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_11"
RESULTS = OUT_DIR / "results"
REPORT = OUT_DIR / "GPT_REVIEW_BUNDLE.md"
TRACK_A = ROOT / "model/std_way/track_a_mpts52"
STRUCT_FEATURES = TRACK_A / "outputs/structural_features.jsonl.gz"
OOF_SCORES = TRACK_A / "outputs/oof_scores.jsonl"

BUDGETS = (1, 5, 20)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_out(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_11: {resolved}")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path = ensure_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_report(title: str, body: str) -> None:
    with ensure_out(REPORT).open("a", encoding="utf-8") as f:
        f.write("\n\n" + f"## opentry_11 自主迭代实验：{title}\n\n" + body.strip() + "\n")


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_json(STRUCT_FEATURES, lines=True, compression="gzip")
    scores = pd.read_json(OOF_SCORES, lines=True)
    score_cols = [c for c in scores.columns if c.startswith("score_")]
    df = df.merge(scores[["sample_id", "rank", *score_cols]], on=["sample_id", "rank"], how="left")
    df["target_rows_ge7"] = df["target_rows_ge7"].astype(bool)
    df["match"] = df["match"].fillna(False).astype(bool)
    df["rank"] = df["rank"].astype(int)
    complex_proxy = df[df["rank"] <= 20].groupby("sample_id")["candidate_rows_ge7"].max().rename("complex_proxy")
    df = df.merge(complex_proxy, on="sample_id", how="left")
    df["complex_proxy"] = df["complex_proxy"].fillna(0).astype(float)
    df["skeleton_hit"] = (
        (df["formula_reduced_match"] > 0.5)
        & (df["sg_number_match"] > 0.5)
        & (df["multiplicity_matches_prompt"] > 0.5)
        & (df["orbit_feasible"] > 0.5)
    ).astype(float)
    df["formula_sg_ok"] = ((df["formula_reduced_match"] > 0.5) & (df["sg_number_match"] > 0.5)).astype(float)
    return df, score_cols


def safe_base_features(df: pd.DataFrame) -> list[str]:
    allowed = [
        "rank",
        "rank_inv",
        "rank_le5",
        "rank_le20",
        "rank_le30",
        "rank_tail31_50",
        "gen_index",
        "temperature",
        "top_k",
        "cif_num_chars",
        "cif_num_lines",
        "atom_site_rows",
        "candidate_rows_ge7",
        "candidate_rows_ge10",
        "declared_sg_number",
        "prompt_sg_number",
        "sg_number_match",
        "sg_symbol_match",
        "sg_number_abs_diff",
        "formula_reduced_match",
        "formula_element_jaccard",
        "formula_l1_prompt",
        "candidate_formula_atoms",
        "prompt_formula_atoms",
        "candidate_prompt_atom_ratio",
        "multiplicity_total",
        "multiplicity_prompt_ratio",
        "multiplicity_formula_ratio",
        "exact_cover_l1_prompt",
        "exact_cover_l1_formula",
        "multiplicity_matches_prompt",
        "multiplicity_matches_formula",
        "orbit_feasible",
        "noninteger_multiplicity_rows",
        "bad_occupancy_rows",
        "bad_coord_rows",
        "multiplicity_min",
        "multiplicity_max",
        "multiplicity_entropy",
        "cell_length_a",
        "cell_length_b",
        "cell_length_c",
        "cell_angle_alpha",
        "cell_angle_beta",
        "cell_angle_gamma",
        "cell_volume",
        "cell_formula_units_Z",
        "cell_valid",
        "length_ratio",
        "length_mean",
        "angle_abs_deviation_sum",
        "volume_per_mult_atom",
        "volume_per_prompt_atom",
        "rows7_hard_bucket",
        "rows7_rank_interaction",
        "geom_parseable",
        "min_pair_dist",
        "nn_dist_mean",
        "nn_dist_p10",
        "nn_radius_ratio_mean",
        "min_radius_ratio",
        "collision_lt_0p5",
        "radius_collision_lt_0p6",
        "close_pair_frac",
        "structural_tail_ok",
        "rows7_tail_ok",
        "complex_proxy",
        "skeleton_hit",
        "formula_sg_ok",
    ]
    return [c for c in allowed if c in df.columns]


def selected_diagnostics(selected: pd.DataFrame) -> dict[str, Any]:
    skel = selected[selected["skeleton_hit"] > 0.5]
    return {
        "valid_rate": float(selected["valid_label"].mean()),
        "formula_consistency": float(selected["formula_reduced_match"].mean()),
        "sg_consistency": float(selected["sg_number_match"].mean()),
        "exact_cover_feasible_rate": float((selected["skeleton_hit"] > 0.5).mean()),
        "skeleton_hit_to_match_conversion": None if len(skel) == 0 else float(skel["match"].mean()),
    }


def sample_stats(ordered: pd.DataFrame) -> dict[str, Any]:
    out = {"sample_id": str(ordered["sample_id"].iloc[0]), "target_rows_ge7": bool(ordered["target_rows_ge7"].iloc[0])}
    for b in BUDGETS:
        out[f"hit@{b}"] = bool(ordered.head(b)["match"].any())
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows7 = [r for r in rows if r["target_rows_ge7"]]
    out: dict[str, Any] = {"samples": len(rows), "rows>=7_samples": len(rows7)}
    for b in BUDGETS:
        out[f"match@{b}"] = float(sum(1 for r in rows if r[f"hit@{b}"]) / max(1, len(rows)))
        out[f"rows>=7_match@{b}"] = float(sum(1 for r in rows7 if r[f"hit@{b}"]) / max(1, len(rows7)))
    return out


def evaluate(df: pd.DataFrame, selector: Callable[[pd.DataFrame], pd.DataFrame], name: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for _, group in df.groupby("sample_id", sort=False):
        ordered = selector(group).head(20).copy()
        rows.append(sample_stats(ordered))
        frames.append(ordered)
    selected = pd.concat(frames, ignore_index=True)
    return {"name": name, "metrics": summarize(rows), "diagnostics": selected_diagnostics(selected), "sample_rows": rows}


def train_hgb_oof(df: pd.DataFrame, cols: list[str], score_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    df[score_col] = np.nan
    fold_meta: list[dict[str, Any]] = []
    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold]
        val = df[df["fold"] == fold]
        y = train["match"].astype(int)
        # Labels are valid for supervised training; no label-derived feature is allowed.
        weights = np.ones(len(train), dtype="float32")
        weights += y.to_numpy(dtype="float32") * 4.0
        weights += ((~train["match"].to_numpy(dtype=bool)) & (train["rank"].to_numpy() <= 20)).astype("float32")
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=1e-3,
                early_stopping=True,
                random_state=20260629 + int(fold),
            ),
        )
        clf.fit(train[cols].astype("float32"), y, histgradientboostingclassifier__sample_weight=weights)
        df.loc[val.index, score_col] = clf.predict_proba(val[cols].astype("float32"))[:, 1]
        fold_meta.append({"fold": int(fold), "train_candidates": int(len(train)), "val_candidates": int(len(val)), "train_positive": int(y.sum())})
        print(f"[safe-hgb] {score_col} fold {fold} done", flush=True)
    return df, {"feature_count": len(cols), "features": cols, "fold_meta": fold_meta}


def metric_delta(metrics: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    keys = [f"match@{b}" for b in BUDGETS] + [f"rows>=7_match@{b}" for b in BUDGETS]
    return {k: float(metrics[k] - base[k]) for k in keys}


def achieved(delta: dict[str, float]) -> bool:
    return sum(1 for b in BUDGETS if delta[f"match@{b}"] >= 0.05) >= 2 or (
        delta["rows>=7_match@5"] >= 0.05 and delta["rows>=7_match@20"] >= 0.05
    )


def fmt_metrics(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"match@{b}"]) for b in BUDGETS)


def fmt_rows7(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"rows>=7_match@{b}"]) for b in BUDGETS)


def fmt_delta(d: dict[str, float], prefix: str = "") -> str:
    return " / ".join(pp(d[f"{prefix}match@{b}"]) for b in BUDGETS)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df, score_cols = load_data()
    baseline = evaluate(df, lambda g: g[g["rank"] <= 20].sort_values("rank"), "baseline")

    base_cols = safe_base_features(df)
    df, meta_base = train_hgb_oof(df, base_cols, "iter02_pure_structural_score")
    stacked_cols = base_cols + [c for c in score_cols if c in df.columns]
    df, meta_stacked = train_hgb_oof(df, stacked_cols, "iter02_stacked_oof_score")

    pure = evaluate(df, lambda g: g.sort_values(["iter02_pure_structural_score", "rank"], ascending=[False, True]), "pure_structural_hgb")
    stacked = evaluate(df, lambda g: g.sort_values(["iter02_stacked_oof_score", "rank"], ascending=[False, True]), "stacked_oof_hgb")
    pure_delta = metric_delta(pure["metrics"], baseline["metrics"])
    stacked_delta = metric_delta(stacked["metrics"], baseline["metrics"])
    stop = achieved(pure_delta) or achieved(stacked_delta)
    result = {
        "time": now_iso(),
        "gpu_necessary": False,
        "gpu_reason": "validation OOF sklearn HGB; no MP-20/MPTS-52 train dataset training",
        "leakage_controls": {
            "excluded": ["match", "rmsd", "label_status", "target_rows_ge7", "parse_ok_label", "skeleton_ok_geometry_wrong", "formula_sg_geometry_wrong", "collision_free_wrong"],
            "pure_structural_uses_score_cols": False,
            "stacked_uses_oof_score_cols": True,
        },
        "data": {
            "samples": int(df["sample_id"].nunique()),
            "candidates": int(len(df)),
            "rows>=7_samples": int(df[["sample_id", "target_rows_ge7"]].drop_duplicates()["target_rows_ge7"].sum()),
        },
        "model_meta": {"pure": meta_base, "stacked": meta_stacked},
        "policies": {
            "baseline": {"metrics": baseline["metrics"], "diagnostics": baseline["diagnostics"]},
            "pure_structural_hgb": {"metrics": pure["metrics"], "diagnostics": pure["diagnostics"]},
            "stacked_oof_hgb": {"metrics": stacked["metrics"], "diagnostics": stacked["diagnostics"]},
        },
        "deltas_vs_baseline": {"pure_structural_hgb": pure_delta, "stacked_oof_hgb": stacked_delta},
        "achieved_stop_threshold": bool(stop),
    }
    write_json(RESULTS / "iteration_02_safe_hgb_search.json", result)

    append_report(
        "迭代 02 inference-safe HGB scorer",
        f"""
时间：{result['time']}

实验逻辑：迭代 01 因 match-derived 特征泄漏作废。本轮重跑 inference-safe HGB：`pure_structural_hgb` 只用推理期可从 prompt/candidate CIF 计算的结构特征；`stacked_oof_hgb` 额外使用已有 Track A OOF score 作为 stacking 信号。两者都排除 match、rmsd、target_rows_ge7、错误类型标签和任何由 StructureMatcher 结果派生的输入。

GPU 必要性判断：本轮没有采用 MP-20/MPTS-52 train 数据集训练模型，只在 MPTS-52 validation OOF 上训练 sklearn HGB；因此不需要 GPU。

核心假设：如果安全结构特征已经足够，pure 或 stacked HGB 应在至少两个 match 指标上超过 baseline +5pp；如果 pure 不够而 stacked 够，说明 Track A OOF 信号有互补但需要 train-data 级别重建；如果都不够，继续排序意义有限。

数据规模：MPTS-52 validation K50 全量 {result['data']['samples']} samples / {result['data']['candidates']} candidates；rows>=7 samples={result['data']['rows>=7_samples']}。pure 特征数={meta_base['feature_count']}；stacked 特征数={meta_stacked['feature_count']}。

baseline：{fmt_metrics(baseline['metrics'])}；rows>=7 = {fmt_rows7(baseline['metrics'])}。

方法变化：5-fold OOF HistGradientBoosting；训练权重只用于 supervised objective，输入特征不包含 match-derived 字段。pure 不使用 `score_*`，stacked 使用 `score_*` 但这些 score 是既有 OOF stacking 信号。

结果 pure_structural_hgb：{fmt_metrics(pure['metrics'])}；delta = {fmt_delta(pure_delta)}。rows>=7 = {fmt_rows7(pure['metrics'])}；rows>=7 delta = {fmt_delta(pure_delta, 'rows>=7_')}。

结果 stacked_oof_hgb：{fmt_metrics(stacked['metrics'])}；delta = {fmt_delta(stacked_delta)}。rows>=7 = {fmt_rows7(stacked['metrics'])}；rows>=7 delta = {fmt_delta(stacked_delta, 'rows>=7_')}。

可信度：比迭代 01 高，因为移除了显式泄漏；但仍是 validation OOF，不是 official full-test，也不是 train-data 级别可部署模型。

和历史实验关系：这是对 Track A/hard-negative scorer 路线的最后一次安全强排序检验。

最终判决：achieved_stop_threshold={stop}。{"达到 +5pp 停止阈值，但只能作为 validation OOF 成功，后续若写论文需 train-data 重建并 official freeze。" if stop else "未达到 +5pp 停止阈值，不能停止，需要继续下一轮非排序方案。"}

下一步：若未达标，转向 train-data 级别 learned geometry repair / skeleton proposer，而不是继续 validation rerank。
""",
    )
    print(json.dumps({"achieved_stop_threshold": stop, "pure_delta": pure_delta, "stacked_delta": stacked_delta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
