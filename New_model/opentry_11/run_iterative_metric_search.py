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
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_11"
RESULTS = OUT_DIR / "results"
REPORT = OUT_DIR / "GPT_REVIEW_BUNDLE.md"
TRACK_A = ROOT / "model/std_way/track_a_mpts52"
STRUCT_FEATURES = TRACK_A / "outputs/structural_features.jsonl.gz"
OOF_SCORES = TRACK_A / "outputs/oof_scores.jsonl"

BUDGETS = (1, 5, 20)
MAX_RANK = 50


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


def load_data() -> pd.DataFrame:
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
    )
    df["formula_sg_ok"] = (df["formula_reduced_match"] > 0.5) & (df["sg_number_match"] > 0.5)
    df["skeleton_ok_geometry_wrong"] = df["skeleton_hit"] & (~df["match"])
    df["formula_sg_geometry_wrong"] = df["formula_sg_ok"] & (~df["match"])
    df["collision_free_wrong"] = (df["radius_collision_lt_0p6"] <= 0) & (~df["match"])
    return df


def sample_stats(ordered: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_id": str(ordered["sample_id"].iloc[0]), "target_rows_ge7": bool(ordered["target_rows_ge7"].iloc[0])}
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


def selected_diagnostics(selected: pd.DataFrame) -> dict[str, Any]:
    skel = selected[selected["skeleton_hit"]]
    return {
        "valid_rate": float(selected["valid_label"].mean()),
        "formula_consistency": float(selected["formula_reduced_match"].mean()),
        "sg_consistency": float(selected["sg_number_match"].mean()),
        "exact_cover_feasible_rate": float(selected["skeleton_hit"].mean()),
        "skeleton_hit_to_match_conversion": None if len(skel) == 0 else float(skel["match"].mean()),
    }


def evaluate_policy(df: pd.DataFrame, selector: Callable[[pd.DataFrame], pd.DataFrame], name: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for _, group in df.groupby("sample_id", sort=False):
        ordered = selector(group).head(20).copy()
        rows.append(sample_stats(ordered))
        frames.append(ordered)
    selected = pd.concat(frames, ignore_index=True)
    return {"name": name, "metrics": summarize(rows), "diagnostics": selected_diagnostics(selected), "sample_rows": rows}


def selected_order_track_a(group: pd.DataFrame, score_col: str) -> pd.DataFrame:
    g = group.copy()
    g["_score"] = pd.to_numeric(g[score_col], errors="coerce").fillna(-1e9)
    selected: list[int] = []
    used: set[int] = set()

    def add(rank: int) -> None:
        if rank not in used and len(selected) < 20:
            selected.append(rank)
            used.add(rank)

    k30 = g[g["rank"] <= 30].sort_values(["_score", "rank"], ascending=[False, True])
    for _, row in k30.iterrows():
        add(int(row["rank"]))
        if len(selected) >= 5:
            break
    rest = g[~g["rank"].isin(used)].copy()
    rest = rest[(rest["rank"] <= 30) | (rest["structural_tail_ok"] > 0)]
    if bool(g["complex_proxy"].iloc[0] > 0):
        rest = rest[(rest["rank"] <= 30) | (rest["rows7_tail_ok"] > 0)]
        rest["_score"] = rest["_score"] - 0.25 * (1.0 - rest["rows7_hard_bucket"])
    rest = rest.sort_values(["_score", "rank"], ascending=[False, True])
    for _, row in rest.iterrows():
        add(int(row["rank"]))
        if len(selected) >= 20:
            break
    for rank in range(1, MAX_RANK + 1):
        add(rank)
        if len(selected) >= 20:
            break
    return g.set_index("rank", drop=False).loc[selected].reset_index(drop=True)


def selected_order_exact_cover(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    g["_score"] = 100.0 * g["skeleton_hit"].astype(float)
    g["_score"] += 10.0 * (g["formula_reduced_match"] > 0.5).astype(float)
    g["_score"] += 5.0 * (g["sg_number_match"] > 0.5).astype(float)
    g["_score"] += 2.0 * (g["radius_collision_lt_0p6"] <= 0).astype(float)
    g["_score"] += g["rank_inv"].astype(float)
    return g.sort_values(["_score", "rank"], ascending=[False, True]).head(20)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "sample_id",
        "material_id",
        "fold",
        "match",
        "rmsd",
        "label_status",
        "target_rows_ge7",
        "parse_ok_label",
    }
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def hard_weights(df: pd.DataFrame) -> np.ndarray:
    neg = ~df["match"].to_numpy(dtype=bool)
    rank = df["rank"].to_numpy()
    w = np.ones(len(df), dtype="float32")
    w += ((neg) & (rank <= 20)).astype("float32") * 1.0
    w += ((neg) & df["formula_sg_geometry_wrong"].to_numpy(dtype=bool)).astype("float32") * 1.0
    w += ((neg) & df["skeleton_ok_geometry_wrong"].to_numpy(dtype=bool)).astype("float32") * 1.5
    w += df["match"].to_numpy(dtype=bool).astype("float32") * 4.0
    return w


def train_hgb_oof(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    df["iter_hgb_score"] = np.nan
    fold_meta: list[dict[str, Any]] = []
    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold]
        val = df[df["fold"] == fold]
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=1e-3,
                early_stopping=True,
                random_state=20260628 + int(fold),
            ),
        )
        x_train = train[cols].astype("float32")
        y_train = train["match"].astype(int)
        clf.fit(x_train, y_train, histgradientboostingclassifier__sample_weight=hard_weights(train))
        pred = clf.predict_proba(val[cols].astype("float32"))[:, 1]
        df.loc[val.index, "iter_hgb_score"] = pred
        fold_meta.append(
            {
                "fold": int(fold),
                "train_candidates": int(len(train)),
                "val_candidates": int(len(val)),
                "train_positive": int(y_train.sum()),
            }
        )
        print(f"[iter-hgb] fold {fold} done", flush=True)
    return df, {"feature_count": len(cols), "features": cols, "fold_meta": fold_meta}


def metric_delta(metrics: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    keys = [f"match@{b}" for b in BUDGETS] + [f"rows>=7_match@{b}" for b in BUDGETS]
    return {k: float(metrics[k] - base[k]) for k in keys}


def fmt_metrics(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"match@{b}"]) for b in BUDGETS)


def fmt_rows7(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"rows>=7_match@{b}"]) for b in BUDGETS)


def fmt_delta(d: dict[str, float], prefix: str = "") -> str:
    return " / ".join(pp(d[f"{prefix}match@{b}"]) for b in BUDGETS)


def union_oracle(policies: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    by_policy = [{r["sample_id"]: r for r in p["sample_rows"]} for p in policies]
    base_rows = baseline["sample_rows"]
    rows: list[dict[str, Any]] = []
    for br in base_rows:
        sid = br["sample_id"]
        out = {"sample_id": sid, "target_rows_ge7": br["target_rows_ge7"]}
        for b in BUDGETS:
            out[f"hit@{b}"] = any(pol.get(sid, {}).get(f"hit@{b}", False) for pol in by_policy)
        rows.append(out)
    return {"name": "route_union_oracle", "metrics": summarize(rows), "sample_rows": rows}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    baseline = evaluate_policy(df, lambda g: g[g["rank"] <= 20].sort_values("rank"), "baseline")
    track_a = evaluate_policy(df, lambda g: selected_order_track_a(g, "score_rows7_specialized_gate"), "track_a_rows7")
    exact = evaluate_policy(df, selected_order_exact_cover, "exact_cover_filter")

    cols = feature_columns(df)
    scored, model_meta = train_hgb_oof(df, cols)
    hgb = evaluate_policy(
        scored,
        lambda g: g.sort_values(["iter_hgb_score", "rank"], ascending=[False, True]).head(20),
        "iter_hgb_candidate_scorer",
    )
    union = union_oracle([baseline, track_a, exact, hgb], baseline)

    policies = {"baseline": baseline, "track_a": track_a, "exact": exact, "hgb": hgb, "union_oracle": union}
    deltas = {name: metric_delta(pol["metrics"], baseline["metrics"]) for name, pol in policies.items() if name != "baseline"}
    achieved = any(
        sum(1 for b in BUDGETS if delta[f"match@{b}"] >= 0.05) >= 2
        or (delta["rows>=7_match@5"] >= 0.05 and delta["rows>=7_match@20"] >= 0.05)
        for delta in deltas.values()
    )
    result = {
        "time": now_iso(),
        "gpu_necessary": False,
        "gpu_reason": "validation OOF sklearn HGB and route-oracle diagnostics; no MP-20/MPTS-52 train dataset training",
        "data": {
            "samples": int(df["sample_id"].nunique()),
            "candidates": int(len(df)),
            "rows>=7_samples": int(df[["sample_id", "target_rows_ge7"]].drop_duplicates()["target_rows_ge7"].sum()),
        },
        "model_meta": model_meta,
        "policies": {name: {"metrics": pol["metrics"], "diagnostics": pol.get("diagnostics")} for name, pol in policies.items()},
        "deltas_vs_baseline": deltas,
        "achieved_stop_threshold": bool(achieved),
    }
    write_json(RESULTS / "iteration_01_hgb_route_search.json", result)

    hgb_d = deltas["hgb"]
    union_d = deltas["union_oracle"]
    append_report(
        "迭代 01 HGB scorer 与 route-union coverage 诊断",
        f"""
时间：{result['time']}

实验逻辑：完成 prompt 1-8 后，按用户要求开始自定义迭代，目标继续提升 match 指标。本轮先验证“更强候选级 scorer + 多路线互补 coverage”是否足够。HGB scorer 是实际 5-fold OOF 训练；route-union oracle 是诊断上限，不是可部署方法。

GPU 必要性判断：本轮没有使用 MP-20/MPTS-52 train 数据集训练，只在 validation OOF 上做 sklearn HGB 和 route coverage 诊断，因此不需要 GPU。

核心假设：如果普通 scorer 还没到上限，HGB 应比 Track A/CPU scorer 明显更强；如果 HGB 不强但 route-union oracle 高，说明需要 sample-level route selector；如果 union oracle 也不够，说明候选/生成 coverage 本身不足。

数据规模：MPTS-52 validation K50，全量 {result['data']['samples']} samples / {result['data']['candidates']} candidates；rows>=7 samples={result['data']['rows>=7_samples']}；HGB 特征数={model_meta['feature_count']}。

baseline：原 GT-SG rank 顺序 = {fmt_metrics(baseline['metrics'])}；rows>=7 = {fmt_rows7(baseline['metrics'])}。

方法变化：训练 HistGradientBoosting candidate scorer，特征包括 rank、结构一致性、formula/SG/exact-cover/collision/local-geometry、Track A OOF score 等 validation OOF 可用信号；同时计算 baseline/TrackA/exact-cover/HGB 四条路线的 route-union oracle。

结果 HGB scorer：{fmt_metrics(hgb['metrics'])}；delta = {fmt_delta(hgb_d)}。rows>=7 = {fmt_rows7(hgb['metrics'])}；rows>=7 delta = {fmt_delta(hgb_d, 'rows>=7_')}。

结果 route-union oracle：{fmt_metrics(union['metrics'])}；delta = {fmt_delta(union_d)}。rows>=7 = {fmt_rows7(union['metrics'])}；rows>=7 delta = {fmt_delta(union_d, 'rows>=7_')}。

可信度：HGB 是 OOF 可信；route-union oracle 使用每个样本事后知道哪条路线命中，只能作为互补上限，不能当真实方法。

和历史实验关系：这轮直接检验“继续做更强 rerank 是否可能超过 +5pp”。若 HGB 不过 +5pp，而 union oracle 明显更高，则后续转向 route selector；若 union oracle 也弱，则说明候选池/geometry coverage 仍是主瓶颈。

最终判决：achieved_stop_threshold={achieved}。{"已达到停止阈值，可以停止迭代。" if achieved else "未达到 +5pp 停止阈值，必须继续下一轮自定义实验。"}

下一步：若未达标，下一轮训练 sample-level route selector 或尝试 CrystaLLM+SymCIF hybrid selector；若 route oracle 上限仍不足，则转向生成/repair 而非排序。
""",
    )
    print(json.dumps({"achieved_stop_threshold": achieved, "hgb_delta": hgb_d, "union_delta": union_d}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
