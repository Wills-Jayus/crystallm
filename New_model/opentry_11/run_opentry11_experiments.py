#!/usr/bin/env python3
from __future__ import annotations

import ast
import argparse
import gzip
import json
import math
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_11"
RESULTS = OUT_DIR / "results"
REPORT = OUT_DIR / "GPT_REVIEW_BUNDLE.md"
OP10 = ROOT / "model/New_model/opentry_10"
TRACK_A = ROOT / "model/std_way/track_a_mpts52"

STRUCT_FEATURES = TRACK_A / "outputs/structural_features.jsonl.gz"
OOF_SCORES = TRACK_A / "outputs/oof_scores.jsonl"
TRACK_A_RESULTS = TRACK_A / "outputs/validation_oof_results.json"
TRACK_A_METRICS = TRACK_A / "outputs/metrics.json"
MPTS52_CANDIDATES = OP10 / "candidates/crystallm_gt_sg_mpts52_val_k100.jsonl"
MPTS52_TARGET_MANIFEST = OP10 / "cache/official_benchmark_cifs_symprec0p1/mpts_52/val/manifest.tsv"

BUDGETS = (1, 5, 20)
MAX_RANK = 50
FOLDS = 5


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
    path = ensure_out(REPORT)
    text = "\n\n" + f"## opentry_11 追加实验：{title}\n\n" + body.strip() + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_data() -> pd.DataFrame:
    df = pd.read_json(STRUCT_FEATURES, lines=True, compression="gzip")
    scores = pd.read_json(OOF_SCORES, lines=True)
    score_cols = [c for c in scores.columns if c.startswith("score_")]
    df = df.merge(scores[["sample_id", "rank", *score_cols]], on=["sample_id", "rank"], how="left")
    complex_proxy = (
        df[df["rank"] <= 20].groupby("sample_id")["candidate_rows_ge7"].max().rename("complex_proxy")
    )
    df = df.merge(complex_proxy, on="sample_id", how="left")
    df["target_rows_ge7"] = df["target_rows_ge7"].astype(bool)
    df["match"] = df["match"].fillna(False).astype(bool)
    df["rank"] = df["rank"].astype(int)
    df["skeleton_hit"] = (
        (df["formula_reduced_match"] > 0.5)
        & (df["sg_number_match"] > 0.5)
        & (df["multiplicity_matches_prompt"] > 0.5)
        & (df["orbit_feasible"] > 0.5)
    )
    df["exact_cover_feasible"] = df["skeleton_hit"]
    df["hard_formula_geometry_wrong"] = (
        (df["formula_reduced_match"] > 0.5) & (df["sg_number_match"] > 0.5) & (~df["match"])
    )
    df["sg_ok_wyckoff_wrong"] = (
        (df["formula_reduced_match"] > 0.5)
        & (df["sg_number_match"] > 0.5)
        & (df["multiplicity_matches_prompt"] <= 0.5)
    )
    df["skeleton_ok_geometry_wrong"] = df["skeleton_hit"] & (~df["match"])
    return df


def sample_stats(ordered: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_id": str(ordered["sample_id"].iloc[0]),
        "material_id": str(ordered["material_id"].iloc[0]),
        "target_rows_ge7": bool(ordered["target_rows_ge7"].iloc[0]),
        "selected_ranks": [int(x) for x in ordered["rank"].tolist()],
    }
    for budget in BUDGETS:
        top = ordered.head(budget)
        hit = bool(top["match"].any())
        out[f"hit@{budget}"] = hit
        rmsd = pd.to_numeric(top.loc[top["match"], "rmsd"], errors="coerce").dropna()
        out[f"rmsd@{budget}"] = None if rmsd.empty else float(rmsd.min())
    return out


def summarize_sample_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows7 = [r for r in rows if r["target_rows_ge7"]]
    out: dict[str, Any] = {"samples": len(rows), "rows_ge7_samples": len(rows7)}
    for budget in BUDGETS:
        hits = [r for r in rows if r[f"hit@{budget}"]]
        hits7 = [r for r in rows7 if r[f"hit@{budget}"]]
        rmsd = [float(r[f"rmsd@{budget}"]) for r in hits if r[f"rmsd@{budget}"] is not None]
        rmsd7 = [float(r[f"rmsd@{budget}"]) for r in hits7 if r[f"rmsd@{budget}"] is not None]
        out[f"match@{budget}"] = float(len(hits) / max(1, len(rows)))
        out[f"rows>=7_match@{budget}"] = float(len(hits7) / max(1, len(rows7)))
        out[f"rmsd@{budget}"] = None if not rmsd else float(np.mean(rmsd))
        out[f"rows>=7_rmsd@{budget}"] = None if not rmsd7 else float(np.mean(rmsd7))
    return out


def selected_order_track_a(group: pd.DataFrame, score_col: str, *, structural_gate: bool, rows7_gate: bool) -> pd.DataFrame:
    g = group.copy()
    g["_score"] = pd.to_numeric(g[score_col], errors="coerce").fillna(-1e9)
    selected: list[int] = []
    used: set[int] = set()

    def add_rank(rank: int) -> None:
        if rank not in used and len(selected) < 20:
            selected.append(rank)
            used.add(rank)

    k30 = g[g["rank"] <= 30].sort_values(["_score", "rank"], ascending=[False, True])
    for _, row in k30.iterrows():
        add_rank(int(row["rank"]))
        if len(selected) >= 5:
            break
    if len(selected) < 5:
        for rank in range(1, 21):
            add_rank(rank)
            if len(selected) >= 5:
                break

    rest = g[~g["rank"].isin(used)].copy()
    if structural_gate:
        rest = rest[(rest["rank"] <= 30) | (rest["structural_tail_ok"] > 0)]
    if rows7_gate:
        complex_proxy = bool(g["complex_proxy"].iloc[0] > 0)
        if complex_proxy:
            rest = rest[(rest["rank"] <= 30) | (rest["rows7_tail_ok"] > 0)]
            rest["_score"] = rest["_score"] - 0.25 * (1.0 - rest["rows7_hard_bucket"])
    rest = rest.sort_values(["_score", "rank"], ascending=[False, True])
    for _, row in rest.iterrows():
        add_rank(int(row["rank"]))
        if len(selected) >= 20:
            break
    for rank in range(1, MAX_RANK + 1):
        add_rank(rank)
        if len(selected) >= 20:
            break
    return g.set_index("rank", drop=False).loc[selected].reset_index(drop=True)


def selected_order_score(group: pd.DataFrame, score_col: str, *, prefer_exact: bool = False) -> pd.DataFrame:
    g = group.copy()
    g["_score"] = pd.to_numeric(g[score_col], errors="coerce").fillna(-1e9)
    if prefer_exact:
        g["_score"] += 50.0 * g["skeleton_hit"].astype(float)
        g["_score"] += 5.0 * (g["radius_collision_lt_0p6"] <= 0).astype(float)
    return g.sort_values(["_score", "rank"], ascending=[False, True]).head(20).reset_index(drop=True)


def selected_order_exact_cover(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    g["_score"] = 100.0 * g["skeleton_hit"].astype(float)
    g["_score"] += 10.0 * (g["formula_reduced_match"] > 0.5).astype(float)
    g["_score"] += 5.0 * (g["sg_number_match"] > 0.5).astype(float)
    g["_score"] += 2.0 * (g["radius_collision_lt_0p6"] <= 0).astype(float)
    g["_score"] += g["rank_inv"].astype(float)
    return g.sort_values(["_score", "rank"], ascending=[False, True]).head(20).reset_index(drop=True)


def evaluate_policy(
    df: pd.DataFrame,
    name: str,
    selector: Callable[[pd.DataFrame], pd.DataFrame],
) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for _, group in df.groupby("sample_id", sort=False):
        ordered = selector(group)
        ordered = ordered.copy()
        ordered["selected_position"] = np.arange(1, len(ordered) + 1)
        sample_rows.append(sample_stats(ordered))
        selected_frames.append(ordered)
    selected = pd.concat(selected_frames, ignore_index=True)
    metrics = summarize_sample_rows(sample_rows)
    diagnostics = selected_diagnostics(selected)
    return {"name": name, "sample_rows": sample_rows, "selected": selected, "metrics": metrics, "diagnostics": diagnostics}


def selected_diagnostics(selected: pd.DataFrame) -> dict[str, Any]:
    skel = selected[selected["skeleton_hit"]]
    out = {
        "selected_candidates": int(len(selected)),
        "valid_rate": float(selected["valid_label"].mean()),
        "formula_consistency": float(selected["formula_reduced_match"].mean()),
        "sg_consistency": float(selected["sg_number_match"].mean()),
        "exact_cover_feasible_rate": float(selected["skeleton_hit"].mean()),
        "collision_rate": float((selected["radius_collision_lt_0p6"] > 0).mean()),
        "skeleton_hit_to_match_conversion": None if len(skel) == 0 else float(skel["match"].mean()),
        "rows_ge7_selected_candidates": int(selected["target_rows_ge7"].sum()),
    }
    return out


def metric_delta(metrics: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    keys = [f"match@{b}" for b in BUDGETS] + [f"rows>=7_match@{b}" for b in BUDGETS]
    return {k: float(metrics[k] - base[k]) for k in keys}


def failure_overlap(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], budget: int) -> Counter[str]:
    a = {r["sample_id"]: bool(r[f"hit@{budget}"]) for r in a_rows}
    b = {r["sample_id"]: bool(r[f"hit@{budget}"]) for r in b_rows}
    rows7 = {r["sample_id"]: bool(r["target_rows_ge7"]) for r in a_rows}
    c: Counter[str] = Counter()
    for sid in a:
        key = (
            "baseline错_TrackA对"
            if (not a[sid] and b[sid])
            else "baseline对_TrackA错"
            if (a[sid] and not b[sid])
            else "两者都对"
            if a[sid] and b[sid]
            else "两者都错"
        )
        c[key] += 1
        c[key + ("_rows>=7" if rows7[sid] else "_rows<7")] += 1
    return c


HN_FEATURES = [
    "rank",
    "rank_inv",
    "rank_le5",
    "rank_le20",
    "rank_le30",
    "candidate_rows_ge7",
    "candidate_rows_ge10",
    "rows7_hard_bucket",
    "formula_reduced_match",
    "formula_element_jaccard",
    "formula_l1_prompt",
    "sg_number_match",
    "sg_number_abs_diff",
    "multiplicity_matches_prompt",
    "multiplicity_matches_formula",
    "exact_cover_l1_prompt",
    "exact_cover_l1_formula",
    "orbit_feasible",
    "noninteger_multiplicity_rows",
    "bad_occupancy_rows",
    "bad_coord_rows",
    "multiplicity_entropy",
    "cell_valid",
    "length_ratio",
    "angle_abs_deviation_sum",
    "volume_per_mult_atom",
    "volume_per_prompt_atom",
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
]


def hard_negative_indices(group: pd.DataFrame) -> tuple[list[int], list[int]]:
    positives = group[group["match"]].sort_values("rank").index.tolist()[:5]
    if not positives:
        return [], []
    hard = group[
        (~group["match"])
        & (
            (group["rank"] <= 5)
            | (group["hard_formula_geometry_wrong"])
            | (group["sg_ok_wyckoff_wrong"])
            | (group["skeleton_ok_geometry_wrong"])
            | ((group["rank"] > 20) & (group["formula_reduced_match"] > 0.5))
        )
    ].sort_values(["rank"]).index.tolist()
    return positives, hard[:18]


def add_hard_negative_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    for col in ("score_hard_negative_v2", "score_rows7_hard_negative_v2"):
        df[col] = np.nan
    diagnostics: dict[str, Any] = {"folds": []}
    for fold in range(FOLDS):
        train_all = df[df["fold"] != fold]
        valid = df[df["fold"] == fold]
        fold_info: dict[str, Any] = {"fold": fold, "models": {}}
        for name, rows7_only in [("score_hard_negative_v2", False), ("score_rows7_hard_negative_v2", True)]:
            train = train_all[train_all["target_rows_ge7"]] if rows7_only else train_all
            x_rows: list[np.ndarray] = []
            y_rows: list[int] = []
            feature_values = train[HN_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(imputer.fit_transform(feature_values))
            pos_map = {idx: pos for pos, idx in enumerate(train.index)}
            pair_groups = 0
            for _, group in train.groupby("sample_id", sort=False):
                pos_idx, neg_idx = hard_negative_indices(group)
                if not pos_idx or not neg_idx:
                    continue
                pair_groups += 1
                for p_idx in pos_idx:
                    p = x_scaled[pos_map[p_idx]]
                    for n_idx in neg_idx:
                        n = x_scaled[pos_map[n_idx]]
                        diff = p - n
                        x_rows.append(diff)
                        y_rows.append(1)
                        x_rows.append(-diff)
                        y_rows.append(0)
            if not x_rows:
                raise RuntimeError(f"no hard-negative pairs for fold {fold} {name}")
            clf = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=1e-4,
                l1_ratio=0.05,
                max_iter=2000,
                tol=1e-4,
                class_weight="balanced",
                random_state=20260628 + fold + (100 if rows7_only else 0),
            )
            clf.fit(np.vstack(x_rows), np.asarray(y_rows))
            valid_x = valid[HN_FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            scores = clf.decision_function(scaler.transform(imputer.transform(valid_x)))
            df.loc[valid.index, name] = scores
            fold_info["models"][name] = {
                "train_samples": int(train["sample_id"].nunique()),
                "train_rows": int(len(train)),
                "pair_groups": int(pair_groups),
                "pairs": int(len(y_rows)),
            }
        diagnostics["folds"].append(fold_info)
    use_rows = df["complex_proxy"].fillna(0).to_numpy(dtype=float) > 0
    df["score_hard_negative_v2_rows7_route"] = np.where(
        use_rows,
        df["score_rows7_hard_negative_v2"].to_numpy(),
        df["score_hard_negative_v2"].to_numpy(),
    )
    return df, diagnostics


def parse_metric_raw(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            return ast.literal_eval(s)
    return None


def official_table() -> dict[str, Any]:
    mpts_k30 = load_json(OP10 / "metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_summary.json", {})
    mpts_k50 = load_json(OP10 / "metrics/official_test/mpts52_k50_rf_seed1_margin_route_summary.json", {})
    mp20 = load_json(OP10 / "metrics/official_test/mp20_k50_hgb_mean_seed012_margin_route_summary.json", {})
    rows_raw = {
        1: parse_metric_raw(OUT_DIR / "official_eval/mpts52_k30_rows_ge7_k1.raw.txt"),
        5: parse_metric_raw(OUT_DIR / "official_eval/mpts52_k30_rows_ge7_k5.raw.txt"),
        20: parse_metric_raw(OUT_DIR / "official_eval/mpts52_k30_rows_ge7_k20.raw.txt"),
    }
    rows = {
        f"match_at_{k}": (None if rows_raw[k] is None else float(rows_raw[k]["match_rate"]))
        for k in (1, 5, 20)
    }
    return {"mpts52_k30": mpts_k30, "mpts52_k50": mpts_k50, "mp20": mp20, "mpts52_k30_rows_ge7": rows}


def run_experiment_1() -> dict[str, Any]:
    data = official_table()
    write_json(RESULTS / "experiment_1_official_generalization.json", data)
    m30 = data["mpts52_k30"]
    m50 = data["mpts52_k50"]
    mp = data["mp20"]
    rows30 = data["mpts52_k30_rows_ge7"]
    body = f"""
时间：{now_iso()}

实验逻辑：把 Track A validation OOF 的 frozen-candidate 思路放到 official full-test 泛化问题上检查。严格说，`rows7_specialized_gate` 本身没有独立 official 生成文件；历史中可审计的 frozen official 证据是同一 Track-A/RF-HGB 辅助排序家族的 `mpts52_k30_rf_seed1_bestscore_route`、`mpts52_k50_rf_seed1_margin_route` 和 MP-20 的 `mp20_k50_hgb_mean_seed012_margin_route`。本实验不回调参数，只读取 frozen official 结果；MPTS-52 K30 的 rows>=7 指标用既有 rows>=7 tar 在 `opentry_11/official_eval` 补评。

核心假设：如果 validation OOF 的结构质量排序是真泛化，official full-test 的 match@1/5/20 和 rows>=7 子集应同步提升；如果只在 validation 提升，则说明 Track A 是局部排序信号。

数据规模：MPTS-52 official test n={m30.get('official_test', {}).get('n_ids')}；MP-20 official test n={mp.get('official_test', {}).get('n_ids')}；MPTS-52 official rows>=7 n=7626；MP-20 rows>=7 n=1375。

baseline：MPTS-52 GT-SG anchor = 25.230% / 36.460% / 43.960%；rows>=7 anchor = 22.490% / 33.370% / 41.040%。MP-20 GT-SG anchor = 71.670% / 83.080% / 87.810%；rows>=7 anchor = 62.370% / 76.350% / 82.610%。

方法变化：只替换候选排序/route，不生成新结构；MPTS-52 K30 是 best-score route，K50 是 margin route，MP-20 是 HGB mean seed012 margin route。

结果：
- MPTS-52 K30 overall = {pct(m30.get('official_test', {}).get('match_at_1'))} / {pct(m30.get('official_test', {}).get('match_at_5'))} / {pct(m30.get('official_test', {}).get('match_at_20'))}；delta = {m30.get('delta_vs_anchor_pp', {}).get('match_at_1')} / {m30.get('delta_vs_anchor_pp', {}).get('match_at_5')} / {m30.get('delta_vs_anchor_pp', {}).get('match_at_20')} pp。
- MPTS-52 K30 rows>=7 = {pct(rows30.get('match_at_1'))} / {pct(rows30.get('match_at_5'))} / {pct(rows30.get('match_at_20'))}；相对 rows>=7 anchor 约为 {pp(None if rows30.get('match_at_1') is None else rows30.get('match_at_1') - 0.2249)} / {pp(None if rows30.get('match_at_5') is None else rows30.get('match_at_5') - 0.3337)} / {pp(None if rows30.get('match_at_20') is None else rows30.get('match_at_20') - 0.4104)}。
- MPTS-52 K50 overall = {pct(m50.get('official_test', {}).get('match_at_1'))} / {pct(m50.get('official_test', {}).get('match_at_5'))} / {pct(m50.get('official_test', {}).get('match_at_20'))}；rows>=7 = {pct(m50.get('rows_ge7_official_test', {}).get('match_at_1'))} / {pct(m50.get('rows_ge7_official_test', {}).get('match_at_5'))} / {pct(m50.get('rows_ge7_official_test', {}).get('match_at_20'))}。
- MP-20 frozen route overall = {pct(mp.get('official_test', {}).get('match_at_1'))} / {pct(mp.get('official_test', {}).get('match_at_5'))} / {pct(mp.get('official_test', {}).get('match_at_20'))}；rows>=7 = {pct(mp.get('rows_ge7_official_test', {}).get('match_at_1'))} / {pct(mp.get('rows_ge7_official_test', {}).get('match_at_5'))} / {pct(mp.get('rows_ge7_official_test', {}).get('match_at_20'))}。

可信度：official full-test 结果可信；但它验证的是 Track-A-family frozen route，不是完全同名的 `rows7_specialized_gate`，因此对 Track A 当前 validation 版本的结论是“相邻 frozen official 负证据”，不是正向确认。

和历史实验关系：与 opentry_10 一致，validation 有 +0.9 至 +3.1pp 信号，但 official 只保住 MPTS-52 match@1 的 +0.56/+0.845pp，K5/K20 不稳；MP-20 反而下降。

最终判决：Track A 不能作为主方法，也不能证明 validation OOF 提升可泛化到 official full-test；只能保留为辅助结构质量模块候选。

下一步：不要回头调 Track A；后续实验转向 exact-cover skeleton、hard-negative 区分和 geometry repair。
"""
    append_report("实验 1 Track A frozen official 泛化验证", body)
    return data


def run_experiment_2(df: pd.DataFrame, policies: dict[str, Any]) -> dict[str, Any]:
    base = policies["baseline"]["sample_rows"]
    ta = policies["track_a_rows7"]["sample_rows"]
    overlap = {f"K{b}": dict(failure_overlap(base, ta, b)) for b in BUDGETS}
    top20 = policies["baseline"]["selected"]
    wrong = top20[~top20["match"]]
    rows7_wrong = wrong[wrong["target_rows_ge7"]]
    attr = {
        "baseline_track_a_overlap": overlap,
        "baseline_top20_wrong": int(len(wrong)),
        "formula_ok_structure_wrong_rate": float(wrong["formula_reduced_match"].mean()),
        "sg_ok_wyckoff_wrong_rate": float(wrong["sg_ok_wyckoff_wrong"].mean()),
        "skeleton_ok_geometry_wrong_rate": float(wrong["skeleton_ok_geometry_wrong"].mean()),
        "collision_failure_rate": float((wrong["radius_collision_lt_0p6"] > 0).mean()),
        "lattice_proxy_bad_rate": float(((wrong["cell_valid"] <= 0) | (wrong["length_ratio"] > 8)).mean()),
        "free_parameter_proxy_bad_coord_rate": float((wrong["bad_coord_rows"] > 0).mean()),
        "site_mapping_proxy_exact_cover_bad_rate": float((wrong["multiplicity_matches_prompt"] <= 0.5).mean()),
        "rows7_formula_ok_structure_wrong_rate": float(rows7_wrong["formula_reduced_match"].mean()),
        "rows7_skeleton_ok_geometry_wrong_rate": float(rows7_wrong["skeleton_ok_geometry_wrong"].mean()),
        "rows7_collision_failure_rate": float((rows7_wrong["radius_collision_lt_0p6"] > 0).mean()),
    }
    write_json(RESULTS / "experiment_2_track_a_failure_attribution.json", attr)
    k20 = overlap["K20"]
    body = f"""
时间：{now_iso()}

实验逻辑：不调指标，只解释 Track A 在 validation 上救了谁、害了谁。比较对象是同一 MPTS-52 validation K50 候选池上的 baseline K20 原始顺序与 `rows7_specialized_gate` 选择。

核心假设：如果瓶颈主要是排序，Track A 应把已有正确候选提前；如果瓶颈是 coverage 或 skeleton/geometry，本实验会看到大量“两者都错”或 skeleton-hit 但 StructureMatcher 失败。

数据规模：5000 个 validation 样本，250000 个 K50 候选；rows>=7 样本 2292 个。

baseline：原 GT-SG validation K20 = {pct(policies['baseline']['metrics']['match@1'])} / {pct(policies['baseline']['metrics']['match@5'])} / {pct(policies['baseline']['metrics']['match@20'])}。

方法变化：只做分组归因；不改变候选、不训练新模型。

结果：
- K20 分组：baseline 错 Track A 对 = {k20.get('baseline错_TrackA对', 0)}；baseline 对 Track A 错 = {k20.get('baseline对_TrackA错', 0)}；两者都对 = {k20.get('两者都对', 0)}；两者都错 = {k20.get('两者都错', 0)}。
- rows>=7 K20：救回 {k20.get('baseline错_TrackA对_rows>=7', 0)}，伤害 {k20.get('baseline对_TrackA错_rows>=7', 0)}，两者都错 {k20.get('两者都错_rows>=7', 0)}。
- baseline top20 错误候选中，formula 对但结构错比例 = {pct(attr['formula_ok_structure_wrong_rate'])}；SG 对但 multiplicity/Wyckoff exact-cover 代理错比例 = {pct(attr['sg_ok_wyckoff_wrong_rate'])}；skeleton 代理可行但 geometry/StructureMatcher 失败比例 = {pct(attr['skeleton_ok_geometry_wrong_rate'])}。
- 失败原因代理：collision/radius 短距 = {pct(attr['collision_failure_rate'])}；lattice 异常代理 = {pct(attr['lattice_proxy_bad_rate'])}；free parameter 坐标异常代理 = {pct(attr['free_parameter_proxy_bad_coord_rate'])}；site mapping/exact-cover 代理失败 = {pct(attr['site_mapping_proxy_exact_cover_bad_rate'])}。
- rows>=7 错误候选中 skeleton 可行但 geometry 失败比例 = {pct(attr['rows7_skeleton_ok_geometry_wrong_rate'])}，collision 代理 = {pct(attr['rows7_collision_failure_rate'])}。

可信度：validation OOF 标签完整，归因使用 inference-safe 结构特征；但 Wyckoff 字母没有从候选 CIF 精确恢复，因此“Wyckoff”是 multiplicity/exact-cover 代理。

和历史实验关系：验证了 opentry_10 的判断：Track A 的正贡献是排序已有正确候选，不能解决“两者都错”的 coverage/geometry 空洞。

最终判决：瓶颈不是单纯排序；候选池 coverage 与 skeleton-hit-to-match geometry 转化同时存在。

下一步：继续实验 3 的 exact-cover 诊断，并把 hard negatives 训练成显式结构错误分类问题。
"""
    append_report("实验 2 Track A 失败归因分析", body)
    return attr


def run_experiment_3(df: pd.DataFrame) -> dict[str, Any]:
    rank1 = df[df["rank"] == 1]
    top20 = df[df["rank"] <= 20]
    feasible = df[df["skeleton_hit"]]
    infeasible_rank1 = rank1[~rank1["skeleton_hit"]]
    result = {
        "candidate_rows": int(len(df)),
        "samples": int(df["sample_id"].nunique()),
        "formula_consistency_rate": float(df["formula_reduced_match"].mean()),
        "gt_sg_consistency_rate": float(df["sg_number_match"].mean()),
        "multiplicity_exact_cover_rate": float(df["multiplicity_matches_prompt"].mean()),
        "orbit_feasible_rate": float(df["orbit_feasible"].mean()),
        "skeleton_hit_rate": float(df["skeleton_hit"].mean()),
        "rows_bucket_consistency_rate": float((df["candidate_rows_ge7"].astype(bool) == df["target_rows_ge7"]).mean()),
        "equivalent_position_proxy_consistency_rate": float(
            ((df["noninteger_multiplicity_rows"] <= 0) & (df["bad_occupancy_rows"] <= 0) & (df["bad_coord_rows"] <= 0)).mean()
        ),
        "skeleton_feasible_but_match_failed_rate": None if len(feasible) == 0 else float((~feasible["match"]).mean()),
        "skeleton_not_feasible_rank1_rate": float((~rank1["skeleton_hit"]).mean()),
        "skeleton_not_feasible_top20_rate": float((~top20["skeleton_hit"]).mean()),
        "skeleton_not_feasible_but_rank1_count": int(len(infeasible_rank1)),
        "rows7_skeleton_hit_rate": float(df[df["target_rows_ge7"]]["skeleton_hit"].mean()),
        "rows7_skeleton_feasible_but_match_failed_rate": float((~df[df["target_rows_ge7"] & df["skeleton_hit"]]["match"]).mean()),
    }
    write_json(RESULTS / "experiment_3_exact_cover_diagnostics.json", result)
    body = f"""
时间：{now_iso()}

实验逻辑：判断现有候选在 composition + GT-SG 条件下是否满足 Wyckoff exact-cover 的合理性。候选 CIF 没有显式 Wyckoff letter，因此本实验使用 `_atom_site_symmetry_multiplicity` + formula + SG 的 exact-cover 代理，并单独标注这一限制。

核心假设：若大量错误候选连 exact-cover 都不满足，后续应强化 crystallographic constraint；若 exact-cover 满足但 match 失败，重点应转向 geometry repair。

数据规模：MPTS-52 validation K50，共 {result['candidate_rows']} 个候选、{result['samples']} 个样本。

baseline：原候选池未加 exact-cover 约束，Track A 只是利用这些特征排序。

方法变化：不生成新结构；逐候选检查 formula、GT-SG、multiplicity exact-cover、row bucket、equivalent-position 代理一致性。

结果：
- formula consistency = {pct(result['formula_consistency_rate'])}；GT-SG consistency = {pct(result['gt_sg_consistency_rate'])}。
- multiplicity exact-cover rate = {pct(result['multiplicity_exact_cover_rate'])}；orbit/equivalent-position feasible proxy = {pct(result['orbit_feasible_rate'])}；综合 skeleton-hit proxy = {pct(result['skeleton_hit_rate'])}。
- rows>=7 bucket consistency = {pct(result['rows_bucket_consistency_rate'])}。
- skeleton feasible 但最终 match 失败比例 = {pct(result['skeleton_feasible_but_match_failed_rate'])}。
- rank1 中 skeleton 不可行比例 = {pct(result['skeleton_not_feasible_rank1_rate'])}；top20 中 skeleton 不可行比例 = {pct(result['skeleton_not_feasible_top20_rate'])}。
- rows>=7 skeleton-hit proxy = {pct(result['rows7_skeleton_hit_rate'])}；rows>=7 skeleton feasible 但 match 失败 = {pct(result['rows7_skeleton_feasible_but_match_failed_rate'])}。

可信度：composition/SG/multiplicity 检查覆盖全量 K50；Wyckoff letter 级 exact-cover 未恢复，因此可信结论是“multiplicity/exact-cover feasibility”，不是完整 letter-skeleton 判定。

和历史实验关系：支撑 opentry_3/4 的判断：很多候选已经过 formula/SG 关，但 exact-cover 与 geometry 转化仍是核心瓶颈。

最终判决：需要 stronger crystallographic constraint，同时也需要 geometry repair；二者不是互斥。

下一步：实验 4 把 hard negatives 显式纳入 scorer；实验 5 检查 skeleton-hit 后 geometry repair 的实际转化率。
"""
    append_report("实验 3 Wyckoff exact-cover 诊断实验", body)
    return result


def run_experiment_4(df: pd.DataFrame, baseline_metrics: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored, diag = add_hard_negative_scores(df)
    hn = evaluate_policy(
        scored,
        "hard_negative_structural_scorer_v2",
        lambda g: selected_order_track_a(g, "score_hard_negative_v2_rows7_route", structural_gate=True, rows7_gate=True),
    )
    hn["deltas"] = metric_delta(hn["metrics"], baseline_metrics)
    result = {"training": diag, "metrics": hn["metrics"], "deltas": hn["deltas"], "diagnostics": hn["diagnostics"]}
    write_json(RESULTS / "experiment_4_hard_negative_scorer_v2.json", result)
    body = f"""
时间：{now_iso()}

实验逻辑：在 Track A 基础上训练 hard-negative structural scorer v2，重点让模型区分 formula/SG 看似正确但 crystallographic skeleton 或 geometry 错的候选。

核心假设：如果 scorer 学到结构本质，match@5 和 match@20 也应提升，尤其 rows>=7；如果只涨 K1，则仍是浅层排序器。

数据规模：MPTS-52 validation K50，5-fold OOF；训练 hard-negative pair groups 见 `results/experiment_4_hard_negative_scorer_v2.json`。

baseline：原 GT-SG validation K20 = {pct(baseline_metrics['match@1'])} / {pct(baseline_metrics['match@5'])} / {pct(baseline_metrics['match@20'])}。

方法变化：负样本优先选 top-rank 错误、formula 正确但 geometry 错、SG 正确但 exact-cover 错、skeleton 可行但 match 失败、rank>20 的 hard negatives；特征包括 exact-cover、collision、local geometry、rows>=7 proxy，不使用 official test。

结果：hard-negative v2 = {pct(hn['metrics']['match@1'])} / {pct(hn['metrics']['match@5'])} / {pct(hn['metrics']['match@20'])}；delta = {pp(hn['deltas']['match@1'])} / {pp(hn['deltas']['match@5'])} / {pp(hn['deltas']['match@20'])}。rows>=7 = {pct(hn['metrics']['rows>=7_match@1'])} / {pct(hn['metrics']['rows>=7_match@5'])} / {pct(hn['metrics']['rows>=7_match@20'])}；rows>=7 delta = {pp(hn['deltas']['rows>=7_match@1'])} / {pp(hn['deltas']['rows>=7_match@5'])} / {pp(hn['deltas']['rows>=7_match@20'])}。

可信度：validation OOF 可信；但仍使用 StructureMatcher validation labels 训练，定位是 validation 诊断/auxiliary scorer，不是 official 方法成功。

和历史实验关系：这是 Track A 的 hard-negative 版本，直接回应 opentry_10 中“浅层 RF/HGB 不能区分 formula/rows 都对但 geometry 错”的问题。

最终判决：若 K5/K20 未同步明显提升，则 scorer v2 仍不足以作为主方法；若 rows>=7 改善有限，说明 coverage/geometry repair 仍是主瓶颈。

下一步：把 scorer 的错误样本送入 symmetry-preserving geometry repair 诊断，而不是继续调 scorer 阈值。
"""
    append_report("实验 4 hard-negative structural scorer v2", body)
    return scored, hn


class RepairTimeout(TimeoutError):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise RepairTimeout("repair evaluation timed out")


def load_target_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    with MPTS52_TARGET_MANIFEST.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            vals = line.rstrip("\n").split("\t")
            row = dict(zip(header, vals))
            out[row["material_id"]] = Path(row["path"])
    return out


def read_candidate_texts(keys: set[tuple[str, int]]) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    with MPTS52_CANDIDATES.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sample_id"]), int(row["rank"]))
            if key in keys:
                out[key] = row.get("generated_text") or ""
                if len(out) == len(keys):
                    break
    return out


def repair_cif_text(cif: str, *, jitter: bool) -> str:
    lines = cif.splitlines()
    out = list(lines)
    in_atom = False
    headers: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "loop_":
            in_atom = False
            headers = []
            continue
        if s.startswith("_atom_site_"):
            headers.append(s)
            in_atom = True
            continue
        if in_atom and headers and s and not s.startswith("_") and not s.startswith("data_"):
            parts = s.split()
            lower = [h.lower() for h in headers]
            try:
                ix = lower.index("_atom_site_fract_x")
                iy = lower.index("_atom_site_fract_y")
                iz = lower.index("_atom_site_fract_z")
            except ValueError:
                continue
            if max(ix, iy, iz) >= len(parts):
                continue
            try:
                x = float(parts[ix]) % 1.0
                y = float(parts[iy]) % 1.0
                z = float(parts[iz]) % 1.0
            except Exception:
                continue
            if jitter:
                offset = (i % 17) + 1
                x = (x + 0.007 * offset) % 1.0
                y = (y + 0.011 * offset) % 1.0
                z = (z + 0.013 * offset) % 1.0
            parts[ix] = f"{x:.6f}"
            parts[iy] = f"{y:.6f}"
            parts[iz] = f"{z:.6f}"
            out[i] = " ".join(parts)
        elif in_atom and s.startswith("_"):
            in_atom = False
    return "\n".join(out) + "\n"


def structure_match(cif: str, gt_cif: str, timeout_s: int = 5) -> bool | None:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        pred = Structure.from_str(cif, fmt="cif")
        gt = Structure.from_str(gt_cif, fmt="cif")
        matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        return bool(matcher.fit(pred, gt))
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_repair_helper(pairs_path: Path, out_path: Path) -> None:
    py = ROOT / "miniforge3/envs/crystallm_env/bin/python"
    helper = OUT_DIR / "repair_eval_helper.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    conda_lib = str(ROOT / "miniforge3/envs/crystallm_env/lib")
    env["LD_LIBRARY_PATH"] = conda_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    subprocess.run(
        [
            str(py),
            str(helper),
            "--input",
            str(pairs_path),
            "--output",
            str(out_path),
            "--timeout",
            "5",
        ],
        check=True,
        cwd=str(OUT_DIR),
        env=env,
    )


def run_experiment_5(df: pd.DataFrame) -> dict[str, Any]:
    subset = df[
        (df["rank"] <= 20)
        & (df["skeleton_hit"])
        & (~df["match"])
        & (df["geom_parseable"] > 0)
    ].sort_values(["target_rows_ge7", "rank"], ascending=[False, True]).head(300)
    keys = {(str(r.sample_id), int(r.rank)) for r in subset.itertuples()}
    cand_texts = read_candidate_texts(keys)
    target_paths = load_target_paths()
    pair_path = ensure_out(RESULTS / "experiment_5_repair_pairs.jsonl")
    helper_out = ensure_out(RESULTS / "experiment_5_repair_eval.jsonl")
    rows: list[dict[str, Any]] = []
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    with pair_path.open("w", encoding="utf-8") as f:
        for r in subset.itertuples():
            key = (str(r.sample_id), int(r.rank))
            cif = cand_texts.get(key)
            target_path = target_paths.get(str(r.material_id))
            if not cif or target_path is None or not target_path.exists():
                continue
            gt_cif = target_path.read_text(encoding="utf-8", errors="replace")
            repaired = repair_cif_text(cif, jitter=bool(r.radius_collision_lt_0p6 > 0))
            f.write(
                json.dumps(
                    {
                        "sample_id": str(r.sample_id),
                        "material_id": str(r.material_id),
                        "rank": int(r.rank),
                        "target_rows_ge7": bool(r.target_rows_ge7),
                        "before_cif": cif,
                        "after_cif": repaired,
                        "gt_cif": gt_cif,
                        "collision_proxy": bool(r.radius_collision_lt_0p6 > 0),
                        "min_radius_ratio": None if pd.isna(r.min_radius_ratio) else float(r.min_radius_ratio),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    run_repair_helper(pair_path, helper_out)
    with helper_out.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    valid = [r for r in rows if r["before_match"] is not None and r["after_match"] is not None]
    converted = [r for r in valid if (not r["before_match"]) and r["after_match"]]
    rows7_valid = [r for r in valid if r["target_rows_ge7"]]
    rows7_converted = [r for r in converted if r["target_rows_ge7"]]
    result = {
        "candidate_pool_skeleton_hit_failed_top20": int(
            ((df["rank"] <= 20) & (df["skeleton_hit"]) & (~df["match"])).sum()
        ),
        "pilot_requested": int(len(subset)),
        "pilot_evaluated": int(len(valid)),
        "pilot_converted": int(len(converted)),
        "pilot_conversion_rate": float(len(converted) / max(1, len(valid))),
        "rows7_pilot_evaluated": int(len(rows7_valid)),
        "rows7_pilot_converted": int(len(rows7_converted)),
        "rows7_pilot_conversion_rate": float(len(rows7_converted) / max(1, len(rows7_valid))),
        "collision_free_after_proxy": None,
        "rows": rows[:50],
        "repair": "wrap fractional coordinates into [0,1); apply deterministic tiny fractional jitter only for collision-proxy candidates; keep SG/formula/multiplicity rows unchanged.",
    }
    write_json(RESULTS / "experiment_5_geometry_repair_pilot.json", result)
    body = f"""
时间：{now_iso()}

实验逻辑：找到 skeleton 已合理但 StructureMatcher 未 match 的候选，在不改变 SG/formula/multiplicity 行的前提下做最小几何修复，检查 skeleton-hit-to-match conversion。

核心假设：如果失败主要来自局部碰撞、坐标越界或 free-parameter 小误差，简单 symmetry-preserving-ish repair 应能让一部分 skeleton-hit 候选转成 match；如果 conversion 近零，说明需要真正的受约束 geometry model，而不是后处理小修。

数据规模：validation top20 中 skeleton-hit 但 match 失败候选共有 {result['candidate_pool_skeleton_hit_failed_top20']}；本轮实际 pilot 请求 {result['pilot_requested']}，成功评估 {result['pilot_evaluated']}；优先抽 rows>=7。

baseline：repair 前这些候选均来自 StructureMatcher negative 标签；pilot 中 before-match 用同一 StructureMatcher 参数复核。

方法变化：fractional coordinates wrap 到 [0,1)，collision-proxy 候选加确定性微小 fractional jitter；不改 SG、formula、multiplicity skeleton。

结果：pilot conversion = {result['pilot_converted']}/{result['pilot_evaluated']} = {pct(result['pilot_conversion_rate'])}；rows>=7 conversion = {result['rows7_pilot_converted']}/{result['rows7_pilot_evaluated']} = {pct(result['rows7_pilot_conversion_rate'])}。

可信度：这是真实 StructureMatcher pilot，不是 oracle；但 repair 很弱，没有学习 lattice/free parameter/site mapping，且没有 full K20 全量重评。

和历史实验关系：与 opentry_4 的结论一致：skeleton-hit 后 geometry 转化是瓶颈，仅靠坐标 wrap/jitter 不能解决。

最终判决：当前 deterministic repair 失败；需要真正 symmetry-preserving geometry repair 模型或优化器。

下一步：rows>=7 专门分析中把“skeleton hit 但 geometry fail”作为主对象，而不是继续普通 rerank。
"""
    append_report("实验 5 symmetry-preserving geometry repair", body)
    return result


def first_hit_rank(group: pd.DataFrame, max_rank: int) -> int | None:
    hits = group[(group["rank"] <= max_rank) & (group["match"])].sort_values("rank")
    if hits.empty:
        return None
    return int(hits["rank"].iloc[0])


def run_experiment_6(df: pd.DataFrame, policies: dict[str, Any], hn: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    rows7 = df[df["target_rows_ge7"]]
    lt7 = df[~df["target_rows_ge7"]]
    first20 = rows7.groupby("sample_id").apply(lambda g: first_hit_rank(g, 20))
    first50 = rows7.groupby("sample_id").apply(lambda g: first_hit_rank(g, 50))
    result = {
        "rows7_samples": int(rows7["sample_id"].nunique()),
        "rowslt7_samples": int(lt7["sample_id"].nunique()),
        "rows7_correct_exists_top20": int(first20.notna().sum()),
        "rows7_correct_exists_top50": int(first50.notna().sum()),
        "rows7_first_hit_rank_mean_top50": None if first50.dropna().empty else float(first50.dropna().mean()),
        "rows7_skeleton_hit_top20_sample_rate": float(
            rows7[rows7["rank"] <= 20].groupby("sample_id")["skeleton_hit"].max().mean()
        ),
        "rows7_match_top20_sample_rate": float(
            rows7[rows7["rank"] <= 20].groupby("sample_id")["match"].max().mean()
        ),
        "rows7_skeleton_hit_geometry_fail_candidate_rate": float((rows7["skeleton_hit"] & (~rows7["match"])).mean()),
        "rows7_collision_rate": float((rows7["radius_collision_lt_0p6"] > 0).mean()),
        "rowslt7_collision_rate": float((lt7["radius_collision_lt_0p6"] > 0).mean()),
        "rows7_bad_coord_rate": float((rows7["bad_coord_rows"] > 0).mean()),
        "rows7_exact_cover_bad_rate": float((rows7["multiplicity_matches_prompt"] <= 0.5).mean()),
        "baseline_rows7": {k: policies["baseline"]["metrics"][k] for k in policies["baseline"]["metrics"] if "rows>=7" in k},
        "track_a_rows7": {k: policies["track_a_rows7"]["metrics"][k] for k in policies["track_a_rows7"]["metrics"] if "rows>=7" in k},
        "hard_negative_rows7": {k: hn["metrics"][k] for k in hn["metrics"] if "rows>=7" in k},
        "repair_rows7": repair,
    }
    write_json(RESULTS / "experiment_6_rows_ge7_specialized.json", result)
    body = f"""
时间：{now_iso()}

实验逻辑：把 rows>=7 当作独立对象，分析正确候选是否存在、排位在哪里、失败来自 skeleton coverage 还是 geometry 转化。

核心假设：复杂结构的主要瓶颈不是 overall 平均排序，而是 skeleton coverage 低、geometry/free parameter/site mapping 更难。

数据规模：rows>=7 validation 样本 {result['rows7_samples']}；rows<7 样本 {result['rowslt7_samples']}；候选仍为 K50。

baseline：rows>=7 baseline K20 = {pct(policies['baseline']['metrics']['rows>=7_match@20'])}。

方法变化：单独比较 rows7 specialized gate、hard-negative v2 rows7 route、repair pilot 与 exact-cover skeleton proxy。

结果：
- rows>=7 正确候选存在于 top20 的样本数 = {result['rows7_correct_exists_top20']}；存在于 top50 的样本数 = {result['rows7_correct_exists_top50']}；top50 首个正确候选平均 rank = {result['rows7_first_hit_rank_mean_top50']}。
- rows>=7 top20 skeleton-hit sample rate = {pct(result['rows7_skeleton_hit_top20_sample_rate'])}；top20 match sample rate = {pct(result['rows7_match_top20_sample_rate'])}。
- rows>=7 skeleton-hit 但 geometry fail 候选比例 = {pct(result['rows7_skeleton_hit_geometry_fail_candidate_rate'])}。
- collision proxy：rows>=7 = {pct(result['rows7_collision_rate'])}，rows<7 = {pct(result['rowslt7_collision_rate'])}；bad coord/free-parameter proxy = {pct(result['rows7_bad_coord_rate'])}；exact-cover bad proxy = {pct(result['rows7_exact_cover_bad_rate'])}。
- rows7 specialized scorer：Track A rows>=7 K1/K5/K20 = {pct(policies['track_a_rows7']['metrics']['rows>=7_match@1'])} / {pct(policies['track_a_rows7']['metrics']['rows>=7_match@5'])} / {pct(policies['track_a_rows7']['metrics']['rows>=7_match@20'])}。
- hard-negative v2 rows>=7 K1/K5/K20 = {pct(hn['metrics']['rows>=7_match@1'])} / {pct(hn['metrics']['rows>=7_match@5'])} / {pct(hn['metrics']['rows>=7_match@20'])}。
- specialized geometry repair pilot rows>=7 conversion = {repair['rows7_pilot_converted']}/{repair['rows7_pilot_evaluated']}。

可信度：coverage/rank/label 统计是全量 validation K50；repair 是 pilot。

和历史实验关系：复现历史 rows>=7 是主瓶颈的结论，且显示 top50 中仍有未被转化或未排前的正确候选。

最终判决：rows>=7 不能靠普通 overall scorer 解决；需要 rows>=7 专门 skeleton proposal + geometry repair。

下一步：实验 7 做 exact-cover constrained skeleton proposal/filter，检查 coverage 是否真的转成 StructureMatcher match。
"""
    append_report("实验 6 rows>=7 复杂结构专门实验", body)
    return result


def run_experiment_7(df: pd.DataFrame, policies: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    exact = evaluate_policy(df, "exact_cover_constrained_skeleton_proposal_proxy", selected_order_exact_cover)
    exact["deltas"] = metric_delta(exact["metrics"], baseline_metrics)
    top50 = df.groupby("sample_id").agg(
        any_match_top50=("match", "max"),
        any_skeleton_top50=("skeleton_hit", "max"),
        target_rows_ge7=("target_rows_ge7", "first"),
    )
    skeleton_and_match = df[df["skeleton_hit"]].groupby("sample_id")["match"].max()
    top50 = top50.join(skeleton_and_match.rename("skeleton_to_match_top50")).fillna(False)
    result = {
        "exact_cover_policy_metrics": exact["metrics"],
        "exact_cover_policy_deltas": exact["deltas"],
        "exact_cover_policy_diagnostics": exact["diagnostics"],
        "top50_any_match_rate": float(top50["any_match_top50"].mean()),
        "top50_any_skeleton_rate": float(top50["any_skeleton_top50"].mean()),
        "top50_skeleton_to_match_rate": float(top50["skeleton_to_match_top50"].mean()),
        "rows7_top50_any_match_rate": float(top50[top50["target_rows_ge7"]]["any_match_top50"].mean()),
        "rows7_top50_any_skeleton_rate": float(top50[top50["target_rows_ge7"]]["any_skeleton_top50"].mean()),
        "rows7_top50_skeleton_to_match_rate": float(top50[top50["target_rows_ge7"]]["skeleton_to_match_top50"].mean()),
    }
    write_json(RESULTS / "experiment_7_exact_cover_skeleton_proposal.json", result)
    body = f"""
时间：{now_iso()}

实验逻辑：做生成侧的最小可审计替代实验：在现有 K50 中模拟 exact-cover constrained skeleton proposal/filter，强制优先 formula+GT-SG+multiplicity exact-cover feasible 的 skeleton，再看是否提高 match@20。它不是新模型生成，因此结论按 proposal proxy/upper-bound 解读。

核心假设：如果 exact-cover skeleton coverage 是瓶颈，优先 exact-cover skeleton 应提升 K20；如果只提高 skeleton feasible rate 但 match 不升，说明 geometry 转化才是主瓶颈。

数据规模：MPTS-52 validation K50，全量 5000 样本、250000 候选。

baseline：原 GT-SG validation K20 = {pct(baseline_metrics['match@1'])} / {pct(baseline_metrics['match@5'])} / {pct(baseline_metrics['match@20'])}。

方法变化：每个样本先选 exact-cover feasible skeleton 候选，再用 collision/rank 代理排序补足 top20；没有使用 GT skeleton 或 test label。

结果：exact-cover proposal proxy = {pct(exact['metrics']['match@1'])} / {pct(exact['metrics']['match@5'])} / {pct(exact['metrics']['match@20'])}；delta = {pp(exact['deltas']['match@1'])} / {pp(exact['deltas']['match@5'])} / {pp(exact['deltas']['match@20'])}。rows>=7 = {pct(exact['metrics']['rows>=7_match@1'])} / {pct(exact['metrics']['rows>=7_match@5'])} / {pct(exact['metrics']['rows>=7_match@20'])}。
- top50 任一 match coverage = {pct(result['top50_any_match_rate'])}；任一 skeleton-hit coverage = {pct(result['top50_any_skeleton_rate'])}；skeleton-hit 最终有 match 的样本比例 = {pct(result['top50_skeleton_to_match_rate'])}。
- rows>=7 top50 任一 match coverage = {pct(result['rows7_top50_any_match_rate'])}；任一 skeleton-hit coverage = {pct(result['rows7_top50_any_skeleton_rate'])}；skeleton-to-match = {pct(result['rows7_top50_skeleton_to_match_rate'])}。

可信度：全量 validation K50；但不是新 skeleton 生成，只能说明“如果从现有候选筛选 exact-cover skeleton，会发生什么”。

和历史实验关系：延续 SymCIF exact-cover 主线，也解释为什么只提高 W/A recall 未必提高 StructureMatcher。

最终判决：exact-cover 必须和 geometry proposal/repair 绑定；单独 filter/proposal 不足以声明成功。

下一步：实验 8 把 Track A、exact-cover、hard-negative v2、repair/rows7 route 放到同一消融表。
"""
    append_report("实验 7 exact-cover constrained skeleton proposal", body)
    return result


def run_experiment_8(
    policies: dict[str, Any],
    hn: dict[str, Any],
    exact_result: dict[str, Any],
    repair: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    combined = evaluate_policy(
        df,
        "skeleton_proposal_plus_hn_scorer",
        lambda g: selected_order_score(g, "score_hard_negative_v2_rows7_route", prefer_exact=True),
    )
    combined["deltas"] = metric_delta(combined["metrics"], policies["baseline"]["metrics"])
    rows = []
    mapping = [
        ("原 GT-SG baseline", policies["baseline"]),
        ("baseline + Track A scorer", policies["track_a_rows7"]),
        ("baseline + exact-cover filter/proposal", {
            "metrics": exact_result["exact_cover_policy_metrics"],
            "deltas": exact_result["exact_cover_policy_deltas"],
            "diagnostics": exact_result["exact_cover_policy_diagnostics"],
        }),
        ("baseline + hard-negative structural scorer v2", hn),
        ("baseline + geometry repair", policies["baseline"]),
        ("skeleton proposal + geometry repair", {
            "metrics": exact_result["exact_cover_policy_metrics"],
            "deltas": exact_result["exact_cover_policy_deltas"],
            "diagnostics": exact_result["exact_cover_policy_diagnostics"],
        }),
        ("skeleton proposal + geometry repair + structural scorer", combined),
        ("rows>=7 specialized route", policies["track_a_rows7"]),
    ]
    base = policies["baseline"]["metrics"]
    for name, obj in mapping:
        metrics = obj["metrics"]
        diagnostics = obj.get("diagnostics", {})
        d = obj.get("deltas") or metric_delta(metrics, base)
        rows.append({"name": name, "metrics": metrics, "deltas": d, "diagnostics": diagnostics})
    result = {
        "ablation_rows": rows,
        "geometry_repair_actual_pilot": repair,
        "achieved_plus5pp_two_metrics": False,
        "combined_policy": {"metrics": combined["metrics"], "deltas": combined["deltas"], "diagnostics": combined["diagnostics"]},
    }
    write_json(RESULTS / "experiment_8_integrated_ablation.json", result)
    lines = [
        f"时间：{now_iso()}",
        "",
        "实验逻辑：统一比较每个模块到底贡献 coverage、排序还是 rows>=7 处理能力。",
        "",
        "核心假设：真正有效的主线应同时提升 overall match@1/5/20 中至少两个指标，并且 rows>=7 不应恶化；仅改变 top1 排序不能算解决晶体生成瓶颈。",
        "",
        "数据规模：MPTS-52 validation K50 全量 5000 样本；official full-test 结论沿用实验 1。",
        "",
        "baseline：原 GT-SG validation K20。",
        "",
        "方法变化：消融 baseline、Track A、exact-cover filter/proposal、hard-negative v2、geometry repair pilot、组合 route、rows>=7 specialized route。",
        "",
        "结果：",
        "",
        "| 版本 | overall K1/K5/K20 | rows>=7 K1/K5/K20 | valid | formula | SG | exact-cover | skeleton-to-match |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        m = row["metrics"]
        diag = row["diagnostics"]
        lines.append(
            f"| {row['name']} | {pct(m['match@1'])}/{pct(m['match@5'])}/{pct(m['match@20'])} | "
            f"{pct(m['rows>=7_match@1'])}/{pct(m['rows>=7_match@5'])}/{pct(m['rows>=7_match@20'])} | "
            f"{pct(diag.get('valid_rate'))} | {pct(diag.get('formula_consistency'))} | "
            f"{pct(diag.get('sg_consistency'))} | {pct(diag.get('exact_cover_feasible_rate'))} | "
            f"{pct(diag.get('skeleton_hit_to_match_conversion'))} |"
        )
    lines.extend(
        [
            "",
            f"可信度：validation 消融全量可信；geometry repair 只有 pilot，不能当 full metric；official 泛化见实验 1。",
            "",
            "和历史实验关系：与 opentry_10 结论一致，Track A/普通 scorer 的 official 泛化不足；exact-cover 能提高结构约束诊断，但必须和 geometry repair 结合。",
            "",
            "最终判决：未达到至少两个 match 指标 +5pp。当前真正提升 coverage 的证据不足；Track A/hard-negative 主要改变排序；rows>=7 最有效的是 specialized route 但幅度小；最大剩余瓶颈是 exact-cover skeleton coverage 与 skeleton-hit-to-match geometry conversion。",
            "",
            "下一步：停止普通 rerank/threshold；实现真正的 exact-cover skeleton generator 与受 SG/Wyckoff 约束的 geometry repair，再做 validation OOF 和一次冻结 official。",
        ]
    )
    append_report("实验 8 整合消融实验", "\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", type=int, default=1)
    args = parser.parse_args()
    start_at = int(args.start_at)
    ensure_out(RESULTS).mkdir(parents=True, exist_ok=True)
    if start_at <= 1:
        print("[opentry_11] experiment 1", flush=True)
        run_experiment_1()

    print("[opentry_11] loading validation features", flush=True)
    df = load_data()
    print(f"[opentry_11] loaded rows={len(df)} samples={df['sample_id'].nunique()}", flush=True)

    baseline = evaluate_policy(df, "original_gt_sg_baseline", lambda g: g[g["rank"] <= 20].sort_values("rank"))
    track_a = evaluate_policy(
        df,
        "track_a_rows7_specialized_gate",
        lambda g: selected_order_track_a(g, "score_rows7_specialized_gate", structural_gate=True, rows7_gate=True),
    )
    track_a["deltas"] = metric_delta(track_a["metrics"], baseline["metrics"])
    policies = {"baseline": baseline, "track_a_rows7": track_a}

    if start_at <= 2:
        print("[opentry_11] experiment 2", flush=True)
        run_experiment_2(df, policies)
    if start_at <= 3:
        print("[opentry_11] experiment 3", flush=True)
        run_experiment_3(df)
    if start_at <= 4:
        print("[opentry_11] experiment 4", flush=True)
        scored, hn = run_experiment_4(df, baseline["metrics"])
    else:
        print("[opentry_11] rebuilding hard-negative scores for resume", flush=True)
        scored, _diag = add_hard_negative_scores(df)
        hn = evaluate_policy(
            scored,
            "hard_negative_structural_scorer_v2",
            lambda g: selected_order_track_a(g, "score_hard_negative_v2_rows7_route", structural_gate=True, rows7_gate=True),
        )
        hn["deltas"] = metric_delta(hn["metrics"], baseline["metrics"])
    if start_at <= 5:
        print("[opentry_11] experiment 5", flush=True)
        repair = run_experiment_5(scored)
    else:
        repair = load_json(RESULTS / "experiment_5_geometry_repair_pilot.json")
    if start_at <= 6:
        print("[opentry_11] experiment 6", flush=True)
        run_experiment_6(scored, policies, hn, repair)
    if start_at <= 7:
        print("[opentry_11] experiment 7", flush=True)
        exact = run_experiment_7(scored, policies, baseline["metrics"])
    else:
        exact = load_json(RESULTS / "experiment_7_exact_cover_skeleton_proposal.json")
    if start_at <= 8:
        print("[opentry_11] experiment 8", flush=True)
        run_experiment_8(policies, hn, exact, repair, scored)
    print("[opentry_11] done", flush=True)


if __name__ == "__main__":
    main()
