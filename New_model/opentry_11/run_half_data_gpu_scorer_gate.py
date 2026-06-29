#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_11"
RESULTS = OUT_DIR / "results"
REPORT = OUT_DIR / "GPT_REVIEW_BUNDLE.md"
STRUCT_FEATURES = ROOT / "model/std_way/track_a_mpts52/outputs/structural_features.jsonl.gz"

SEED = 20260628
BUDGETS = (1, 5, 20)
GATE_PP = 0.05


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
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n" + f"## opentry_11 追加实验：{title}\n\n" + body.strip() + "\n")


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ScorerMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_data() -> pd.DataFrame:
    df = pd.read_json(STRUCT_FEATURES, lines=True, compression="gzip")
    df["target_rows_ge7"] = df["target_rows_ge7"].astype(bool)
    df["match"] = df["match"].fillna(False).astype(bool)
    df["rank"] = df["rank"].astype(int)
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
    complex_proxy = df[df["rank"] <= 20].groupby("sample_id")["candidate_rows_ge7"].max().rename("complex_proxy")
    df = df.merge(complex_proxy, on="sample_id", how="left")
    df["complex_proxy"] = df["complex_proxy"].fillna(0).astype(bool)
    return df


def half_sample_ids(df: pd.DataFrame) -> set[str]:
    sample_table = df[["sample_id", "target_rows_ge7"]].drop_duplicates().copy()
    selected: set[str] = set()
    for _, group in sample_table.groupby("target_rows_ge7", sort=False):
        ids = sorted(str(x) for x in group["sample_id"].tolist())
        selected.update(sid for i, sid in enumerate(ids) if i % 2 == 0)
    return selected


def select_stage_df(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage == "full":
        return df.copy()
    if stage != "half":
        raise ValueError(stage)
    selected = half_sample_ids(df)
    return df[df["sample_id"].astype(str).isin(selected)].copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    allow = [
        "parse_ok_label",
        "valid_label",
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
        "skeleton_hit",
        "formula_sg_ok",
        "complex_proxy",
    ]
    return [c for c in allow if c in df.columns]


def build_xy(
    df: pd.DataFrame,
    cols: list[str],
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = df[cols].astype("float32").to_numpy(copy=True)
    x[~np.isfinite(x)] = np.nan
    if mean is None:
        mean = np.nanmean(x, axis=0)
        mean[~np.isfinite(mean)] = 0.0
    inds = np.where(np.isnan(x))
    if inds[0].size:
        x[inds] = np.take(mean, inds[1])
    if std is None:
        std = np.nanstd(x, axis=0)
        std[~np.isfinite(std)] = 1.0
        std[std < 1e-6] = 1.0
    x = ((x - mean) / std).astype("float32")
    y = df["match"].astype("float32").to_numpy()
    return x, y, mean.astype("float32"), std.astype("float32")


def hard_weights(df: pd.DataFrame, rows7_focus: bool) -> np.ndarray:
    neg = ~df["match"].to_numpy(dtype=bool)
    rank = df["rank"].to_numpy()
    weights = np.ones(len(df), dtype="float32")
    weights += ((neg) & (rank <= 20)).astype("float32") * 1.0
    weights += ((neg) & df["formula_sg_geometry_wrong"].to_numpy(dtype=bool)).astype("float32") * 1.2
    weights += ((neg) & df["skeleton_ok_geometry_wrong"].to_numpy(dtype=bool)).astype("float32") * 2.0
    weights += ((neg) & df["collision_free_wrong"].to_numpy(dtype=bool)).astype("float32") * 0.5
    if rows7_focus:
        weights *= np.where(df["target_rows_ge7"].to_numpy(dtype=bool), 2.0, 0.25).astype("float32")
    return weights


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cols: list[str],
    device: torch.device,
    rows7_focus: bool,
    epochs: int,
    fold_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train, y_train, mean, std = build_xy(train_df, cols)
    x_val, _, _, _ = build_xy(val_df, cols, mean=mean, std=std)
    weights = hard_weights(train_df, rows7_focus=rows7_focus)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight_value = min(20.0, neg / max(1.0, pos))

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype("float32")),
        torch.from_numpy(weights.astype("float32")),
    )
    generator = torch.Generator()
    generator.manual_seed(fold_seed)
    loader = DataLoader(
        dataset,
        batch_size=8192,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ScorerMLP(len(cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device),
        reduction="none",
    )
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        count = 0
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_fn(model(xb), yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        losses.append(total / max(1, count))

    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_val), 65536):
            xb = torch.from_numpy(x_val[start : start + 65536]).to(device)
            chunks.append(model(xb).detach().cpu().numpy())
    scores = np.concatenate(chunks).astype("float32")
    meta = {
        "train_candidates": int(len(train_df)),
        "train_samples": int(train_df["sample_id"].nunique()),
        "train_positive_candidates": int(pos),
        "pos_weight": float(pos_weight_value),
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
        "epochs": epochs,
    }
    return scores, meta


def sample_stats(ordered: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_id": str(ordered["sample_id"].iloc[0]),
        "target_rows_ge7": bool(ordered["target_rows_ge7"].iloc[0]),
    }
    for budget in BUDGETS:
        top = ordered.head(budget)
        out[f"hit@{budget}"] = bool(top["match"].any())
    return out


def summarize_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows7 = [r for r in rows if r["target_rows_ge7"]]
    out: dict[str, Any] = {"samples": len(rows), "rows>=7_samples": len(rows7)}
    for budget in BUDGETS:
        hits = sum(1 for r in rows if r[f"hit@{budget}"])
        hits7 = sum(1 for r in rows7 if r[f"hit@{budget}"])
        out[f"match@{budget}"] = float(hits / max(1, len(rows)))
        out[f"rows>=7_match@{budget}"] = float(hits7 / max(1, len(rows7)))
    return out


def selected_diagnostics(selected: pd.DataFrame) -> dict[str, Any]:
    skel = selected[selected["skeleton_hit"]]
    return {
        "selected_candidates": int(len(selected)),
        "valid_rate": float(selected["valid_label"].mean()),
        "formula_consistency": float(selected["formula_reduced_match"].mean()),
        "sg_consistency": float(selected["sg_number_match"].mean()),
        "exact_cover_feasible_rate": float(selected["skeleton_hit"].mean()),
        "collision_rate": float((selected["radius_collision_lt_0p6"] > 0).mean()),
        "skeleton_hit_to_match_conversion": None if len(skel) == 0 else float(skel["match"].mean()),
    }


def evaluate(df: pd.DataFrame, name: str, score_col: str | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for _, group in df.groupby("sample_id", sort=False):
        if score_col is None:
            ordered = group.sort_values(["rank"], ascending=[True]).head(20).copy()
        else:
            ordered = group.sort_values([score_col, "rank"], ascending=[False, True]).head(20).copy()
        ordered["selected_position"] = np.arange(1, len(ordered) + 1)
        rows.append(sample_stats(ordered))
        selected_frames.append(ordered)
    selected = pd.concat(selected_frames, ignore_index=True)
    return {"name": name, "metrics": summarize_samples(rows), "diagnostics": selected_diagnostics(selected)}


def metric_delta(metrics: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    keys = [f"match@{b}" for b in BUDGETS] + [f"rows>=7_match@{b}" for b in BUDGETS]
    return {k: float(metrics[k] - base[k]) for k in keys}


def fmt_overall(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"match@{b}"]) for b in BUDGETS)


def fmt_rows7(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"rows>=7_match@{b}"]) for b in BUDGETS)


def fmt_delta(d: dict[str, float], prefix: str = "") -> str:
    return " / ".join(pp(d[f"{prefix}match@{b}"]) for b in BUDGETS)


def gate_pass(delta: dict[str, float]) -> tuple[bool, dict[str, Any]]:
    overall_good = sum(1 for b in BUDGETS if delta[f"match@{b}"] >= GATE_PP)
    rows7_good = (delta["rows>=7_match@5"] >= GATE_PP) and (delta["rows>=7_match@20"] >= GATE_PP)
    passed = overall_good >= 2 or rows7_good
    return passed, {"overall_metrics_over_5pp": overall_good, "rows7_k5_k20_over_5pp": rows7_good, "threshold": GATE_PP}


def run_stage(full_df: pd.DataFrame, stage: str, device: torch.device, epochs: int) -> dict[str, Any]:
    df = select_stage_df(full_df, stage).reset_index(drop=True)
    cols = feature_columns(df)
    general_scores = np.full(len(df), np.nan, dtype="float32")
    rows7_scores = np.full(len(df), np.nan, dtype="float32")
    fold_meta: list[dict[str, Any]] = []

    for fold in sorted(df["fold"].unique()):
        train_df = df[df["fold"] != fold].copy()
        val_df = df[df["fold"] == fold].copy()
        if len(train_df) == 0 or len(val_df) == 0:
            continue
        scores, meta = train_one_fold(
            train_df,
            val_df,
            cols,
            device,
            rows7_focus=False,
            epochs=epochs,
            fold_seed=SEED + int(fold),
        )
        general_scores[val_df.index.to_numpy()] = scores
        meta.update({"fold": int(fold), "model": "general"})
        fold_meta.append(meta)
        print(f"[{stage}] fold {fold} general done loss {meta['loss_first']:.4f}->{meta['loss_last']:.4f}", flush=True)

        rows7_train = train_df[train_df["target_rows_ge7"]].copy()
        if len(rows7_train) > 0:
            scores7, meta7 = train_one_fold(
                rows7_train,
                val_df,
                cols,
                device,
                rows7_focus=True,
                epochs=epochs + 2,
                fold_seed=SEED + 100 + int(fold),
            )
            rows7_scores[val_df.index.to_numpy()] = scores7
            meta7.update({"fold": int(fold), "model": "rows>=7_specialized"})
            fold_meta.append(meta7)
            print(f"[{stage}] fold {fold} rows>=7 done loss {meta7['loss_first']:.4f}->{meta7['loss_last']:.4f}", flush=True)

    df = df.copy()
    df["gpu_general_score"] = general_scores
    rows7_route_score = np.where(df["complex_proxy"].to_numpy(dtype=bool), rows7_scores, general_scores)
    rows7_route_score = np.where(np.isfinite(rows7_route_score), rows7_route_score, general_scores)
    df["gpu_rows7_route_score"] = rows7_route_score

    baseline = evaluate(df, "baseline_rank", None)
    general = evaluate(df, "gpu_general_hard_negative_mlp", "gpu_general_score")
    rows7 = evaluate(df, "gpu_rows7_specialized_route", "gpu_rows7_route_score")
    general_delta = metric_delta(general["metrics"], baseline["metrics"])
    rows7_delta = metric_delta(rows7["metrics"], baseline["metrics"])
    general_pass, general_gate = gate_pass(general_delta)
    rows7_pass, rows7_gate = gate_pass(rows7_delta)

    result = {
        "time": now_iso(),
        "stage": stage,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "seed": SEED,
        "data": {
            "candidates": int(len(df)),
            "samples": int(df["sample_id"].nunique()),
            "rows>=7_samples": int(df[["sample_id", "target_rows_ge7"]].drop_duplicates()["target_rows_ge7"].sum()),
            "features": cols,
            "feature_count": len(cols),
            "half_selection": "stratified by target_rows_ge7, sorted sample_id, even index" if stage == "half" else "full",
        },
        "fold_meta": fold_meta,
        "policies": {"baseline": baseline, "gpu_general": general, "gpu_rows7_route": rows7},
        "deltas_vs_baseline": {"gpu_general": general_delta, "gpu_rows7_route": rows7_delta},
        "gate": {
            "gpu_general": {"pass": general_pass, **general_gate},
            "gpu_rows7_route": {"pass": rows7_pass, **rows7_gate},
        },
    }

    write_json(RESULTS / f"experiment_4_6_gpu_scorer_{stage}.json", result)
    score_path = ensure_out(RESULTS / f"experiment_4_6_gpu_scorer_{stage}_scores.jsonl.gz")
    with gzip.open(score_path, "wt", encoding="utf-8") as f:
        for row in df[["sample_id", "material_id", "rank", "fold", "target_rows_ge7", "complex_proxy", "match", "gpu_general_score", "gpu_rows7_route_score"]].itertuples(index=False):
            f.write(json.dumps(row._asdict(), ensure_ascii=False) + "\n")
    return result


def append_stage_reports(result: dict[str, Any], full_planned: bool) -> None:
    stage = result["stage"]
    data = result["data"]
    base = result["policies"]["baseline"]["metrics"]
    gen = result["policies"]["gpu_general"]["metrics"]
    row = result["policies"]["gpu_rows7_route"]["metrics"]
    gen_d = result["deltas_vs_baseline"]["gpu_general"]
    row_d = result["deltas_vs_baseline"]["gpu_rows7_route"]
    gen_diag = result["policies"]["gpu_general"]["diagnostics"]
    row_diag = result["policies"]["gpu_rows7_route"]["diagnostics"]
    gen_gate = result["gate"]["gpu_general"]
    row_gate = result["gate"]["gpu_rows7_route"]
    full_note = "half-data gate 已通过，将继续补跑全量。" if full_planned else "half-data gate 未通过，因此不补跑全量，避免把弱信号扩成 full run。"

    append_report(
        f"实验 4C GPU hard-negative scorer half-data gate（{stage}）",
        f"""
时间：{result['time']}

实验逻辑：按更新后的目标，先判断实验 4 是否需要 GPU。实验 4 涉及训练 structural scorer，因此需要 GPU 补充；本轮先用 deterministic 50% validation samples，而不是小 pilot 或直接全量。

核心假设：如果 hard-negative neural scorer 真学到晶体学结构错误，half-data OOF 至少应在两个 overall match 指标达到 +5pp，或在 rows>=7 match@5/match@20 同时达到 +5pp。

数据规模：stage={stage}；样本 {data['samples']}；候选 {data['candidates']}；rows>=7 样本 {data['rows>=7_samples']}；特征 {data['feature_count']} 个；设备 {result['cuda_device_name']}；PyTorch {result['torch_version']}。half 子集按 rows>=7/rows<7 分层后 sample_id 稳定排序隔位抽取。

baseline：原 GT-SG rank 顺序 = {fmt_overall(base)}；rows>=7 = {fmt_rows7(base)}。

方法变化：GPU MLP + weighted BCE；hard-negative 加权 top20 错误、formula+SG 正确但 geometry 错、skeleton-hit 但不 match、collision-free 但不 match。特征不使用 target_rows_ge7、rank/rank_inv/rank_le*、CIF 字符数/行数、atom rows 作为输入。

结果：GPU general scorer = {fmt_overall(gen)}；delta = {fmt_delta(gen_d)}。rows>=7 = {fmt_rows7(gen)}；rows>=7 delta = {fmt_delta(gen_d, 'rows>=7_')}。

诊断：valid rate = {pct(gen_diag['valid_rate'])}；formula consistency = {pct(gen_diag['formula_consistency'])}；SG consistency = {pct(gen_diag['sg_consistency'])}；exact-cover feasible = {pct(gen_diag['exact_cover_feasible_rate'])}；skeleton-hit-to-match conversion = {pct(gen_diag['skeleton_hit_to_match_conversion'])}。

可信度：这是实际 GPU 训练 + 5-fold OOF；但仍只在已有候选池内排序，不是新 CIF 生成，也不是 official full-test。

和历史实验关系：替代前一轮 CPU sklearn 实验 4 的严格 half-data GPU 版本；直接检验 scorer 是否只是普通 rerank。

最终判决：gate_pass={gen_gate['pass']}；overall >= +5pp 指标数={gen_gate['overall_metrics_over_5pp']}；rows>=7 K5/K20 是否均 >= +5pp={gen_gate['rows7_k5_k20_over_5pp']}。{full_note}

下一步：若未过 gate，实验 4 不进入全量；后续优先转向 geometry repair 和 skeleton proposal。
""",
    )

    append_report(
        f"实验 6C GPU rows>=7 specialized scorer half-data gate（{stage}）",
        f"""
时间：{now_iso()}

实验逻辑：实验 6 的分析部分不需要 GPU，但 specialized scorer 属于训练实验，因此必须先做 half-data GPU gate。本轮 rows>=7 model 只用训练 fold 中 target rows>=7 样本训练；推理时不用 target_rows_ge7，而用候选池里的 complex_proxy 决定是否走 specialized route。

核心假设：如果 rows>=7 主要是排序/结构质量识别问题，specialized route 应显著提升 rows>=7 match@5/match@20；如果仍不过 +5pp，说明复杂结构瓶颈主要在 skeleton coverage 或 geometry conversion。

数据规模：stage={stage}；样本 {data['samples']}；候选 {data['candidates']}；rows>=7 样本 {data['rows>=7_samples']}。

baseline：rows>=7 原 GT-SG rank 顺序 = {fmt_rows7(base)}。

方法变化：GPU rows>=7 specialized MLP；非 complex_proxy 样本回退 general scorer，避免把简单结构强行交给 rows>=7 模型。

结果：rows>=7 specialized route overall = {fmt_overall(row)}；overall delta = {fmt_delta(row_d)}。rows>=7 = {fmt_rows7(row)}；rows>=7 delta = {fmt_delta(row_d, 'rows>=7_')}。

诊断：valid rate = {pct(row_diag['valid_rate'])}；formula consistency = {pct(row_diag['formula_consistency'])}；SG consistency = {pct(row_diag['sg_consistency'])}；exact-cover feasible = {pct(row_diag['exact_cover_feasible_rate'])}；skeleton-hit-to-match conversion = {pct(row_diag['skeleton_hit_to_match_conversion'])}。

可信度：实际 GPU 训练 + OOF；但它仍是候选池内排序，不能声称提高 skeleton 或 geometry coverage。

和历史实验关系：这是实验 6 中 rows>=7 scorer 的严格 half-data GPU 补充；用于判断是否值得全量扩展。

最终判决：gate_pass={row_gate['pass']}；overall >= +5pp 指标数={row_gate['overall_metrics_over_5pp']}；rows>=7 K5/K20 是否均 >= +5pp={row_gate['rows7_k5_k20_over_5pp']}。{full_note}

下一步：若未过 gate，rows>=7 路线不能继续普通 scorer，应转向 exact-cover skeleton proposal + symmetry-preserving geometry repair。
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--no-auto-full", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    seed_everything(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    full_df = load_data()
    half_result = run_stage(full_df, "half", device, epochs=args.epochs)
    half_pass = bool(half_result["gate"]["gpu_general"]["pass"] or half_result["gate"]["gpu_rows7_route"]["pass"])
    append_stage_reports(half_result, full_planned=half_pass and not args.no_auto_full)

    final = {"half": half_result, "full": None, "full_run_reason": None}
    if half_pass and not args.no_auto_full:
        full_result = run_stage(full_df, "full", device, epochs=args.epochs)
        append_stage_reports(full_result, full_planned=True)
        final["full"] = full_result
        final["full_run_reason"] = "half-data gate passed"
    else:
        final["full_run_reason"] = "half-data gate failed or --no-auto-full set"
    write_json(RESULTS / "experiment_4_6_gpu_scorer_half_gate_summary.json", final)
    print(json.dumps({"half_gate_pass": half_pass, "full_run_reason": final["full_run_reason"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
