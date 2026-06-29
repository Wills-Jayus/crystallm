#!/usr/bin/env python3
from __future__ import annotations

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

TRACK_A = ROOT / "model/std_way/track_a_mpts52"
STRUCT_FEATURES = TRACK_A / "outputs/structural_features.jsonl.gz"

BUDGETS = (1, 5, 20)
MAX_RANK = 50
SEED = 20260628


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


class MLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
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
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "sample_id",
        "material_id",
        "fold",
        "match",
        "rmsd",
        "label_status",
        "rank",
        "rank_inv",
        "rank_le5",
        "rank_le20",
        "rank_le30",
        "rank_tail31_50",
        "gen_index",
        "cif_num_chars",
        "cif_num_lines",
    }
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def build_xy(df: pd.DataFrame, cols: list[str], mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def hard_weights(df: pd.DataFrame, *, rows7_focus: bool) -> np.ndarray:
    weights = np.ones(len(df), dtype="float32")
    neg = ~df["match"].to_numpy(dtype=bool)
    rank = df["rank"].to_numpy()
    weights += ((neg) & (rank <= 20)).astype("float32") * 1.0
    weights += ((neg) & df["formula_sg_geometry_wrong"].to_numpy(dtype=bool)).astype("float32") * 1.5
    weights += ((neg) & df["skeleton_ok_geometry_wrong"].to_numpy(dtype=bool)).astype("float32") * 2.0
    weights += ((neg) & df["collision_free_wrong"].to_numpy(dtype=bool)).astype("float32") * 0.5
    if rows7_focus:
        weights *= np.where(df["target_rows_ge7"].to_numpy(dtype=bool), 2.0, 0.35).astype("float32")
    return weights


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cols: list[str],
    device: torch.device,
    *,
    rows7_focus: bool,
    epochs: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train, y_train, mean, std = build_xy(train_df, cols)
    x_val, _, _, _ = build_xy(val_df, cols, mean=mean, std=std)
    weights = hard_weights(train_df, rows7_focus=rows7_focus)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight_value = negatives / max(1.0, positives)

    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype("float32")),
        torch.from_numpy(weights.astype("float32")),
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + int(train_df["fold"].min()) + (17 if rows7_focus else 0))
    loader = DataLoader(ds, batch_size=8192, shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")

    model = MLP(len(cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        total = 0.0
        count = 0
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = (loss_fn(logits, yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        losses.append(total / max(1, count))

    model.eval()
    val_scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_val), 65536):
            xb = torch.from_numpy(x_val[start : start + 65536]).to(device)
            val_scores.append(model(xb).detach().cpu().numpy())
    scores = np.concatenate(val_scores).astype("float32")
    meta = {
        "epochs": epochs,
        "train_rows": int(len(train_df)),
        "train_samples": int(train_df["sample_id"].nunique()),
        "train_positive_rows": int(positives),
        "pos_weight": float(pos_weight_value),
        "loss_first": float(losses[0]) if losses else None,
        "loss_last": float(losses[-1]) if losses else None,
    }
    return scores, meta


def sample_stats(ordered: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_id": str(ordered["sample_id"].iloc[0]),
        "target_rows_ge7": bool(ordered["target_rows_ge7"].iloc[0]),
        "selected_ranks": [int(x) for x in ordered["rank"].tolist()],
    }
    for budget in BUDGETS:
        top = ordered.head(budget)
        out[f"hit@{budget}"] = bool(top["match"].any())
        rmsd = pd.to_numeric(top.loc[top["match"], "rmsd"], errors="coerce").dropna()
        out[f"rmsd@{budget}"] = None if rmsd.empty else float(rmsd.min())
    return out


def summarize_sample_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows7 = [r for r in rows if r["target_rows_ge7"]]
    out: dict[str, Any] = {"samples": len(rows), "rows>=7_samples": len(rows7)}
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
        "rows>=7_selected_candidates": int(selected["target_rows_ge7"].sum()),
    }


def evaluate_policy(df: pd.DataFrame, score_col: str | None, name: str) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    sort_cols = ["rank"] if score_col is None else [score_col, "rank"]
    ascending = [True] if score_col is None else [False, True]
    for _, group in df.groupby("sample_id", sort=False):
        ordered = group.sort_values(sort_cols, ascending=ascending).head(20).copy()
        ordered["selected_position"] = np.arange(1, len(ordered) + 1)
        sample_rows.append(sample_stats(ordered))
        selected_frames.append(ordered)
    selected = pd.concat(selected_frames, ignore_index=True)
    return {
        "name": name,
        "metrics": summarize_sample_rows(sample_rows),
        "diagnostics": selected_diagnostics(selected),
        "sample_rows_head": sample_rows[:20],
    }


def metric_delta(metrics: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    keys = [f"match@{b}" for b in BUDGETS] + [f"rows>=7_match@{b}" for b in BUDGETS]
    return {k: float(metrics[k] - base[k]) for k in keys}


def format_metrics(metrics: dict[str, Any]) -> str:
    return " / ".join(pct(metrics[f"match@{b}"]) for b in BUDGETS)


def format_rows7(metrics: dict[str, Any]) -> str:
    return " / ".join(pct(metrics[f"rows>=7_match@{b}"]) for b in BUDGETS)


def format_delta(delta: dict[str, float], prefix: str = "") -> str:
    keys = [f"{prefix}match@{b}" for b in BUDGETS]
    return " / ".join(pp(delta[k]) for k in keys)


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    seed_everything(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    cols = feature_columns(df)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    general_scores = np.full(len(df), np.nan, dtype="float32")
    rows7_scores = np.full(len(df), np.nan, dtype="float32")
    fold_meta: list[dict[str, Any]] = []

    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold].copy()
        val = df[df["fold"] == fold].copy()

        scores, meta = train_model(train, val, cols, device, rows7_focus=False, epochs=16)
        general_scores[val.index.to_numpy()] = scores
        meta.update({"fold": int(fold), "model": "general"})
        fold_meta.append(meta)

        rows7_train = train[train["target_rows_ge7"]].copy()
        if len(rows7_train) > 0:
            scores7, meta7 = train_model(rows7_train, val, cols, device, rows7_focus=True, epochs=18)
            rows7_scores[val.index.to_numpy()] = scores7
            meta7.update({"fold": int(fold), "model": "rows>=7_specialized"})
            fold_meta.append(meta7)

    df["gpu_general_score"] = general_scores
    df["gpu_rows7_score"] = np.where(df["target_rows_ge7"], rows7_scores, general_scores)
    df["gpu_rows7_score"] = np.where(np.isfinite(df["gpu_rows7_score"]), df["gpu_rows7_score"], df["gpu_general_score"])

    baseline = evaluate_policy(df, None, "baseline_rank")
    general = evaluate_policy(df, "gpu_general_score", "gpu_general_hard_negative_mlp")
    rows7_route = evaluate_policy(df, "gpu_rows7_score", "gpu_rows7_specialized_route")

    result = {
        "time": now_iso(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "seed": SEED,
        "data": {
            "candidates": int(len(df)),
            "samples": int(df["sample_id"].nunique()),
            "rows>=7_samples": int(df.drop_duplicates("sample_id")["target_rows_ge7"].sum()),
            "features": cols,
            "feature_count": len(cols),
            "excluded_shallow_rank_features": ["rank", "rank_inv", "rank_le5", "rank_le20", "rank_le30", "rank_tail31_50"],
        },
        "fold_meta": fold_meta,
        "policies": {
            "baseline": baseline,
            "gpu_general_hard_negative_mlp": general,
            "gpu_rows7_specialized_route": rows7_route,
        },
        "deltas_vs_baseline": {
            "gpu_general_hard_negative_mlp": metric_delta(general["metrics"], baseline["metrics"]),
            "gpu_rows7_specialized_route": metric_delta(rows7_route["metrics"], baseline["metrics"]),
        },
    }
    write_json(RESULTS / "experiment_4b_gpu_neural_hard_negative_scorer.json", result)

    score_path = ensure_out(RESULTS / "experiment_4b_gpu_neural_scores.jsonl.gz")
    with gzip.open(score_path, "wt", encoding="utf-8") as f:
        for row in df[["sample_id", "material_id", "rank", "fold", "target_rows_ge7", "match", "gpu_general_score", "gpu_rows7_score"]].itertuples(index=False):
            f.write(
                json.dumps(
                    {
                        "sample_id": row.sample_id,
                        "material_id": row.material_id,
                        "rank": int(row.rank),
                        "fold": int(row.fold),
                        "target_rows_ge7": bool(row.target_rows_ge7),
                        "match": bool(row.match),
                        "gpu_general_score": float(row.gpu_general_score),
                        "gpu_rows7_score": float(row.gpu_rows7_score),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    base_m = baseline["metrics"]
    gen_m = general["metrics"]
    row_m = rows7_route["metrics"]
    gen_d = result["deltas_vs_baseline"]["gpu_general_hard_negative_mlp"]
    row_d = result["deltas_vs_baseline"]["gpu_rows7_specialized_route"]
    gen_diag = general["diagnostics"]
    row_diag = rows7_route["diagnostics"]

    append_report(
        "实验 4B GPU hard-negative neural structural scorer 补充",
        f"""
时间：{result['time']}

实验逻辑：针对用户指出的“实验 4 应该是 GPU 训练而不是 CPU sklearn”问题，补做一个明确使用 PyTorch/CUDA 的 hard-negative neural structural scorer。它仍然不是 CrystaLLM/SymCIF 生成模型训练，也不生成新 CIF；它只回答一个更窄的问题：在已有 MPTS-52 validation K50 候选上，去掉 rank/rank_inv/rank_le* 等浅层 rank 特征后，GPU MLP 是否能用 formula/SG/exact-cover/collision/local geometry 等结构特征把 hard negatives 排下去。

核心假设：如果结构特征足够，GPU scorer 在 5-fold OOF 上应同时提升 match@5 和 match@20；如果只小幅提升或 rows>=7 仍弱，说明瓶颈仍不是普通 scorer。

数据规模：候选 {len(df)} 个，样本 {df['sample_id'].nunique()} 个，rows>=7 样本 {int(df.drop_duplicates('sample_id')['target_rows_ge7'].sum())} 个；5-fold OOF；特征数 {len(cols)}；设备 {result['cuda_device_name']}；PyTorch {result['torch_version']}。

baseline：原 GT-SG validation rank 顺序 = {format_metrics(base_m)}；rows>=7 = {format_rows7(base_m)}。

方法变化：用 GPU MLP + weighted BCE 训练 candidate-level scorer；hard-negative 加权包括 top20 错误候选、formula+SG 正确但 geometry 错、skeleton-hit 但 StructureMatcher 不 match、collision-free 但不 match。主模型不使用 rank、rank_inv、rank_le5/20/30、CIF 字符数/行数作为输入。

结果：GPU hard-negative MLP = {format_metrics(gen_m)}；delta = {format_delta(gen_d)}。rows>=7 = {format_rows7(gen_m)}；rows>=7 delta = {format_delta(gen_d, 'rows>=7_')}。

诊断：valid rate = {pct(gen_diag['valid_rate'])}；formula consistency = {pct(gen_diag['formula_consistency'])}；SG consistency = {pct(gen_diag['sg_consistency'])}；exact-cover feasible = {pct(gen_diag['exact_cover_feasible_rate'])}；skeleton-hit-to-match conversion = {pct(gen_diag['skeleton_hit_to_match_conversion'])}。

可信度：这是实际 GPU 训练和 OOF 评估，可信度高于前一轮 CPU sklearn 口径；但标签仍来自 validation StructureMatcher，因此它是 validation scorer 诊断，不是 official full-test 成功，也不是生成侧 coverage 提升。

和历史实验关系：它替代/补充实验 4 的 CPU sklearn 版本，边界更清楚：这是 GPU neural scorer 补充，不是主生成模型。

最终判决：如果 delta 未达到至少两个指标 +5pp，则 hard-negative neural scorer 仍不足以作为主线，只能作为辅助排序模块。

下一步：把同一训练框架改成 rows>=7 专门 route，并继续检查是否解决复杂结构瓶颈。
""",
    )

    append_report(
        "实验 6B GPU rows>=7 specialized scorer 补充",
        f"""
时间：{now_iso()}

实验逻辑：针对 prompt 中 rows>=7 必须独立实验的要求，在 GPU hard-negative scorer 基础上补做 rows>=7 专门训练。训练时只用 rows>=7 train folds 学 rows>=7 scorer；评估 route 对 rows>=7 样本使用 specialized score，对 rows<7 样本回退 general score。

核心假设：如果复杂结构主要是排序/结构质量识别问题，rows>=7 专门 GPU scorer 应明显提升 rows>=7 match@5/match@20；如果提升仍小，说明必须转向 skeleton proposal + geometry repair。

数据规模：rows>=7 train/eval 来自同一 MPTS-52 validation K50 OOF；rows>=7 样本 {int(df.drop_duplicates('sample_id')['target_rows_ge7'].sum())} 个，rows>=7 候选 {int(df['target_rows_ge7'].sum())} 个；每 fold 独立训练 rows>=7 MLP。

baseline：rows>=7 原 GT-SG validation = {format_rows7(base_m)}。

方法变化：GPU rows>=7 specialized MLP；非 rows>=7 样本不强行用 specialized 模型，避免为了复杂结构牺牲简单结构。

结果：rows>=7 specialized route overall = {format_metrics(row_m)}；overall delta = {format_delta(row_d)}。rows>=7 = {format_rows7(row_m)}；rows>=7 delta = {format_delta(row_d, 'rows>=7_')}。

诊断：valid rate = {pct(row_diag['valid_rate'])}；formula consistency = {pct(row_diag['formula_consistency'])}；SG consistency = {pct(row_diag['sg_consistency'])}；exact-cover feasible = {pct(row_diag['exact_cover_feasible_rate'])}；skeleton-hit-to-match conversion = {pct(row_diag['skeleton_hit_to_match_conversion'])}。

可信度：实际 GPU 训练 + 5-fold OOF；但仍只在已有候选池内排序，不提高 true skeleton/geometry coverage。

和历史实验关系：这是实验 6 中 CPU/proxy rows7 specialized scorer 的严格补充；如果它仍然提升有限，则更支持此前“rows>=7 不是普通 scorer 能解决”的结论。

最终判决：若 rows>=7 match@5/match@20 仍未显著提高，则 rows>=7 需要 exact-cover skeleton proposal 与 symmetry-preserving geometry repair，而不是继续 rerank。

下一步：继续补审实验 5/7 的生成侧和 geometry repair 缺口；不能把本实验写成最终主方法。
""",
    )

    print(json.dumps(result["policies"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
