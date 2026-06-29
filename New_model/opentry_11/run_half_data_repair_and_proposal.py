#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_11"
RESULTS = OUT_DIR / "results"
REPORT = OUT_DIR / "GPT_REVIEW_BUNDLE.md"

STRUCT_FEATURES = ROOT / "model/std_way/track_a_mpts52/outputs/structural_features.jsonl.gz"
MPTS52_CANDIDATES = ROOT / "model/New_model/opentry_10/candidates/crystallm_gt_sg_mpts52_val_k100.jsonl"
MPTS52_TARGET_MANIFEST = ROOT / "model/New_model/opentry_10/cache/official_benchmark_cifs_symprec0p1/mpts_52/val/manifest.tsv"
V5_RUN = ROOT / "runs/symcif_v5_multidataset_wa_decoder/mpts52/val"

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


def half_ids_from_table(sample_table: pd.DataFrame, rows_col: str) -> set[str]:
    selected: set[str] = set()
    for _, group in sample_table.groupby(rows_col, sort=False):
        ids = sorted(str(x) for x in group["sample_id"].tolist())
        selected.update(sid for i, sid in enumerate(ids) if i % 2 == 0)
    return selected


def load_structural_features() -> pd.DataFrame:
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
    return df


def select_half_structural(df: pd.DataFrame) -> pd.DataFrame:
    table = df[["sample_id", "target_rows_ge7"]].drop_duplicates()
    selected = half_ids_from_table(table, "target_rows_ge7")
    return df[df["sample_id"].astype(str).isin(selected)].copy().reset_index(drop=True)


def load_target_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    with MPTS52_TARGET_MANIFEST.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip("\n").split("\t")))
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


def evaluate_rank_policy(df: pd.DataFrame, match_col: str = "match") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for _, group in df.groupby("sample_id", sort=False):
        ordered = group.sort_values("rank").head(20).copy()
        selected_frames.append(ordered)
        row = {"sample_id": str(ordered["sample_id"].iloc[0]), "target_rows_ge7": bool(ordered["target_rows_ge7"].iloc[0])}
        for budget in BUDGETS:
            row[f"hit@{budget}"] = bool(ordered.head(budget)[match_col].any())
        rows.append(row)
    rows7 = [r for r in rows if r["target_rows_ge7"]]
    metrics: dict[str, Any] = {"samples": len(rows), "rows>=7_samples": len(rows7)}
    for budget in BUDGETS:
        metrics[f"match@{budget}"] = float(sum(1 for r in rows if r[f"hit@{budget}"]) / max(1, len(rows)))
        metrics[f"rows>=7_match@{budget}"] = float(sum(1 for r in rows7 if r[f"hit@{budget}"]) / max(1, len(rows7)))
    selected = pd.concat(selected_frames, ignore_index=True)
    skel = selected[selected["skeleton_hit"]]
    diagnostics = {
        "valid_rate": float(selected["valid_label"].mean()),
        "formula_consistency": float(selected["formula_reduced_match"].mean()),
        "sg_consistency": float(selected["sg_number_match"].mean()),
        "exact_cover_feasible_rate": float(selected["skeleton_hit"].mean()),
        "skeleton_hit_to_match_conversion": None if len(skel) == 0 else float(skel[match_col].mean()),
    }
    return {"metrics": metrics, "diagnostics": diagnostics}


def metric_delta(metrics: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    keys = [f"match@{b}" for b in BUDGETS] + [f"rows>=7_match@{b}" for b in BUDGETS]
    return {k: float(metrics[k] - base[k]) for k in keys}


def gate_pass(delta: dict[str, float]) -> tuple[bool, dict[str, Any]]:
    overall = sum(1 for b in BUDGETS if delta[f"match@{b}"] >= GATE_PP)
    rows7 = delta["rows>=7_match@5"] >= GATE_PP and delta["rows>=7_match@20"] >= GATE_PP
    return overall >= 2 or rows7, {"overall_metrics_over_5pp": overall, "rows7_k5_k20_over_5pp": rows7, "threshold": GATE_PP}


def fmt_metrics(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"match@{b}"]) for b in BUDGETS)


def fmt_rows7(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"rows>=7_match@{b}"]) for b in BUDGETS)


def fmt_delta(d: dict[str, float], prefix: str = "") -> str:
    return " / ".join(pp(d[f"{prefix}match@{b}"]) for b in BUDGETS)


def run_repair_helper(input_path: Path, output_path: Path, workers: int) -> None:
    py = ROOT / "miniforge3/envs/crystallm_env/bin/python"
    helper = OUT_DIR / "repair_after_match_helper.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    conda_lib = str(ROOT / "miniforge3/envs/crystallm_env/lib")
    env["LD_LIBRARY_PATH"] = conda_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    subprocess.run(
        [
            str(py),
            str(helper),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--workers",
            str(workers),
            "--timeout",
            "5",
        ],
        check=True,
        cwd=str(OUT_DIR),
        env=env,
    )


def run_experiment_5_half(df: pd.DataFrame, workers: int) -> dict[str, Any]:
    half = select_half_structural(df)
    baseline = evaluate_rank_policy(half, "match")
    pool = half[
        (half["rank"] <= 20)
        & half["skeleton_hit"]
        & (~half["match"])
        & (half["geom_parseable"] > 0)
    ].copy()
    keys = {(str(r.sample_id), int(r.rank)) for r in pool.itertuples()}
    cand_texts = read_candidate_texts(keys)
    target_paths = load_target_paths()
    pair_path = ensure_out(RESULTS / "experiment_5_half_repair_pairs.jsonl")
    eval_path = ensure_out(RESULTS / "experiment_5_half_repair_eval.jsonl")
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with pair_path.open("w", encoding="utf-8") as f:
        for r in pool.itertuples():
            key = (str(r.sample_id), int(r.rank))
            cif = cand_texts.get(key)
            gt_path = target_paths.get(str(r.material_id))
            if not cif or gt_path is None or not gt_path.exists():
                continue
            after = repair_cif_text(cif, jitter=bool(r.radius_collision_lt_0p6 > 0))
            f.write(
                json.dumps(
                    {
                        "sample_id": str(r.sample_id),
                        "material_id": str(r.material_id),
                        "rank": int(r.rank),
                        "target_rows_ge7": bool(r.target_rows_ge7),
                        "after_cif": after,
                        "gt_path": str(gt_path),
                        "collision_proxy": bool(r.radius_collision_lt_0p6 > 0),
                        "min_radius_ratio": None if pd.isna(r.min_radius_ratio) else float(r.min_radius_ratio),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    run_repair_helper(pair_path, eval_path, workers=workers)

    rows: list[dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    eval_map = {(str(r["sample_id"]), int(r["rank"])): r for r in rows}
    half["repair_match"] = half["match"].astype(bool)
    converted_keys = set()
    for key, row in eval_map.items():
        if row.get("after_match") is True:
            converted_keys.add(key)
    if converted_keys:
        key_series = list(zip(half["sample_id"].astype(str), half["rank"].astype(int)))
        half["repair_match"] = [bool(m) or (key in converted_keys) for m, key in zip(half["match"], key_series)]
    repaired = evaluate_rank_policy(half, "repair_match")
    delta = metric_delta(repaired["metrics"], baseline["metrics"])
    passed, gate = gate_pass(delta)
    valid = [r for r in rows if r.get("after_match") is not None]
    converted = [r for r in valid if r.get("after_match") is True]
    rows7_valid = [r for r in valid if r.get("target_rows_ge7")]
    rows7_converted = [r for r in converted if r.get("target_rows_ge7")]
    result = {
        "time": now_iso(),
        "stage": "half",
        "gpu_necessary": False,
        "gpu_reason": "deterministic geometry repair has no trainable model; StructureMatcher evaluation is CPU-bound",
        "data": {
            "samples": int(half["sample_id"].nunique()),
            "candidates": int(len(half)),
            "rows>=7_samples": int(half[["sample_id", "target_rows_ge7"]].drop_duplicates()["target_rows_ge7"].sum()),
            "repair_pool_candidates": int(len(pool)),
            "repair_pool_samples": int(pool["sample_id"].nunique()),
            "repair_pairs_written": int(written),
        },
        "baseline": baseline,
        "repaired": repaired,
        "delta_vs_baseline": delta,
        "gate": {"pass": passed, **gate},
        "repair_eval": {
            "evaluated_valid": int(len(valid)),
            "converted": int(len(converted)),
            "conversion_rate": float(len(converted) / max(1, len(valid))),
            "rows>=7_evaluated_valid": int(len(rows7_valid)),
            "rows>=7_converted": int(len(rows7_converted)),
            "rows>=7_conversion_rate": float(len(rows7_converted) / max(1, len(rows7_valid))),
            "after_match_none": int(sum(1 for r in rows if r.get("after_match") is None)),
        },
        "artifacts": {"pairs": str(pair_path), "eval": str(eval_path)},
    }
    write_json(RESULTS / "experiment_5_half_geometry_repair.json", result)
    append_repair_report(result)
    return result


def append_repair_report(result: dict[str, Any]) -> None:
    base_m = result["baseline"]["metrics"]
    rep_m = result["repaired"]["metrics"]
    delta = result["delta_vs_baseline"]
    diag = result["repaired"]["diagnostics"]
    ev = result["repair_eval"]
    gate = result["gate"]
    data = result["data"]
    full_note = "half-data gate 通过，应补跑全量。" if gate["pass"] else "half-data gate 未通过，因此不补跑全量。"
    append_report(
        "实验 5B symmetry-preserving geometry repair half-data gate",
        f"""
时间：{result['time']}

实验逻辑：按更新后的目标重做实验 5。当前 repair 是 deterministic symmetry-preserving-ish 后处理，不训练模型，因此本实验判断为不需要 GPU；但因为它使用 MPTS-52 validation 数据，必须先用全量样本的一半，而不是 300 个 pilot。

核心假设：如果 skeleton-hit 失败主要来自坐标越界或轻微 collision，wrap fractional coordinates + collision jitter 应把一批 negative candidate 转成 StructureMatcher match，并提升 match@5/match@20，尤其 rows>=7。

数据规模：half samples={data['samples']}；half candidates={data['candidates']}；rows>=7 samples={data['rows>=7_samples']}；repair pool={data['repair_pool_candidates']} candidates / {data['repair_pool_samples']} samples；实际写入 repair pairs={data['repair_pairs_written']}。

baseline：原 GT-SG rank 顺序 = {fmt_metrics(base_m)}；rows>=7 = {fmt_rows7(base_m)}。

方法变化：对 half-data top20 中 skeleton-hit 且 StructureMatcher negative 的候选做 repair；不改变 SG/formula/multiplicity 行，只把 fractional coordinates wrap 到 [0,1)，对 collision-proxy 候选加确定性微小 jitter；repair 后用 StructureMatcher 重新评估，并回填 match@k。

结果：repair 后 = {fmt_metrics(rep_m)}；delta = {fmt_delta(delta)}。rows>=7 = {fmt_rows7(rep_m)}；rows>=7 delta = {fmt_delta(delta, 'rows>=7_')}。

repair conversion：valid evaluated={ev['evaluated_valid']}；converted={ev['converted']}；conversion rate={pct(ev['conversion_rate'])}；rows>=7 converted={ev['rows>=7_converted']}/{ev['rows>=7_evaluated_valid']}，conversion={pct(ev['rows>=7_conversion_rate'])}；after_match_none={ev['after_match_none']}。

诊断：valid rate = {pct(diag['valid_rate'])}；formula consistency = {pct(diag['formula_consistency'])}；SG consistency = {pct(diag['sg_consistency'])}；exact-cover feasible = {pct(diag['exact_cover_feasible_rate'])}；skeleton-hit-to-match conversion = {pct(diag['skeleton_hit_to_match_conversion'])}。

可信度：半量样本、全 repair pool 评估，可信度高于 300-candidate pilot；但 repair 本身很弱，不是 learned geometry model。

和历史实验关系：把前一轮 pilot 的 0 conversion 放大到 half-data gate；用于判断 deterministic repair 是否值得全量。

最终判决：gate_pass={gate['pass']}；overall >= +5pp 指标数={gate['overall_metrics_over_5pp']}；rows>=7 K5/K20 是否均 >= +5pp={gate['rows7_k5_k20_over_5pp']}。{full_note}

下一步：若 deterministic repair 不过 gate，则实验 5 的下一步不是全量后处理，而是设计真正受 SG/Wyckoff 约束的 learned/optimized geometry repair。
""",
    )


def material_from_v5_sample(sample_id: str) -> str:
    return sample_id.split("__")[-1]


def load_v5_records(name: str) -> pd.DataFrame:
    gen_path = V5_RUN / "generations" / f"{name}.jsonl"
    met_path = V5_RUN / "metrics" / f"{name}_metrics.jsonl"
    rows: list[dict[str, Any]] = []
    with gen_path.open("r", encoding="utf-8") as gf, met_path.open("r", encoding="utf-8") as mf:
        for g_line, m_line in zip(gf, mf):
            g = json.loads(g_line)
            m = json.loads(m_line)
            rows.append(
                {
                    "sample_id": str(g["sample_id"]),
                    "material_id": material_from_v5_sample(str(g["sample_id"])),
                    "gen_index": int(g["gen_index"]),
                    "generation_score": float(g.get("generation_score") or -1e9),
                    "row_count_target": int(g.get("row_count_target") or 0),
                    "target_rows_ge7": int(g.get("row_count_target") or 0) >= 7,
                    "skeleton_hit": bool(g.get("skeleton_hit")),
                    "wa_hit": bool(g.get("wa_hit")),
                    "formula_ok": bool(m.get("formula_ok")),
                    "space_group_ok": bool(m.get("space_group_ok")),
                    "multiplicity_ok": bool(m.get("multiplicity_ok")),
                    "readable": bool(m.get("pymatgen_readable")),
                    "valid": bool(m.get("valid")),
                    "match": bool(m.get("match_ok")),
                }
            )
    return pd.DataFrame(rows)


def evaluate_generated(df: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for _, group in df.groupby("sample_id", sort=False):
        ordered = group.sort_values(["generation_score", "gen_index"], ascending=[False, True]).copy()
        selected_frames.append(ordered.head(20))
        row = {"sample_id": str(ordered["sample_id"].iloc[0]), "target_rows_ge7": bool(ordered["target_rows_ge7"].iloc[0])}
        for budget in BUDGETS:
            row[f"hit@{budget}"] = bool(ordered.head(budget)["match"].any())
        rows.append(row)
    rows7 = [r for r in rows if r["target_rows_ge7"]]
    metrics: dict[str, Any] = {"samples": len(rows), "rows>=7_samples": len(rows7)}
    for budget in BUDGETS:
        metrics[f"match@{budget}"] = float(sum(1 for r in rows if r[f"hit@{budget}"]) / max(1, len(rows)))
        metrics[f"rows>=7_match@{budget}"] = float(sum(1 for r in rows7 if r[f"hit@{budget}"]) / max(1, len(rows7)))
    selected = pd.concat(selected_frames, ignore_index=True)
    skel = selected[selected["skeleton_hit"]]
    diagnostics = {
        "valid_rate": float(selected["valid"].mean()),
        "formula_consistency": float(selected["formula_ok"].mean()),
        "sg_consistency": float(selected["space_group_ok"].mean()),
        "exact_cover_feasible_rate": float(selected["multiplicity_ok"].mean()),
        "skeleton_hit_rate": float(selected["skeleton_hit"].mean()),
        "wa_hit_rate": float(selected["wa_hit"].mean()),
        "skeleton_hit_to_match_conversion": None if len(skel) == 0 else float(skel["match"].mean()),
        "candidate_count": int(len(df)),
        "selected_candidate_count": int(len(selected)),
    }
    return {"metrics": metrics, "diagnostics": diagnostics}


def baseline_for_materials(struct_df: pd.DataFrame, material_ids: set[str]) -> dict[str, Any]:
    sub = struct_df[struct_df["material_id"].astype(str).isin(material_ids)].copy()
    return evaluate_rank_policy(sub, "match")


def run_experiment_7_half(struct_df: pd.DataFrame) -> dict[str, Any]:
    a1 = load_v5_records("v5_a1_exact_cover_sg_formula_e08")
    pool = load_v5_records("v5_fullgen_eval_pool")
    table = a1[["sample_id", "target_rows_ge7"]].drop_duplicates()
    selected = half_ids_from_table(table, "target_rows_ge7")
    a1_half = a1[a1["sample_id"].isin(selected)].copy()
    pool_half = pool[pool["sample_id"].isin(selected)].copy()
    a1_material_ids = set(a1_half["material_id"].astype(str))
    pool_material_ids = set(pool_half["material_id"].astype(str))
    baseline = baseline_for_materials(struct_df, a1_material_ids)
    pool_baseline = baseline_for_materials(struct_df, pool_material_ids)
    a1_eval = evaluate_generated(a1_half)
    pool_eval = evaluate_generated(pool_half)
    a1_delta = metric_delta(a1_eval["metrics"], baseline["metrics"])
    pool_delta = metric_delta(pool_eval["metrics"], pool_baseline["metrics"])
    a1_pass, a1_gate = gate_pass(a1_delta)
    pool_pass, pool_gate = gate_pass(pool_delta)
    result = {
        "time": now_iso(),
        "stage": "half",
        "gpu_necessary": False,
        "gpu_reason": "this half-data audit reuses existing SymCIF v5 generated/evaluated artifacts; retraining a neural W/A decoder would need GPU, but this experiment evaluates generation-side candidates rather than training a new model",
        "data": {
            "samples": int(a1_half["sample_id"].nunique()),
            "rows>=7_samples": int(a1_half[["sample_id", "target_rows_ge7"]].drop_duplicates()["target_rows_ge7"].sum()),
            "a1_candidates": int(len(a1_half)),
            "pool_candidates": int(len(pool_half)),
        },
        "baseline": baseline,
        "pool_baseline": pool_baseline,
        "exact_cover_a1": a1_eval,
        "fullgen_pool": pool_eval,
        "deltas_vs_baseline": {"exact_cover_a1": a1_delta, "fullgen_pool": pool_delta},
        "gate": {
            "exact_cover_a1": {"pass": a1_pass, **a1_gate},
            "fullgen_pool": {"pass": pool_pass, **pool_gate},
        },
    }
    write_json(RESULTS / "experiment_7_half_exact_cover_generation.json", result)
    append_proposal_report(result)
    return result


def append_proposal_report(result: dict[str, Any]) -> None:
    data = result["data"]
    base_m = result["baseline"]["metrics"]
    pool_base_m = result.get("pool_baseline", result["baseline"])["metrics"]
    a1_m = result["exact_cover_a1"]["metrics"]
    pool_m = result["fullgen_pool"]["metrics"]
    a1_d = result["deltas_vs_baseline"]["exact_cover_a1"]
    pool_d = result["deltas_vs_baseline"]["fullgen_pool"]
    a1_diag = result["exact_cover_a1"]["diagnostics"]
    pool_diag = result["fullgen_pool"]["diagnostics"]
    a1_gate = result["gate"]["exact_cover_a1"]
    pool_gate = result["gate"]["fullgen_pool"]
    full_note = "至少一个 generation route half-data gate 通过，应补跑/汇总全量。" if (a1_gate["pass"] or pool_gate["pass"]) else "half-data gate 未通过，因此不补跑全量 generation 汇总。"
    append_report(
        "实验 7C exact-cover constrained skeleton proposal half-data gate 修正版",
        f"""
时间：{result['time']}

实验逻辑：按更新后的目标重做实验 7。前一轮只在现有 CrystaLLM K50 候选中做 exact-cover filter/proxy，不满足“生成侧实验”。本轮改用已有 SymCIF v5 MPTS-52 validation generation artifacts：`v5_a1_exact_cover_sg_formula_e08` 是 exact-cover constrained skeleton/WA proposal，`v5_fullgen_eval_pool` 是 exact-cover 与 geometry-aware route 的生成池。本节修正 7B 中 fullgen pool 与 A1 样本数不一致时共用 baseline 的口径；A1 结论不变，fullgen pool delta 改为使用自己的同材料 baseline。

GPU 必要性判断：本轮不重新训练模型，只评估已有 generation/evaluation artifacts，因此不需要新 GPU 训练；如果下一步要重训 neural W/A decoder 或 learned skeleton proposer，则需要 GPU，并且也要先 half-data gate。

核心假设：如果 exact-cover constrained skeleton proposal 真提高 coverage，它应相对同材料的 CrystaLLM GT-SG baseline 在 match@20 或至少两个 match 指标上达到 +5pp，并且 rows>=7 不应只停留在 skeleton_hit。

数据规模：half samples={data['samples']}；rows>=7 samples={data['rows>=7_samples']}；exact-cover A1 candidates={data['a1_candidates']}；fullgen pool candidates={data['pool_candidates']}。half 子集按 rows>=7/rows<7 分层后 sample_id 稳定排序隔位抽取。

baseline：A1 同材料 CrystaLLM GT-SG rank 顺序 = {fmt_metrics(base_m)}；rows>=7 = {fmt_rows7(base_m)}。fullgen pool 因缺少部分样本，使用自己的同材料 baseline = {fmt_metrics(pool_base_m)}；rows>=7 = {fmt_rows7(pool_base_m)}。

方法变化：使用真正生成侧 SymCIF exact-cover candidates，而不是在 CrystaLLM K50 内重排；排序按 generation_score desc + gen_index asc；报告 K1/K5/K20，其中 K20 是该生成池可用候选内的 top20。

结果 A1 exact-cover：{fmt_metrics(a1_m)}；delta = {fmt_delta(a1_d)}。rows>=7 = {fmt_rows7(a1_m)}；rows>=7 delta = {fmt_delta(a1_d, 'rows>=7_')}。

结果 fullgen pool：{fmt_metrics(pool_m)}；delta = {fmt_delta(pool_d)}。rows>=7 = {fmt_rows7(pool_m)}；rows>=7 delta = {fmt_delta(pool_d, 'rows>=7_')}。

诊断 A1：formula={pct(a1_diag['formula_consistency'])}；SG={pct(a1_diag['sg_consistency'])}；exact-cover/multiplicity={pct(a1_diag['exact_cover_feasible_rate'])}；skeleton_hit={pct(a1_diag['skeleton_hit_rate'])}；WA_hit={pct(a1_diag['wa_hit_rate'])}；skeleton-hit-to-match={pct(a1_diag['skeleton_hit_to_match_conversion'])}。

诊断 fullgen pool：formula={pct(pool_diag['formula_consistency'])}；SG={pct(pool_diag['sg_consistency'])}；exact-cover/multiplicity={pct(pool_diag['exact_cover_feasible_rate'])}；skeleton_hit={pct(pool_diag['skeleton_hit_rate'])}；WA_hit={pct(pool_diag['wa_hit_rate'])}；skeleton-hit-to-match={pct(pool_diag['skeleton_hit_to_match_conversion'])}。

可信度：这是生成侧 artifact 的 half-data 复算，强于现有 K50 filter proxy；但它复用历史生成结果，不是本轮新训练。

和历史实验关系：与 SymCIF v5 报告一致，exact-cover 能提高 skeleton 可行性，但未稳定转化为 StructureMatcher match，尤其 rows>=7 仍弱。

最终判决：A1 gate_pass={a1_gate['pass']}；fullgen_pool gate_pass={pool_gate['pass']}。{full_note}

下一步：若 gate 未过，实验 7 的瓶颈不是“是否 exact-cover”本身，而是 exact-cover skeleton 到 geometry/StructureMatcher match 的转化；后续应与实验 5 的 learned geometry repair 绑定。
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--skip-proposal", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_structural_features()
    outputs: dict[str, Any] = {}
    if not args.skip_repair:
        outputs["repair"] = run_experiment_5_half(df, workers=args.workers)
    if not args.skip_proposal:
        outputs["proposal"] = run_experiment_7_half(df)
    write_json(RESULTS / "experiment_5_7_half_repair_proposal_summary.json", outputs)
    print(json.dumps({k: v.get("gate") for k, v in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
