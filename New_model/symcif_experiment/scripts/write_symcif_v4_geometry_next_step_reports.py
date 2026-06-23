#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_breakdown(rel: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with (ROOT / rel).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("k")) == "20":
                out[str(row["subset"])] = row
    return out


def pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        x = float(value)
    except Exception:
        return "N/A"
    if math.isnan(x):
        return "N/A"
    return f"{x * 100:.1f}%"


def num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        x = float(value)
    except Exception:
        return "N/A"
    if math.isnan(x):
        return "N/A"
    return f"{x:.{digits}f}"


def overall(summary: dict[str, Any], k: int) -> dict[str, Any]:
    return summary["overall"][f"top{k}"]


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    align = ["---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return lines


def metric_row(label: str, summary: dict[str, Any], k: int) -> list[str]:
    m = overall(summary, k)
    return [
        label,
        f"@{k}",
        pct(m.get("match_at_k")),
        num(m.get("RMSE")),
        pct(m.get("readable")),
        pct(m.get("formula_ok")),
        pct(m.get("atom_count_ok")),
        pct(m.get("SG_ok")),
        pct(m.get("valid")),
        pct(m.get("strict_valid")),
        pct(m.get("strict_valid_any_at_k")),
        pct(m.get("wa_hit_at_k")),
        pct(m.get("eval_timeout")),
    ]


def geometry_quality_row(label: str, summary: dict[str, Any], k: int) -> list[str]:
    m = overall(summary, k)
    return [
        label,
        f"@{k}",
        pct(m.get("match_at_k")),
        num(m.get("RMSE")),
        pct(m.get("readable")),
        pct(m.get("formula_ok")),
        pct(m.get("atom_count_ok")),
        pct(m.get("SG_ok")),
        pct(m.get("valid")),
        pct(m.get("strict_valid")),
        pct(m.get("bond_lengths_reasonable")),
        num(m.get("bond_length_score")),
    ]


def best_training_row(summary: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in summary["history"] if "val_loss" in row]
    return min(rows, key=lambda row: float(row["val_loss"]))


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_renderer_fix_report() -> dict[str, Any]:
    out_dir = REPORTS / "symcif_v4_renderer_evaluator_fix"
    before = load_json("reports/symcif_v4_full_eval_current/full_eval_summary.json")
    after = load_json("reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/full_eval_summary.json")

    rows: list[list[str]] = []
    for k in (1, 5, 20):
        rows.append(metric_row("before renderer/evaluator fix", before, k))
        rows.append(metric_row("after renderer/evaluator fix", after, k))

    payload = {
        "before": before,
        "after": after,
        "conclusion": {
            "strict_valid_top20_before": overall(before, 20)["strict_valid"],
            "strict_valid_top20_after": overall(after, 20)["strict_valid"],
            "strict_valid_any_top20_after": overall(after, 20)["strict_valid_any_at_k"],
            "match_top20_before": overall(before, 20)["match_at_k"],
            "match_top20_after": overall(after, 20)["match_at_k"],
        },
    }
    write_json(out_dir / "summary.json", payload)

    lines = [
        "# symcif v4 renderer/evaluator 兼容性修复报告",
        "",
        "## 1. 结论",
        "",
        "修复已生效：v4 expanded CIF 不再被 legacy evaluator 的 multiplicity/formula 检查全部打成 `valid=0`。`strict_valid@20` 从 0.0% 恢复到 14.7%，`strict_valid_any@20` 恢复到 68.2%。",
        "",
        "代价是兼容检查后评测更重，`eval_timeout@20` 从 3.0% 升到 9.0%，所以 match@20 从 56.6% 小幅降到 55.6%。这个下降更像 evaluator timeout/检查成本变化，不是 WA 搜索能力退化。",
        "",
        "## 2. Before / After",
        "",
        *md_table(
            [
                "run",
                "k",
                "match",
                "RMSE",
                "readable",
                "formula_ok",
                "atom_count_ok",
                "SG_ok",
                "valid",
                "strict_valid",
                "strict_any",
                "WA hit",
                "eval_timeout",
            ],
            rows,
        ),
        "",
        "## 3. 修改点",
        "",
        "- `src/symcif_v4/orbit_engine.py` 的 CIF data block 改为包含 formula id，避免 sample id 与 formula 不一致。",
        "- 渲染输出新增 `_chemical_formula_structural`、`_cell_formula_units_Z`、`_cell_volume`。",
        "- `_atom_site` loop 新增 `_atom_site_symmetry_multiplicity`，展开坐标行统一写 1，使 legacy evaluator 的 multiplicity loop 能正常读取。",
        "",
        "## 4. Artifacts",
        "",
        "- after-fix eval: `reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/`",
        "- summary json: `reports/symcif_v4_renderer_evaluator_fix/summary.json`",
        "- before-fix reference: `reports/symcif_v4_full_eval_current/`",
    ]
    write_text(out_dir / "summary.md", lines)
    return payload


def write_gtwa_geometry_report() -> dict[str, Any]:
    out_dir = REPORTS / "symcif_v4_geometry_model_gtwa"
    old = load_json("reports/symcif_v4_geometry_bottleneck/full_eval_summary.json")
    no_over = load_json("reports/symcif_v4_geometry_model_gtwa/no_oversampling/full_eval_summary.json")
    over = load_json("reports/symcif_v4_geometry_model_gtwa/oversampling/full_eval_summary.json")
    train_no = load_json("runs/symcif_v4_geometry_model_no_oversampling/training_summary.json")
    train_over = load_json("runs/symcif_v4_geometry_model_oversampling/training_summary.json")
    best_no = best_training_row(train_no)
    best_over = best_training_row(train_over)

    eval_rows: list[list[str]] = []
    for label, summary in [
        ("GT-WA + current geometry", old),
        ("GT-WA + geometry model no-over", no_over),
        ("GT-WA + geometry model oversampling", over),
    ]:
        for k in (1, 5, 20):
            eval_rows.append(geometry_quality_row(label, summary, k))

    no_b = load_breakdown("reports/symcif_v4_geometry_model_gtwa/no_oversampling/full_eval_breakdown.csv")
    over_b = load_breakdown("reports/symcif_v4_geometry_model_gtwa/oversampling/full_eval_breakdown.csv")
    subset_rows: list[list[str]] = []
    for subset in ("overall", "n_sites>=6", "num_elements>=4", "SG=2", "SG=65", "SG=71", "SG=127"):
        a = no_b[subset]
        b = over_b[subset]
        subset_rows.append(
            [
                subset,
                a["samples"],
                pct(a["match_at_k"]),
                num(a["RMSE"]),
                pct(a["valid"]),
                pct(a["strict_valid_any_at_k"]),
                pct(b["match_at_k"]),
                num(b["RMSE"]),
                pct(b["valid"]),
                pct(b["strict_valid_any_at_k"]),
            ]
        )

    payload = {
        "current_geometry_reference": old,
        "no_oversampling": no_over,
        "oversampling": over,
        "training": {
            "no_oversampling": {"summary": train_no, "best_eval_epoch": best_no},
            "oversampling": {"summary": train_over, "best_eval_epoch": best_over},
        },
    }
    write_json(out_dir / "summary.json", payload)

    lines = [
        "# symcif v4 GT-WA geometry model 实验报告",
        "",
        "## 1. 结论",
        "",
        "GT-WA 条件下，新的 orbit-level geometry model 明显突破原 geometry bottleneck：match@20 从 56.4% 提升到 70.2%/70.4%，RMSE@20 从 0.0973 降到约 0.0833。说明 WA 已知时，lattice/free_params 的预测和原型补全确实是有效增益来源。",
        "",
        "oversampling 的整体收益很小：overall match@20 只从 70.2% 到 70.4%；但在 `n_sites>=6`、`SG=65` 等目标复杂子集有局部改善，仍值得在更大数据规模下继续验证。",
        "",
        "## 2. 训练摘要",
        "",
        *md_table(
            ["variant", "complex_weight", "best_epoch", "best_val_loss", "val_lattice_loss", "val_coord_loss", "checkpoint"],
            [
                [
                    "no oversampling",
                    train_no["config"]["complex_weight"],
                    best_no["epoch"],
                    num(best_no["val_loss"]),
                    num(best_no["val_lattice_loss"]),
                    num(best_no["val_coord_loss"]),
                    "`runs/symcif_v4_geometry_model_no_oversampling/ckpt_best.pt`",
                ],
                [
                    "complex oversampling",
                    train_over["config"]["complex_weight"],
                    best_over["epoch"],
                    num(best_over["val_loss"]),
                    num(best_over["val_lattice_loss"]),
                    num(best_over["val_coord_loss"]),
                    "`runs/symcif_v4_geometry_model_oversampling/ckpt_best.pt`",
                ],
            ],
        ),
        "",
        "## 3. GT-WA evaluation",
        "",
        *md_table(
            [
                "run",
                "k",
                "match",
                "RMSE",
                "readable",
                "formula_ok",
                "atom_count_ok",
                "SG_ok",
                "valid",
                "strict_valid",
                "bond_reasonable",
                "bond_score",
            ],
            eval_rows,
        ),
        "",
        "## 4. Complex subset breakdown",
        "",
        *md_table(
            [
                "subset",
                "samples",
                "no-over match@20",
                "no-over RMSE",
                "no-over valid",
                "no-over strict_any",
                "over match@20",
                "over RMSE",
                "over valid",
                "over strict_any",
            ],
            subset_rows,
        ),
        "",
        "## 5. Artifacts",
        "",
        "- no-over eval: `reports/symcif_v4_geometry_model_gtwa/no_oversampling/`",
        "- oversampling eval: `reports/symcif_v4_geometry_model_gtwa/oversampling/`",
        "- no-over checkpoint: `runs/symcif_v4_geometry_model_no_oversampling/ckpt_best.pt`",
        "- oversampling checkpoint: `runs/symcif_v4_geometry_model_oversampling/ckpt_best.pt`",
        "- training script: `scripts/train_symcif_v4_geometry_model.py`",
        "- eval script: `scripts/run_symcif_v4_geometry_model_eval.py`",
    ]
    write_text(out_dir / "summary.md", lines)
    return payload


def write_full_pipeline_report() -> dict[str, Any]:
    out_dir = REPORTS / "symcif_v4_full_pipeline_geometry_model"
    current = load_json("reports/symcif_v4_full_eval_current/full_eval_summary.json")
    current_fixed = load_json("reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/full_eval_summary.json")
    upper = load_json("reports/symcif_v4_wa_upper_bound/full_eval_summary.json")
    no_over = load_json("reports/symcif_v4_full_pipeline_geometry_model/no_oversampling/full_eval_summary.json")
    over = load_json("reports/symcif_v4_full_pipeline_geometry_model/oversampling/full_eval_summary.json")
    best = load_json("reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/full_eval_summary.json")

    rows: list[list[str]] = []
    for label, summary in [
        ("v4 current before renderer fix", current),
        ("v4 current after renderer fix", current_fixed),
        ("WA upper-bound", upper),
        ("geometry no-over, WA20 x geom1", no_over),
        ("geometry oversampling, WA20 x geom1", over),
        ("geometry no-over, WA15 x geom2", best),
    ]:
        for k in (1, 5, 20):
            m = overall(summary, k)
            rows.append(
                [
                    label,
                    f"@{k}",
                    pct(m.get("match_at_k")),
                    num(m.get("RMSE")),
                    pct(m.get("valid")),
                    pct(m.get("strict_valid")),
                    pct(m.get("strict_valid_any_at_k")),
                    pct(m.get("wa_hit_at_k")),
                    pct(m.get("skeleton_hit_at_k")),
                    pct(m.get("eval_timeout")),
                ]
            )

    payload = {
        "current_before_fix": current,
        "current_after_fix": current_fixed,
        "wa_upper_bound": upper,
        "no_oversampling_wa20_geom1": no_over,
        "oversampling_wa20_geom1": over,
        "best_no_oversampling_wa15_geom2": best,
    }
    write_json(out_dir / "summary.json", payload)

    lines = [
        "# symcif v4 full pipeline + geometry model 实验报告",
        "",
        "## 1. 结论",
        "",
        "接入 geometry model 后，full pipeline 的最优配置达到 match@20=62.8%，超过当前 v4 full 的 56.6%，也超过 60% 的阶段目标。这个最优来自 `WA top15 + 每个 WA 最多 2 个 geometry variant` 的输出预算。",
        "",
        "如果只做 `WA top20 x 每个 WA 1 个 geometry`，match@20 为 59.4%/59.2%，还没稳定越过 60%。因此 full pipeline 的有效增益不只是单点 geometry 质量，而是需要在 top20 预算里保留 geometry diversity。",
        "",
        "## 2. Full pipeline 对比",
        "",
        *md_table(
            [
                "run",
                "k",
                "match",
                "RMSE",
                "valid",
                "strict_valid",
                "strict_any",
                "WA hit",
                "skeleton hit",
                "eval_timeout",
            ],
            rows,
        ),
        "",
        "## 3. 与上限的距离",
        "",
        "- 当前最佳 full pipeline：match@20=62.8%，RMSE@20=0.0887。",
        "- WA upper-bound：match@20=82.4%，RMSE@20=0.0157。",
        "- 剩余差距约 19.6 个 match 点，主要来自复杂结构 geometry/free_params 仍弱，以及 `WA15 x geom2` 预算下 WA hit@20 从 86.6% 降到 85.2%。",
        "",
        "## 4. Artifacts",
        "",
        "- no-over WA20 x geom1: `reports/symcif_v4_full_pipeline_geometry_model/no_oversampling/`",
        "- oversampling WA20 x geom1: `reports/symcif_v4_full_pipeline_geometry_model/oversampling/`",
        "- best WA15 x geom2: `reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/`",
        "- current fixed reference: `reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/`",
    ]
    write_text(out_dir / "summary.md", lines)
    return payload


def write_complex_subset_report() -> dict[str, Any]:
    out_dir = REPORTS / "symcif_v4_complex_subset_oversampling"
    out_dir.mkdir(parents=True, exist_ok=True)
    current = load_breakdown("reports/symcif_v4_full_eval_current/full_eval_breakdown.csv")
    upper = load_breakdown("reports/symcif_v4_wa_upper_bound/full_eval_breakdown.csv")
    old_geom = load_breakdown("reports/symcif_v4_geometry_bottleneck/full_eval_breakdown.csv")
    gt_no = load_breakdown("reports/symcif_v4_geometry_model_gtwa/no_oversampling/full_eval_breakdown.csv")
    gt_over = load_breakdown("reports/symcif_v4_geometry_model_gtwa/oversampling/full_eval_breakdown.csv")
    full_best = load_breakdown("reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/full_eval_breakdown.csv")

    subsets = ("overall", "n_sites>=6", "num_elements>=4", "SG=2", "SG=65", "SG=71", "SG=127")
    comparison_rows: list[dict[str, Any]] = []
    table_rows: list[list[str]] = []
    oversampling_rows: list[list[str]] = []
    for subset in subsets:
        row = {
            "subset": subset,
            "samples": int(current[subset]["samples"]),
            "current_match20": float(current[subset]["match_at_k"]),
            "current_wa_hit20": float(current[subset]["wa_hit_at_k"]),
            "wa_upper_match20": float(upper[subset]["match_at_k"]),
            "old_gtwa_geometry_match20": float(old_geom[subset]["match_at_k"]),
            "gtwa_no_over_match20": float(gt_no[subset]["match_at_k"]),
            "gtwa_no_over_rmse": float(gt_no[subset]["RMSE"]),
            "gtwa_no_over_valid": float(gt_no[subset]["valid"]),
            "gtwa_over_match20": float(gt_over[subset]["match_at_k"]),
            "gtwa_over_rmse": float(gt_over[subset]["RMSE"]),
            "gtwa_over_valid": float(gt_over[subset]["valid"]),
            "full_best_match20": float(full_best[subset]["match_at_k"]),
            "full_best_rmse": float(full_best[subset]["RMSE"]),
            "full_best_wa_hit20": float(full_best[subset]["wa_hit_at_k"]),
        }
        comparison_rows.append(row)
        table_rows.append(
            [
                subset,
                row["samples"],
                pct(row["current_match20"]),
                pct(row["current_wa_hit20"]),
                pct(row["wa_upper_match20"]),
                pct(row["old_gtwa_geometry_match20"]),
                pct(row["gtwa_no_over_match20"]),
                pct(row["gtwa_over_match20"]),
                pct(row["full_best_match20"]),
                pct(row["full_best_wa_hit20"]),
                num(row["full_best_rmse"]),
            ]
        )
        oversampling_rows.append(
            [
                subset,
                pct(row["gtwa_no_over_match20"]),
                pct(row["gtwa_over_match20"]),
                f"{(row['gtwa_over_match20'] - row['gtwa_no_over_match20']) * 100:+.1f}pt",
                num(row["gtwa_no_over_rmse"]),
                num(row["gtwa_over_rmse"]),
                pct(row["gtwa_no_over_valid"]),
                pct(row["gtwa_over_valid"]),
            ]
        )

    with (out_dir / "comparison_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    write_json(out_dir / "summary.json", comparison_rows)

    lines = [
        "# symcif v4 complex subset oversampling 实验报告",
        "",
        "## 1. 结论",
        "",
        "复杂子集的主要瓶颈仍是 geometry，而不是 WA：`n_sites>=6` 的当前 WA hit@20 有 72.6%，WA upper-bound match@20 有 61.1%，但原 geometry bottleneck 只有 17.7%。新 geometry model 在 GT-WA 下把该子集提升到 31.4%/32.0%，证明方向有效，但 full pipeline 仍只有 19.4%。",
        "",
        "oversampling 不是单独的充分解。它对 `n_sites>=6` 从 31.4% 到 32.0%，对 `SG=65` 从 75.0% 到 83.3%，但对 `num_elements>=4` 从 56.8% 降到 55.7%。更合理的结论是：保留复杂样本加权，但需要更大训练集和更多 geometry variants，而不是仅靠当前小规模 oversampling。",
        "",
        "## 2. Subset 对比",
        "",
        *md_table(
            [
                "subset",
                "samples",
                "current match@20",
                "current WA hit@20",
                "WA upper match@20",
                "old GT-WA geometry",
                "GT-WA no-over",
                "GT-WA over",
                "full best",
                "full best WA hit",
                "full best RMSE",
            ],
            table_rows,
        ),
        "",
        "## 3. Oversampling effect under GT-WA",
        "",
        *md_table(
            [
                "subset",
                "no-over match@20",
                "over match@20",
                "delta",
                "no-over RMSE",
                "over RMSE",
                "no-over valid",
                "over valid",
            ],
            oversampling_rows,
        ),
        "",
        "## 4. Artifacts",
        "",
        "- comparison CSV: `reports/symcif_v4_complex_subset_oversampling/comparison_rows.csv`",
        "- GT-WA no-over: `reports/symcif_v4_geometry_model_gtwa/no_oversampling/full_eval_breakdown.csv`",
        "- GT-WA oversampling: `reports/symcif_v4_geometry_model_gtwa/oversampling/full_eval_breakdown.csv`",
        "- full best: `reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/full_eval_breakdown.csv`",
    ]
    write_text(out_dir / "summary.md", lines)
    return {"comparison_rows": comparison_rows}


def write_final_summary() -> dict[str, Any]:
    current = load_json("reports/symcif_v4_full_eval_current/full_eval_summary.json")
    fixed = load_json("reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/full_eval_summary.json")
    upper = load_json("reports/symcif_v4_wa_upper_bound/full_eval_summary.json")
    old_geom = load_json("reports/symcif_v4_geometry_bottleneck/full_eval_summary.json")
    gt_no = load_json("reports/symcif_v4_geometry_model_gtwa/no_oversampling/full_eval_summary.json")
    gt_over = load_json("reports/symcif_v4_geometry_model_gtwa/oversampling/full_eval_summary.json")
    full_no = load_json("reports/symcif_v4_full_pipeline_geometry_model/no_oversampling/full_eval_summary.json")
    full_over = load_json("reports/symcif_v4_full_pipeline_geometry_model/oversampling/full_eval_summary.json")
    full_best = load_json("reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/full_eval_summary.json")

    rows = [
        [
            "CrystaLLM baseline same-as-v4",
            "16.2%",
            "33.8%",
            "44.6%",
            "0.1164",
            "19.7%",
            "eval_runs/baseline_reeval_same_as_v4_20260522",
        ],
        [
            "CrystaLLM baseline_minprompt same-as-v4",
            "22.4%",
            "38.6%",
            "47.8%",
            "0.1188",
            "23.0%",
            "eval_runs/baseline_minprompt_reeval_same_as_v4_20260522",
        ],
        [
            "v4 current before fix",
            pct(overall(current, 1)["match_at_k"]),
            pct(overall(current, 5)["match_at_k"]),
            pct(overall(current, 20)["match_at_k"]),
            num(overall(current, 20)["RMSE"]),
            pct(overall(current, 20)["strict_valid"]),
            "reports/symcif_v4_full_eval_current/",
        ],
        [
            "v4 current after renderer fix",
            pct(overall(fixed, 1)["match_at_k"]),
            pct(overall(fixed, 5)["match_at_k"]),
            pct(overall(fixed, 20)["match_at_k"]),
            num(overall(fixed, 20)["RMSE"]),
            pct(overall(fixed, 20)["strict_valid"]),
            "reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/",
        ],
        [
            "GT-WA + old geometry",
            pct(overall(old_geom, 1)["match_at_k"]),
            pct(overall(old_geom, 5)["match_at_k"]),
            pct(overall(old_geom, 20)["match_at_k"]),
            num(overall(old_geom, 20)["RMSE"]),
            pct(overall(old_geom, 20)["strict_valid"]),
            "reports/symcif_v4_geometry_bottleneck/",
        ],
        [
            "GT-WA + geometry model no-over",
            pct(overall(gt_no, 1)["match_at_k"]),
            pct(overall(gt_no, 5)["match_at_k"]),
            pct(overall(gt_no, 20)["match_at_k"]),
            num(overall(gt_no, 20)["RMSE"]),
            pct(overall(gt_no, 20)["strict_valid"]),
            "reports/symcif_v4_geometry_model_gtwa/no_oversampling/",
        ],
        [
            "GT-WA + geometry model over",
            pct(overall(gt_over, 1)["match_at_k"]),
            pct(overall(gt_over, 5)["match_at_k"]),
            pct(overall(gt_over, 20)["match_at_k"]),
            num(overall(gt_over, 20)["RMSE"]),
            pct(overall(gt_over, 20)["strict_valid"]),
            "reports/symcif_v4_geometry_model_gtwa/oversampling/",
        ],
        [
            "full geometry no-over WA20x1",
            pct(overall(full_no, 1)["match_at_k"]),
            pct(overall(full_no, 5)["match_at_k"]),
            pct(overall(full_no, 20)["match_at_k"]),
            num(overall(full_no, 20)["RMSE"]),
            pct(overall(full_no, 20)["strict_valid"]),
            "reports/symcif_v4_full_pipeline_geometry_model/no_oversampling/",
        ],
        [
            "full geometry over WA20x1",
            pct(overall(full_over, 1)["match_at_k"]),
            pct(overall(full_over, 5)["match_at_k"]),
            pct(overall(full_over, 20)["match_at_k"]),
            num(overall(full_over, 20)["RMSE"]),
            pct(overall(full_over, 20)["strict_valid"]),
            "reports/symcif_v4_full_pipeline_geometry_model/oversampling/",
        ],
        [
            "full geometry no-over WA15x2 best",
            pct(overall(full_best, 1)["match_at_k"]),
            pct(overall(full_best, 5)["match_at_k"]),
            pct(overall(full_best, 20)["match_at_k"]),
            num(overall(full_best, 20)["RMSE"]),
            pct(overall(full_best, 20)["strict_valid"]),
            "reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/",
        ],
        [
            "WA upper-bound",
            pct(overall(upper, 1)["match_at_k"]),
            pct(overall(upper, 5)["match_at_k"]),
            pct(overall(upper, 20)["match_at_k"]),
            num(overall(upper, 20)["RMSE"]),
            pct(overall(upper, 20)["strict_valid"]),
            "reports/symcif_v4_wa_upper_bound/",
        ],
    ]

    payload = {
        "current_before_fix": current,
        "current_after_fix": fixed,
        "wa_upper_bound": upper,
        "gtwa_old_geometry": old_geom,
        "gtwa_geometry_no_oversampling": gt_no,
        "gtwa_geometry_oversampling": gt_over,
        "full_geometry_no_oversampling": full_no,
        "full_geometry_oversampling": full_over,
        "full_geometry_best": full_best,
    }
    write_json(REPORTS / "symcif_v4_geometry_next_step_summary.json", payload)

    lines = [
        "# symcif v4 geometry next-step decision summary",
        "",
        "## 1. 总结论",
        "",
        "本轮实验完成了 renderer/evaluator 修复、GT-WA geometry model、predicted-WA full pipeline 接入、复杂子集 oversampling 对照。结论是：v4 路线值得继续，而且已经具备进入更大数据规模正式实验的条件，但必须把 geometry diversity 和复杂子集加权作为主配置的一部分。",
        "",
        "最关键的结果是 full pipeline best 达到 match@20=62.8%，高于同条件 CrystaLLM baseline 的 44.6%/47.8%，也高于修复前 v4 current 的 56.6%。GT-WA geometry 从 56.4% 提升到 70.4%，说明 geometry/free_params/lattice 子模块是有效突破点。",
        "",
        "## 2. 核心指标",
        "",
        *md_table(
            ["experiment", "match@1", "match@5", "match@20", "RMSE@20", "strict_valid@20", "artifact"],
            rows,
        ),
        "",
        "## 3. 判定",
        "",
        "- renderer/evaluator: 已修。`valid/strict_valid` 不再全 0；但 timeout 增加，需要后续优化 evaluator 性能。",
        "- geometry model: 成功。GT-WA match@20 从 56.4% 到 70.2%/70.4%，达到原计划 70-75% 区间下沿。",
        "- full pipeline: 成功但有条件。`WA20 x geom1` 只有 59.2%-59.4%；`WA15 x geom2` 达到 62.8%，说明 top20 输出预算必须分配给 geometry 多样性。",
        "- complex subset: 仍是主风险。`n_sites>=6` 在 GT-WA 下从 17.7% 到 32.0%，但 full best 只有 19.4%，距离 WA upper-bound 61.1% 还很远。",
        "- oversampling: 当前小规模只提供局部收益。建议保留加权策略，但不要把它视为单独解法。",
        "",
        "## 4. 下一步建议",
        "",
        "1. 进入大规模训练 pilot：用 CrystaLLM-small 同量级数据、2M 结构数据或 MPTS-52 数据训练 v4 geometry/WA pipeline，保持同 evaluator 对比。",
        "2. 固定两个 full-pipeline 配置一起跑：`WA20 x geom1` 用于公平单 geometry 对照，`WA15 x geom2` 用于最佳 end-to-end 结果。",
        "3. 数据策略保留复杂样本加权：重点覆盖 `n_sites>=6`、`num_elements>=4`、`SG=65/71/127`，并记录每个子集的 WA hit、GT-WA geometry、full pipeline 三段损失。",
        "4. 优化 evaluator/renderer throughput：修复后 `eval_timeout@20` 对 current full 到 9.0%，会干扰小幅增益判断。",
        "5. 保留 CrystaLLM baseline/minprompt 作为同条件 control；当前 best 已超过 baseline，但大数据训练能否超过 CrystaLLM 原模型仍需要正式同数据、同 prompt/evaluator 验证。",
        "",
        "## 5. 分报告",
        "",
        "- renderer/evaluator fix: `reports/symcif_v4_renderer_evaluator_fix/summary.md`",
        "- GT-WA geometry model: `reports/symcif_v4_geometry_model_gtwa/summary.md`",
        "- full pipeline geometry model: `reports/symcif_v4_full_pipeline_geometry_model/summary.md`",
        "- complex subset oversampling: `reports/symcif_v4_complex_subset_oversampling/summary.md`",
    ]
    write_text(REPORTS / "symcif_v4_geometry_next_step_summary.md", lines)
    return payload


def main() -> None:
    write_renderer_fix_report()
    write_gtwa_geometry_report()
    write_full_pipeline_report()
    write_complex_subset_report()
    write_final_summary()


if __name__ == "__main__":
    main()
