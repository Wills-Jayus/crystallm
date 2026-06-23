#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


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


def row_by(rows: list[dict[str, Any]], mode: str, n: int) -> dict[str, Any]:
    for row in rows:
        if row.get("mode") == mode and int(row.get("n", -1)) == n:
            return row
    return {}


def load_breakdown(path: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with (ROOT / path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["k"] == "20":
                out[row["subset"]] = row
    return out


def write_full_model_report() -> dict[str, Any]:
    reports = ROOT / "reports"
    out_dir = reports / "full_model_vs_crystallm_small"
    out_dir.mkdir(parents=True, exist_ok=True)

    v4_current = load_json("reports/symcif_v4_full_eval_current/full_eval_summary.json")["overall"]
    large_rows = load_json("eval_runs/symcif_v2_full_large_constrained_n20_20260522/summary_with_n5.json")
    baseline_same_rows = load_json("eval_runs/baseline_reeval_same_as_v4_20260522/summary_with_n5.json")
    baseline_minprompt_same_rows = load_json(
        "eval_runs/baseline_minprompt_reeval_same_as_v4_20260522/summary_with_n5.json"
    )
    small_n20 = load_json("eval_runs/symcif_v2_constrained_eval_t1_topk10_n20/summary.json")
    fixed_n5 = load_json("eval_runs/symcif_v2_fixed_cell_match5_20260521/summary.json")
    v2full_n5 = load_json("eval_runs/symcif_v3_vs_v2full_n5_20260521/summary.json")

    large1 = row_by(large_rows, "symcif_v2_full_large_constrained", 1)
    large5 = row_by(large_rows, "symcif_v2_full_large_constrained", 5)
    large20 = row_by(large_rows, "symcif_v2_full_large_constrained", 20)
    baseline_same1 = row_by(baseline_same_rows, "baseline_same_as_v4", 1)
    baseline_same5 = row_by(baseline_same_rows, "baseline_same_as_v4", 5)
    baseline_same20 = row_by(baseline_same_rows, "baseline_same_as_v4", 20)
    baseline_min_same1 = row_by(baseline_minprompt_same_rows, "baseline_minprompt_same_as_v4", 1)
    baseline_min_same5 = row_by(baseline_minprompt_same_rows, "baseline_minprompt_same_as_v4", 5)
    baseline_min_same20 = row_by(baseline_minprompt_same_rows, "baseline_minprompt_same_as_v4", 20)
    fixed5 = row_by(fixed_n5, "symcif_v2_constrained_fixed_cell", 5)
    v2full5 = row_by(v2full_n5, "symcif_v2_full3500_constrained", 5)
    small1 = row_by(small_n20, "symcif_v2_constrained", 1)
    small20 = row_by(small_n20, "symcif_v2_constrained", 20)

    train_curve = [
        (250, 1.0502, 1.0527),
        (500, 0.7484, 0.7540),
        (750, 0.6405, 0.6640),
        (1000, 0.5613, 0.6648),
        (1250, 0.4172, 0.7685),
        (1500, 0.2437, 1.0151),
        (1750, 0.1345, 1.2812),
        (2000, 0.0921, 1.4444),
    ]

    summary_data = {
        "paths": {
            "large_config": "configs/exp_symcif_v2_full_large.yaml",
            "large_checkpoint": "runs/exp_symcif_v2_full_large/ckpt_best.pt",
            "large_training_log": "runs/logs/exp_symcif_v2_full_large_20260522_085542.log",
            "large_eval": "eval_runs/symcif_v2_full_large_constrained_n20_20260522",
        },
        "training_curve": train_curve,
        "baseline_same_as_v4": baseline_same_rows,
        "baseline_minprompt_same_as_v4": baseline_minprompt_same_rows,
        "v2_full_large": large_rows,
        "v4_current": v4_current,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    compare_rows = [
        [
            "baseline same-as-v4",
            pct(baseline_same1.get("match_rate_n1")),
            pct(baseline_same5.get("match_rate_n20")),
            pct(baseline_same20.get("match_rate_n20")),
            num(baseline_same20.get("RMSE")),
            pct(baseline_same20.get("pymatgen_readable")),
            pct(baseline_same20.get("formula_ok")),
            pct(baseline_same20.get("space_group_ok")),
            pct(baseline_same20.get("valid")),
            "eval_runs/baseline_reeval_same_as_v4_20260522",
        ],
        [
            "baseline_minprompt same-as-v4",
            pct(baseline_min_same1.get("match_rate_n1")),
            pct(baseline_min_same5.get("match_rate_n20")),
            pct(baseline_min_same20.get("match_rate_n20")),
            num(baseline_min_same20.get("RMSE")),
            pct(baseline_min_same20.get("pymatgen_readable")),
            pct(baseline_min_same20.get("formula_ok")),
            pct(baseline_min_same20.get("space_group_ok")),
            pct(baseline_min_same20.get("valid")),
            "eval_runs/baseline_minprompt_reeval_same_as_v4_20260522",
        ],
        [
            "v2_constrained small n20",
            pct(small1.get("match_rate_n1")),
            "N/A",
            pct(small20.get("match_rate_n20")),
            num(small20.get("RMSE")),
            pct(small20.get("pymatgen_readable")),
            pct(small20.get("formula_ok")),
            pct(small20.get("space_group_ok")),
            pct(small20.get("valid")),
            "eval_runs/symcif_v2_constrained_eval_t1_topk10_n20",
        ],
        [
            "v2_fixed_cell best n5",
            pct(fixed5.get("match_rate_n1")),
            pct(fixed5.get("match_rate_n20")),
            "N/A",
            num(fixed5.get("RMSE")),
            pct(fixed5.get("pymatgen_readable")),
            pct(fixed5.get("formula_ok")),
            pct(fixed5.get("space_group_ok")),
            pct(fixed5.get("valid")),
            "eval_runs/symcif_v2_fixed_cell_match5_20260521",
        ],
        [
            "v2_full3500 constrained n5",
            pct(v2full5.get("match_rate_n1")),
            pct(v2full5.get("match_rate_n20")),
            "N/A",
            num(v2full5.get("RMSE")),
            pct(v2full5.get("pymatgen_readable")),
            pct(v2full5.get("formula_ok")),
            pct(v2full5.get("space_group_ok")),
            pct(v2full5.get("valid")),
            "eval_runs/symcif_v3_vs_v2full_n5_20260521",
        ],
        [
            "v2_full_large constrained n20",
            pct(large1.get("match_rate_n1")),
            pct(large5.get("match_rate_n20")),
            pct(large20.get("match_rate_n20")),
            num(large20.get("RMSE")),
            pct(large20.get("pymatgen_readable")),
            pct(large20.get("formula_ok")),
            pct(large20.get("space_group_ok")),
            pct(large20.get("valid")),
            "eval_runs/symcif_v2_full_large_constrained_n20_20260522",
        ],
        [
            "v4 current full pipeline",
            pct(v4_current["top1"]["match_at_k"]),
            pct(v4_current["top5"]["match_at_k"]),
            pct(v4_current["top20"]["match_at_k"]),
            num(v4_current["top20"]["RMSE"]),
            pct(v4_current["top20"]["readable"]),
            pct(v4_current["top20"]["formula_ok"]),
            pct(v4_current["top20"]["SG_ok"]),
            pct(v4_current["top20"]["valid"]),
            "reports/symcif_v4_full_eval_current",
        ],
    ]

    lines = [
        "# Full Model vs CrystaLLM-small 实验报告",
        "",
        "## 1. 训练配置",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        "| route | `symcif_v2_constrained` / full train set |",
        "| data path | `data/tokens_symcif_v2` |",
        "| model | 12 layers, 12 heads, 768 embedding, 85.26M params |",
        "| tokenizer | CrystaLLM `CIFTokenizer` |",
        "| batch/block | batch_size=64, block_size=1024 |",
        "| lr | 6e-4 cosine decay, min_lr=6e-5, warmup=100 |",
        "| steps | 2000, eval every 250 |",
        "| seed | 1337 |",
        "| dtype/device | bfloat16, train on cuda:0 |",
        "| sampling | n=20, temp_discrete=1.0/topk=10, temp_coord=0.7/topk=5, temp_cell=0.5/topk=5 |",
        "| config | `configs/exp_symcif_v2_full_large.yaml` |",
        "| best checkpoint | `runs/exp_symcif_v2_full_large/ckpt_best.pt` |",
        "| log | `runs/logs/exp_symcif_v2_full_large_20260522_085542.log` |",
        "",
        "## 2. 训练曲线",
        "",
        "| step | train loss | val loss |",
        "| ---: | ---: | ---: |",
    ]
    for step, train, val in train_curve:
        mark = " **best**" if step == 750 else ""
        lines.append(f"| {step} | {train:.4f} | {val:.4f}{mark} |")
    lines.extend(
        [
            "",
            "验证损失在 step 750 达到最优 0.6640，之后持续过拟合；评测使用 `ckpt_best.pt`。",
            "",
            "## 3. 同 evaluator 对比",
            "",
            "| model / route | match@1 | match@5 | match@20 | RMSE | readable | formula_ok | SG_ok | valid | artifact |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in compare_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 4. 结论",
            "",
            "- 同等 evaluator 条件下，baseline match@20=44.6%，baseline_minprompt match@20=47.8%；v4 current match@20=56.6%，优势明确。",
            "- 新的大模型没有超过既有 v2 小模型：match@20=47.2%，低于既有 `symcif_v2_constrained` n20 的 49.2%，也低于 v4 current 的 56.6%。",
            "- 过拟合很明显：train loss 从 0.6405 降到 0.0921，但 val loss 从 0.6640 升到 1.4444。下一轮如果继续 v2 大模型，需要更强正则、早停、数据扩增或更保守训练步数。",
            "- v4 current 已经是本轮最高的 end-to-end 结果，且 WA upper-bound 仍有 82.4% 的 top20 上限；相比继续单纯放大 v2，优先补 v4 geometry/free-params/lattice 更直接。",
            "",
            "## 5. 输出",
            "",
            "- generation/eval：`eval_runs/symcif_v2_full_large_constrained_n20_20260522/`",
            "- same-condition baseline：`eval_runs/baseline_reeval_same_as_v4_20260522/`",
            "- same-condition baseline_minprompt：`eval_runs/baseline_minprompt_reeval_same_as_v4_20260522/`",
            "- n=1/5/20 聚合：`eval_runs/symcif_v2_full_large_constrained_n20_20260522/summary_with_n5.json`",
            "- 本报告 JSON：`reports/full_model_vs_crystallm_small/summary.json`",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "large1": large1,
        "large5": large5,
        "large20": large20,
        "baseline20": baseline_same20,
        "baseline_minprompt20": baseline_min_same20,
        "small20": small20,
    }


def write_decision_summary(extra: dict[str, Any]) -> None:
    reports = ROOT / "reports"
    v4_current = load_json("reports/symcif_v4_full_eval_current/full_eval_summary.json")["overall"]
    v4_upper = load_json("reports/symcif_v4_wa_upper_bound/full_eval_summary.json")["overall"]
    v4_geom = load_json("reports/symcif_v4_geometry_bottleneck/full_eval_summary.json")["overall"]
    current_break = load_breakdown("reports/symcif_v4_full_eval_current/full_eval_breakdown.csv")
    upper_break = load_breakdown("reports/symcif_v4_wa_upper_bound/full_eval_breakdown.csv")
    geom_break = load_breakdown("reports/symcif_v4_geometry_bottleneck/full_eval_breakdown.csv")
    large1 = extra["large1"]
    large5 = extra["large5"]
    large20 = extra["large20"]
    baseline20 = extra["baseline20"]
    baseline_minprompt20 = extra["baseline_minprompt20"]
    small20 = extra["small20"]

    lines = [
        "# Next Step Decision Summary",
        "",
        "## 1. 当前 v4 end-to-end 结果",
        "",
        "| experiment | match@1 | match@5 | match@20 | RMSE@20 | readable@20 | formula_ok@20 | atom_count_ok@20 | SG_ok@20 | bond_score@20 | bond_reasonable@20 | valid/strict_valid | WA hit@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in [
        ("v4 current", v4_current),
        ("WA upper-bound", v4_upper),
        ("GT-WA + current geometry", v4_geom),
    ]:
        t1, t5, t20 = data["top1"], data["top5"], data["top20"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    pct(t1["match_at_k"]),
                    pct(t5["match_at_k"]),
                    pct(t20["match_at_k"]),
                    num(t20["RMSE"]),
                    pct(t20["readable"]),
                    pct(t20["formula_ok"]),
                    pct(t20["atom_count_ok"]),
                    pct(t20["SG_ok"]),
                    num(t20["bond_length_score"]),
                    pct(t20["bond_lengths_reasonable"]),
                    f"{pct(t20['valid'])}/{pct(t20['strict_valid'])}",
                    pct(t20["wa_hit_at_k"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            (
                "v4 current 已经超过同等 evaluator 条件下的 baseline "
                f"match@20={pct(baseline20.get('match_rate_n20'))}，也超过 baseline_minprompt "
                f"match@20={pct(baseline_minprompt20.get('match_rate_n20'))}。"
            ),
            "",
            "## 2. Breakdown",
            "",
            "| subset | samples | current match@20 | current WA hit@20 | current RMSE | WA upper match@20 | GT-WA/current-geometry match@20 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for subset in ["overall", "n_sites>=6", "num_elements>=4", "SG=2", "SG=65", "SG=71", "SG=127"]:
        c = current_break[subset]
        u = upper_break[subset]
        g = geom_break[subset]
        lines.append(
            "| "
            + " | ".join(
                [
                    subset,
                    c["samples"],
                    pct(c["match_at_k"]),
                    pct(c["wa_hit_at_k"]),
                    num(c["RMSE"]),
                    pct(u["match_at_k"]),
                    pct(g["match_at_k"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 3. 瓶颈判断",
            "",
            "- WA candidate/search/ranking：不是当前主瓶颈。当前 WA@20=86.6%，WA upper-bound match@20=82.4%；WA 命中后到 evaluator 成功只损失约 4.2 个百分点。",
            "- geometry/free_params/lattice：是主瓶颈。GT WA + current geometry 的 match@20=56.4%，几乎等于 v4 current 的 56.6%；说明即使 WA 完全正确，当前 retrieval/fallback geometry 也只能做到约 56%。",
            "- renderer/validator：需要修，但不是重建 match 的首要瓶颈。v4 expanded CIF 的 `valid/strict_valid` 为 0，主要因为 legacy evaluator 的 multiplicity loop 检查不适配 OrbitEngine 展开 CIF；readable/formula/SG/atom_count 与 StructureMatcher 结果仍然可用于本轮决策。",
            "- data scale：复杂子集仍弱，尤其 n_sites>=6 current match@20=13.1%，WA upper-bound=61.1%，GT-WA/current-geometry=17.7%；这说明复杂结构需要 geometry model 加 oversampling/data scale 验证。",
            "",
            "## 4. v2/full-data 大模型结论",
            "",
            (
                "新 v2 full_large："
                f"match@1={pct(large1.get('match_rate_n1'))}，"
                f"match@5={pct(large5.get('match_rate_n20'))}，"
                f"match@20={pct(large20.get('match_rate_n20'))}，"
                f"RMSE@20={num(large20.get('RMSE'))}。"
                f"它略高于同条件 baseline n20={pct(baseline20.get('match_rate_n20'))}，"
                f"但低于同条件 baseline_minprompt n20={pct(baseline_minprompt20.get('match_rate_n20'))}，"
                f"也低于既有 v2_constrained n20={pct(small20.get('match_rate_n20'))} 和 v4 current=56.6%。训练曲线显示 step 750 后明显过拟合。"
            ),
            "",
            "## 5. 是否继续 v4",
            "",
            "继续 v4。按判定规则，current v4 full match@20 已超过 baseline；同时 WA upper-bound 高、full pipeline 低，最短路径不是继续深挖 WA@200，而是补 orbit-level geometry/free_params/lattice model。",
            "",
            "## 6. 下一步最短路径",
            "",
            "1. 训练 orbit-level geometry model：condition on formula + SG + WA，预测 lattice 与 free parameters；先用 GT WA 评测，目标把 GT-WA geometry bottleneck 从 56.4% 提到 70-75% 以上。",
            "2. 接入 predicted WA top20：用同一 evaluator 跑 match@1/5/20，直接对比当前 v4 full 56.6% 和 WA upper-bound 82.4%。",
            "3. 修 renderer/evaluator 兼容：OrbitEngine expanded CIF 增加或适配 multiplicity 信息，避免 `valid/strict_valid` 被 legacy 格式检查全部打零。",
            "4. 对复杂子集做 data scale/oversampling 验证：优先 n_sites>=6、num_elements>=4、SG=65/71/127；不要只加局部规则。",
            "5. v2 路线保留为 baseline/control：若继续训练大模型，使用早停和更强正则；但当前不应把它作为主线。",
            "",
            "## 7. Artifacts",
            "",
            "- v4 current：`reports/symcif_v4_full_eval_current/`",
            "- WA upper-bound：`reports/symcif_v4_wa_upper_bound/`",
            "- geometry bottleneck：`reports/symcif_v4_geometry_bottleneck/`",
            "- full model comparison：`reports/full_model_vs_crystallm_small/summary.md`",
            "- same-condition baseline：`eval_runs/baseline_reeval_same_as_v4_20260522/`",
            "- same-condition baseline_minprompt：`eval_runs/baseline_minprompt_reeval_same_as_v4_20260522/`",
            "- new v2 large eval：`eval_runs/symcif_v2_full_large_constrained_n20_20260522/`",
            "",
        ]
    )
    (reports / "next_step_decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    extra = write_full_model_report()
    write_decision_summary(extra)
    print(ROOT / "reports/full_model_vs_crystallm_small/summary.md")
    print(ROOT / "reports/next_step_decision_summary.md")


if __name__ == "__main__":
    main()
