#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_12"
RESULTS = OUT_DIR / "results"
REPORT = ROOT / "model/New_model/opentry_11/GPT_REVIEW_BUNDLE.md"

EXP1 = RESULTS / "experiment_1_c2s3c15_official.json"
EXP2 = RESULTS / "experiment_2_c2s3c15_attribution.json"
EXP3 = RESULTS / "experiment_3_mp20_transfer_c2s3c15.json"
EXP4 = RESULTS / "experiment_4_learned_geometry_repair_audit.json"
EXP5 = RESULTS / "experiment_5_skeleton_proposer_audit.json"
EXP6 = RESULTS / "experiment_6_rows_ge7_specialized_audit.json"
STRICT_ABLATION = ROOT / "model/New_model/opentry_11/results/experiment_8_strict_integrated_ablation.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_opentry12(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_12: {resolved}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = ensure_opentry12(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_report_once(marker: Path, title: str, body: str) -> None:
    marker = ensure_opentry12(marker)
    if marker.exists():
        return
    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n\n" + f"## opentry_12 实验：{title}\n\n" + body.strip() + "\n")
    write_json(marker, {"time": now_iso(), "title": title, "report": str(REPORT)})


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def f6(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{float(v):.6f}"


def triplet(d: dict[str, Any], prefix: str = "match@") -> dict[str, float | None]:
    return {f"{prefix}{k}": d.get(f"{prefix}{k}") for k in (1, 5, 20)}


def rmse_triplet(d: dict[str, Any], prefix: str = "RMSE@") -> dict[str, float | None]:
    return {f"{prefix}{k}": d.get(f"{prefix}{k}") or d.get(f"rmsd@{k}") for k in (1, 5, 20)}


def rows_triplet(d: dict[str, Any]) -> dict[str, float | None]:
    return {f"rows>=7_match@{k}": d.get(f"rows>=7_match@{k}") for k in (1, 5, 20)}


def rows_rmse_triplet(d: dict[str, Any]) -> dict[str, float | None]:
    return {f"rows>=7_RMSE@{k}": d.get(f"rows>=7_RMSE@{k}") or d.get(f"rows>=7_rmsd@{k}") for k in (1, 5, 20)}


def format_match(row: dict[str, Any]) -> str:
    m = row["overall_match"]
    r = row["rows_ge7_match"]
    return (
        f"overall {pct(m.get('match@1'))}/{pct(m.get('match@5'))}/{pct(m.get('match@20'))}; "
        f"rows>=7 {pct(r.get('rows>=7_match@1'))}/{pct(r.get('rows>=7_match@5'))}/{pct(r.get('rows>=7_match@20'))}"
    )


def format_quality(row: dict[str, Any]) -> str:
    q = row.get("quality", {})
    return (
        f"valid={pct(q.get('valid_rate'))}, formula={pct(q.get('formula_consistency'))}, "
        f"SG={pct(q.get('sg_consistency'))}, exact-cover={pct(q.get('exact_cover_feasible_rate'))}, "
        f"skeleton->match={pct(q.get('skeleton_hit_to_match_conversion'))}"
    )


def strict_row(name: str, payload: dict[str, Any], role: str, boundary: str) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    diag = payload.get("diagnostics", {})
    return {
        "name": name,
        "role": role,
        "boundary": boundary,
        "scope": "MPTS-52 validation K50 full",
        "overall_match": triplet(metrics),
        "overall_rmse": rmse_triplet(metrics, prefix="rmsd@"),
        "rows_ge7_match": rows_triplet(metrics),
        "rows_ge7_rmse": rows_rmse_triplet(metrics),
        "quality": {
            "valid_rate": diag.get("valid_rate"),
            "formula_consistency": diag.get("formula_consistency"),
            "sg_consistency": diag.get("sg_consistency"),
            "exact_cover_feasible_rate": diag.get("exact_cover_feasible_rate"),
            "skeleton_hit_to_match_conversion": diag.get("skeleton_hit_to_match_conversion"),
        },
        "delta": payload.get("deltas"),
        "gate": "fail_main_gate" if name != "原 GT-SG baseline" else "reference",
    }


def symcif_v5_row(exp5: dict[str, Any]) -> dict[str, Any]:
    all_full = exp5["full_validation_missing_as_fail"]["all"]
    rows7 = exp5["full_validation_missing_as_fail"]["rows_ge7"]
    cdiag = exp5["candidate_level_diagnostics"]
    return {
        "name": "SymCIF v5 neural skeleton/geometry proposer",
        "role": "main_method_candidate_diagnostic",
        "boundary": "generation-side proposer, but current pool coverage fails main gate",
        "scope": "MPTS-52 validation full, missing artifacts as fail",
        "overall_match": {
            "match@1": all_full.get("top1_match_coverage"),
            "match@5": all_full.get("top5_match_coverage"),
            "match@20": all_full.get("top20_match_coverage"),
        },
        "overall_rmse": {
            "RMSE@1": all_full.get("top1_RMSE"),
            "RMSE@5": all_full.get("top5_RMSE"),
            "RMSE@20": all_full.get("top20_RMSE"),
        },
        "rows_ge7_match": {
            "rows>=7_match@1": rows7.get("top1_match_coverage"),
            "rows>=7_match@5": rows7.get("top5_match_coverage"),
            "rows>=7_match@20": rows7.get("top20_match_coverage"),
        },
        "rows_ge7_rmse": {
            "rows>=7_RMSE@1": rows7.get("top1_RMSE"),
            "rows>=7_RMSE@5": rows7.get("top5_RMSE"),
            "rows>=7_RMSE@20": rows7.get("top20_RMSE"),
        },
        "quality": {
            "valid_rate": cdiag.get("valid_rate"),
            "formula_consistency": cdiag.get("formula_consistency"),
            "sg_consistency": cdiag.get("sg_consistency"),
            "exact_cover_feasible_rate": cdiag.get("exact_cover_feasible_rate"),
            "skeleton_hit_to_match_conversion": cdiag.get("candidate_skeleton_to_match_conversion"),
        },
        "gate": "fail_main_gate",
    }


def c2s3c15_official_row(exp1: dict[str, Any]) -> dict[str, Any]:
    q = exp1["candidate_quality"]
    return {
        "name": "C2S3C15 frozen official",
        "role": "auxiliary_hybrid_official",
        "boundary": "frozen auxiliary route; cannot be claimed as core scientific method",
        "scope": "MPTS-52 official full-test once",
        "overall_match": triplet(exp1["all"]),
        "overall_rmse": rmse_triplet(exp1["all"]),
        "rows_ge7_match": {
            f"rows>=7_match@{k}": exp1["rows_ge7"].get(f"match@{k}") for k in (1, 5, 20)
        },
        "rows_ge7_rmse": {
            f"rows>=7_RMSE@{k}": exp1["rows_ge7"].get(f"RMSE@{k}") for k in (1, 5, 20)
        },
        "quality": {
            "valid_rate": q.get("slot_valid_rate"),
            "formula_consistency": q.get("slot_formula_consistency_rate"),
            "sg_consistency": q.get("slot_sg_consistency_rate"),
            "exact_cover_feasible_rate": q.get("symcif_slot_exact_cover_feasible_rate"),
            "skeleton_hit_to_match_conversion": q.get("symcif_skeleton_hit_to_match_conversion"),
        },
        "delta": {
            "overall": exp1.get("delta_vs_crystallm_baseline_all"),
            "rows_ge7": exp1.get("delta_vs_crystallm_baseline_rows_ge7"),
        },
        "gate": "passes_auxiliary_mpts52_official_k5_k20",
    }


def mp20_transfer_row(exp3: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "C2S3C15 MP-20 transfer",
        "role": "auxiliary_hybrid_transfer_check",
        "boundary": "same frozen idea, validation-like only; MP-20 overall does not pass +5pp gate",
        "scope": "MP-20 validation-like replay",
        "overall_match": triplet(exp3["hybrid_all"]),
        "overall_rmse": rmse_triplet(exp3["hybrid_all"]),
        "rows_ge7_match": {
            f"rows>=7_match@{k}": exp3["hybrid_rows_ge7"].get(f"match@{k}") for k in (1, 5, 20)
        },
        "rows_ge7_rmse": {
            f"rows>=7_RMSE@{k}": exp3["hybrid_rows_ge7"].get(f"RMSE@{k}") for k in (1, 5, 20)
        },
        "quality": {
            "valid_rate": exp3["quality_by_source"]["S"].get("valid_rate"),
            "formula_consistency": exp3["quality_by_source"]["S"].get("formula_ok_rate")
            or exp3["quality_by_source"]["S"].get("formula_consistency_rate"),
            "sg_consistency": exp3["quality_by_source"]["S"].get("space_group_ok_rate")
            or exp3["quality_by_source"]["S"].get("sg_consistency_rate"),
            "exact_cover_feasible_rate": exp3["quality_by_source"]["S"].get("exact_cover_feasible_rate")
            or exp3["quality_by_source"]["S"].get("exact_cover_feasible_rate_known_slots"),
            "skeleton_hit_to_match_conversion": exp3["quality_by_source"]["S"].get("skeleton_hit_to_match_conversion"),
        },
        "delta": {
            "overall": exp3.get("delta_vs_crystallm_baseline_all"),
            "rows_ge7": exp3.get("delta_vs_crystallm_baseline_rows_ge7"),
        },
        "gate": "rows_ge7_passes_but_overall_fails_transfer_gate",
    }


def learned_geometry_row(exp4: dict[str, Any]) -> dict[str, Any]:
    gtwa = exp4["mp20_gtwa_learned_geometry_k5"]
    all_m = gtwa["all"]
    rows7 = gtwa["rows_ge7"]
    return {
        "name": "learned geometry repair under GT-WA",
        "role": "main_method_component_oracle",
        "boundary": "oracle GT-WA/GT-skeleton component; not inference pipeline result",
        "scope": "MP-20 GT-WA geometry K5 replay",
        "overall_match": {"match@1": all_m.get("match@1"), "match@5": all_m.get("match@5"), "match@20": None},
        "overall_rmse": {"RMSE@1": all_m.get("RMSE@1"), "RMSE@5": all_m.get("RMSE@5"), "RMSE@20": None},
        "rows_ge7_match": {
            "rows>=7_match@1": rows7.get("match@1"),
            "rows>=7_match@5": rows7.get("match@5"),
            "rows>=7_match@20": None,
        },
        "rows_ge7_rmse": {
            "rows>=7_RMSE@1": rows7.get("RMSE@1"),
            "rows>=7_RMSE@5": rows7.get("RMSE@5"),
            "rows>=7_RMSE@20": None,
        },
        "quality": {
            "valid_rate": all_m.get("valid_any@5"),
            "formula_consistency": all_m.get("formula_ok_any@5"),
            "sg_consistency": all_m.get("sg_ok_any@5"),
            "exact_cover_feasible_rate": 1.0,
            "skeleton_hit_to_match_conversion": all_m.get("skeleton_hit_to_match_conversion@5"),
        },
        "rows_ge7_quality": {
            "valid_rate": rows7.get("valid_any@5"),
            "formula_consistency": rows7.get("formula_ok_any@5"),
            "sg_consistency": rows7.get("sg_ok_any@5"),
            "exact_cover_feasible_rate": 1.0,
            "skeleton_hit_to_match_conversion": rows7.get("skeleton_hit_to_match_conversion@5"),
        },
        "gate": "component_continue_not_benchmark_success",
    }


def compute_result() -> dict[str, Any]:
    exp1 = read_json(EXP1)
    exp2 = read_json(EXP2)
    exp3 = read_json(EXP3)
    exp4 = read_json(EXP4)
    exp5 = read_json(EXP5)
    exp6 = read_json(EXP6)
    strict = read_json(STRICT_ABLATION)

    full = strict["full_validation"]
    validation_rows = [
        strict_row("原 GT-SG baseline", full["baseline"], "reference_baseline", "reference"),
        strict_row("baseline + Track A scorer", full["track_a"], "auxiliary_scorer_diagnostic", "sorting_only"),
        strict_row("baseline + exact-cover filter/proxy", full["exact_cover_filter_proxy"], "proxy_diagnostic", "not_generation_side_success"),
        strict_row("baseline + hard-negative structural scorer v2", full["cpu_hard_negative_v2"], "auxiliary_scorer_diagnostic", "ordinary_scorer_fails_gate"),
        strict_row("skeleton proposal + geometry repair + structural scorer proxy", full["proxy_combined"], "proxy_diagnostic", "proxy_fails_coverage_gate"),
        symcif_v5_row(exp5),
        learned_geometry_row(exp4),
    ]
    auxiliary_rows = [c2s3c15_official_row(exp1), mp20_transfer_row(exp3)]

    final_judgment = {
        "main_method_success": False,
        "auxiliary_hybrid_success_mpts52": True,
        "mp20_overall_transfer_success": False,
        "ordinary_scorer_stopped": True,
        "coverage_solved": False,
        "geometry_repair_component_continue": True,
        "next_mainline": "rows>=7-specialized exact-cover skeleton proposer plus learned geometry repair; evaluate on validation/half-train gate before any official",
        "reason": (
            "Only C2S3C15 passes +5pp on MPTS-52 official K5/K20, but it is a frozen auxiliary route. "
            "Strict validation ablations and SymCIF v5 proposer do not solve overall coverage; learned geometry works only under oracle GT-WA."
        ),
    }
    attribution = exp2.get("baseline_vs_hybrid", {})
    all_gain5 = attribution.get("all", {}).get("net_gain_match@5_count") or 0
    all_gain20 = attribution.get("all", {}).get("net_gain_match@20_count") or 0
    rows7_gain5 = attribution.get("rows_ge7", {}).get("net_gain_match@5_count") or 0
    rows7_gain20 = attribution.get("rows_ge7", {}).get("net_gain_match@20_count") or 0

    return {
        "experiment": "experiment_7_main_method_ablation_and_hybrid_boundary",
        "time": now_iso(),
        "inputs": {
            "experiment_1": str(EXP1),
            "experiment_2": str(EXP2),
            "experiment_3": str(EXP3),
            "experiment_4": str(EXP4),
            "experiment_5": str(EXP5),
            "experiment_6": str(EXP6),
            "strict_ablation": str(STRICT_ABLATION),
        },
        "cpu_policy": {
            "policy": "bounded_cpu_use",
            "logical_cpus": os.cpu_count(),
            "used_parallel_workers": 1,
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "validation_ablation_rows": validation_rows,
        "auxiliary_hybrid_rows": auxiliary_rows,
        "experiment_2_attribution_key": {
            "rows_ge7_k5_net_gain_share": (rows7_gain5 / all_gain5) if all_gain5 else None,
            "rows_ge7_k20_net_gain_share": (rows7_gain20 / all_gain20) if all_gain20 else None,
            "s_only_after_c17_rows_ge7": attribution.get("rows_ge7", {}).get("s_only_after_c17_count"),
        },
        "experiment_6_rows_ge7_key": {
            "crystallm_k50_top50_rows_ge7": exp6["crystallm_k50_rows_ge7"].get("top50_match"),
            "symcif_v5_top50_rows_ge7": exp6["symcif_v5_rows_ge7_full_missing_as_fail"].get("top50_match"),
            "rows_ge7_primary_bottleneck": exp6["decision"].get("rows_ge7_primary_bottleneck"),
        },
        "module_roles": {
            "main_method_candidates": [
                "Wyckoff/exact-cover constrained skeleton proposal",
                "SymCIF neural skeleton/geometry proposer",
                "symmetry-preserving learned geometry repair",
                "rows>=7-specialized generation/repair",
            ],
            "auxiliary_or_diagnostic_only": [
                "C2S3C15 frozen hybrid route",
                "Track A scorer",
                "hard-negative scorer v2",
                "ordinary rerank/fusion/HGB/RF routes",
            ],
            "stopped": [
                "ordinary threshold tuning",
                "C/S ratio search after C2S3C15 freeze",
                "deterministic wrap/jitter geometry repair",
            ],
        },
        "final_judgment": final_judgment,
    }


def row_line(row: dict[str, Any]) -> str:
    rmse = row["overall_rmse"]
    rrmse = row["rows_ge7_rmse"]
    return (
        f"- {row['name']} [{row['role']}]: {format_match(row)}；"
        f"RMSE overall {f6(rmse.get('RMSE@1') or rmse.get('rmsd@1'))}/"
        f"{f6(rmse.get('RMSE@5') or rmse.get('rmsd@5'))}/"
        f"{f6(rmse.get('RMSE@20') or rmse.get('rmsd@20'))}；"
        f"rows>=7 {f6(rrmse.get('rows>=7_RMSE@1'))}/{f6(rrmse.get('rows>=7_RMSE@5'))}/{f6(rrmse.get('rows>=7_RMSE@20'))}；"
        f"{format_quality(row)}；判定={row['gate']}。"
    )


def report(result: dict[str, Any]) -> str:
    validation_lines = "\n".join(row_line(row) for row in result["validation_ablation_rows"])
    auxiliary_lines = "\n".join(row_line(row) for row in result["auxiliary_hybrid_rows"])
    key2 = result["experiment_2_attribution_key"]
    key6 = result["experiment_6_rows_ge7_key"]
    judge = result["final_judgment"]

    return f"""
时间：{result['time']}

实验逻辑：把实验 1-6 的结果汇总成最终消融与边界判定，不再做新调参、不重新跑 official、不新增 scorer。核心是区分主方法候选、辅助 hybrid、诊断 scorer，并检查每条路线是否真的在 overall 和 rows>=7 的 match@1/5/20、RMSE、valid/formula/SG/exact-cover/skeleton-to-match conversion 上过 gate。

为什么做：前序结果已经显示 `C2S3C15` 在 MPTS-52 official K5/K20 过 +5pp，但它是 frozen auxiliary route；而 SymCIF/exact-cover/geometry repair 才是论文主线候选。实验 7 的作用是防止把普通 rerank 或 hybrid 包装成主贡献。

核心假设：如果主方法已经成立，generation-side proposer + learned geometry repair 应在 validation 上同时提高至少两个 match 指标，并且 rows>=7 不应只靠辅助 route；如果只有 C2S3C15 成立，则只能写成 auxiliary result，主方法仍需继续。

数据规模：综合 MPTS-52 validation K50 full 5000 samples/250000 candidates、MPTS-52 official full-test 8096 samples、MP-20 validation-like 9047 samples、SymCIF v5 MPTS-52 validation pool 25840 candidates/4574 covered samples、MP-20 GT-WA learned geometry 8874 samples。所有结果来自既有 JSON/metrics replay。

CPU/资源控制：bounded CPU use；本实验 used_parallel_workers=1，只读 JSON 汇总；不跑 StructureMatcher，不训练 GPU 模型。

validation 主线/诊断消融：
{validation_lines}

auxiliary hybrid 边界：
{auxiliary_lines}

收益归因关键点：实验 2 显示 C2S3C15 official 的 rows>=7 占 K5 净收益 {pct(key2.get('rows_ge7_k5_net_gain_share'))}、K20 净收益 {pct(key2.get('rows_ge7_k20_net_gain_share'))}；rows>=7 中 SymCIF top3 在 C1-C17 失败后独立救回 {key2.get('s_only_after_c17_rows_ge7')} 个样本。

rows>=7 关键点：实验 6 显示 CrystaLLM K50 rows>=7 top50={pct(key6.get('crystallm_k50_top50_rows_ge7'))}，SymCIF v5 rows>=7 top50={pct(key6.get('symcif_v5_top50_rows_ge7'))}，主瓶颈是 {key6.get('rows_ge7_primary_bottleneck')}。这说明复杂结构不是普通排序能解决。

和历史实验关系：Track A、hard-negative scorer、exact-cover filter/proxy、strict integrated ablation 都没有达到两个指标 +5pp；deterministic repair conversion=0；SymCIF v5 exact-cover feasible 高但 coverage/skeleton-to-match 还不够；learned geometry repair在 GT-WA 条件下有强信号但不是 inference result。

最终判决：main_method_success={judge['main_method_success']}；auxiliary_hybrid_success_mpts52={judge['auxiliary_hybrid_success_mpts52']}；mp20_overall_transfer_success={judge['mp20_overall_transfer_success']}；coverage_solved={judge['coverage_solved']}。因此 C2S3C15 只能作为 auxiliary hybrid official result；Track A/RF/HGB/hard-negative scorer 只保留为诊断或停止；主线继续 exact-cover constrained skeleton proposer + symmetry-preserving learned geometry repair，尤其 rows>=7 专门路线。

下一步：若继续迭代，必须先写明失败原因和预期提升，使用 MP-20/MPTS-52 train 数据训练 rows>=7-specialized skeleton proposer；先过 validation/half-train gate，再考虑 frozen official。禁止继续调 C/S 比例、ordinary rerank、threshold tuning 或 official feedback。
"""


def main() -> None:
    result = compute_result()
    write_json(RESULTS / "experiment_7_main_ablation_boundary.json", result)
    append_report_once(
        RESULTS / "experiment_7_report_appended.json",
        "实验 7 主方法消融与 hybrid 边界说明",
        report(result),
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "main_method_success": result["final_judgment"]["main_method_success"],
                "auxiliary_hybrid_success_mpts52": result["final_judgment"]["auxiliary_hybrid_success_mpts52"],
                "mp20_overall_transfer_success": result["final_judgment"]["mp20_overall_transfer_success"],
                "next_mainline": result["final_judgment"]["next_mainline"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
