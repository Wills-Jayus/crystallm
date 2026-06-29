#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

TRUE_ANCHOR_PATH = (
    NEW_MODEL
    / "opentry_7"
    / "metrics"
    / "crystallm_a_gt_sg_mpts_52_test_k20.json"
)
INTERNAL_BASELINE_PATH = (
    NEW_MODEL
    / "opentry_7"
    / "metrics"
    / "pure_crystallm_gt_sg_mpts_52_test_k20.json"
)
C2S3C15_PATH = (
    NEW_MODEL / "opentry_12" / "results" / "experiment_1_c2s3c15_official.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def triplet(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "match@1": float(metrics["match@1"]),
        "match@5": float(metrics["match@5"]),
        "match@20": float(metrics["match@20"]),
    }


def rmse_triplet(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "RMSE@1": float(metrics["RMSE@1"]),
        "RMSE@5": float(metrics["RMSE@5"]),
        "RMSE@20": float(metrics["RMSE@20"]),
    }


def pp(delta: float) -> float:
    return round(delta * 100.0, 3)


def fmt_pct(x: float) -> str:
    return f"{x * 100.0:.3f}"


def fmt_pp(x: float) -> str:
    return f"{x * 100.0:+.3f}pp"


def fmt_triplet(metrics: dict[str, Any], prefix: str = "match") -> str:
    keys = [f"{prefix}@1", f"{prefix}@5", f"{prefix}@20"]
    return " / ".join(fmt_pct(float(metrics[k])) for k in keys)


def fmt_rmse_triplet(metrics: dict[str, Any]) -> str:
    keys = ["RMSE@1", "RMSE@5", "RMSE@20"]
    return " / ".join(f"{float(metrics[k]):.6f}" for k in keys)


def fmt_delta_triplet(delta: dict[str, float]) -> str:
    return " / ".join(fmt_pp(delta[k]) for k in ["match@1", "match@5", "match@20"])


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {k: float(a[k]) - float(b[k]) for k in ["match@1", "match@5", "match@20"]}


def same_matcher(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return matcher_config(a) == matcher_config(b)


def matcher_config(payload: dict[str, Any]) -> Any:
    return payload.get("matcher", payload.get("structure_matcher"))


def no_append_duplicate(marker: str, section: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    if marker in text:
        return
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(marker)
        f.write("\n")
        f.write(section.rstrip())
        f.write("\n")


def quality_from_c2s(c2s: dict[str, Any]) -> dict[str, Any]:
    q = c2s["candidate_quality"]
    return {
        "valid_any@1": q["valid_any@1"],
        "valid_any@5": q["valid_any@5"],
        "valid_any@20": q["valid_any@20"],
        "slot_valid_rate": q["slot_valid_rate"],
        "formula_ok_any@1": q["formula_ok_any@1"],
        "formula_ok_any@5": q["formula_ok_any@5"],
        "formula_ok_any@20": q["formula_ok_any@20"],
        "slot_formula_consistency_rate": q["slot_formula_consistency_rate"],
        "space_group_ok_any@1": q["space_group_ok_any@1"],
        "space_group_ok_any@5": q["space_group_ok_any@5"],
        "space_group_ok_any@20": q["space_group_ok_any@20"],
        "slot_sg_consistency_rate": q["slot_sg_consistency_rate"],
        "known_exact_cover_any@1": q["known_exact_cover_any@1"],
        "known_exact_cover_any@5": q["known_exact_cover_any@5"],
        "known_exact_cover_any@20": q["known_exact_cover_any@20"],
        "symcif_slot_exact_cover_feasible_rate": q[
            "symcif_slot_exact_cover_feasible_rate"
        ],
        "symcif_skeleton_hit_candidates": q["symcif_skeleton_hit_candidates"],
        "symcif_skeleton_hit_to_match_conversion": q[
            "symcif_skeleton_hit_to_match_conversion"
        ],
        "source_counts": q["source_counts"],
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    true_anchor = read_json(TRUE_ANCHOR_PATH)
    internal = read_json(INTERNAL_BASELINE_PATH)
    c2s = read_json(C2S3C15_PATH)

    exp1 = {
        "experiment": "opentry_13_exp1_baseline_protocol_audit",
        "created_at_utc": now,
        "question": (
            "Check whether opentry_12 C2S3C15 low baseline and the global "
            "GT-SG CrystaLLM-a anchor share split/evaluator/input/candidate source."
        ),
        "paths": {
            "true_gt_sg_anchor": str(TRUE_ANCHOR_PATH),
            "low_internal_baseline": str(INTERNAL_BASELINE_PATH),
            "c2s3c15_official": str(C2S3C15_PATH),
        },
        "protocol_comparison": {
            "same_dataset": true_anchor.get("dataset") == internal.get("dataset"),
            "same_split": true_anchor.get("split") == internal.get("split"),
            "same_sample_count_all": true_anchor["all"]["samples"]
            == internal["all"]["samples"]
            == c2s["all"]["samples"],
            "same_sample_count_rows_ge7": true_anchor["rows_ge7"]["samples"]
            == internal["rows_ge7"]["samples"]
            == c2s["rows_ge7"]["samples"],
            "same_evaluator": same_matcher(true_anchor, internal)
            and same_matcher(true_anchor, c2s),
            "same_input_condition": (
                "Nominal composition + GT-SG for all compared files; however "
                "the 17.181/24.345/31.522 line is a pure-model candidate source, "
                "not the CrystaLLM-a GT-SG anchor."
            ),
            "same_candidate_source": False,
            "candidate_sources": {
                "true_gt_sg_anchor": true_anchor.get("candidate_source"),
                "low_internal_baseline": internal.get("candidate_source"),
                "c2s3c15": c2s.get("candidate_source"),
            },
            "same_model_family_or_checkpoint": False,
        },
        "metrics": {
            "true_gt_sg_anchor": {
                "all_match": triplet(true_anchor["all"]),
                "rows_ge7_match": triplet(true_anchor["rows_ge7"]),
                "all_rmse": rmse_triplet(true_anchor["all"]),
                "rows_ge7_rmse": rmse_triplet(true_anchor["rows_ge7"]),
            },
            "low_internal_baseline": {
                "all_match": triplet(internal["all"]),
                "rows_ge7_match": triplet(internal["rows_ge7"]),
                "all_rmse": rmse_triplet(internal["all"]),
                "rows_ge7_rmse": rmse_triplet(internal["rows_ge7"]),
            },
            "c2s3c15": {
                "all_match": triplet(c2s["all"]),
                "rows_ge7_match": triplet(c2s["rows_ge7"]),
                "all_rmse": rmse_triplet(c2s["all"]),
                "rows_ge7_rmse": rmse_triplet(c2s["rows_ge7"]),
            },
        },
        "verdict": {
            "low_baseline_is_main_anchor": False,
            "use_low_baseline_for_main_claim": False,
            "main_anchor": "CrystaLLM-a GT-SG MPTS-52 official full test: 25.23 / 36.46 / 43.96",
            "reason": (
                "The low baseline shares split/evaluator but uses a different "
                "candidate source/model and is explicitly the pure-model line."
            ),
        },
    }

    c2s_delta_true_all = delta(c2s["all"], true_anchor["all"])
    c2s_delta_true_rows = delta(c2s["rows_ge7"], true_anchor["rows_ge7"])
    c2s_delta_internal_all = delta(c2s["all"], internal["all"])
    c2s_delta_internal_rows = delta(c2s["rows_ge7"], internal["rows_ge7"])
    passes_plus5 = sum(v >= 0.05 for v in c2s_delta_true_all.values()) >= 2
    rows_not_degraded = all(v >= 0.0 for v in c2s_delta_true_rows.values())

    exp2 = {
        "experiment": "opentry_13_exp2_c2s3c15_true_gt_sg_anchor_replay_audit",
        "created_at_utc": now,
        "restriction": (
            "Fixed C2S3C15; no C/S ratio tuning, threshold tuning, scorer, "
            "or official-result feedback."
        ),
        "dataset": {
            "name": true_anchor.get("dataset"),
            "split": true_anchor.get("split"),
            "samples_all": true_anchor["all"]["samples"],
            "samples_rows_ge7": true_anchor["rows_ge7"]["samples"],
                "matcher": matcher_config(true_anchor),
        },
        "systems": {
            "internal_low_baseline_pure_model": {
                "source": str(INTERNAL_BASELINE_PATH),
                "candidate_source": internal.get("candidate_source"),
                "all_match": triplet(internal["all"]),
                "all_rmse": rmse_triplet(internal["all"]),
                "rows_ge7_match": triplet(internal["rows_ge7"]),
                "rows_ge7_rmse": rmse_triplet(internal["rows_ge7"]),
                "quality_diagnostics": {
                    "available": False,
                    "reason": "opentry_7 metric JSON stores match/RMSE only, not valid/formula/SG/exact-cover diagnostics.",
                },
            },
            "true_gt_sg_anchor_crystallm_a": {
                "source": str(TRUE_ANCHOR_PATH),
                "candidate_source": true_anchor.get("candidate_source"),
                "all_match": triplet(true_anchor["all"]),
                "all_rmse": rmse_triplet(true_anchor["all"]),
                "rows_ge7_match": triplet(true_anchor["rows_ge7"]),
                "rows_ge7_rmse": rmse_triplet(true_anchor["rows_ge7"]),
                "quality_diagnostics": {
                    "available": False,
                    "reason": "opentry_7 metric JSON stores match/RMSE only, not valid/formula/SG/exact-cover diagnostics.",
                },
            },
            "c2s3c15": {
                "source": str(C2S3C15_PATH),
                "candidate_source": c2s.get("candidate_source"),
                "all_match": triplet(c2s["all"]),
                "all_rmse": rmse_triplet(c2s["all"]),
                "rows_ge7_match": triplet(c2s["rows_ge7"]),
                "rows_ge7_rmse": rmse_triplet(c2s["rows_ge7"]),
                "quality_diagnostics": quality_from_c2s(c2s),
            },
        },
        "deltas": {
            "c2s3c15_vs_true_gt_sg_anchor_all": c2s_delta_true_all,
            "c2s3c15_vs_true_gt_sg_anchor_rows_ge7": c2s_delta_true_rows,
            "c2s3c15_vs_internal_low_baseline_all": c2s_delta_internal_all,
            "c2s3c15_vs_internal_low_baseline_rows_ge7": c2s_delta_internal_rows,
        },
        "gate": {
            "requires_at_least_two_match_metrics_plus5pp_vs_true_anchor": passes_plus5,
            "requires_rows_ge7_not_degraded": rows_not_degraded,
            "passes_main_claim_gate": passes_plus5 and rows_not_degraded,
        },
        "verdict": {
            "classification": "auxiliary_hybrid_result",
            "reason": (
                "C2S3C15 is below the true GT-SG anchor on match@1/5/20 "
                "overall and rows>=7, so it cannot support the main claim."
            ),
        },
    }

    write_json(RESULT_DIR / "experiment_1_baseline_protocol_audit.json", exp1)
    write_json(RESULT_DIR / "experiment_2_c2s3c15_true_anchor_replay.json", exp2)

    exp1_section = f"""## opentry_13 实验 1：主 baseline 口径复核

结果文件：`model/New_model/opentry_13/results/experiment_1_baseline_protocol_audit.json`

- 为什么做：opentry_12 C2S3C15 official 使用了 `17.181 / 24.345 / 31.522` 作为 baseline，但全局主 GT-SG CrystaLLM-a anchor 是 `25.23 / 36.46 / 43.96`。本实验只复核 split、evaluator、input condition 与 candidate source，防止用低 baseline 宣称主目标。
- 核心假设：如果两个 baseline 不是同一 candidate source / model checkpoint，即使 split 和 evaluator 相同，也不能混用为主比较 anchor。
- 数据规模：MPTS-52 official test，overall `{true_anchor['all']['samples']}` 个样本；rows>=7 `{true_anchor['rows_ge7']['samples']}` 个样本。
- baseline：低 internal baseline 是 pure CrystaLLM GT-SG line，match@1/5/20 = `{fmt_triplet(internal['all'])}`；true GT-SG anchor 是 CrystaLLM-a GT-SG，match@1/5/20 = `{fmt_triplet(true_anchor['all'])}`。
- 口径核查：split 相同，均为 MPTS-52 official test；evaluator 相同，StructureMatcher 参数为 `{matcher_config(true_anchor)}`；输入条件名义上均为 composition + GT-SG；candidate source 不同，低 baseline 来自 `{internal.get('candidate_source')}`，true anchor 来自 `{true_anchor.get('candidate_source')}`，C2S3C15 来自 `{c2s.get('candidate_source')}`。
- 结果：`17.181 / 24.345 / 31.522` 不是全局主 GT-SG CrystaLLM-a anchor，而是 opentry_7 pure model line。C2S3C15 不能使用该低 baseline 作为主目标达成依据。
- 可信度：高。结论直接来自 opentry_7 机器 JSON、opentry_12 机器 JSON 与全局 bundle 中已登记的 anchor，且 sample count / matcher 参数一致可核验。
- 和历史实验关系：opentry_12 的 C2S3C15 仍可作为辅助 hybrid 结果，但不能替代 opentry_7/opentry_10 登记的主 anchor。
- 最终判决：主比较必须使用 true GT-SG CrystaLLM-a anchor：`25.23 / 36.46 / 43.96`。
- 下一步：在 fixed C2S3C15 条件下重算其相对 true anchor 的 delta，并决定是否降级。
"""

    exp2_section = f"""## opentry_13 实验 2：C2S3C15 true GT-SG anchor replay / audit

结果文件：`model/New_model/opentry_13/results/experiment_2_c2s3c15_true_anchor_replay.json`

- 为什么做：固定 C2S3C15，不再调整 C/S 比例、不做 threshold tuning、不接 scorer，只回答它在 true GT-SG anchor 口径下是否真的超过主目标。
- 核心假设：如果 C2S3C15 只是相对 pure-model low baseline 有提升，而相对 CrystaLLM-a GT-SG anchor 没有 +5pp，则只能是 auxiliary hybrid result。
- 数据规模：MPTS-52 official test，overall `{true_anchor['all']['samples']}`；rows>=7 `{true_anchor['rows_ge7']['samples']}`；candidate records `{c2s['candidate_quality']['evaluated_candidate_records']}`。
- baseline：internal low baseline match@1/5/20 = `{fmt_triplet(internal['all'])}`，rows>=7 = `{fmt_triplet(internal['rows_ge7'])}`；true GT-SG anchor match@1/5/20 = `{fmt_triplet(true_anchor['all'])}`，rows>=7 = `{fmt_triplet(true_anchor['rows_ge7'])}`。
- 方法变化：无新方法。仅把 opentry_12 fixed C2S3C15 official 结果改挂到 true GT-SG anchor 上审计。
- 结果：C2S3C15 overall match@1/5/20 = `{fmt_triplet(c2s['all'])}`，RMSE = `{fmt_rmse_triplet(c2s['all'])}`；rows>=7 match@1/5/20 = `{fmt_triplet(c2s['rows_ge7'])}`，RMSE = `{fmt_rmse_triplet(c2s['rows_ge7'])}`。相对 true anchor overall delta = `{fmt_delta_triplet(c2s_delta_true_all)}`；rows>=7 delta = `{fmt_delta_triplet(c2s_delta_true_rows)}`。相对 internal low baseline overall delta = `{fmt_delta_triplet(c2s_delta_internal_all)}`。
- 结构质量：C2S3C15 valid any@1/5/20 = `{fmt_pct(c2s['candidate_quality']['valid_any@1'])}` / `{fmt_pct(c2s['candidate_quality']['valid_any@5'])}` / `{fmt_pct(c2s['candidate_quality']['valid_any@20'])}`；formula consistency slot rate = `{fmt_pct(c2s['candidate_quality']['slot_formula_consistency_rate'])}`；SG consistency slot rate = `{fmt_pct(c2s['candidate_quality']['slot_sg_consistency_rate'])}`；known exact-cover any@20 = `{fmt_pct(c2s['candidate_quality']['known_exact_cover_any@20'])}`；SymCIF slot exact-cover feasible = `{fmt_pct(c2s['candidate_quality']['symcif_slot_exact_cover_feasible_rate'])}`。
- 可信度：高。没有使用 official 结果回调；只读取冻结 C2S3C15 official JSON 与 opentry_7 anchor JSON。internal baseline 和 true anchor 的 valid/formula/SG/exact-cover 诊断在 opentry_7 指标 JSON 中未记录，因此报告为不可用，不做外推。
- 和历史实验关系：opentry_12 中 C2S3C15 相对 pure baseline 提升 match@5/20，但在 true CrystaLLM-a GT-SG anchor 下 match@1/5/20 全部为负 delta。
- 最终判决：C2S3C15 未达到 true anchor +5pp，也没有 rows>=7 不恶化；降级为 auxiliary hybrid result，禁止作为论文主方法 claim。
- 下一步：转向 rows>=7-specialized skeleton proposer，目标是让候选池中真实增加可 match 的 skeleton/geometry，而不是继续调 C/S 比例或 scorer。
"""

    no_append_duplicate("<!-- OPENTRY13_EXP1_BASELINE_AUDIT -->", exp1_section)
    no_append_duplicate("<!-- OPENTRY13_EXP2_C2S3C15_TRUE_ANCHOR -->", exp2_section)
    print(RESULT_DIR / "experiment_1_baseline_protocol_audit.json")
    print(RESULT_DIR / "experiment_2_c2s3c15_true_anchor_replay.json")


if __name__ == "__main__":
    main()
