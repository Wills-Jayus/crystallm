#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

OP12_SCRIPT = NEW_MODEL / "opentry_12" / "run_opentry12_experiments.py"
TRUE_CRYSTALLM_TAR = NEW_MODEL / "opentry_7" / "generations" / "crystallm_a_gt_sg_mpts_52_test.tar.gz"
TRUE_ANCHOR_METRICS = NEW_MODEL / "opentry_7" / "metrics" / "crystallm_a_gt_sg_mpts_52_test_k20.json"
PURE_SOURCE_C2S3C15 = NEW_MODEL / "opentry_12" / "results" / "experiment_1_c2s3c15_official.json"
PURE_SOURCE_C2S3C15_CANDIDATES = NEW_MODEL / "opentry_12" / "candidates" / "c2s3c15_mpts52_test_k20.jsonl.gz"

TRUE_CANDIDATE_NAME = "c2s3c15_true_anchor_source_mpts52_test_k20"
TRUE_RESULT_NAME = "experiment_1_true_anchor_source_c2s3c15"
EVAL_NAME_DEFAULT = "c2s3c15_true_anchor_source_mpts52_test_20260629"

BUDGETS = (1, 5, 20)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_op12_module():
    spec = importlib.util.spec_from_file_location("opentry12_runner_for_true_anchor", OP12_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {OP12_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_op12(module: Any) -> None:
    module.OUT_DIR = OUT_DIR
    module.RESULTS = OUT_DIR / "results"
    module.CACHE = OUT_DIR / "cache"
    module.CANDIDATES = OUT_DIR / "candidates"
    module.OFFICIAL_EVAL = OUT_DIR / "official_eval"
    module.CRYSTALLM_TAR = TRUE_CRYSTALLM_TAR
    module.CRYSTALLM_BASELINE = TRUE_ANCHOR_METRICS
    module.MAX_CPU_WORKERS = int(os.environ.get("OPENTRY13_MAX_CPU_WORKERS", "8"))


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f}pp"


def triplet(summary: dict[str, Any], prefix: str = "match@") -> str:
    return " / ".join(pct(summary.get(f"{prefix}{k}")) for k in BUDGETS)


def rmse_triplet(summary: dict[str, Any]) -> str:
    return " / ".join(
        "NA" if summary.get(f"RMSE@{k}") is None else f"{float(summary[f'RMSE@{k}']):.6f}"
        for k in BUDGETS
    )


def delta_triplet(delta: dict[str, Any]) -> str:
    return " / ".join(pp(delta.get(f"match@{k}")) for k in BUDGETS)


def delta_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {f"match@{k}": float(a[f"match@{k}"]) - float(b[f"match@{k}"]) for k in BUDGETS}


def append_or_replace_report(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker in text:
        start = text.index(marker)
        next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
        if next_marker == -1:
            REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement, encoding="utf-8")
        else:
            REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement + text[next_marker:], encoding="utf-8")
        return
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(replacement)


def build_true_anchor_result(official: dict[str, Any], route_manifest: dict[str, Any]) -> dict[str, Any]:
    true_anchor = read_json(TRUE_ANCHOR_METRICS)
    pure_c2s = read_json(PURE_SOURCE_C2S3C15)
    delta_true_all = delta_metrics(official["all"], true_anchor["all"])
    delta_true_rows = delta_metrics(official["rows_ge7"], true_anchor["rows_ge7"])
    delta_pure_old_all = delta_metrics(official["all"], pure_c2s["all"])
    delta_pure_old_rows = delta_metrics(official["rows_ge7"], pure_c2s["rows_ge7"])
    match5_20_positive = delta_true_all["match@5"] >= 0.05 and delta_true_all["match@20"] >= 0.05
    rows_not_down = all(delta_true_rows[f"match@{k}"] >= 0.0 for k in BUDGETS)
    any_k5_k20_negative = delta_true_all["match@5"] < 0.0 or delta_true_all["match@20"] < 0.0
    if match5_20_positive and rows_not_down:
        verdict = "keep_as_auxiliary_hybrid_result"
        reason = "true-anchor-source C2S3C15 passes +5pp on match@5 and match@20 and rows>=7 is not degraded."
    elif any_k5_k20_negative:
        verdict = "stop_c2s3c15_direction"
        reason = "true-anchor-source C2S3C15 has a negative delta on match@5 or match@20 versus the true anchor."
    elif 0.0 <= delta_true_all["match@5"] <= 0.03 or 0.0 <= delta_true_all["match@20"] <= 0.03:
        verdict = "diagnostic_only"
        reason = "true-anchor-source C2S3C15 improves only 0 to +3pp on at least one key metric."
    else:
        verdict = "not_official_main_method"
        reason = "true-anchor-source C2S3C15 does not satisfy the saved acceptance gate."
    return {
        "created_at_utc": now_iso(),
        "experiment": TRUE_RESULT_NAME,
        "question": "Fixed C2S3C15 replay using true CrystaLLM-a GT-SG anchor candidates as C source.",
        "dataset": "mpts_52",
        "split": "test",
        "route": "C2S3C15",
        "sequence": ["C", "C", "S", "S", "S"] + ["C"] * 15,
        "restrictions": [
            "true CrystaLLM-a GT-SG candidate source for C slots",
            "SymCIF fullgen test pool for S slots",
            "fixed C1,C2,S1,S2,S3,C3...C17 order",
            "no ratio search",
            "no threshold tuning",
            "no scorer/reranker",
        ],
        "paths": {
            "true_anchor_metrics": str(TRUE_ANCHOR_METRICS),
            "true_anchor_c_source": str(TRUE_CRYSTALLM_TAR),
            "old_pure_source_c2s3c15": str(PURE_SOURCE_C2S3C15),
            "old_pure_source_c2s3c15_candidates": str(PURE_SOURCE_C2S3C15_CANDIDATES),
            "candidate_source": official["candidate_source"],
            "sample_metrics": official["sample_metrics"],
            "candidate_eval": official["candidate_eval"],
        },
        "route_manifest": route_manifest,
        "systems": {
            "true_crystallm_a_gt_sg_C20_anchor": {
                "all": true_anchor["all"],
                "rows_ge7": true_anchor["rows_ge7"],
                "candidate_source": true_anchor.get("candidate_source"),
            },
            "old_pure_source_C2S3C15": {
                "all": pure_c2s["all"],
                "rows_ge7": pure_c2s["rows_ge7"],
                "candidate_source": pure_c2s.get("candidate_source"),
                "delta_vs_true_anchor_all": delta_metrics(pure_c2s["all"], true_anchor["all"]),
                "delta_vs_true_anchor_rows_ge7": delta_metrics(pure_c2s["rows_ge7"], true_anchor["rows_ge7"]),
            },
            "true_anchor_source_C2S3C15": {
                "all": official["all"],
                "rows_ge7": official["rows_ge7"],
                "candidate_quality": official["candidate_quality"],
                "candidate_source": official["candidate_source"],
                "delta_vs_true_anchor_all": delta_true_all,
                "delta_vs_true_anchor_rows_ge7": delta_true_rows,
                "delta_vs_old_pure_source_C2S3C15_all": delta_pure_old_all,
                "delta_vs_old_pure_source_C2S3C15_rows_ge7": delta_pure_old_rows,
            },
        },
        "acceptance_gate": {
            "match5_and_match20_plus5pp_vs_true_anchor": match5_20_positive,
            "rows_ge7_not_degraded": rows_not_down,
            "any_match5_or_match20_negative": any_k5_k20_negative,
            "verdict": verdict,
            "reason": reason,
        },
    }


def report_body(result: dict[str, Any]) -> str:
    true_anchor = result["systems"]["true_crystallm_a_gt_sg_C20_anchor"]
    old_c2s = result["systems"]["old_pure_source_C2S3C15"]
    new_c2s = result["systems"]["true_anchor_source_C2S3C15"]
    q = new_c2s["candidate_quality"]
    route_manifest = result["route_manifest"]
    return f"""## opentry_13 实验 1：true-anchor-source C2S3C15 fixed hybrid 重放

结果文件：`model/New_model/opentry_13/results/{TRUE_RESULT_NAME}.json`

- 实验逻辑：固定 `C1,C2,S1,S2,S3,C3...C17`，把 C source 从旧 pure CrystaLLM GT-SG 切换为 true CrystaLLM-a GT-SG anchor；SymCIF source 仍用已有 MPTS-52 test fullgen pool。只重放固定候选序列，不搜索比例、不调阈值、不接 scorer。
- 为什么做：opentry_12 的 C2S3C15 用的是低 baseline/pure candidate source，不能作为主 anchor 比较；本实验直接检查复杂结构互补信号在 true anchor 上是否仍能转化为 match@5/20 增益。
- 数据规模：MPTS-52 official test {new_c2s['all']['samples']} samples；rows>=7 {new_c2s['rows_ge7']['samples']} samples；route records={q['evaluated_candidate_records']}；source counts={q['source_counts']}。
- baseline：true CrystaLLM-a GT-SG C20 overall match@1/5/20 = `{triplet(true_anchor['all'])}`，rows>=7 = `{triplet(true_anchor['rows_ge7'])}`；RMSE overall = `{rmse_triplet(true_anchor['all'])}`，rows>=7 = `{rmse_triplet(true_anchor['rows_ge7'])}`。
- 比较对象：old pure-source C2S3C15 overall = `{triplet(old_c2s['all'])}`，rows>=7 = `{triplet(old_c2s['rows_ge7'])}`；true-anchor-source C2S3C15 overall = `{triplet(new_c2s['all'])}`，rows>=7 = `{triplet(new_c2s['rows_ge7'])}`。
- 结果：true-anchor-source C2S3C15 相对 true anchor overall delta = `{delta_triplet(new_c2s['delta_vs_true_anchor_all'])}`；rows>=7 delta = `{delta_triplet(new_c2s['delta_vs_true_anchor_rows_ge7'])}`；相对 old pure-source C2S3C15 overall delta = `{delta_triplet(new_c2s['delta_vs_old_pure_source_C2S3C15_all'])}`。
- RMSE：true-anchor-source C2S3C15 overall RMSE@1/5/20 = `{rmse_triplet(new_c2s['all'])}`；rows>=7 RMSE@1/5/20 = `{rmse_triplet(new_c2s['rows_ge7'])}`。
- valid/formula/SG/exact-cover：valid any@1/5/20 = `{pct(q['valid_any@1'])}` / `{pct(q['valid_any@5'])}` / `{pct(q['valid_any@20'])}`；formula any@1/5/20 = `{pct(q['formula_ok_any@1'])}` / `{pct(q['formula_ok_any@5'])}` / `{pct(q['formula_ok_any@20'])}`；SG any@1/5/20 = `{pct(q['space_group_ok_any@1'])}` / `{pct(q['space_group_ok_any@5'])}` / `{pct(q['space_group_ok_any@20'])}`；known exact-cover any@20 = `{pct(q['known_exact_cover_any@20'])}`；SymCIF slot exact-cover feasible = `{pct(q['symcif_slot_exact_cover_feasible_rate'])}`。
- SymCIF skeleton-to-match：SymCIF candidates={q['symcif_candidates']}，skeleton-hit candidates={q['symcif_skeleton_hit_candidates']}，skeleton-hit-to-match conversion=`{pct(q['symcif_skeleton_hit_to_match_conversion'])}`。
- 缺失与 fallback：missing SymCIF samples full-C fallback={route_manifest['missing_symcif_samples_full_crystallm_fallback']}；partial SymCIF samples slot fallback={route_manifest['partial_symcif_samples_slot_fallback']}；missing CrystaLLM samples={route_manifest['missing_crystallm_samples']}。无 SymCIF artifact 的样本统一回退 true-anchor C20；S 槽不足时该槽回退下一条 C。
- 可信度：高。候选顺序固定，C/S 路由不使用 match/RMSE/StructureMatcher/test feedback/GT-WA/GT-skeleton；唯一变化是把 C source 改成 true anchor source，并重新跑同一 StructureMatcher evaluator。
- 和历史实验关系：该实验修正 opentry_12/13 旧报告的 anchor-source 口径；old pure-source C2S3C15 只能说明相对低 baseline 的互补，不再作为主比较。
- 最终判决：`{result['acceptance_gate']['verdict']}`。原因：{result['acceptance_gate']['reason']} 无论该结果如何，C2S3C15 仍只能作为 auxiliary/diagnostic，不能写成论文主方法。
- 下一步：若判决为 stop 或 diagnostic，停止 C2S3C15 调参，把主线继续放在 predicted skeleton renderer/site mapping 与 geometry repair gate。
"""


def run(args: argparse.Namespace) -> None:
    module = load_op12_module()
    configure_op12(module)
    for path in (module.RESULTS, module.CACHE, module.CANDIDATES, module.OFFICIAL_EVAL):
        path.mkdir(parents=True, exist_ok=True)

    targets = module.load_targets(refresh=args.refresh_targets)
    route_path = module.CANDIDATES / f"{TRUE_CANDIDATE_NAME}.jsonl.gz"
    manifest_path = module.CANDIDATES / f"{TRUE_CANDIDATE_NAME}.manifest.json"

    if route_path.exists() and manifest_path.exists() and args.reuse_existing:
        route_manifest = read_json(manifest_path)
    else:
        crystallm = module.load_crystallm_candidates()
        symcif = module.load_symcif_candidates(include_diagnostics=False)
        route_records, route_manifest = module.build_route_candidate_records(targets, crystallm, symcif)
        tmp_path = module.write_route_candidates(route_records, route_manifest)
        tmp_manifest = module.CANDIDATES / "c2s3c15_mpts52_test_k20.manifest.json"
        tmp_path.rename(route_path)
        route_manifest = read_json(tmp_manifest)
        route_manifest["candidate_file"] = str(route_path)
        route_manifest["crystallm_source_kind"] = "true_crystallm_a_gt_sg_anchor"
        route_manifest["true_anchor_source_note"] = "C slots use crystallm_a_gt_sg_mpts_52_test.tar.gz, not pure_crystallm_gt_sg_mpts_52_test.tar.gz."
        write_json(manifest_path, route_manifest)

    summary_path = module.OFFICIAL_EVAL / args.eval_name / "summary.json"
    if summary_path.exists() and args.reuse_existing:
        official = read_json(summary_path)
    elif args.audit_only:
        audit = {
            "created_at_utc": now_iso(),
            "audit_pass": TRUE_CRYSTALLM_TAR.exists()
            and TRUE_ANCHOR_METRICS.exists()
            and Path(route_manifest["candidate_file"]).exists(),
            "route_manifest": route_manifest,
            "true_anchor_metrics": str(TRUE_ANCHOR_METRICS),
        }
        write_json(RESULT_DIR / f"{TRUE_RESULT_NAME}_audit.json", audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    else:
        official = module.evaluate_official_once(
            route_path,
            targets,
            eval_name=args.eval_name,
            workers=module.clamp_cpu_workers(args.workers),
            timeout_s=args.candidate_timeout_seconds,
            sample_timeout_s=args.sample_timeout_seconds,
            resume=args.resume,
        )

    result = build_true_anchor_result(official, route_manifest)
    write_json(RESULT_DIR / f"{TRUE_RESULT_NAME}.json", result)
    append_or_replace_report("<!-- OPENTRY13_EXP1_TRUE_ANCHOR_C2S3C15 -->", report_body(result))
    print(
        json.dumps(
            {
                "result": str(RESULT_DIR / f"{TRUE_RESULT_NAME}.json"),
                "verdict": result["acceptance_gate"],
                "true_anchor_source_all": result["systems"]["true_anchor_source_C2S3C15"]["all"],
                "delta_vs_true_anchor_all": result["systems"]["true_anchor_source_C2S3C15"]["delta_vs_true_anchor_all"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run opentry_13 true-anchor-source C2S3C15 replay")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--refresh-targets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--eval-name", default=EVAL_NAME_DEFAULT)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
