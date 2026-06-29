#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_12"
RESULTS = OUT_DIR / "results"
REPORT = ROOT / "model/New_model/opentry_11/GPT_REVIEW_BUNDLE.md"

TARGETS = ROOT / "model/New_model/opentry_7/cache/mpts_52_val_targets.jsonl"
SYMCIF_GEN = ROOT / "runs/symcif_v5_multidataset_wa_decoder/mpts52/val/generations/v5_fullgen_eval_pool.jsonl"
SYMCIF_MET = ROOT / "runs/symcif_v5_multidataset_wa_decoder/mpts52/val/metrics/v5_fullgen_eval_pool_metrics.jsonl"
HALF_AUDIT = ROOT / "model/New_model/opentry_11/results/experiment_7_half_exact_cover_generation.json"

BUDGETS = (1, 5, 20, 50)
BASELINE_FULL = {
    "match@1": 0.30020,
    "match@5": 0.40480,
    "match@20": 0.48000,
    "rows>=7_match@1": 0.05323,
    "rows>=7_match@5": 0.09991,
    "rows>=7_match@20": 0.14747,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_opentry12(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_12: {resolved}")
    return resolved


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


def ratio(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def material_id_from_sample_id(sample_id: str) -> str:
    return str(sample_id).split("__")[-1]


def load_targets() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with TARGETS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["material_id"])] = row
    return out


def load_candidates() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with SYMCIF_GEN.open("r", encoding="utf-8") as gf, SYMCIF_MET.open("r", encoding="utf-8") as mf:
        for gen_line, met_line in zip(gf, mf):
            if not gen_line.strip() or not met_line.strip():
                continue
            gen = json.loads(gen_line)
            met = json.loads(met_line)
            mid = material_id_from_sample_id(str(gen["sample_id"]))
            groups[mid].append(
                {
                    "sample_id": gen["sample_id"],
                    "material_id": mid,
                    "gen_index": int(gen.get("gen_index", 0)),
                    "generation_score": float(gen.get("generation_score") if gen.get("generation_score") is not None else -1.0e30),
                    "row_count_target": gen.get("row_count_target"),
                    "match": bool(met.get("match_ok")),
                    "rms": met.get("rms"),
                    "valid": bool(met.get("valid")),
                    "formula_ok": bool(met.get("formula_ok")),
                    "space_group_ok": bool(met.get("space_group_ok")),
                    "exact_cover_feasible": bool(met.get("multiplicity_ok")) if met.get("multiplicity_ok") is not None else None,
                    "skeleton_hit": gen.get("skeleton_hit"),
                    "wa_hit": gen.get("wa_hit"),
                    "row_count_hit": gen.get("row_count_hit"),
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda r: (float(r.get("generation_score") or -1.0e30), -int(r.get("gen_index") or 0)), reverse=True)
    return dict(groups)


def summarize(groups: dict[str, list[dict[str, Any]]], targets: dict[str, dict[str, Any]], rows_ge7_only: bool, missing_as_fail: bool) -> dict[str, Any]:
    mids = sorted(targets) if missing_as_fail else sorted(groups)
    if rows_ge7_only:
        mids = [mid for mid in mids if int(targets.get(mid, {}).get("row_count") or 0) >= 7]
    out: dict[str, Any] = {
        "samples": len(mids),
        "covered_samples": sum(1 for mid in mids if mid in groups),
        "missing_samples": sum(1 for mid in mids if mid not in groups),
        "mean_candidates_per_covered_sample": mean([float(len(groups[mid])) for mid in mids if mid in groups]),
    }
    for k in BUDGETS:
        hit = 0
        rms_vals: list[float] = []
        valid_any = formula_any = sg_any = exact_any = skel_any = wa_any = 0
        skel_any_and_match = 0
        for mid in mids:
            top = groups.get(mid, [])[:k]
            matched = [r for r in top if r.get("match") is True]
            if matched:
                hit += 1
                rms = [float(r["rms"]) for r in matched if r.get("rms") is not None]
                if rms:
                    rms_vals.append(min(rms))
            valid_any += int(any(r.get("valid") is True for r in top))
            formula_any += int(any(r.get("formula_ok") is True for r in top))
            sg_any += int(any(r.get("space_group_ok") is True for r in top))
            exact_any += int(any(r.get("exact_cover_feasible") is True for r in top))
            skel = any(r.get("skeleton_hit") is True for r in top)
            skel_any += int(skel)
            skel_any_and_match += int(skel and bool(matched))
            wa_any += int(any(r.get("wa_hit") is True for r in top))
        out[f"top{k}_match_coverage"] = ratio(hit, len(mids))
        out[f"top{k}_hits"] = hit
        out[f"top{k}_RMSE"] = mean(rms_vals)
        out[f"top{k}_valid_any"] = ratio(valid_any, len(mids))
        out[f"top{k}_formula_ok_any"] = ratio(formula_any, len(mids))
        out[f"top{k}_sg_ok_any"] = ratio(sg_any, len(mids))
        out[f"top{k}_exact_cover_feasible_any"] = ratio(exact_any, len(mids))
        out[f"top{k}_skeleton_hit_any"] = ratio(skel_any, len(mids))
        out[f"top{k}_wa_hit_any"] = ratio(wa_any, len(mids))
        out[f"top{k}_sample_skeleton_to_match_conversion"] = ratio(skel_any_and_match, skel_any)
    return out


def candidate_level_diagnostics(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [r for arr in groups.values() for r in arr]
    skel = [r for r in rows if r.get("skeleton_hit") is True]
    return {
        "candidate_count": len(rows),
        "samples": len(groups),
        "valid_rate": ratio(sum(r.get("valid") is True for r in rows), len(rows)),
        "formula_consistency": ratio(sum(r.get("formula_ok") is True for r in rows), len(rows)),
        "sg_consistency": ratio(sum(r.get("space_group_ok") is True for r in rows), len(rows)),
        "exact_cover_feasible_rate": ratio(sum(r.get("exact_cover_feasible") is True for r in rows), len(rows)),
        "skeleton_hit_rate": ratio(len(skel), len(rows)),
        "wa_hit_rate": ratio(sum(r.get("wa_hit") is True for r in rows), len(rows)),
        "candidate_skeleton_to_match_conversion": ratio(sum(r.get("match") is True for r in skel), len(skel)),
    }


def build_experiment() -> dict[str, Any]:
    targets = load_targets()
    groups = load_candidates()
    half = json.loads(HALF_AUDIT.read_text(encoding="utf-8"))
    full_all_overlap = summarize(groups, targets, rows_ge7_only=False, missing_as_fail=False)
    full_rows7_overlap = summarize(groups, targets, rows_ge7_only=True, missing_as_fail=False)
    full_all_missing_fail = summarize(groups, targets, rows_ge7_only=False, missing_as_fail=True)
    full_rows7_missing_fail = summarize(groups, targets, rows_ge7_only=True, missing_as_fail=True)
    result = {
        "time": now_iso(),
        "experiment": "experiment_5_neural_skeleton_geometry_proposer_audit",
        "dataset": "mpts_52",
        "split": "val",
        "cpu_safeguard": {
            "structurematcher_rerun": False,
            "parallel_workers": 1,
            "thread_env_defaults": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
        },
        "inputs": {
            "targets": str(TARGETS),
            "symcif_generation": str(SYMCIF_GEN),
            "symcif_metrics": str(SYMCIF_MET),
            "half_audit": str(HALF_AUDIT),
        },
        "target_samples": len(targets),
        "target_rows_ge7_samples": sum(int(r.get("row_count") or 0) >= 7 for r in targets.values()),
        "covered_samples": len(groups),
        "covered_rows_ge7_samples": sum(1 for mid in groups if int(targets.get(mid, {}).get("row_count") or 0) >= 7),
        "candidate_level_diagnostics": candidate_level_diagnostics(groups),
        "overlap": {"all": full_all_overlap, "rows_ge7": full_rows7_overlap},
        "full_validation_missing_as_fail": {"all": full_all_missing_fail, "rows_ge7": full_rows7_missing_fail},
        "baseline_full_validation": BASELINE_FULL,
        "half_audit_summary": half,
        "decision": {
            "top50_overall_exceeds_baseline20_by_5pp_missing_fail": (
                full_all_missing_fail["top50_match_coverage"] - BASELINE_FULL["match@20"] >= 0.05
            ),
            "top50_rows7_exceeds_baseline20_by_5pp_missing_fail": (
                full_rows7_missing_fail["top50_match_coverage"] - BASELINE_FULL["rows>=7_match@20"] >= 0.05
            ),
            "covered_artifact_missing_samples": len(targets) - len(groups),
            "main_blocker": "top50 proposer coverage and skeleton-to-match conversion are not enough for overall +5pp unless paired with stronger proposer/geometry repair",
        },
    }
    write_json(RESULTS / "experiment_5_skeleton_proposer_audit.json", result)
    return result


def write_report(result: dict[str, Any]) -> None:
    all_full = result["full_validation_missing_as_fail"]["all"]
    r7_full = result["full_validation_missing_as_fail"]["rows_ge7"]
    all_overlap = result["overlap"]["all"]
    r7_overlap = result["overlap"]["rows_ge7"]
    diag = result["candidate_level_diagnostics"]
    half_pool = result["half_audit_summary"]["fullgen_pool"]
    body = f"""
时间：{result['time']}

实验逻辑：实验 5 不训练新模型，先审计现有 neural/fullgen skeleton-geometry proposer 的 coverage 上限。读取 SymCIF v5 MPTS-52 validation fullgen pool 的 generation/metrics，按 generation_score 固定排序，计算 top1/top5/top20/top50 oracle coverage、rows>=7 coverage、exact-cover feasible、skeleton-to-match conversion。缺失 artifact 的样本在 full-validation 口径下按 fail 处理。

为什么做：实验 4 说明 learned geometry repair 在 GT-WA/GT-skeleton 条件下有效，但 inference 主线还缺能提出正确 skeleton/geometry 的 proposer。若 top50 coverage 不足，任何 scorer/rerank 都无法解决 K20。

核心假设：如果 neural skeleton/geometry proposer 真能扩展候选池，top50 match coverage 应明显超过 CrystaLLM K20 baseline，rows>=7 top50 也应提升；如果 skeleton_hit 低或 skeleton-to-match conversion 低，说明还需要更强 skeleton proposer 和 learned geometry repair 联动。

数据规模：MPTS-52 validation targets={result['target_samples']}；rows>=7 targets={result['target_rows_ge7_samples']}；SymCIF covered samples={result['covered_samples']}；covered rows>=7={result['covered_rows_ge7_samples']}；candidate_count={diag['candidate_count']}。missing samples={result['decision']['covered_artifact_missing_samples']}。

CPU/资源控制：parallel_workers=1；未运行 StructureMatcher；只复用已有 metric labels；线程环境默认限制为 OMP/MKL/OPENBLAS/NUMEXPR=1。

baseline：CrystaLLM validation full C20 = {pct(result['baseline_full_validation']['match@1'])} / {pct(result['baseline_full_validation']['match@5'])} / {pct(result['baseline_full_validation']['match@20'])}；rows>=7 = {pct(result['baseline_full_validation']['rows>=7_match@1'])} / {pct(result['baseline_full_validation']['rows>=7_match@5'])} / {pct(result['baseline_full_validation']['rows>=7_match@20'])}。

方法变化：无 route/rerank/scorer；只评估 proposer pool 的 oracle coverage。top50 使用每个样本所有可用候选，因当前 pool 平均候选数少于 50，top50 实际等于 available-pool coverage。

full-validation missing-as-fail 结果：overall top1/top5/top20/top50 coverage = {pct(all_full['top1_match_coverage'])} / {pct(all_full['top5_match_coverage'])} / {pct(all_full['top20_match_coverage'])} / {pct(all_full['top50_match_coverage'])}；rows>=7 = {pct(r7_full['top1_match_coverage'])} / {pct(r7_full['top5_match_coverage'])} / {pct(r7_full['top20_match_coverage'])} / {pct(r7_full['top50_match_coverage'])}。

overlap-only 结果：covered samples 上 top50 coverage={pct(all_overlap['top50_match_coverage'])}；rows>=7 overlap top50={pct(r7_overlap['top50_match_coverage'])}。overlap 结果用于看 proposer 本身，不代表 full validation。

exact-cover / skeleton 指标：candidate exact-cover feasible={pct(diag['exact_cover_feasible_rate'])}；formula={pct(diag['formula_consistency'])}；SG={pct(diag['sg_consistency'])}；valid={pct(diag['valid_rate'])}；candidate skeleton_hit={pct(diag['skeleton_hit_rate'])}；candidate WA_hit={pct(diag['wa_hit_rate'])}；candidate skeleton-to-match conversion={pct(diag['candidate_skeleton_to_match_conversion'])}。sample top50 skeleton-to-match conversion overall={pct(all_full['top50_sample_skeleton_to_match_conversion'])}，rows>=7={pct(r7_full['top50_sample_skeleton_to_match_conversion'])}。

与 half audit 关系：opentry_11 half fullgen_pool overlap 结果为 match@1/5/20={pct(half_pool['metrics']['match@1'])}/{pct(half_pool['metrics']['match@5'])}/{pct(half_pool['metrics']['match@20'])}，rows>=7={pct(half_pool['metrics']['rows>=7_match@1'])}/{pct(half_pool['metrics']['rows>=7_match@5'])}/{pct(half_pool['metrics']['rows>=7_match@20'])}；本实验扩展到 full validation artifact 并补 top50 coverage。

可信度：这是 validation artifact replay，不是 official，不训练新模型，也不使用 GT label 做 inference。限制是当前 fullgen pool 对 426 个 validation 样本缺失，且每样本平均候选数不足 50，所以 top50 是现有候选池上限而非真正 50-sample proposer。

和历史实验关系：承接实验 4 的 learned geometry repair 信号；本实验检验 proposer 是否能给 repair 提供足够 skeleton/geometry candidates。结果与 opentry_11 实验 7C 的半量结论一致：exact-cover 很高，但 skeleton_hit/WA_hit 和 overall coverage 不足。

最终判决：当前 proposer pool 不能作为主方法成功。它对 rows>=7 有局部信号，但 full-validation overall top50 仍不足以证明 candidate pool coverage 已解决；必须训练/构造更强 neural skeleton proposer，并与 learned geometry repair 联动。

下一步：实验 6 rows>=7 专门路线应聚焦 rows>=7 skeleton proposal + geometry repair；若新训练 proposer，必须使用 MP-20/MPTS-52 train 数据并定义 half-train gate。
"""
    append_report_once(RESULTS / "experiment_5_report_appended.json", "实验 5 neural skeleton / geometry proposer coverage 审计", body)


def main() -> None:
    result = build_experiment()
    write_report(result)
    print(json.dumps(
        {
            "experiment": result["experiment"],
            "target_samples": result["target_samples"],
            "covered_samples": result["covered_samples"],
            "top50_full": result["full_validation_missing_as_fail"]["all"]["top50_match_coverage"],
            "top50_rows_ge7_full": result["full_validation_missing_as_fail"]["rows_ge7"]["top50_match_coverage"],
            "candidate_skeleton_to_match": result["candidate_level_diagnostics"]["candidate_skeleton_to_match_conversion"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
