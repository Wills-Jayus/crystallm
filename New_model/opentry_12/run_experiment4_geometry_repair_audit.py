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

MP20_TARGETS = ROOT / "model/New_model/opentry_7/cache/mp_20_val_targets.jsonl"
DETERMINISTIC_REPAIR = ROOT / "model/New_model/opentry_11/results/experiment_5_half_geometry_repair.json"
GTWA_K5_SUMMARY = ROOT / "model/New_model/symcif_experiment/reports/symcif_v4_mp20_k5_dev/gtwa_geometry_k5_eval.json"
GTWA_K5_METRICS = ROOT / "model/New_model/symcif_experiment/reports/symcif_v4_mp20_k5_dev/gtwa_geometry_k5_eval_gpu/metrics/baseline_per_generation_metrics.jsonl"
GEOMETRY_NEXT_STEP = ROOT / "model/New_model/symcif_experiment/reports/symcif_v4_geometry_next_step_summary.json"

BUDGETS = (1, 5)


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


def load_mp20_row_counts() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with MP20_TARGETS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["material_id"])] = {
                    "row_count": int(row["row_count"]),
                    "n_sites": int(row["n_sites"]),
                    "sample_id": row["sample_id"],
                }
    return out


def summarize_metric_rows(rows_by_sample: dict[str, list[dict[str, Any]]], row_meta: dict[str, dict[str, Any]], rows_ge7_only: bool = False) -> dict[str, Any]:
    sample_items: list[tuple[str, list[dict[str, Any]]]] = []
    for sample_id, rows in rows_by_sample.items():
        mid = material_id_from_sample_id(sample_id)
        meta = row_meta.get(mid)
        if meta is None:
            continue
        if rows_ge7_only and int(meta["row_count"]) < 7:
            continue
        sample_items.append((sample_id, rows))
    out: dict[str, Any] = {"samples": len(sample_items)}
    for k in BUDGETS:
        hits = 0
        rms_vals: list[float] = []
        valid_any = 0
        strict_valid_any = 0
        formula_any = 0
        sg_any = 0
        for _sid, rows in sample_items:
            top = rows[:k]
            matched = [r for r in top if r.get("match_ok") is True]
            if matched:
                hits += 1
                rms = [float(r["rms"]) for r in matched if r.get("rms") is not None]
                if rms:
                    rms_vals.append(min(rms))
            valid_any += int(any(r.get("valid") is True for r in top))
            strict_valid_any += int(any(r.get("strict_valid") is True for r in top))
            formula_any += int(any((r.get("formula_ok") is True or r.get("SG_ok") is True) for r in top))
            sg_any += int(any((r.get("space_group_ok") is True or r.get("SG_ok") is True) for r in top))
        out[f"match@{k}"] = ratio(hits, len(sample_items))
        out[f"RMSE@{k}"] = mean(rms_vals)
        out[f"hits@{k}"] = hits
        out[f"skeleton_hit_to_match_conversion@{k}"] = ratio(hits, len(sample_items))
        out[f"valid_any@{k}"] = ratio(valid_any, len(sample_items))
        out[f"strict_valid_any@{k}"] = ratio(strict_valid_any, len(sample_items))
        out[f"formula_ok_any@{k}"] = ratio(formula_any, len(sample_items))
        out[f"sg_ok_any@{k}"] = ratio(sg_any, len(sample_items))
    return out


def load_gtwa_k5_metrics() -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    geometry_sources: dict[str, int] = defaultdict(int)
    with GTWA_K5_METRICS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            groups[str(row["sample_id"])].append(row)
            geometry_sources[str(row.get("geometry_source") or "unknown")] += 1
    for rows in groups.values():
        rows.sort(key=lambda r: int(r.get("rank") or r.get("gen_index") or 0))
    return dict(groups), dict(geometry_sources)


def top_metric(block: dict[str, Any], key: str, top: str) -> float | None:
    try:
        return float(block[key]["overall"][top]["match_at_k"])
    except Exception:
        return None


def build_experiment() -> dict[str, Any]:
    row_meta = load_mp20_row_counts()
    deterministic = json.loads(DETERMINISTIC_REPAIR.read_text(encoding="utf-8"))
    gtwa_summary = json.loads(GTWA_K5_SUMMARY.read_text(encoding="utf-8"))
    next_step = json.loads(GEOMETRY_NEXT_STEP.read_text(encoding="utf-8"))
    rows_by_sample, geometry_sources = load_gtwa_k5_metrics()
    gtwa_k5_all = summarize_metric_rows(rows_by_sample, row_meta, rows_ge7_only=False)
    gtwa_k5_rows7 = summarize_metric_rows(rows_by_sample, row_meta, rows_ge7_only=True)

    old_top20 = top_metric(next_step, "gtwa_old_geometry", "top20")
    learned_top20 = top_metric(next_step, "gtwa_geometry_no_oversampling", "top20")
    learned_over_top20 = top_metric(next_step, "gtwa_geometry_oversampling", "top20")
    full_best_top20 = top_metric(next_step, "full_geometry_best", "top20")
    wa_upper_top20 = top_metric(next_step, "wa_upper_bound", "top20")
    result = {
        "time": now_iso(),
        "experiment": "experiment_4_train_data_learned_geometry_repair_audit",
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
            "deterministic_repair": str(DETERMINISTIC_REPAIR),
            "gtwa_k5_summary": str(GTWA_K5_SUMMARY),
            "gtwa_k5_metrics": str(GTWA_K5_METRICS),
            "geometry_next_step_summary": str(GEOMETRY_NEXT_STEP),
            "mp20_targets": str(MP20_TARGETS),
        },
        "deterministic_half_repair": deterministic,
        "mp20_gtwa_learned_geometry_k5": {
            "source_summary_samples": gtwa_summary["source_summary"]["samples"],
            "metric_samples": len(rows_by_sample),
            "rows_ge7_samples": gtwa_k5_rows7["samples"],
            "all": gtwa_k5_all,
            "rows_ge7": gtwa_k5_rows7,
            "geometry_source_counts": geometry_sources,
            "training_config": gtwa_summary["source_summary"].get("training_config"),
            "condition": "GT-WA/GT-skeleton fixed; evaluates geometry/free-parameter/lattice repair only",
        },
        "v4_500_sample_top20_geometry_model": {
            "gtwa_old_geometry_match@20": old_top20,
            "gtwa_learned_no_oversampling_match@20": learned_top20,
            "gtwa_learned_oversampling_match@20": learned_over_top20,
            "gtwa_learned_no_over_delta_vs_old": (learned_top20 - old_top20) if learned_top20 is not None and old_top20 is not None else None,
            "gtwa_learned_over_delta_vs_old": (learned_over_top20 - old_top20) if learned_over_top20 is not None and old_top20 is not None else None,
            "full_pipeline_best_match@20": full_best_top20,
            "wa_upper_bound_match@20": wa_upper_top20,
            "samples": next_step["gtwa_old_geometry"]["samples"],
            "condition": "500-sample v4 evaluator; top20 available; not MP-20 official",
        },
        "decision": {
            "deterministic_repair_failed": deterministic["repair_eval"]["converted"] == 0,
            "learned_geometry_has_signal_under_gtwa": bool(gtwa_k5_all["match@5"] and gtwa_k5_all["match@5"] > 0.8),
            "learned_geometry_is_not_inference_pipeline": True,
            "next_step": "train/evaluate inference-time skeleton/WA proposer plus learned geometry repair; do not claim GT-WA repair as main result",
        },
    }
    write_json(RESULTS / "experiment_4_learned_geometry_repair_audit.json", result)
    return result


def write_report(result: dict[str, Any]) -> None:
    det = result["deterministic_half_repair"]
    gtwa = result["mp20_gtwa_learned_geometry_k5"]
    top20 = result["v4_500_sample_top20_geometry_model"]
    body = f"""
时间：{result['time']}

实验逻辑：实验 4 不再继续 weak wrap/jitter，而是审计已有 train-data 级 learned/optimized geometry repair 证据：先用 opentry_11 half-data deterministic repair 作为失败 baseline，再用 MP-20 GT-WA learned geometry K5 artifact 重算 rows>=7 conversion，并引用 v4 500-sample top20 geometry model 结果。全程不重新跑 StructureMatcher，不启动新训练。

为什么做：实验 5B 已证明 deterministic repair conversion=0，说明坐标 wrap/jitter 不能把 skeleton-hit negative 转成 match。下一步必须确认真正 learned/optimized geometry repair 是否在固定 composition+GT-SG+GT-WA/GT-skeleton 条件下能提升 skeleton-hit-to-match conversion。

核心假设：如果失败主要来自 lattice/free parameters/site geometry，GT-WA/GT-skeleton 固定时 learned geometry repair 应有较高 match conversion；如果在 GT-WA 下仍低，则 geometry repair 主线也不值得继续。

数据规模：deterministic repair half-data samples={det['data']['samples']}，repair pool={det['data']['repair_pool_candidates']} candidates；MP-20 GT-WA learned geometry metric samples={gtwa['metric_samples']}，rows>=7 samples={gtwa['rows_ge7_samples']}，topK=1/5；v4 top20 summary samples={top20['samples']}。

CPU/资源控制：parallel_workers=1；未运行 StructureMatcher；只读既有 metrics/summary；线程环境默认限制为 OMP/MKL/OPENBLAS/NUMEXPR=1。

baseline：deterministic repair before = {pct(det['baseline']['metrics']['match@1'])} / {pct(det['baseline']['metrics']['match@5'])} / {pct(det['baseline']['metrics']['match@20'])}；rows>=7 = {pct(det['baseline']['metrics']['rows>=7_match@1'])} / {pct(det['baseline']['metrics']['rows>=7_match@5'])} / {pct(det['baseline']['metrics']['rows>=7_match@20'])}。repair 后 delta 全为 0；conversion={det['repair_eval']['converted']}/{det['repair_eval']['evaluated_valid']}，rows>=7 conversion={det['repair_eval']['rows>=7_converted']}/{det['repair_eval']['rows>=7_evaluated_valid']}。

方法变化：从非学习 deterministic repair 转为审计 train-data learned geometry model/prototype repair。MP-20 GT-WA 条件固定 WA/skeleton，只评估 geometry/free-parameter/lattice 转化；这不是 inference pipeline 成功，也不使用 test feedback。

MP-20 GT-WA learned geometry K5 结果：overall match@1/5 = {pct(gtwa['all']['match@1'])} / {pct(gtwa['all']['match@5'])}；skeleton-hit-to-match conversion@1/5 = {pct(gtwa['all']['skeleton_hit_to_match_conversion@1'])} / {pct(gtwa['all']['skeleton_hit_to_match_conversion@5'])}；RMSE@1/5 = {gtwa['all']['RMSE@1']:.6f} / {gtwa['all']['RMSE@5']:.6f}；strict_valid_any@5={pct(gtwa['all']['strict_valid_any@5'])}。

rows>=7 结果：MP-20 GT-WA learned geometry rows>=7 match@1/5 = {pct(gtwa['rows_ge7']['match@1'])} / {pct(gtwa['rows_ge7']['match@5'])}；conversion@1/5 = {pct(gtwa['rows_ge7']['skeleton_hit_to_match_conversion@1'])} / {pct(gtwa['rows_ge7']['skeleton_hit_to_match_conversion@5'])}；RMSE@1/5 = {gtwa['rows_ge7']['RMSE@1']:.6f} / {gtwa['rows_ge7']['RMSE@5']:.6f}。

top20 补充：v4 500-sample GT-WA old geometry match@20={pct(top20['gtwa_old_geometry_match@20'])}；learned geometry no-over={pct(top20['gtwa_learned_no_oversampling_match@20'])}，delta={pp(top20['gtwa_learned_no_over_delta_vs_old'])}；oversampling={pct(top20['gtwa_learned_oversampling_match@20'])}，delta={pp(top20['gtwa_learned_over_delta_vs_old'])}；full-pipeline best match@20={pct(top20['full_pipeline_best_match@20'])}；WA upper-bound={pct(top20['wa_upper_bound_match@20'])}。

可信度：deterministic half-data failure 是 MPTS-52 validation half 的真实 repair pool；MP-20 GT-WA learned geometry 是 8874/8874 级别 summary 与 44370 metric rows，可信度高于 smoke；top20 来自 500-sample v4 evaluator，只能作为 top20 辅助证据。限制是 GT-WA/GT-skeleton 是 oracle 条件，不能当 inference 主方法结果。

和历史实验关系：直接回应 opentry_11 实验 5B 的 0 conversion。结果说明“learned geometry repair 在 oracle skeleton/WA 下有效”，但真正主线仍缺 inference-time skeleton/WA proposer。

最终判决：继续 learned/optimized geometry repair 主线，但只能作为与 skeleton/WA proposer 绑定的主方法组件；不能把 GT-WA repair 本身写成 benchmark 成功。下一步必须做实验 5 neural skeleton/geometry proposer，检查 top50 coverage 与 exact-cover feasible 是否能把 oracle geometry repair 信号转成 inference gain。

下一步：做实验 5 的 proposer 审计/训练设计，优先用已有 fullgen/top50 labels 评估 coverage；若要新训练，必须明确 MP-20/MPTS-52 train 数据规模、GPU 配置、half/full gate 和停止条件。
"""
    append_report_once(RESULTS / "experiment_4_report_appended.json", "实验 4 train-data learned geometry repair 审计", body)


def main() -> None:
    result = build_experiment()
    write_report(result)
    print(json.dumps(
        {
            "experiment": result["experiment"],
            "deterministic_conversion": result["deterministic_half_repair"]["repair_eval"]["conversion_rate"],
            "mp20_gtwa_match@5": result["mp20_gtwa_learned_geometry_k5"]["all"]["match@5"],
            "mp20_gtwa_rows_ge7_match@5": result["mp20_gtwa_learned_geometry_k5"]["rows_ge7"]["match@5"],
            "v4_gtwa_top20_delta_no_over": result["v4_500_sample_top20_geometry_model"]["gtwa_learned_no_over_delta_vs_old"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
