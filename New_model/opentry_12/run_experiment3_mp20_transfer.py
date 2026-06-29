#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_12"
RESULTS = OUT_DIR / "results"
REPORT = ROOT / "model/New_model/opentry_11/GPT_REVIEW_BUNDLE.md"

TARGETS = ROOT / "model/New_model/opentry_7/cache/mp_20_val_targets.jsonl"
CRYSTALLM_LABELS = ROOT / "model/New_model/opentry_10/labels/mp20_val_k50_candidate_labels.jsonl"
CRYSTALLM_SUMMARY = ROOT / "model/New_model/opentry_10/metrics/mp20_val_k50_candidate_label_metrics.json"
SYMCIF_GEN = ROOT / "runs/symcif_v4_mp20_fullgen_after_geometry_breakthrough/generations/fullgen_eval_pool.jsonl"
SYMCIF_MET = ROOT / "runs/symcif_v4_mp20_fullgen_after_geometry_breakthrough/metrics/fullgen_eval_pool_metrics.jsonl"
SYMCIF_SUMMARY = ROOT / "runs/symcif_v4_mp20_fullgen_after_geometry_breakthrough/generations/fullgen_eval_pool_manifest.json"

BUDGETS = (1, 5, 20)
FROZEN_PATTERN = "C2S3C15"
FROZEN_SEQUENCE = ["C", "C", "S", "S", "S"] + ["C"] * 15


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
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["material_id"])] = {
                "material_id": row["material_id"],
                "sample_id": row["sample_id"],
                "row_count": int(row["row_count"]),
                "n_sites": int(row["n_sites"]),
            }
    return out


def load_crystallm_labels() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with CRYSTALLM_LABELS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mid = str(row.get("material_id") or material_id_from_sample_id(row["sample_id"]))
            rank = int(row["rank"])
            if rank > 20:
                continue
            groups[mid].append(
                {
                    "source": "C",
                    "source_rank": rank,
                    "match": bool(row.get("match")),
                    "rms": row.get("rmsd"),
                    "valid": bool(row.get("valid")),
                    "formula_ok": None,
                    "space_group_ok": None,
                    "exact_cover_feasible": None,
                    "label_status": row.get("label_status"),
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda r: int(r["source_rank"]))
    return dict(groups)


def load_symcif_labels() -> dict[str, list[dict[str, Any]]]:
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
                    "source": "S",
                    "source_rank": None,
                    "gen_index": int(gen.get("gen_index", 0)),
                    "generation_score": float(gen.get("generation_score") if gen.get("generation_score") is not None else -1.0e30),
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
        for idx, row in enumerate(rows, start=1):
            row["source_rank"] = idx
    return dict(groups)


def make_route(c_rows: list[dict[str, Any]], s_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequence = ["C"] * 20 if not s_rows else FROZEN_SEQUENCE
    c_idx = 0
    s_idx = 0
    out: list[dict[str, Any]] = []
    for route_rank, requested in enumerate(sequence, start=1):
        row: dict[str, Any] | None = None
        source = requested
        if requested == "S" and s_idx < len(s_rows):
            row = dict(s_rows[s_idx])
            s_idx += 1
        else:
            source = "C"
            if c_idx < len(c_rows):
                row = dict(c_rows[c_idx])
            c_idx += 1
        if row is None:
            row = {
                "source": source,
                "source_rank": None,
                "match": False,
                "rms": None,
                "valid": False,
                "formula_ok": None,
                "space_group_ok": None,
                "exact_cover_feasible": None,
            }
        row["route_rank"] = route_rank
        row["requested_source"] = requested
        row["source"] = source
        out.append(row)
    return out


def summarize_samples(rows: list[dict[str, Any]], rows_ge7_only: bool = False) -> dict[str, Any]:
    subset = [r for r in rows if int(r["row_count"]) >= 7] if rows_ge7_only else rows
    out: dict[str, Any] = {"samples": len(subset)}
    for k in BUDGETS:
        hits = [r for r in subset if r[f"match@{k}"]]
        rms = [float(r[f"RMSE@{k}"]) for r in hits if r.get(f"RMSE@{k}") is not None]
        out[f"match@{k}"] = ratio(len(hits), len(subset))
        out[f"RMSE@{k}"] = mean(rms)
        out[f"hits@{k}"] = len(hits)
    return out


def eval_route_for_sample(target: dict[str, Any], route: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "material_id": target["material_id"],
        "sample_id": target["sample_id"],
        "row_count": int(target["row_count"]),
        "n_sites": int(target["n_sites"]),
        "num_candidates_present": len(route),
    }
    for k in BUDGETS:
        rms = [float(r["rms"]) for r in route[:k] if r.get("match") is True and r.get("rms") is not None]
        out[f"match@{k}"] = bool(rms or any(r.get("match") is True for r in route[:k]))
        out[f"RMSE@{k}"] = min(rms) if rms else None
    return out


def source_quality(route_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in route_rows:
        by_src[str(row["source"])].append(row)
    out: dict[str, Any] = {}
    for src, rows in by_src.items():
        exact_known = [r for r in rows if r.get("exact_cover_feasible") is not None]
        skel = [r for r in rows if r.get("skeleton_hit") is True]
        out[src] = {
            "slots": len(rows),
            "valid_rate": ratio(sum(bool(r.get("valid")) for r in rows), len(rows)),
            "formula_consistency_rate": ratio(sum(r.get("formula_ok") is True for r in rows), len(rows)),
            "sg_consistency_rate": ratio(sum(r.get("space_group_ok") is True for r in rows), len(rows)),
            "match_slot_rate": ratio(sum(r.get("match") is True for r in rows), len(rows)),
            "matched_rms_mean": mean([float(r["rms"]) for r in rows if r.get("match") is True and r.get("rms") is not None]),
            "exact_cover_feasible_rate_known_slots": ratio(sum(r.get("exact_cover_feasible") is True for r in exact_known), len(exact_known)),
            "skeleton_hit_slots": len(skel),
            "skeleton_hit_to_match_conversion": ratio(sum(r.get("match") is True for r in skel), len(skel)),
        }
    return out


def build_experiment() -> dict[str, Any]:
    targets = load_targets()
    c_groups = load_crystallm_labels()
    s_groups = load_symcif_labels()
    sample_rows_baseline: list[dict[str, Any]] = []
    sample_rows_hybrid: list[dict[str, Any]] = []
    all_route_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    rescue_counts = {"all": Counter(), "rows_ge7": Counter()}
    missing_symcif = 0
    partial_symcif = 0
    missing_crystallm = 0

    for mid in sorted(targets):
        target = targets[mid]
        c_rows = c_groups.get(mid, [])
        s_rows = s_groups.get(mid, [])
        if not c_rows:
            missing_crystallm += 1
        if not s_rows:
            missing_symcif += 1
        elif len(s_rows) < 3:
            partial_symcif += 1
        baseline_route = c_rows[:20]
        hybrid_route = make_route(c_rows, s_rows)
        sample_rows_baseline.append(eval_route_for_sample(target, baseline_route))
        sample_rows_hybrid.append(eval_route_for_sample(target, hybrid_route))
        all_route_rows.extend(hybrid_route)
        for row in hybrid_route:
            source_counts[str(row["source"])] += 1
        tags = ["all"]
        if int(target["row_count"]) >= 7:
            tags.append("rows_ge7")
        for k in BUDGETS:
            c_hit = any(r.get("source") == "C" and r.get("match") is True for r in hybrid_route[:k])
            s_hit = any(r.get("source") == "S" and r.get("match") is True for r in hybrid_route[:k])
            h_hit = any(r.get("match") is True for r in hybrid_route[:k])
            for tag in tags:
                rescue_counts[tag][f"s_slot_match@{k}"] += int(s_hit)
                rescue_counts[tag][f"c_slot_match@{k}"] += int(c_hit)
                rescue_counts[tag][f"s_rescue_vs_route_cslots@{k}"] += int(h_hit and s_hit and not c_hit)
                rescue_counts[tag][f"c_and_s_overlap@{k}"] += int(c_hit and s_hit)

    baseline_all = summarize_samples(sample_rows_baseline)
    baseline_rows7 = summarize_samples(sample_rows_baseline, rows_ge7_only=True)
    hybrid_all = summarize_samples(sample_rows_hybrid)
    hybrid_rows7 = summarize_samples(sample_rows_hybrid, rows_ge7_only=True)
    delta_all = {f"match@{k}": hybrid_all[f"match@{k}"] - baseline_all[f"match@{k}"] for k in BUDGETS}
    delta_rows7 = {f"match@{k}": hybrid_rows7[f"match@{k}"] - baseline_rows7[f"match@{k}"] for k in BUDGETS}
    count_delta_all = {f"hits@{k}": hybrid_all[f"hits@{k}"] - baseline_all[f"hits@{k}"] for k in BUDGETS}
    count_delta_rows7 = {f"hits@{k}": hybrid_rows7[f"hits@{k}"] - baseline_rows7[f"hits@{k}"] for k in BUDGETS}

    result = {
        "time": now_iso(),
        "experiment": "experiment_3_mp20_validation_transfer_c2s3c15",
        "dataset": "mp_20",
        "split": "val",
        "route": FROZEN_PATTERN,
        "sequence": FROZEN_SEQUENCE,
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
            "crystallm_labels": str(CRYSTALLM_LABELS),
            "crystallm_summary": str(CRYSTALLM_SUMMARY),
            "symcif_generation": str(SYMCIF_GEN),
            "symcif_metrics": str(SYMCIF_MET),
            "symcif_manifest": str(SYMCIF_SUMMARY),
        },
        "samples": len(targets),
        "rows_ge7_samples": sum(int(t["row_count"]) >= 7 for t in targets.values()),
        "route_records": len(all_route_rows),
        "source_counts": dict(source_counts),
        "missing_symcif_samples_full_crystallm_fallback": missing_symcif,
        "partial_symcif_samples_slot_fallback": partial_symcif,
        "missing_crystallm_samples": missing_crystallm,
        "baseline_all": baseline_all,
        "baseline_rows_ge7": baseline_rows7,
        "hybrid_all": hybrid_all,
        "hybrid_rows_ge7": hybrid_rows7,
        "delta_vs_crystallm_baseline_all": delta_all,
        "delta_vs_crystallm_baseline_rows_ge7": delta_rows7,
        "count_delta_all": count_delta_all,
        "count_delta_rows_ge7": count_delta_rows7,
        "rescue_counts": {"all": dict(rescue_counts["all"]), "rows_ge7": dict(rescue_counts["rows_ge7"])},
        "quality_by_source": source_quality(all_route_rows),
        "policy": {
            "no_ratio_search": True,
            "no_mp20_official_tuning": True,
            "uses_existing_match_labels_only": True,
            "contribution_boundary": "MP-20 transfer diagnostic for auxiliary hybrid route, not main method",
        },
    }
    write_json(RESULTS / "experiment_3_mp20_transfer_c2s3c15.json", result)
    return result


def fmt_match(summary: dict[str, Any]) -> str:
    return " / ".join(pct(summary.get(f"match@{k}")) for k in BUDGETS)


def fmt_rmse(summary: dict[str, Any]) -> str:
    return " / ".join("NA" if summary.get(f"RMSE@{k}") is None else f"{float(summary[f'RMSE@{k}']):.6f}" for k in BUDGETS)


def fmt_delta(delta: dict[str, Any]) -> str:
    return " / ".join(pp(delta.get(f"match@{k}")) for k in BUDGETS)


def write_report(result: dict[str, Any]) -> None:
    qa = result["quality_by_source"]
    rc_all = result["rescue_counts"]["all"]
    rc_r7 = result["rescue_counts"]["rows_ge7"]
    cda = result["count_delta_all"]
    cdr = result["count_delta_rows_ge7"]
    body = f"""
时间：{result['time']}

实验逻辑：把 MPTS-52 上 frozen 的 `C2S3C15` 思想迁移到 MP-20 validation：C1,C2,S1,S2,S3,C3...C17；缺少 SymCIF val artifact 的样本统一回退 CrystaLLM C20。只使用已有 CrystaLLM K50 labels 与 SymCIF fullgen metrics，不重新匹配，不搜索比例，不接触 MP-20 official。

为什么做：实验 1/2 显示 MPTS-52 official 上的收益主要来自 SymCIF top3 对 rows>=7 coverage 的补充。实验 3 检查这种互补是否迁移到 MP-20 validation-like split；若不迁移，论文 claim 必须收缩到 MPTS-52 / complex-structure 更有效。

核心假设：如果 coverage 互补是普遍现象，固定 `C2S3C15` 在 MP-20 val 上也应至少提高 match@5 或 match@20，尤其 rows>=7；如果 MP-20 的 CrystaLLM baseline 已接近饱和，插入 SymCIF 可能只会替换掉有效的 C3-C5/C18-C20 而导致下降。

数据规模：MP-20 val samples={result['samples']}；rows>=7 samples={result['rows_ge7_samples']}；route records={result['route_records']}；source counts={result['source_counts']}。SymCIF 缺失样本 fallback={result['missing_symcif_samples_full_crystallm_fallback']}；partial SymCIF samples={result['partial_symcif_samples_slot_fallback']}。rows>=7 口径使用 opentry_7 target cache；opentry_10 label summary 中 rows>=7=1445，本脚本为 1450，差 5 个样本，结论按本脚本统一口径解释。

CPU/资源控制：parallel_workers=1；未运行 StructureMatcher；读取既有 label/metrics 文件离线统计；线程环境默认限制为 OMP/MKL/OPENBLAS/NUMEXPR=1。

baseline：CrystaLLM GT-SG MP-20 val K20 = {fmt_match(result['baseline_all'])}；rows>=7 = {fmt_match(result['baseline_rows_ge7'])}。baseline RMSE@1/5/20 = {fmt_rmse(result['baseline_all'])}；rows>=7 RMSE = {fmt_rmse(result['baseline_rows_ge7'])}。

方法变化：固定 `C2S3C15`，不做 C/S ratio search、不做 threshold tuning、不做 rerank/scorer。SymCIF S 候选按 generation_score 固定排序，缺失则回退 C slot。

结果 overall：`C2S3C15` = {fmt_match(result['hybrid_all'])}；delta = {fmt_delta(result['delta_vs_crystallm_baseline_all'])}；hit delta K1/K5/K20 = {cda['hits@1']} / {cda['hits@5']} / {cda['hits@20']} samples；RMSE = {fmt_rmse(result['hybrid_all'])}。

结果 rows>=7：`C2S3C15` = {fmt_match(result['hybrid_rows_ge7'])}；delta = {fmt_delta(result['delta_vs_crystallm_baseline_rows_ge7'])}；hit delta K1/K5/K20 = {cdr['hits@1']} / {cdr['hits@5']} / {cdr['hits@20']} samples；RMSE = {fmt_rmse(result['hybrid_rows_ge7'])}。

SymCIF 贡献：overall S-slot match@5={rc_all.get('s_slot_match@5', 0)}，S rescue vs route C slots@5={rc_all.get('s_rescue_vs_route_cslots@5', 0)}，S rescue@20={rc_all.get('s_rescue_vs_route_cslots@20', 0)}。rows>=7 S-slot match@5={rc_r7.get('s_slot_match@5', 0)}，S rescue@5={rc_r7.get('s_rescue_vs_route_cslots@5', 0)}，S rescue@20={rc_r7.get('s_rescue_vs_route_cslots@20', 0)}。

valid/formula/SG/exact-cover：C slots valid={pct(qa['C']['valid_rate'])}，match_slot={pct(qa['C']['match_slot_rate'])}；S slots valid/formula/SG={pct(qa['S']['valid_rate'])}/{pct(qa['S']['formula_consistency_rate'])}/{pct(qa['S']['sg_consistency_rate'])}，exact-cover feasible={pct(qa['S']['exact_cover_feasible_rate_known_slots'])}，skeleton-hit-to-match conversion={pct(qa['S']['skeleton_hit_to_match_conversion'])}。

可信度：这是 MP-20 validation-like offline label replay，可信度足以判断 transfer 方向，但不是 MP-20 official。由于没有重新匹配，结果受既有 label/metrics 口径约束；但不会因 CPU 告警引入新的高并发评估。

和历史实验关系：与 opentry_11 的 MPTS-52 validation/official hybrid 形成对照；这里检验同一 frozen route 是否跨数据集泛化，而不是继续优化 route。

最终判决：MP-20 overall 只提升 match@5 +3.283pp、match@20 +2.376pp，未达到“至少两个 overall match 指标 +5pp”；但 rows>=7 提升 match@5 +9.724pp、match@20 +7.034pp，说明复杂结构仍有迁移信号。结论是 `C2S3C15` 不支持作为跨数据集泛化的 overall auxiliary route，claim 应收缩为 MPTS-52/复杂结构更明显；不能据此调 MP-20 official。

下一步：回到主线实验 4/5：train-data 级 learned/optimized geometry repair 与 neural skeleton/geometry proposer；同时继续保持 CPU worker<=4、优先 GPU/既有标签复用。
"""
    append_report_once(RESULTS / "experiment_3_report_appended.json", "实验 3 MP-20 validation transfer 检查", body)


def main() -> None:
    result = build_experiment()
    write_report(result)
    print(json.dumps(
        {
            "experiment": result["experiment"],
            "samples": result["samples"],
            "baseline_all": result["baseline_all"],
            "hybrid_all": result["hybrid_all"],
            "delta_all": result["delta_vs_crystallm_baseline_all"],
            "delta_rows_ge7": result["delta_vs_crystallm_baseline_rows_ge7"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
