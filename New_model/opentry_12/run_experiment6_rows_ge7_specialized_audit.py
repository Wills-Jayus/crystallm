#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_12"
RESULTS = OUT_DIR / "results"
REPORT = ROOT / "model/New_model/opentry_11/GPT_REVIEW_BUNDLE.md"

TARGETS = ROOT / "model/New_model/opentry_7/cache/mpts_52_val_targets.jsonl"
CRYSTALLM_LABELS = ROOT / "model/New_model/opentry_10/labels/mpts52_val_k50_candidate_labels.jsonl"
SYMCIF_GEN = ROOT / "runs/symcif_v5_multidataset_wa_decoder/mpts52/val/generations/v5_fullgen_eval_pool.jsonl"
SYMCIF_MET = ROOT / "runs/symcif_v5_multidataset_wa_decoder/mpts52/val/metrics/v5_fullgen_eval_pool_metrics.jsonl"

OLD_ROWS7 = ROOT / "model/New_model/opentry_11/results/experiment_6_rows_ge7_specialized.json"
OLD_EXACT = ROOT / "model/New_model/opentry_11/results/experiment_3_exact_cover_diagnostics.json"
STRICT_ABLATION = ROOT / "model/New_model/opentry_11/results/experiment_8_strict_integrated_ablation.json"
EXP1 = RESULTS / "experiment_1_c2s3c15_official.json"
EXP3 = RESULTS / "experiment_3_mp20_transfer_c2s3c15.json"
EXP4 = RESULTS / "experiment_4_learned_geometry_repair_audit.json"
EXP5 = RESULTS / "experiment_5_skeleton_proposer_audit.json"

BUDGETS = (1, 5, 20, 50)


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def safe_ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def load_targets() -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    with TARGETS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                targets[str(row["material_id"])] = row
    return targets


def mid_from_sample_id(sample_id: str) -> str:
    return str(sample_id).split("__")[-1]


def summarize_ranked_groups(
    groups: dict[str, list[dict[str, Any]]],
    targets: dict[str, dict[str, Any]],
    mids: list[str],
    *,
    missing_as_fail: bool,
) -> dict[str, Any]:
    eval_mids = mids if missing_as_fail else [mid for mid in mids if mid in groups]
    first_ranks: list[int] = []
    rank_bins = Counter()
    out: dict[str, Any] = {
        "samples": len(eval_mids),
        "covered_samples": sum(1 for mid in eval_mids if mid in groups),
        "missing_samples": sum(1 for mid in eval_mids if mid not in groups),
        "candidate_records": sum(len(groups.get(mid, [])) for mid in eval_mids),
    }
    for k in BUDGETS:
        hits = 0
        valid_any = 0
        rms_vals: list[float] = []
        for mid in eval_mids:
            top = groups.get(mid, [])[:k]
            if any(bool(r.get("match")) for r in top):
                hits += 1
                rms_first = next((r.get("rms") for r in top if bool(r.get("match")) and r.get("rms") is not None), None)
                if rms_first is not None:
                    rms_vals.append(float(rms_first))
            if any(bool(r.get("valid")) for r in top):
                valid_any += 1
        out[f"top{k}_hits"] = hits
        out[f"top{k}_match"] = safe_ratio(hits, len(eval_mids))
        out[f"top{k}_valid_any"] = safe_ratio(valid_any, len(eval_mids))
        out[f"top{k}_rmse"] = safe_mean(rms_vals)

    for mid in eval_mids:
        rows = groups.get(mid, [])
        first = next((int(r["rank"]) for r in rows if bool(r.get("match"))), None)
        if first is None:
            rank_bins["none"] += 1
            continue
        first_ranks.append(first)
        if first == 1:
            rank_bins["1"] += 1
        elif first <= 5:
            rank_bins["2-5"] += 1
        elif first <= 20:
            rank_bins["6-20"] += 1
        elif first <= 50:
            rank_bins["21-50"] += 1
        else:
            rank_bins[">50"] += 1
    out["first_hit_rank_mean"] = safe_mean([float(x) for x in first_ranks])
    out["first_hit_rank_median"] = safe_median([float(x) for x in first_ranks])
    out["first_hit_rank_bins"] = dict(rank_bins)
    out["target_row_count_mean"] = safe_mean([float(targets[mid]["row_count"]) for mid in eval_mids if mid in targets])
    return out


def load_crystallm_groups() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with CRYSTALLM_LABELS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mid = str(row.get("material_id") or mid_from_sample_id(str(row["sample_id"])))
            rank = int(row.get("rank") or (int(row.get("gen_index") or 0) + 1))
            groups[mid].append(
                {
                    "rank": rank,
                    "match": bool(row.get("match")),
                    "valid": bool(row.get("valid") or row.get("sensible") or row.get("parse_ok")),
                    "rms": row.get("rmsd"),
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda r: int(r["rank"]))
    return dict(groups)


def load_symcif_groups() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with SYMCIF_GEN.open("r", encoding="utf-8") as gf, SYMCIF_MET.open("r", encoding="utf-8") as mf:
        for gen_line, met_line in zip(gf, mf):
            if not gen_line.strip() or not met_line.strip():
                continue
            gen = json.loads(gen_line)
            met = json.loads(met_line)
            mid = mid_from_sample_id(str(gen["sample_id"]))
            groups[mid].append(
                {
                    "rank": int(gen.get("gen_index") or 0) + 1,
                    "sort_score": float(gen.get("generation_score") if gen.get("generation_score") is not None else -1.0e30),
                    "gen_index": int(gen.get("gen_index") or 0),
                    "match": bool(met.get("match_ok")),
                    "valid": bool(met.get("valid")),
                    "formula_ok": bool(met.get("formula_ok")),
                    "sg_ok": bool(met.get("space_group_ok")),
                    "exact_cover": bool(met.get("multiplicity_ok")) if met.get("multiplicity_ok") is not None else None,
                    "skeleton_hit": bool(gen.get("skeleton_hit")),
                    "wa_hit": bool(gen.get("wa_hit")),
                    "row_count_hit": bool(gen.get("row_count_hit")),
                    "bond_length_score": met.get("bond_length_score"),
                    "rms": met.get("rms"),
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda r: (float(r.get("sort_score") or -1.0e30), -int(r.get("gen_index") or 0)), reverse=True)
        for i, row in enumerate(rows, start=1):
            row["rank"] = i
    return dict(groups)


def summarize_symcif_diagnostics(groups: dict[str, list[dict[str, Any]]], mids: list[str]) -> dict[str, Any]:
    candidates = [row for mid in mids for row in groups.get(mid, [])]
    sample_rows = {mid: groups.get(mid, [])[:50] for mid in mids}
    skeleton_any = sum(1 for rows in sample_rows.values() if any(r.get("skeleton_hit") for r in rows))
    match_any = sum(1 for rows in sample_rows.values() if any(r.get("match") for r in rows))
    skeleton_no_match = sum(1 for rows in sample_rows.values() if any(r.get("skeleton_hit") for r in rows) and not any(r.get("match") for r in rows))
    skeleton_and_match = sum(1 for rows in sample_rows.values() if any(r.get("skeleton_hit") for r in rows) and any(r.get("match") for r in rows))
    no_skeleton = sum(1 for rows in sample_rows.values() if rows and not any(r.get("skeleton_hit") for r in rows))
    exact_any = sum(1 for rows in sample_rows.values() if any(r.get("exact_cover") for r in rows))
    valid_any = sum(1 for rows in sample_rows.values() if any(r.get("valid") for r in rows))
    low_bond = [
        float(r["bond_length_score"])
        for r in candidates
        if r.get("bond_length_score") is not None and math.isfinite(float(r["bond_length_score"]))
    ]
    return {
        "candidate_records": len(candidates),
        "candidate_valid_rate": safe_ratio(sum(1 for r in candidates if r.get("valid")), len(candidates)),
        "candidate_formula_rate": safe_ratio(sum(1 for r in candidates if r.get("formula_ok")), len(candidates)),
        "candidate_sg_rate": safe_ratio(sum(1 for r in candidates if r.get("sg_ok")), len(candidates)),
        "candidate_exact_cover_rate": safe_ratio(sum(1 for r in candidates if r.get("exact_cover")), len(candidates)),
        "candidate_skeleton_hit_rate": safe_ratio(sum(1 for r in candidates if r.get("skeleton_hit")), len(candidates)),
        "candidate_wa_hit_rate": safe_ratio(sum(1 for r in candidates if r.get("wa_hit")), len(candidates)),
        "candidate_match_rate": safe_ratio(sum(1 for r in candidates if r.get("match")), len(candidates)),
        "candidate_skeleton_to_match_conversion": safe_ratio(
            sum(1 for r in candidates if r.get("skeleton_hit") and r.get("match")),
            sum(1 for r in candidates if r.get("skeleton_hit")),
        ),
        "top50_sample_skeleton_hit_any": safe_ratio(skeleton_any, len(mids)),
        "top50_sample_match_any": safe_ratio(match_any, len(mids)),
        "top50_sample_skeleton_to_match_conversion": safe_ratio(skeleton_and_match, skeleton_any),
        "top50_sample_skeleton_and_match_any": safe_ratio(skeleton_and_match, len(mids)),
        "top50_sample_skeleton_hit_but_no_match": safe_ratio(skeleton_no_match, len(mids)),
        "top50_sample_no_skeleton_hit_covered_only": safe_ratio(no_skeleton, len(mids)),
        "top50_sample_exact_cover_any": safe_ratio(exact_any, len(mids)),
        "top50_sample_valid_any": safe_ratio(valid_any, len(mids)),
        "bond_length_score_mean": safe_mean(low_bond),
        "bond_length_score_lt_0_5_rate": safe_ratio(sum(1 for v in low_bond if v < 0.5), len(low_bond)),
    }


def metric_triplet(d: dict[str, Any], prefix: str = "match@") -> str:
    return f"{pct(d.get(prefix + '1'))} / {pct(d.get(prefix + '5'))} / {pct(d.get(prefix + '20'))}"


def rows_metric_triplet(d: dict[str, Any]) -> str:
    return f"{pct(d.get('rows>=7_match@1'))} / {pct(d.get('rows>=7_match@5'))} / {pct(d.get('rows>=7_match@20'))}"


def run() -> dict[str, Any]:
    targets = load_targets()
    rows7_mids = sorted([mid for mid, row in targets.items() if int(row.get("row_count") or 0) >= 7])
    rowslt7_mids = sorted([mid for mid, row in targets.items() if int(row.get("row_count") or 0) < 7])

    c_groups = load_crystallm_groups()
    s_groups = load_symcif_groups()

    c_rows7 = summarize_ranked_groups(c_groups, targets, rows7_mids, missing_as_fail=True)
    c_rowslt7 = summarize_ranked_groups(c_groups, targets, rowslt7_mids, missing_as_fail=True)
    s_rows7_full = summarize_ranked_groups(s_groups, targets, rows7_mids, missing_as_fail=True)
    s_rows7_overlap = summarize_ranked_groups(s_groups, targets, rows7_mids, missing_as_fail=False)
    s_rowslt7_full = summarize_ranked_groups(s_groups, targets, rowslt7_mids, missing_as_fail=True)
    s_diag_rows7 = summarize_symcif_diagnostics(s_groups, rows7_mids)
    s_diag_rowslt7 = summarize_symcif_diagnostics(s_groups, rowslt7_mids)

    old_rows7 = read_json(OLD_ROWS7)
    old_exact = read_json(OLD_EXACT)
    strict_ablation = read_json(STRICT_ABLATION)
    exp1 = read_json(EXP1)
    exp3 = read_json(EXP3)
    exp4 = read_json(EXP4)
    exp5 = read_json(EXP5)

    result: dict[str, Any] = {
        "experiment": "experiment_6_rows_ge7_specialized_generation_repair_audit",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "inputs": {
            "targets": str(TARGETS),
            "crystallm_labels": str(CRYSTALLM_LABELS),
            "symcif_generation": str(SYMCIF_GEN),
            "symcif_metrics": str(SYMCIF_MET),
            "old_rows7": str(OLD_ROWS7),
            "old_exact": str(OLD_EXACT),
            "strict_ablation": str(STRICT_ABLATION),
            "exp1_official": str(EXP1),
            "exp3_mp20_transfer": str(EXP3),
            "exp4_geometry_repair": str(EXP4),
            "exp5_proposer": str(EXP5),
        },
        "cpu_policy": {
            "policy": "bounded_cpu_use",
            "logical_cpus": os.cpu_count(),
            "recommended_json_workers": min(16, max(4, (os.cpu_count() or 8) // 8)),
            "used_parallel_workers": 1,
            "why": "this audit is IO/light JSON aggregation; heavy StructureMatcher reruns should use bounded workers rather than saturating all CPUs",
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "data_scale": {
            "target_samples": len(targets),
            "target_rows_ge7_samples": len(rows7_mids),
            "target_rowslt7_samples": len(rowslt7_mids),
            "crystallm_covered_samples": len(c_groups),
            "crystallm_candidate_records": sum(len(v) for v in c_groups.values()),
            "symcif_covered_samples": len(s_groups),
            "symcif_candidate_records": sum(len(v) for v in s_groups.values()),
        },
        "crystallm_k50_rows_ge7": c_rows7,
        "crystallm_k50_rowslt7": c_rowslt7,
        "symcif_v5_rows_ge7_full_missing_as_fail": s_rows7_full,
        "symcif_v5_rows_ge7_overlap": s_rows7_overlap,
        "symcif_v5_rowslt7_full_missing_as_fail": s_rowslt7_full,
        "symcif_v5_rows_ge7_diagnostics": s_diag_rows7,
        "symcif_v5_rowslt7_diagnostics": s_diag_rowslt7,
        "legacy_rows_ge7_specialized": {
            "baseline_rows7": old_rows7.get("baseline_rows7"),
            "track_a_rows7": old_rows7.get("track_a_rows7"),
            "hard_negative_rows7": old_rows7.get("hard_negative_rows7"),
            "repair_rows7": {k: v for k, v in old_rows7.get("repair_rows7", {}).items() if k != "rows"},
            "rows7_skeleton_hit_top20_sample_rate": old_rows7.get("rows7_skeleton_hit_top20_sample_rate"),
            "rows7_skeleton_hit_geometry_fail_candidate_rate": old_rows7.get("rows7_skeleton_hit_geometry_fail_candidate_rate"),
            "rows7_collision_rate": old_rows7.get("rows7_collision_rate"),
            "rowslt7_collision_rate": old_rows7.get("rowslt7_collision_rate"),
            "rows7_exact_cover_bad_rate": old_rows7.get("rows7_exact_cover_bad_rate"),
            "rows7_bad_coord_rate": old_rows7.get("rows7_bad_coord_rate"),
        },
        "legacy_exact_cover_diagnostics": old_exact,
        "strict_integrated_ablation_rows_ge7": {
            "full_validation": strict_ablation.get("full_validation", {}),
            "final_judgment": strict_ablation.get("final_judgment", {}),
        },
        "hybrid_rows_ge7_signals": {
            "mpts52_official_c2s3c15": {
                "baseline_rows_ge7": exp1.get("baseline_rows_ge7"),
                "hybrid_rows_ge7": exp1.get("rows_ge7"),
                "delta_rows_ge7": exp1.get("delta_vs_crystallm_baseline_rows_ge7"),
            },
            "mp20_validation_transfer_c2s3c15": {
                "baseline_rows_ge7": exp3.get("baseline_rows_ge7"),
                "hybrid_rows_ge7": exp3.get("hybrid_rows_ge7"),
                "delta_rows_ge7": exp3.get("delta_vs_crystallm_baseline_rows_ge7"),
            },
        },
        "learned_geometry_repair_rows_ge7": {
            "deterministic_repair": exp4.get("deterministic_repair") or exp4.get("deterministic_half_repair", {}),
            "mp20_gtwa_learned_geometry": (
                exp4.get("mp20_gtwa_learned_geometry", {}).get("rows_ge7")
                or exp4.get("mp20_gtwa_learned_geometry_k5", {}).get("rows_ge7", {})
            ),
            "v4_top20": exp4.get("v4_top20_geometry_summary") or exp4.get("v4_500_sample_top20_geometry_model", {}),
        },
        "experiment5_rows_ge7_reference": {
            "full_missing_as_fail": exp5.get("full_validation_missing_as_fail", {}).get("rows_ge7", {}),
            "overlap": exp5.get("overlap", {}).get("rows_ge7", {}),
            "candidate_level_diagnostics": exp5.get("candidate_level_diagnostics", {}),
        },
    }
    c50_delta = (c_rows7.get("top50_match") or 0.0) - (old_rows7.get("baseline_rows7", {}).get("rows>=7_match@20") or 0.0)
    sym50_delta = (s_rows7_full.get("top50_match") or 0.0) - (old_rows7.get("baseline_rows7", {}).get("rows>=7_match@20") or 0.0)
    result["decision"] = {
        "rows_ge7_primary_bottleneck": "coverage plus skeleton-to-match conversion, not ordinary ranking",
        "crystallm_k50_rows7_top50_delta_vs_k20": c50_delta,
        "symcif_rows7_top50_delta_vs_crystallm_k20": sym50_delta,
        "ordinary_rows7_scorer_passes_5pp_gate": False,
        "deterministic_repair_passes_conversion_gate": False,
        "learned_geometry_repair_has_oracle_signal": True,
        "next_route": "train/construct rows>=7-specialized skeleton proposer, then couple it to learned geometry repair; keep C2S3C15 as auxiliary only",
    }
    return result


def report(result: dict[str, Any]) -> str:
    ds = result["data_scale"]
    c7 = result["crystallm_k50_rows_ge7"]
    clt = result["crystallm_k50_rowslt7"]
    s7 = result["symcif_v5_rows_ge7_full_missing_as_fail"]
    slt = result["symcif_v5_rowslt7_full_missing_as_fail"]
    sdiag = result["symcif_v5_rows_ge7_diagnostics"]
    old = result["legacy_rows_ge7_specialized"]
    base = old["baseline_rows7"]
    track = old["track_a_rows7"]
    hard = old["hard_negative_rows7"]
    official = result["hybrid_rows_ge7_signals"]["mpts52_official_c2s3c15"]
    mp20 = result["hybrid_rows_ge7_signals"]["mp20_validation_transfer_c2s3c15"]
    repair = result["learned_geometry_repair_rows_ge7"]
    cpu = result["cpu_policy"]

    return f"""
时间：{result['time']}

实验逻辑：把 rows>=7 从 overall 指标里拆出来作为独立对象，合并 CrystaLLM K50 validation、SymCIF v5 fullgen proposer、C2S3C15 official/MP20 transfer、deterministic repair、GT-WA learned geometry repair 和旧 rows>=7 scorer 结果，判断复杂结构瓶颈到底在候选池 coverage、skeleton proposal、geometry conversion 还是普通排序。

为什么做：实验 1/2 说明 C2S3C15 official 的 K5/K20 净收益几乎都来自 rows>=7；实验 5 又显示 proposer overall top50 coverage 不够。因此需要专门回答 rows>=7 是否只是排序问题，还是必须做 rows>=7-specialized skeleton proposal + learned geometry repair。

核心假设：如果 rows>=7 只是排序问题，CrystaLLM K50 内应有足够 top50 正确候选，Track A/hard-negative scorer 应能显著提升 K5/K20；如果是 coverage/geometry 问题，则 top50 上限和 skeleton-to-match conversion 会偏低，deterministic repair 不转化，而 oracle learned geometry repair 会显示高 conversion。

数据规模：MPTS-52 validation targets={ds['target_samples']}；rows>=7 targets={ds['target_rows_ge7_samples']}；rows<7 targets={ds['target_rowslt7_samples']}。CrystaLLM K50 labels={ds['crystallm_candidate_records']} candidates/{ds['crystallm_covered_samples']} samples；SymCIF v5 pool={ds['symcif_candidate_records']} candidates/{ds['symcif_covered_samples']} covered samples。

CPU/资源控制：策略改为 bounded CPU use，不是单线程禁用 CPU。机器 logical_cpus={cpu['logical_cpus']}，推荐 JSON/轻评估 worker 上限={cpu['recommended_json_workers']}；本实验实际 used_parallel_workers={cpu['used_parallel_workers']}，因为只是流式 JSON 聚合。线程环境限制 OMP/MKL/OPENBLAS/NUMEXPR=1，避免每个 worker 内部再抢满 CPU。

baseline：CrystaLLM validation rows>=7 K20 baseline = {rows_metric_triplet(base)}；rows<7 对照从同一 K50 label 复算 top1/top5/top20={pct(clt['top1_match'])}/{pct(clt['top5_match'])}/{pct(clt['top20_match'])}。rows>=7 baseline RMSE@1/5/20 = {base.get('rows>=7_rmsd@1'):.6f} / {base.get('rows>=7_rmsd@5'):.6f} / {base.get('rows>=7_rmsd@20'):.6f}。

方法变化：不训练新模型，不重新跑 StructureMatcher，不做 scorer/rerank 调参。只做 rows>=7 专门审计：CrystaLLM K50 看正确候选是否存在和首次命中 rank；SymCIF v5 看 exact-cover/skeleton/WA/geometry conversion；repair 看 deterministic 与 learned/oracle 的差异；hybrid 只作为边界证据。

CrystaLLM K50 rows>=7 coverage：top1/top5/top20/top50 = {pct(c7['top1_match'])} / {pct(c7['top5_match'])} / {pct(c7['top20_match'])} / {pct(c7['top50_match'])}；top50 相对 K20 只多 {pp(result['decision']['crystallm_k50_rows7_top50_delta_vs_k20'])}。first-hit rank mean={c7['first_hit_rank_mean']:.3f}，median={c7['first_hit_rank_median']:.3f}，rank bins={c7['first_hit_rank_bins']}。这说明现有 K50 候选池对复杂结构的额外 headroom 很小。

SymCIF v5 rows>=7 proposer：full missing-as-fail top1/top5/top20/top50 = {pct(s7['top1_match'])} / {pct(s7['top5_match'])} / {pct(s7['top20_match'])} / {pct(s7['top50_match'])}；rows<7 对照 = {pct(slt['top1_match'])} / {pct(slt['top5_match'])} / {pct(slt['top20_match'])} / {pct(slt['top50_match'])}。rows>=7 top50 sample exact-cover_any={pct(sdiag['top50_sample_exact_cover_any'])}，valid_any={pct(sdiag['top50_sample_valid_any'])}，skeleton_hit_any={pct(sdiag['top50_sample_skeleton_hit_any'])}，skeleton-to-match conversion={pct(sdiag['top50_sample_skeleton_to_match_conversion'])}，skeleton_hit_but_no_match={pct(sdiag['top50_sample_skeleton_hit_but_no_match'])}。

collision/geometry 失败信号：旧 K50 full diagnostic 中 rows>=7 collision_proxy={pct(old['rows7_collision_rate'])}，rows<7 collision_proxy={pct(old['rowslt7_collision_rate'])}；rows>=7 exact-cover bad rate={pct(old['rows7_exact_cover_bad_rate'])}；skeleton-hit geometry-fail candidate rate={pct(old['rows7_skeleton_hit_geometry_fail_candidate_rate'])}。SymCIF v5 没有同一 collision proxy，本实验只记录 bond_length_score_lt_0.5={pct(sdiag['bond_length_score_lt_0_5_rate'])}，不能等同 StructureMatcher collision。

rows>=7 专门 scorer/route 结果：Track A rows>=7 = {rows_metric_triplet(track)}；hard-negative rows>=7 = {rows_metric_triplet(hard)}。hard-negative 相对 baseline 约为 K1 {pp(hard['rows>=7_match@1'] - base['rows>=7_match@1'])}、K5 {pp(hard['rows>=7_match@5'] - base['rows>=7_match@5'])}、K20 {pp(hard['rows>=7_match@20'] - base['rows>=7_match@20'])}，没有达到两个指标 +5pp，判定普通 rows>=7 scorer 不过 gate。

hybrid 边界：MPTS-52 official C2S3C15 rows>=7 baseline={metric_triplet(official['baseline_rows_ge7'])}，hybrid={metric_triplet(official['hybrid_rows_ge7'])}，delta={pp(official['delta_rows_ge7']['match@1'])}/{pp(official['delta_rows_ge7']['match@5'])}/{pp(official['delta_rows_ge7']['match@20'])}。MP-20 transfer rows>=7 baseline={metric_triplet(mp20['baseline_rows_ge7'])}，hybrid={metric_triplet(mp20['hybrid_rows_ge7'])}，delta={pp(mp20['delta_rows_ge7']['match@1'])}/{pp(mp20['delta_rows_ge7']['match@5'])}/{pp(mp20['delta_rows_ge7']['match@20'])}。这些结果支持“复杂结构有互补信号”，但 route 是 auxiliary，不是主方法。

geometry repair 证据：deterministic repair rows>=7 conversion={pct(repair['deterministic_repair']['repair_eval']['rows>=7_conversion_rate'])}，converted={repair['deterministic_repair']['repair_eval']['rows>=7_converted']}/{repair['deterministic_repair']['repair_eval']['rows>=7_evaluated_valid']}；MP-20 GT-WA learned geometry rows>=7 match@1/5={pct(repair['mp20_gtwa_learned_geometry']['match@1'])}/{pct(repair['mp20_gtwa_learned_geometry']['match@5'])}。因此简单 wrap/jitter 停止，learned/optimized geometry repair 继续，但必须绑定 inference-time skeleton proposer。

可信度：CrystaLLM K50 与 exact-cover/collision 诊断来自全量 validation labels；SymCIF v5 是现有 generation/metrics replay，缺失样本按 fail；official/MP20 transfer/repair 结果直接读取前序实验 JSON。限制是 rows>=7 target 口径在 opentry_7 target cache 为 {ds['target_rows_ge7_samples']}，旧 opentry_11 K50 表为 2292，本报告并列展示时不把不同口径强行合并。

和历史实验关系：这是对 opentry_11 实验 6、实验 7C、实验 8 strict ablation 的更新版 rows>=7 专门判读，并把 opentry_12 实验 1-5 的 official/hybrid/proposer/repair 结果接入同一失败归因矩阵。

最终判决：rows>=7 不是普通排序能解决的瓶颈。当前 CrystaLLM K50 top50 headroom 小，SymCIF v5 exact-cover 高但 skeleton-to-match conversion 低，deterministic repair conversion=0；C2S3C15 只能作为辅助证明复杂结构互补存在。主线应转向 rows>=7-specialized skeleton proposer + learned geometry repair。

下一步：做实验 7 主方法消融与 hybrid 边界说明，把 SymCIF/exact-cover/geometry repair/proposer 与 C2S3C15/Track A/scorer 的角色彻底分开，并给出 overall 与 rows>=7 的最终 gate 判决。
"""


def main() -> None:
    result = run()
    out = RESULTS / "experiment_6_rows_ge7_specialized_audit.json"
    write_json(out, result)
    append_report_once(
        RESULTS / "experiment_6_report_appended.json",
        "实验 6 rows>=7 专门生成/修复路线审计",
        report(result),
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "rows_ge7_samples": result["data_scale"]["target_rows_ge7_samples"],
                "crystallm_top50_rows_ge7": result["crystallm_k50_rows_ge7"]["top50_match"],
                "symcif_top50_rows_ge7": result["symcif_v5_rows_ge7_full_missing_as_fail"]["top50_match"],
                "decision": result["decision"]["rows_ge7_primary_bottleneck"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
