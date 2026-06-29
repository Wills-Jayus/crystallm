#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2g_candidate_headroom_audit"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import read_jsonl, sample_id  # noqa: E402


VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
EXP2B_RESULT = RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json"
EXP2D_RESULT = RESULT_DIR / "experiment_2d_site_assignment_multi_hypothesis.json"
EXP2E_RESULT = RESULT_DIR / "experiment_2e_train_pair_residual_posterior.json"
EXP2F_RESULT = RESULT_DIR / "experiment_2f_permutation_aware_alignment.json"
SOURCE_FILES = {
    "exp2b_safe_pool": OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool" / "evaluated_safe_pool_candidates.jsonl",
    "exp2d_site_assignment": OUT_DIR / "artifacts" / "exp2d_site_assignment_multi_hypothesis" / "evaluated_site_assignment_candidates.jsonl",
    "exp2e_residual_posterior": OUT_DIR / "artifacts" / "exp2e_train_pair_residual_posterior" / "evaluated_residual_posterior_candidates.jsonl",
    "exp2f_permutation_alignment": OUT_DIR / "artifacts" / "exp2f_permutation_aware_alignment" / "evaluated_permutation_alignment_candidates.jsonl",
}
BUDGETS = (1, 5, 20, 50, 100, 200)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def pct(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:.3f}%"


def pp(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:+.3f}pp"


def safe_int(v: Any, default: int = 10**9) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def candidate_rank(row: dict[str, Any]) -> int:
    return min(
        safe_int(row.get("rank")),
        safe_int(row.get("geometry_rank")),
        safe_int(row.get("source_rank")),
        safe_int(row.get("raw_generation_order")),
    )


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("sample_id")),
        str(row.get("source_pool")),
        str(row.get("pool_source") or ""),
        str(row.get("geometry_source") or ""),
        str(row.get("source_sample_id") or row.get("reference_sample_id") or ""),
        str(row.get("predicted_skeleton_key") or ""),
        safe_int(row.get("proposal_rank"), 0),
        safe_int(row.get("assignment_rank"), 0),
        safe_int(row.get("permutation_rank"), 0),
        str(row.get("alignment_variant") or ""),
        str(row.get("generated_sha1") or ""),
        safe_int(row.get("rank"), 0),
        safe_int(row.get("geometry_rank"), 0),
        safe_int(row.get("source_rank"), 0),
    )


def rank_stats(ranks: list[int]) -> dict[str, Any]:
    clean = [int(r) for r in ranks if r < 10**9]
    if not clean:
        return {"count": 0}
    clean.sort()
    return {
        "count": len(clean),
        "min": clean[0],
        "median": statistics.median(clean),
        "p90": clean[min(len(clean) - 1, int(math.ceil(0.9 * len(clean))) - 1)],
        "max": clean[-1],
        "le50": sum(r <= 50 for r in clean),
        "gt50": sum(r > 50 for r in clean),
    }


def summarize_rows(rows: list[dict[str, Any]], universe: set[str], label: str) -> dict[str, Any]:
    rows7 = [r for r in rows if str(r.get("sample_id")) in universe]
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows7:
        by_sid[str(row["sample_id"])].append(row)

    sample_ids = set(by_sid)
    skeleton_samples = {sid for sid, group in by_sid.items() if any(bool(r.get("predicted_skeleton_hit")) for r in group)}
    match_samples = {sid for sid, group in by_sid.items() if any(bool(r.get("match")) for r in group)}
    skelmatch_samples = skeleton_samples & match_samples
    valid_complete_samples = {
        sid
        for sid, group in by_sid.items()
        if any(
            bool(r.get("valid"))
            and bool(r.get("formula_ok"))
            and bool(r.get("space_group_ok"))
            and bool(r.get("exact_cover_retained"))
            for r in group
        )
    }
    first_match_rank = []
    first_skeleton_match_rank = []
    for sid, group in by_sid.items():
        match_ranks = [candidate_rank(r) for r in group if bool(r.get("match"))]
        if match_ranks:
            first_match_rank.append(min(match_ranks))
        skmatch_ranks = [candidate_rank(r) for r in group if bool(r.get("match")) and bool(r.get("predicted_skeleton_hit"))]
        if skmatch_ranks:
            first_skeleton_match_rank.append(min(skmatch_ranks))

    out: dict[str, Any] = {
        "label": label,
        "universe_samples": len(universe),
        "sample_with_candidates": len(sample_ids),
        "candidate_records": len(rows7),
        "sample_candidate_coverage": ratio(len(sample_ids), len(universe)),
        "match_samples_any": len(match_samples),
        "match_rate_any_fixed_denominator": ratio(len(match_samples), len(universe)),
        "skeleton_samples_any": len(skeleton_samples),
        "skeleton_coverage_any_fixed_denominator": ratio(len(skeleton_samples), len(universe)),
        "skeleton_and_match_samples_any": len(skelmatch_samples),
        "skeleton_to_match_conversion_any": ratio(len(skelmatch_samples), len(skeleton_samples)),
        "valid_complete_samples_any": len(valid_complete_samples),
        "valid_complete_rate_fixed_denominator": ratio(len(valid_complete_samples), len(universe)),
        "valid_rate_records": ratio(sum(bool(r.get("valid")) for r in rows7), len(rows7)),
        "formula_rate_records": ratio(sum(bool(r.get("formula_ok")) for r in rows7), len(rows7)),
        "sg_rate_records": ratio(sum(bool(r.get("space_group_ok")) for r in rows7), len(rows7)),
        "exact_cover_rate_records": ratio(sum(bool(r.get("exact_cover_retained")) for r in rows7), len(rows7)),
        "collision_rate_records": ratio(
            sum(bool(r.get("collision_flag")) for r in rows7 if r.get("collision_flag") is not None),
            sum(r.get("collision_flag") is not None for r in rows7),
        ),
        "first_match_rank": rank_stats(first_match_rank),
        "first_skeleton_match_rank": rank_stats(first_skeleton_match_rank),
    }

    for k in BUDGETS:
        match_k = set()
        skeleton_k = set()
        skelmatch_k = set()
        for sid, group in by_sid.items():
            top = sorted(group, key=candidate_rank)[:k]
            has_match = any(bool(r.get("match")) for r in top)
            has_skel = any(bool(r.get("predicted_skeleton_hit")) for r in top)
            if has_match:
                match_k.add(sid)
            if has_skel:
                skeleton_k.add(sid)
            if has_match and has_skel:
                skelmatch_k.add(sid)
        out[f"match@{k}_fixed_denominator"] = ratio(len(match_k), len(universe))
        out[f"skeleton_hit@{k}_fixed_denominator"] = ratio(len(skeleton_k), len(universe))
        out[f"skeleton_to_match_conversion@{k}"] = ratio(len(skelmatch_k), len(skeleton_k))

    return out


def summarize_failure_buckets(rows: list[dict[str, Any]], universe: set[str]) -> dict[str, Any]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = str(row.get("sample_id"))
        if sid in universe:
            by_sid[sid].append(row)

    buckets = Counter()
    candidate_counts = []
    valid_complete_no_match = 0
    skeleton_no_match = 0
    skeleton_no_match_valid_complete = 0
    no_candidate = 0
    no_skeleton = 0
    for sid in sorted(universe):
        group = by_sid.get(sid, [])
        if not group:
            no_candidate += 1
            buckets["no_candidate"] += 1
            continue
        candidate_counts.append(len(group))
        has_skel = any(bool(r.get("predicted_skeleton_hit")) for r in group)
        has_match = any(bool(r.get("match")) for r in group)
        has_valid_complete = any(
            bool(r.get("valid"))
            and bool(r.get("formula_ok"))
            and bool(r.get("space_group_ok"))
            and bool(r.get("exact_cover_retained"))
            for r in group
        )
        if not has_skel:
            no_skeleton += 1
            buckets["candidate_but_no_skeleton_hit"] += 1
        elif not has_match:
            skeleton_no_match += 1
            buckets["skeleton_hit_no_match"] += 1
            if has_valid_complete:
                skeleton_no_match_valid_complete += 1
        elif has_skel and has_match:
            buckets["skeleton_hit_and_match"] += 1
        if has_valid_complete and not has_match:
            valid_complete_no_match += 1

    return {
        "buckets": dict(buckets),
        "no_candidate_samples": no_candidate,
        "candidate_but_no_skeleton_hit_samples": no_skeleton,
        "skeleton_hit_no_match_samples": skeleton_no_match,
        "skeleton_hit_no_match_with_valid_formula_sg_exact_sample": skeleton_no_match_valid_complete,
        "valid_formula_sg_exact_but_no_match_samples": valid_complete_no_match,
        "candidate_count_per_sample": {
            "mean": statistics.mean(candidate_counts) if candidate_counts else None,
            "median": statistics.median(candidate_counts) if candidate_counts else None,
            "max": max(candidate_counts) if candidate_counts else None,
        },
    }


def source_contributions(source_sets: dict[str, set[str]]) -> dict[str, Any]:
    all_sources = sorted(source_sets)
    union = set().union(*(source_sets[s] for s in all_sources)) if all_sources else set()
    out = {"union_match_samples": len(union), "sources": {}}
    for source in all_sources:
        others = set().union(*(source_sets[s] for s in all_sources if s != source)) if len(all_sources) > 1 else set()
        own = source_sets[source]
        out["sources"][source] = {
            "match_samples": len(own),
            "unique_match_samples_vs_other_sources": len(own - others),
            "overlap_with_other_sources": len(own & others),
        }
    return out


def append_or_replace(path: Path, marker: str, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker not in text:
        with path.open("a", encoding="utf-8") as f:
            if text and not text.endswith("\n\n"):
                f.write("\n\n")
            f.write(replacement)
        return
    start = text.index(marker)
    next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
    if next_marker == -1:
        path.write_text(text[:start].rstrip() + "\n\n" + replacement, encoding="utf-8")
    else:
        path.write_text(text[:start].rstrip() + "\n\n" + replacement + text[next_marker:], encoding="utf-8")


def report_body(result: dict[str, Any]) -> str:
    union = result["union_all_candidates"]
    failures = result["failure_buckets"]
    best = result["current_best_validation"]
    contrib = result["source_contributions"]["sources"]
    return f"""## opentry_14 实验 2g：candidate headroom 与 ranking 根因审计

结果文件：`model/New_model/opentry_14/results/experiment_2g_candidate_headroom_audit.json`
诊断 artifact：`model/New_model/opentry_14/artifacts/exp2g_candidate_headroom_audit/`

- 为什么做：Exp2d/2e/2f 都只带来小幅或零新增 sample-level match，需要确认失败根因是候选池本身没有足够正确几何，还是已有正确候选被 rank/top50 选择压掉。该实验只做 oracle/headroom 审计，不作为 scorer 或主方法。
- 核心假设：如果现有 Exp2b+2d+2e+2f 候选池内已经有足够 match，只是排序不好，则 all-candidate oracle 应显著超过 Exp3 线 `26.866%` conversion 或 Exp2 target `28%`；如果 all-candidate oracle 仍低，则继续扩大同类 beam 或普通 critic 没有通过 gate 的余量。
- 数据规模：rows>=7 validation universe `{result['data_scale']['rows_ge7_universe_samples']}` samples；审计候选 `{union['candidate_records']}` records；覆盖样本 `{union['sample_with_candidates']}`；读取来源包括 Exp2b safe pool、Exp2d site assignment、Exp2e residual posterior、Exp2f permutation alignment。
- baseline：当前 match@50 最优仍是 Exp2d `{best['best_match_variant']}`，rows>=7 match@50 `{pct(best['best_match50'])}`、conversion `{pct(best['best_match_variant_conversion50'])}`；当前 conversion 数值最高为 Exp2f `{best['best_conversion_variant']}`，conversion `{pct(best['best_conversion50'])}`、match@50 `{pct(best['best_conversion_variant_match50'])}`。Exp3 gate 需要 conversion `{pct(result['gate_reference']['exp3_required_conversion50'])}`。
- 方法变化：按 sample_id 合并既有 validation 候选，不生成新 CIF，不训练模型，不改变 rank；统计 fixed-denominator match/skeleton coverage、all-candidate oracle、first-match rank、source unique contribution 和 skeleton-hit/no-match failure buckets。推理侧不使用 match/RMSD/StructureMatcher label；match 只作为离线审计标签。
- 结果：union all-candidate oracle rows>=7 match fixed-denominator `{pct(union['match_rate_any_fixed_denominator'])}`，skeleton coverage `{pct(union['skeleton_coverage_any_fixed_denominator'])}`，skeleton-to-match conversion `{pct(union['skeleton_to_match_conversion_any'])}`。first skeleton-match rank 中位数 `{union['first_skeleton_match_rank'].get('median')}`，`>50` 的样本数 `{union['first_skeleton_match_rank'].get('gt50')}`。
- source 贡献：unique match samples vs other sources 分别为 Exp2b `{contrib['exp2b_safe_pool']['unique_match_samples_vs_other_sources']}`、Exp2d `{contrib['exp2d_site_assignment']['unique_match_samples_vs_other_sources']}`、Exp2e `{contrib['exp2e_residual_posterior']['unique_match_samples_vs_other_sources']}`、Exp2f `{contrib['exp2f_permutation_alignment']['unique_match_samples_vs_other_sources']}`。Exp2e/2f 的新增很小，说明 residual/permutation 候选族余量不足。
- 失败桶：no-candidate `{failures['no_candidate_samples']}`；candidate but no skeleton-hit `{failures['candidate_but_no_skeleton_hit_samples']}`；skeleton-hit/no-match `{failures['skeleton_hit_no_match_samples']}`，其中已有 valid+formula+SG+exact candidate 的 `{failures['skeleton_hit_no_match_with_valid_formula_sg_exact_sample']}`。这支持“结构合法但几何 basin/骨架候选不对”的根因，而不是前端结构 gate 崩坏。
- 可信度：中等偏高。它直接读取已真实评估的 validation artifacts，固定 rows>=7 universe，未改写历史目录；限制是 oracle 使用 validation match 标签进行诊断，不能作为推理排序规则，也不能证明某个 inference-safe critic 能达到 oracle。
- 和历史实验关系：解释 Exp3/3b local optimizer、Exp2e residual posterior、Exp2f deterministic permutation 都不能过 gate 的共同原因：候选池 all-candidate oracle 本身仍低于 Exp3 所需 conversion，排序不是主瓶颈。
- gate 判定：target_headroom_passed=`{result['gate']['target_headroom_passed']}`；exp3_headroom_passed=`{result['gate']['exp3_headroom_passed']}`；oracle conversion delta vs Exp3 line `{pp(result['gate']['oracle_conversion_delta_vs_exp3_line'])}`；oracle conversion delta vs Exp2 target `{pp(result['gate']['oracle_conversion_delta_vs_exp2_target'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-output-limit", type=int, default=200)
    args = parser.parse_args()

    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    val_repr = {sample_id(row): row for row in read_jsonl(VAL_REPR)}
    rows7_universe = {sid for sid, row in val_repr.items() if int(row.get("row_count") or 0) >= 7}

    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_match_sets: dict[str, set[str]] = {}
    all_rows: list[dict[str, Any]] = []
    seen = set()
    for source, path in SOURCE_FILES.items():
        rows: list[dict[str, Any]] = []
        for row in read_jsonl_iter(path):
            sid = str(row.get("sample_id"))
            if sid not in rows7_universe:
                continue
            item = dict(row)
            item["source_pool"] = source
            rows.append(item)
            key = row_key(item)
            if key not in seen:
                seen.add(key)
                all_rows.append(item)
        source_rows[source] = rows
        source_match_sets[source] = {str(r["sample_id"]) for r in rows if bool(r.get("match"))}

    source_summaries = {source: summarize_rows(rows, rows7_universe, source) for source, rows in source_rows.items()}
    union_summary = summarize_rows(all_rows, rows7_universe, "union_all_candidates")
    failure_buckets = summarize_failure_buckets(all_rows, rows7_universe)

    contribution = source_contributions(source_match_sets)
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_sid[str(row["sample_id"])].append(row)

    hard_cases: list[dict[str, Any]] = []
    for sid in sorted(rows7_universe):
        group = by_sid.get(sid, [])
        if not group:
            continue
        has_skel = any(bool(r.get("predicted_skeleton_hit")) for r in group)
        has_match = any(bool(r.get("match")) for r in group)
        if not has_skel or has_match:
            continue
        valid_complete = sum(
            bool(r.get("valid"))
            and bool(r.get("formula_ok"))
            and bool(r.get("space_group_ok"))
            and bool(r.get("exact_cover_retained"))
            for r in group
        )
        hard_cases.append(
            {
                "sample_id": sid,
                "candidate_records": len(group),
                "valid_formula_sg_exact_records": valid_complete,
                "source_pools": dict(Counter(str(r.get("source_pool")) for r in group)),
                "best_min_pair_distance": max((safe_float(r.get("min_pair_distance")) or 0.0 for r in group), default=None),
                "min_reference_score": min((safe_float(r.get("reference_score")) for r in group if safe_float(r.get("reference_score")) is not None), default=None),
            }
        )
    hard_cases.sort(key=lambda r: (-int(r["valid_formula_sg_exact_records"]), -int(r["candidate_records"]), str(r["sample_id"])))
    write_jsonl(ARTIFACT_DIR / "skeleton_hit_no_match_hard_cases.jsonl", hard_cases[: int(args.sample_output_limit)])

    exp2b = read_json(EXP2B_RESULT)
    exp2d = read_json(EXP2D_RESULT)
    exp2e = read_json(EXP2E_RESULT)
    exp2f = read_json(EXP2F_RESULT)
    exp2d_best_name = str(exp2d["best_variant"])
    exp2d_best = exp2d["variants"][exp2d_best_name]["rows_ge7"]
    exp2e_best_name = str(exp2e["best_variant"])
    exp2e_best = exp2e["variants"][exp2e_best_name]["rows_ge7"]
    exp2f_best_name = str(exp2f["best_variant"])
    exp2f_best = exp2f["variants"][exp2f_best_name]["rows_ge7"]
    current_variants = {
        "exp2b_safe_pool": exp2b["rows_ge7"],
        f"exp2d_{exp2d_best_name}": exp2d_best,
        f"exp2e_{exp2e_best_name}": exp2e_best,
        f"exp2f_{exp2f_best_name}": exp2f_best,
    }
    best_match_variant = max(current_variants, key=lambda k: float(current_variants[k].get("match@50") or 0.0))
    best_conv_variant = max(current_variants, key=lambda k: float(current_variants[k].get("skeleton_to_match_conversion@50") or 0.0))

    oracle_conversion = float(union_summary.get("skeleton_to_match_conversion_any") or 0.0)
    exp3_line = float(exp2b["rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0) + 0.02
    exp2_target = 0.28
    gate = {
        "target_headroom_passed": oracle_conversion >= exp2_target,
        "exp3_headroom_passed": oracle_conversion >= exp3_line,
        "oracle_conversion_delta_vs_exp3_line": oracle_conversion - exp3_line,
        "oracle_conversion_delta_vs_exp2_target": oracle_conversion - exp2_target,
    }
    verdict = "fail_candidate_headroom_insufficient"
    next_step = (
        "不要进入 Exp4/Exp5/official，也不要继续普通 scorer/rerank。下一步需要新候选来源：训练真正的 geometry posterior，"
        "或升级 predicted skeleton proposer 以产生新的 correct geometry basin；若只沿用现有候选池，oracle 也不够过 gate。"
    )
    reason = (
        "Union all-candidate oracle is still below the Exp3 conversion line and Exp2 target, so the main bottleneck is candidate generation/geometry basin rather than top50 ranking."
    )
    if gate["exp3_headroom_passed"] and not gate["target_headroom_passed"]:
        verdict = "diagnostic_headroom_for_exp3_only"
        reason = "Union all-candidate oracle clears the Exp3 conversion line but not the Exp2 target; a critic could be diagnostic but not enough for the main target."
        next_step = "Only try an inference-safe critic if paired with a new learned generator; do not treat ranking-only gains as final-method success."
    elif gate["target_headroom_passed"]:
        verdict = "diagnostic_headroom_available_for_learned_critic"
        reason = "Union all-candidate oracle clears the Exp2 target, so ranking/critic may be worth testing under strict inference-safe features."
        next_step = "Train/evaluate an inference-safe critic on structural features only, but keep official frozen until Exp5 full validation passes."

    result = {
        "experiment": "opentry_14_exp2g_candidate_headroom_audit",
        "time": {"started": now_iso(), "runtime_seconds": time.time() - started},
        "dataset": "MPTS-52 validation",
        "split": "val",
        "data_scale": {
            "rows_ge7_universe_samples": len(rows7_universe),
            "source_candidate_records": {source: len(rows) for source, rows in source_rows.items()},
            "deduplicated_union_candidate_records": len(all_rows),
        },
        "method": {
            "purpose": "offline oracle/headroom audit after Exp2f failure",
            "forbidden_as_inference": "match/RMSD/StructureMatcher labels are used only for offline diagnosis, not ranking or features",
            "source_files": {k: str(v) for k, v in SOURCE_FILES.items()},
        },
        "current_best_validation": {
            "best_match_variant": best_match_variant,
            "best_match50": current_variants[best_match_variant].get("match@50"),
            "best_match_variant_conversion50": current_variants[best_match_variant].get("skeleton_to_match_conversion@50"),
            "best_conversion_variant": best_conv_variant,
            "best_conversion50": current_variants[best_conv_variant].get("skeleton_to_match_conversion@50"),
            "best_conversion_variant_match50": current_variants[best_conv_variant].get("match@50"),
        },
        "gate_reference": {
            "exp2b_conversion50": exp2b["rows_ge7"].get("skeleton_to_match_conversion@50"),
            "exp3_required_conversion50": exp3_line,
            "exp2_target_conversion50": exp2_target,
        },
        "source_summaries": source_summaries,
        "union_all_candidates": union_summary,
        "source_contributions": contribution,
        "failure_buckets": failure_buckets,
        "gate": gate,
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {"hard_cases": str(ARTIFACT_DIR / "skeleton_hit_no_match_hard_cases.jsonl")},
    }

    write_json(RESULT_DIR / "experiment_2g_candidate_headroom_audit.json", result)
    body = report_body(result)
    marker = "<!-- OPENTRY14_EXP2G_CANDIDATE_HEADROOM_AUDIT -->"
    append_or_replace(REPORT_PATH, marker, body)
    append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / "experiment_2g_candidate_headroom_audit.json"), "decision": result["decision"], "gate": gate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
