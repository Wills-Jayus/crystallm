#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool"

for _path in (OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_exp3_rows7_skeleton_proposer as exp3  # noqa: E402


EXP2_RESULT = RESULT_DIR / "experiment_2_joint_geometry_repair.json"
EXP2_EVAL = OUT_DIR / "artifacts" / "exp2_joint_geometry_repair" / "evaluated_joint_repair_candidates.jsonl"
EXP4_EVAL = OP13 / "artifacts" / "exp4_rows_ge7_multi_geometry_proposal" / "evaluated_ranked_candidates.jsonl"
EXP3_RESULT = OP13 / "results" / "experiment_3_rows_ge7_skeleton_proposer_validation_gate.json"
EXP4_RESULT = OP13 / "results" / "experiment_4_rows_ge7_multi_geometry_proposal.json"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

BUDGETS = (1, 5, 20, 50)


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


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def normalize_hydrated_candidate(row: dict[str, Any], *, sample_meta: dict[str, Any], local_rank: int) -> dict[str, Any]:
    skeleton_hit = str(row.get("skeleton_key") or "") == str(row.get("target_skeleton_key") or "")
    exact_cover = row.get("exact_cover_feasible")
    return {
        "sample_id": str(row["sample_id"]),
        "material_id": str(row.get("material_id") or str(row["sample_id"]).split("__")[-1]),
        "row_count": int(sample_meta.get("row_count") or 0),
        "sg": int(sample_meta.get("sg") or 0),
        "rank": int(local_rank),
        "pool_source": "symcif_v5_hydrated",
        "source_rank": int(row.get("gen_index") or local_rank - 1) + 1,
        "generation_score": row.get("generation_score"),
        "proposal_rank": row.get("proposal_rank"),
        "proposal_source": row.get("proposal_source"),
        "geometry_source": str(row.get("source_experiment") or row.get("source_experiment") or "symcif_v5_hydrated"),
        "reference_sample_id": row.get("source_sample_id"),
        "generated_sha1": row.get("generated_sha1"),
        "source_eval_path": str(exp3.SYMCIF_MET),
        "source_generation_path": str(exp3.SYMCIF_GEN),
        "predicted_skeleton_key": str(row.get("skeleton_key") or ""),
        "target_skeleton_key": str(row.get("target_skeleton_key") or ""),
        "predicted_skeleton_hit": bool(skeleton_hit),
        "match": bool(row.get("match")),
        "rms": row.get("rms"),
        "valid": bool(row.get("valid")),
        "formula_ok": bool(row.get("formula_ok")),
        "space_group_ok": bool(row.get("sg_ok")),
        "site_count_ok": True,
        "exact_cover_retained": bool(exact_cover is True),
        "legal_cif": bool(row.get("valid") or row.get("formula_ok") or row.get("sg_ok")),
        "collision_flag": None,
        "volume_per_atom_ok": None,
        "selection_rule": "fixed_top10_hydrated_then_top40_prototype; no match/RMSD used for ranking",
    }


def normalize_prototype_candidate(row: dict[str, Any], *, local_rank: int) -> dict[str, Any]:
    out = dict(row)
    out["rank"] = int(local_rank)
    out["pool_source"] = "opentry13_prototype_multi_geometry"
    out["source_rank"] = int(row.get("rank") or local_rank)
    out["source_eval_path"] = str(EXP4_EVAL)
    out["selection_rule"] = "fixed_top10_hydrated_then_top40_prototype; no match/RMSD used for ranking"
    out["space_group_ok"] = bool(row.get("space_group_ok"))
    out["exact_cover_retained"] = bool(row.get("exact_cover_retained"))
    out["predicted_skeleton_hit"] = bool(row.get("predicted_skeleton_hit"))
    out["match"] = bool(row.get("match"))
    out["valid"] = bool(row.get("valid"))
    out["formula_ok"] = bool(row.get("formula_ok"))
    out["legal_cif"] = bool(row.get("legal_cif"))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = sorted({str(r["sample_id"]) for r in rows})
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    out: dict[str, Any] = {
        "samples": len(sample_ids),
        "candidate_records": len(rows),
        "valid_rate": ratio(sum(bool(r.get("valid")) for r in rows), len(rows)),
        "formula_consistency": ratio(sum(bool(r.get("formula_ok")) for r in rows), len(rows)),
        "sg_consistency": ratio(sum(bool(r.get("space_group_ok")) for r in rows), len(rows)),
        "exact_cover_retained": ratio(sum(bool(r.get("exact_cover_retained")) for r in rows), len(rows)),
        "legal_cif_rate": ratio(sum(bool(r.get("legal_cif")) for r in rows), len(rows)),
        "collision_rate": ratio(sum(bool(r.get("collision_flag")) for r in rows if r.get("collision_flag") is not None), sum(r.get("collision_flag") is not None for r in rows)),
    }
    for k in BUDGETS:
        match_hits = valid_any = formula_any = sg_any = exact_any = skeleton_any = skeleton_and_match = 0
        rms_vals: list[float] = []
        for sid in sample_ids:
            top = sorted(by_sid[sid], key=lambda r: int(r.get("rank") or 10**9))[:k]
            match_any = any(bool(r.get("match")) for r in top)
            skel_any = any(bool(r.get("predicted_skeleton_hit")) for r in top)
            match_hits += int(match_any)
            valid_any += int(any(bool(r.get("valid")) for r in top))
            formula_any += int(any(bool(r.get("formula_ok")) for r in top))
            sg_any += int(any(bool(r.get("space_group_ok")) for r in top))
            exact_any += int(any(bool(r.get("exact_cover_retained")) for r in top))
            skeleton_any += int(skel_any)
            skeleton_and_match += int(skel_any and match_any)
            matched_rms = [float(r["rms"]) for r in top if bool(r.get("match")) and r.get("rms") is not None]
            if matched_rms:
                rms_vals.append(min(matched_rms))
        out[f"match@{k}"] = ratio(match_hits, len(sample_ids))
        out[f"RMSE@{k}"] = mean(rms_vals)
        out[f"valid_any@{k}"] = ratio(valid_any, len(sample_ids))
        out[f"formula_ok_any@{k}"] = ratio(formula_any, len(sample_ids))
        out[f"sg_ok_any@{k}"] = ratio(sg_any, len(sample_ids))
        out[f"exact_cover_any@{k}"] = ratio(exact_any, len(sample_ids))
        out[f"skeleton_hit_coverage@{k}"] = ratio(skeleton_any, len(sample_ids))
        out[f"skeleton_to_match_conversion@{k}"] = ratio(skeleton_and_match, skeleton_any)
    return out


def candidate_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(r.get("pool_source") or "unknown") for r in rows)
    return dict(counter)


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


def report_body(result: dict[str, Any]) -> str:
    rows7 = result["rows_ge7"]
    overall = result["overall"]
    diag = result["failure_analysis"]
    gate = result["gate"]
    return f"""## opentry_14 实验 2b：failure-analysis guided safe geometry pool

结果文件：`model/New_model/opentry_14/results/experiment_2b_safe_pool_after_failure_analysis.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2b_hydrated_prototype_safe_pool/evaluated_safe_pool_candidates.jsonl`

- 为什么做：实验 2 的 joint MLP 在 rows>=7 上 match@50 只有 `{pct(diag['exp2_joint_rows_ge7'].get('match@50'))}`、conversion@50 只有 `{pct(diag['exp2_joint_rows_ge7'].get('skeleton_to_match_conversion@50'))}`。失败分析显示它把已有可 match 的 hydrated/source-prototype geometry 替换成单一 repaired geometry，导致 rows>=7 只剩 `8` 个 matched samples；因此本实验不再让 MLP 覆盖安全候选，而是构造固定配额的 safe geometry pool。
- 核心假设：如果根因是破坏性单一 repair 覆盖，而不是 skeleton coverage 不足，那么保留原 SymCIF v5 hydrated geometry，再补入 prototype multi-geometry，应该在不使用 match/RMSD 排序的情况下恢复并超过实验 2 最低 gate。
- 数据规模：validation samples `{result['data_scale']['validation_samples']}`，rows>=7 `{result['data_scale']['validation_rows_ge7_samples']}`；candidate records `{result['data_scale']['candidate_records']}`；topK `50`；固定配额为 hydrated top `{result['method']['hydrated_quota']}` + prototype top `{result['method']['prototype_quota']}`。
- baseline：实验 2 最低 gate 使用 rows>=7 CrystaLLM K50 validation top50 `{pct(result['baselines']['crystallm_rows_ge7_top50'])}`，允许近线下限为 `{pct(result['baselines']['crystallm_rows_ge7_top50_minus_0p5pp'])}`；历史 opentry_13 hydrated rows>=7 match@50 `{pct(result['baselines']['opentry13_predicted_skeleton_hydrated_rows_ge7_match50'])}`，multi-geometry rows>=7 match@50 `{pct(result['baselines']['opentry13_multi_geometry_rows_ge7_match50'])}`。
- 方法变化：排名规则固定为每个样本先取 rows>=7 proposer 映射到的 SymCIF v5 hydrated candidates top10，再取 opentry_13 prototype multi-geometry top40；候选内部顺序只使用 proposer rank、generation score 和原 structural rank，不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB/rerank。
- 结果 overall：match@1/5/20/50 = `{pct(overall.get('match@1'))} / {pct(overall.get('match@5'))} / {pct(overall.get('match@20'))} / {pct(overall.get('match@50'))}`；formula `{pct(overall.get('formula_consistency'))}`，SG `{pct(overall.get('sg_consistency'))}`，exact-cover `{pct(overall.get('exact_cover_retained'))}`。
- 结果 rows>=7：match@1/5/20/50 = `{pct(rows7.get('match@1'))} / {pct(rows7.get('match@5'))} / {pct(rows7.get('match@20'))} / {pct(rows7.get('match@50'))}`；RMSE@50 `{rows7.get('RMSE@50')}`；skeleton-hit@50 `{pct(rows7.get('skeleton_hit_coverage@50'))}`；skeleton-to-match conversion@50 `{pct(rows7.get('skeleton_to_match_conversion@50'))}`；valid_any@50 `{pct(rows7.get('valid_any@50'))}`；formula `{pct(rows7.get('formula_consistency'))}`，SG `{pct(rows7.get('sg_consistency'))}`，exact-cover `{pct(rows7.get('exact_cover_retained'))}`。
- 可信度：中等偏高。所有 counted candidates 都有已有 validation metrics 或 opentry_13 StructureMatcher evaluation；本实验只做固定配额组合并重新汇总，不用 match/RMSD 参与排序。限制是 hydrated metrics 是复用既有 SymCIF v5 validation evaluation，不是重新跑 StructureMatcher；该方法是 validation safe-pool gate，不是 official claim。
- 和历史实验关系：opentry_13 exp3 hydrated rows>=7 top50 已有 `15.112%`，opentry_13 exp4 prototype multi-geometry top50 为 `13.084%`，二者互补；opentry_14 exp2 joint MLP 破坏 rows>=7 geometry。本实验把历史两类 inference-safe geometry pool 合并，先恢复最低 conversion gate。
- gate 判定：minimum_passed=`{gate['minimum_passed']}`，target_passed=`{gate['target_passed']}`；rows>=7 match@50 delta vs near CrystaLLM K50 lower bound `{pp(gate['rows_ge7_match50_delta_vs_minimum'])}`；conversion@50 delta vs 22.3% `{pp(gate['rows_ge7_conversion50_delta_vs_minimum'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def append_or_replace_report(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker not in text:
        with REPORT_PATH.open("a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(replacement)
        return
    start = text.index(marker)
    next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
    if next_marker == -1:
        REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement, encoding="utf-8")
    else:
        REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement + text[next_marker:], encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydrated-quota", type=int, default=10)
    parser.add_argument("--prototype-quota", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    if int(args.hydrated_quota) + int(args.prototype_quota) != int(args.top_k):
        raise ValueError("hydrated_quota + prototype_quota must equal top_k")

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = exp3.read_jsonl(exp3.TRAIN_REPR)
    val_rows = exp3.read_jsonl(exp3.VAL_REPR)
    train_index = exp3.build_train_index(train_rows)
    hydrated_groups = exp3.load_hydrated_candidates()
    sample_meta = {exp3.sample_id(r): r for r in val_rows}

    prototype_by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl_iter(EXP4_EVAL):
        prototype_by_sid[str(row["sample_id"])].append(row)
    for rows in prototype_by_sid.values():
        rows.sort(key=lambda r: int(r.get("rank") or 10**9))

    out_rows: list[dict[str, Any]] = []
    source_counts = Counter()
    for record in val_rows:
        sid = exp3.sample_id(record)
        local_rank = 1
        proposals = exp3.propose_for_target(record, train_index, top_n=50)
        hydrated = exp3.hydrate_by_proposal(proposals, hydrated_groups.get(sid, []), max_candidates=50)
        for cand in hydrated[: int(args.hydrated_quota)]:
            row = normalize_hydrated_candidate(cand, sample_meta=record, local_rank=local_rank)
            out_rows.append(row)
            source_counts[row["pool_source"]] += 1
            local_rank += 1
        for cand in prototype_by_sid.get(sid, [])[: int(args.prototype_quota)]:
            row = normalize_prototype_candidate(cand, local_rank=local_rank)
            out_rows.append(row)
            source_counts[row["pool_source"]] += 1
            local_rank += 1

    out_rows.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    write_jsonl(ARTIFACT_DIR / "evaluated_safe_pool_candidates.jsonl", out_rows)

    rows_ge7 = [r for r in out_rows if int(r.get("row_count") or 0) >= 7]
    rows_lt7 = [r for r in out_rows if int(r.get("row_count") or 0) < 7]
    overall = summarize(out_rows)
    rows7 = summarize(rows_ge7)
    rowslt7 = summarize(rows_lt7)

    exp2_result = read_json(EXP2_RESULT)
    exp3_result = read_json(EXP3_RESULT)
    exp4_result = read_json(EXP4_RESULT)
    crystallm_rows7_top50 = float(exp3_result["baseline_reference"]["crystallm_k50_rows_ge7"]["top50_match"])
    min_match50 = crystallm_rows7_top50 - 0.005
    min_gate = bool(
        (rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.223
        and (rows7.get("match@50") or 0.0) >= min_match50
        and (rows7.get("formula_consistency") or 0.0) >= 0.95
        and (rows7.get("sg_consistency") or 0.0) >= 0.90
        and (rows7.get("exact_cover_retained") or 0.0) >= 0.95
    )
    target_gate = bool(
        (rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28
        or (
            (rows7.get("match@5") or 0.0) >= exp4_result["baselines"]["crystallm_rows_ge7_reference"]["top5_match"] + 0.05
            and (rows7.get("match@20") or 0.0) >= exp4_result["baselines"]["crystallm_rows_ge7_reference"]["top20_match"] + 0.05
        )
    )
    result = {
        "experiment": "opentry_14_exp2b_failure_analysis_guided_safe_geometry_pool",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "fixed_quota_hydrated_plus_prototype_safe_pool",
            "hydrated_quota": int(args.hydrated_quota),
            "prototype_quota": int(args.prototype_quota),
            "top_k": int(args.top_k),
            "inference_inputs": ["composition", "GT-SG", "predicted exact-cover skeleton proposals", "existing hydrated SymCIF v5 geometry", "train prototype multi-geometry"],
            "ranking_rule": "fixed top10 hydrated followed by top40 prototype structural-rank candidates",
            "not_used_for_ranking": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
        },
        "data_scale": {
            "validation_samples": overall["samples"],
            "validation_rows_ge7_samples": rows7["samples"],
            "candidate_records": len(out_rows),
            "rows_ge7_candidate_records": len(rows_ge7),
            "rows_lt7_candidate_records": len(rows_lt7),
            "source_counts": dict(source_counts),
        },
        "failure_analysis": {
            "root_cause": "single joint MLP repair destructively replaced safe hydrated/prototype geometries; rows>=7 labels are noisy and param head does not recover matchable geometry",
            "exp2_joint_rows_ge7": exp2_result["rows_ge7"],
            "exp2_joint_overall": exp2_result["overall"],
            "opentry13_multi_geometry_rows_ge7": exp4_result["rows_ge7"],
            "opentry13_hydrated_rows_ge7_match50": exp3_result["rows_ge7"].get("top50_hydrated_match_coverage"),
        },
        "baselines": {
            "crystallm_rows_ge7_top50": crystallm_rows7_top50,
            "crystallm_rows_ge7_top50_minus_0p5pp": min_match50,
            "opentry13_predicted_skeleton_hydrated_rows_ge7_match50": exp3_result["rows_ge7"].get("top50_hydrated_match_coverage"),
            "opentry13_multi_geometry_rows_ge7_match50": exp4_result["rows_ge7"].get("match@50"),
        },
        "overall": overall,
        "rows_ge7": rows7,
        "rows_lt7": rowslt7,
        "gate": {
            "minimum_passed": min_gate,
            "target_passed": target_gate,
            "passed": min_gate,
            "rows_ge7_match50_delta_vs_minimum": (rows7.get("match@50") or 0.0) - min_match50,
            "rows_ge7_conversion50_delta_vs_minimum": (rows7.get("skeleton_to_match_conversion@50") or 0.0) - 0.223,
            "minimum_standard": {
                "rows_ge7_conversion50": 0.223,
                "rows_ge7_match50_near_crystallm_k50": crystallm_rows7_top50,
                "rows_ge7_match50_allowed_lower_bound": min_match50,
                "rows_ge7_formula_consistency": 0.95,
                "rows_ge7_sg_consistency": 0.90,
                "rows_ge7_exact_cover_retained": 0.95,
            },
        },
        "decision": {
            "verdict": "pass_minimum_gate" if min_gate else "fail_validation_gate",
            "reason": "Safe pool passes the experiment-2 minimum validation gate; target gate is recorded separately." if min_gate else "Safe pool still does not pass the minimum validation gate.",
            "next_step": "Allowed to test experiment 3 local optimizer on this fixed safe pool; do not claim official success before full validation." if min_gate else "Do not enter experiment 3/full validation/official; continue repair/data-alignment work.",
        },
        "artifacts": {
            "evaluated_safe_pool_candidates": str(ARTIFACT_DIR / "evaluated_safe_pool_candidates.jsonl"),
            "source_hydrated_generation": str(exp3.SYMCIF_GEN),
            "source_hydrated_metrics": str(exp3.SYMCIF_MET),
            "source_prototype_eval": str(EXP4_EVAL),
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json", result)
    append_or_replace_report("<!-- OPENTRY14_EXP2B_SAFE_POOL_AFTER_FAILURE_ANALYSIS -->", report_body(result))
    print(json.dumps({"output": str(RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json"), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
