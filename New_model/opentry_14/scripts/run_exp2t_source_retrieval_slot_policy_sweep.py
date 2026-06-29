#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2t_source_retrieval_slot_policy_sweep"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (OUT_DIR / "scripts", NEW_MODEL / "opentry_13" / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2f_permutation_aware_alignment import by_sid_ranked, sample_sets, split_safe_pool  # noqa: E402
from run_exp2j_chemical_site_order_assignment import (  # noqa: E402
    EXP2B_EVAL,
    EXP2D_SITE_EVAL,
    append_or_replace,
    pct,
    pp,
    read_json,
    read_jsonl_iter,
    ratio,
    write_json,
    write_jsonl,
)
from run_exp4_rows_ge7_multi_geometry_proposal import assign_structural_ranks, summarize  # noqa: E402


EXP2O_RESULT = RESULT_DIR / "experiment_2o_expanded_pairwise_local_chemistry_assignment.json"
EXP2S_RESULT = RESULT_DIR / "experiment_2s_source_retrieval_supplement.json"


def build_rows(
    *,
    name: str,
    hq: int,
    sq: int,
    oq: int,
    rq: int,
    pq: int,
    hydrated: dict[str, list[dict[str, Any]]],
    site_by: dict[str, list[dict[str, Any]]],
    exp2o_by: dict[str, list[dict[str, Any]]],
    retrieval_by: dict[str, list[dict[str, Any]]],
    prototype: dict[str, list[dict[str, Any]]],
    rows_lt7: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in rows_lt7:
        item = dict(row)
        item["selection_variant"] = name
        rows.append(item)
    for sid in sorted(set(hydrated) | set(site_by) | set(exp2o_by) | set(retrieval_by) | set(prototype)):
        selected: list[dict[str, Any]] = []
        selected.extend(hydrated.get(sid, [])[:hq])
        selected.extend(site_by.get(sid, [])[:sq])
        selected.extend(exp2o_by.get(sid, [])[:oq])
        selected.extend(retrieval_by.get(sid, [])[:rq])
        selected.extend(prototype.get(sid, [])[:pq])
        for rank, row in enumerate(selected[:top_k], start=1):
            item = dict(row)
            item["rank"] = rank
            item["selection_variant"] = name
            rows.append(item)
    rows.sort(key=lambda r: (str(r["sample_id"]), int(r.get("rank") or 10**9)))
    return rows


def report_body(result: dict[str, Any]) -> str:
    best = result["variants"][result["best_variant"]]["rows_ge7"]
    base = result["baseline_exp2o"]["rows_ge7"]
    gate = result["gate"]
    oracle = result["oracle_diagnostic"]
    return f"""## opentry_14 实验 2t：source retrieval slot-policy sweep

结果文件：`model/New_model/opentry_14/results/{result['result_filename']}`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2t_source_retrieval_slot_policy_sweep/`

- 为什么做：Exp2s source retrieval 产生了 `38` 个 Exp2o 外新 match，但默认 structural-rank supplement 让 conversion@50 下降。需要判断问题是否只是 slot policy，还是 source retrieval 候选本身无法 inference-safe 转化。
- 核心假设：如果 retrieval 的新增 match 只是被过多低质量 skeleton-hit 稀释，那么固定少量 retrieval slots、保留 Exp2o pair-chem slots，应同时维持 Exp2o conversion 并获得 match@50 小幅提升。
- 数据规模：复用 Exp2o evaluated candidates `{result['data_scale']['exp2o_pairchem_candidates']}`、Exp2s retrieval evaluated candidates `{result['data_scale']['source_retrieval_candidates']}`、safe-pool records `{result['data_scale']['safe_pool_records']}`、site-assignment records `{result['data_scale']['site_assignment_records']}`；sweep variants `{len(result['variants'])}`。
- baseline：Exp2o best `{result['baseline_exp2o']['best_variant']}` rows>=7 match@50 `{pct(base.get('match@50'))}`、conversion `{pct(base.get('skeleton_to_match_conversion@50'))}`、collision `{pct(base.get('collision_rate'))}`。
- 方法变化：不重新生成 CIF，不使用 match/RMSD/StructureMatcher label 参与排序；只改变固定 slot 配额：hydrated/site/Exp2o-pairchem/source-retrieval/prototype。各池内部保留原 inference-safe rank 或 structural rank。
- 结果 best variant `{result['best_variant']}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；valid `{pct(best.get('valid_rate'))}`；formula `{pct(best.get('formula_consistency'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`；collision `{pct(best.get('collision_rate'))}`。
- oracle 诊断：source-retrieval-only match/skelmatch `{oracle['source_retrieval_only_match_samples']}` / `{oracle['source_retrieval_only_skelmatch_samples']}`；相对 Exp2o 新增 match/skelmatch `{oracle['new_match_samples_vs_exp2o']}` / `{oracle['new_skelmatch_samples_vs_exp2o']}`。
- 可信度：中等偏高。所有候选已真实评估，sweep 只改固定配额；限制是 slot policy 仍是 validation-side heuristic，不能作为主方法成功。
- 和历史实验关系：这是 Exp2s 的排序/slot 归因，不是新 generator；直接判断 retrieval oracle 能否被 inference-safe fixed policy 吃到。
- gate 判定：passed=`{gate['passed']}`；best match delta vs Exp2o `{pp(gate['rows_ge7_match50_delta_vs_exp2o'])}`；best conversion delta vs Exp2o `{pp(gate['rows_ge7_conversion50_delta_vs_exp2o'])}`；best collision delta vs Exp2o `{pp(gate['rows_ge7_collision_delta_vs_exp2o'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    exp2o = read_json(EXP2O_RESULT)
    exp2s = read_json(EXP2S_RESULT)
    exp2o_rows7 = exp2o["variants"][exp2o["best_variant"]]["rows_ge7"]
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    site_rows = list(read_jsonl_iter(EXP2D_SITE_EVAL))
    exp2o_rows = assign_structural_ranks(list(read_jsonl_iter(Path(exp2o["artifacts"]["evaluated_candidates"]))), 50)
    retrieval_rows = assign_structural_ranks(list(read_jsonl_iter(Path(exp2s["artifacts"]["evaluated_candidates"]))), 50)
    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by = by_sid_ranked(assign_structural_ranks(site_rows, 50))
    exp2o_by = by_sid_ranked(exp2o_rows)
    retrieval_by = by_sid_ranked(retrieval_rows)
    specs = [
        ("base_like_o30", 10, 5, 30, 0, 5),
        ("src2_o28", 10, 5, 28, 2, 5),
        ("src5_o25", 10, 5, 25, 5, 5),
        ("src8_o22", 10, 5, 22, 8, 5),
        ("src10_o20", 10, 5, 20, 10, 5),
        ("s10_o25", 10, 10, 25, 0, 5),
        ("s10_src2_o23", 10, 10, 23, 2, 5),
        ("s10_src5_o20", 10, 10, 20, 5, 5),
        ("h8_src5_o30_p2", 8, 5, 30, 5, 2),
    ]
    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for name, hq, sq, oq, rq, pq in specs:
        rows = build_rows(
            name=name,
            hq=hq,
            sq=sq,
            oq=oq,
            rq=rq,
            pq=pq,
            hydrated=hydrated,
            site_by=site_by,
            exp2o_by=exp2o_by,
            retrieval_by=retrieval_by,
            prototype=prototype,
            rows_lt7=rows_lt7,
            top_k=50,
        )
        path = ARTIFACT_DIR / f"selected_{name}_candidates.jsonl"
        write_jsonl(path, rows)
        selected_paths[name] = str(path)
        rows7 = [r for r in rows if int(r.get("row_count") or 0) >= 7]
        rowslt7 = [r for r in rows if int(r.get("row_count") or 0) < 7]
        variants[name] = {"overall": summarize(rows), "rows_ge7": summarize(rows7), "rows_lt7": summarize(rowslt7)}

    best_variant = max(
        variants,
        key=lambda name: (
            float(variants[name]["rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0),
            float(variants[name]["rows_ge7"].get("match@50") or 0.0),
            -float(variants[name]["rows_ge7"].get("collision_rate") or 1.0),
        ),
    )
    best_rows7 = variants[best_variant]["rows_ge7"]
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(exp2o_rows7.get("match@50") or 0.0)
    conv_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2o_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    coll_delta = float(best_rows7.get("collision_rate") or 0.0) - float(exp2o_rows7.get("collision_rate") or 0.0)
    passed = bool(match_delta >= 0.0 and conv_delta >= 0.0 and float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28)
    exp2o_sets = sample_sets([r for r in exp2o_rows if int(r.get("row_count") or 0) >= 7])
    retrieval_sets = sample_sets([r for r in retrieval_rows if int(r.get("row_count") or 0) >= 7])
    if passed:
        verdict = "pass_slot_policy_non_regression"
        reason = "Fixed source-retrieval slots preserve Exp2o conversion while adding match@50."
        next_step = "Run Exp3 local optimizer only if this becomes the new Exp2 best."
    else:
        verdict = "fail_slot_policy_tradeoff"
        reason = "No fixed slot policy improves match@50 and preserves Exp2o conversion simultaneously."
        next_step = "Stop heuristic retrieval/slot tuning; move to learned source/assignment-aware generation or a model-based critic."
    result = {
        "experiment": "opentry_14_exp2t_source_retrieval_slot_policy_sweep",
        "result_filename": "experiment_2t_source_retrieval_slot_policy_sweep.json",
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "fixed_slot_policy_sweep_over_exp2o_and_source_retrieval",
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB"],
        },
        "data_scale": {
            "safe_pool_records": len(safe_rows),
            "site_assignment_records": len(site_rows),
            "exp2o_pairchem_candidates": len(exp2o_rows),
            "source_retrieval_candidates": len(retrieval_rows),
            "selected_variant_paths": selected_paths,
        },
        "baseline_exp2o": {"result_path": str(EXP2O_RESULT), "best_variant": exp2o["best_variant"], "rows_ge7": exp2o_rows7},
        "source_exp2s": {"result_path": str(EXP2S_RESULT), "decision": exp2s["decision"], "gate": exp2s["gate"]},
        "oracle_diagnostic": {
            "source_retrieval_only_match_samples": len(retrieval_sets["match"]),
            "source_retrieval_only_skelmatch_samples": len(retrieval_sets["skelmatch"]),
            "new_match_samples_vs_exp2o": len(retrieval_sets["match"] - exp2o_sets["match"]),
            "new_skelmatch_samples_vs_exp2o": len(retrieval_sets["skelmatch"] - exp2o_sets["skelmatch"]),
        },
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "passed": passed,
            "rows_ge7_match50_delta_vs_exp2o": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2o": conv_delta,
            "rows_ge7_collision_delta_vs_exp2o": coll_delta,
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / result["result_filename"], result)
    marker = "<!-- OPENTRY14_EXP2T_SOURCE_RETRIEVAL_SLOT_POLICY_SWEEP -->"
    body = report_body(result)
    append_or_replace(REPORT_PATH, marker, body)
    append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / result["result_filename"]), "best_variant": best_variant, "gate": result["gate"], "decision": result["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
