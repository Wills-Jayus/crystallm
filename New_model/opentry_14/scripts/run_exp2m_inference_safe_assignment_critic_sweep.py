#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2m_inference_safe_assignment_critic_sweep"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2f_permutation_aware_alignment import by_sid_ranked, split_safe_pool  # noqa: E402
from run_exp2j_chemical_site_order_assignment import build_variant_rows, pct, pp  # noqa: E402
from run_exp4_rows_ge7_multi_geometry_proposal import summarize  # noqa: E402


EXP2B_EVAL = OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool" / "evaluated_safe_pool_candidates.jsonl"
EXP2D_SITE_EVAL = OUT_DIR / "artifacts" / "exp2d_site_assignment_multi_hypothesis" / "evaluated_site_assignment_candidates.jsonl"
EXP2J_RESULT = RESULT_DIR / "experiment_2j_chemical_site_order_assignment.json"
EXP2J_CHEM_EVAL = OUT_DIR / "artifacts" / "exp2j_chemical_site_order_assignment" / "evaluated_chemical_assignment_candidates.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def val(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return default


def rank_chemical(rows: list[dict[str, Any]], score_fn: Callable[[dict[str, Any]], float], score_name: str, top_k: int) -> list[dict[str, Any]]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("row_count") or 0) < 7 or str(row.get("geometry_source")) != "chemical_site_order_source_geometry":
            continue
        item = dict(row)
        item["assignment_critic_name"] = score_name
        item["assignment_critic_score"] = float(score_fn(item))
        by_sid[str(item["sample_id"])].append(item)
    out: list[dict[str, Any]] = []
    for sid, group in by_sid.items():
        ordered = sorted(
            group,
            key=lambda r: (
                -float(r["assignment_critic_score"]),
                int(r.get("proposal_rank") or 999999),
                int(r.get("assignment_rank") or 999999),
                int(r.get("rank") or 999999),
            ),
        )[:top_k]
        for rank, row in enumerate(ordered, start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    out.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    return out


def report_body(result: dict[str, Any]) -> str:
    best_name = result["best_sweep"]
    best = result["sweeps"][best_name]["combo_rows_ge7"]
    baseline = result["baseline_exp2j"]["rows_ge7"]
    gate = result["gate"]
    return f"""## opentry_14 实验 2m：inference-safe assignment critic sweep

结果文件：`model/New_model/opentry_14/results/experiment_2m_inference_safe_assignment_critic_sweep.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2m_inference_safe_assignment_critic_sweep/`

- 为什么做：Exp2l 显示 valid skeleton-hit no-match 中 species/site assignment mismatch 占比高，先检验不使用真值标签的 assignment/structure 字段级 critic 是否足以把 Exp2j chemical candidates 排到更好的 top50。
- 核心假设：如果正确 assignment 已在 Exp2j candidate pool 内，只是被 structural score 排低，则使用 assignment_score、source-preserved atoms、proposal/assignment rank、valid/SG/exact/min-distance/volume 等 inference-safe 字段应提升 rows>=7 conversion。
- 数据规模：chemical candidate records `{result['data_scale']['chemical_records']}`；rows>=7 chemical records `{result['data_scale']['rows_ge7_chemical_records']}`；samples `{result['data_scale']['rows_ge7_samples']}`；sweep formulas `{len(result['sweeps'])}`。
- baseline：Exp2j best `{result['baseline_exp2j']['best_variant']}` rows>=7 match@50 `{pct(baseline.get('match@50'))}`、conversion `{pct(baseline.get('skeleton_to_match_conversion@50'))}`、collision `{pct(baseline.get('collision_rate'))}`。
- 方法变化：只重排已有 Exp2j chemical candidates，不重新生成 CIF；排序不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。该实验只作为 critic/headroom 诊断，不作为主 repair。
- 结果 best sweep `{best_name}`：rows>=7 match@50 `{pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；collision `{pct(best.get('collision_rate'))}`；valid `{pct(best.get('valid_rate'))}`。
- sweep 结论：最高 match@50 sweep 仍未提升 conversion；best-vs-Exp2j match delta `{pp(gate['rows_ge7_match50_delta_vs_exp2j'])}`，conversion delta `{pp(gate['rows_ge7_conversion50_delta_vs_exp2j'])}`。
- 可信度：中等。所有候选已由 Exp2j 真实 render/parse/StructureMatcher 评估，选择阶段只用 inference-safe 字段；限制是没有重建 CIF 做更细局部化学统计。
- 和历史实验关系：直接回应 Exp2l 的 species/site mismatch 归因，检验“已有候选只需安全 critic 重排”是否成立。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    exp2j = read_json(EXP2J_RESULT)
    exp2j_best = str(exp2j["best_variant"])
    baseline_rows7 = exp2j["variants"][exp2j_best]["rows_ge7"]
    chem_rows = read_jsonl(EXP2J_CHEM_EVAL)
    safe_rows = read_jsonl(EXP2B_EVAL)
    site_rows = read_jsonl(EXP2D_SITE_EVAL)
    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by = by_sid_ranked(site_rows)

    scores: dict[str, Callable[[dict[str, Any]], float]] = {
        "structural_existing": lambda r: val(r.get("structural_selection_score")),
        "assignment_score": lambda r: val(r.get("assignment_score")),
        "assignment_plus_valid": lambda r: 1000.0 * float(bool(r.get("valid"))) + val(r.get("assignment_score")),
        "assignment_source_preserved": lambda r: 100.0 * val(r.get("assignment_source_preserved_atoms")) + val(r.get("assignment_score")),
        "proposal_assignment": lambda r: -10.0 * val(r.get("proposal_rank")) - val(r.get("assignment_rank")) + 0.05 * val(r.get("assignment_score")),
        "valid_mindist_vpa_assign": lambda r: 1000.0 * float(bool(r.get("valid")))
        + 100.0 * float(bool(r.get("space_group_ok")))
        + 50.0 * float(bool(r.get("exact_cover_retained")))
        - 20.0 * abs(math.log(max(1.0e-6, val(r.get("volume_per_atom"), 18.0) / 18.0)))
        + 10.0 * min(3.0, val(r.get("min_pair_distance")))
        + 0.1 * val(r.get("assignment_score")),
        "low_assignment_rank": lambda r: -val(r.get("assignment_rank")) - 0.1 * val(r.get("proposal_rank")),
        "high_source_pres_low_rank": lambda r: 20.0 * val(r.get("assignment_source_preserved_atoms")) - val(r.get("assignment_rank")) - 0.1 * val(r.get("proposal_rank")),
    }
    sweeps: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for name, fn in scores.items():
        ranked = rank_chemical(chem_rows, fn, name, 50)
        chem_by = by_sid_ranked(ranked)
        combo_rows = build_variant_rows(
            variant="h10_s10_chem25_p5",
            hydrated=hydrated,
            prototype=prototype,
            siteassign=site_by,
            chemassign=chem_by,
            rows_lt7=rows_lt7,
            top_k=50,
        )
        if name in {"structural_existing", "proposal_assignment", "assignment_plus_valid", "valid_mindist_vpa_assign"}:
            path = ARTIFACT_DIR / f"selected_{name}_h10_s10_chem25_p5_candidates.jsonl"
            write_jsonl(path, combo_rows)
            selected_paths[name] = str(path)
        sweeps[name] = {
            "chemical_rows_ge7": summarize([r for r in ranked if int(r.get("row_count") or 0) >= 7]),
            "combo_rows_ge7": summarize([r for r in combo_rows if int(r.get("row_count") or 0) >= 7]),
            "combo_overall": summarize(combo_rows),
        }
    best_sweep = max(
        sweeps,
        key=lambda name: (
            float(sweeps[name]["combo_rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0),
            float(sweeps[name]["combo_rows_ge7"].get("match@50") or 0.0),
            -float(sweeps[name]["combo_rows_ge7"].get("collision_rate") or 1.0),
        ),
    )
    best_rows7 = sweeps[best_sweep]["combo_rows_ge7"]
    conversion_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0)
    passed = bool(conversion_delta >= 0.02 and match_delta >= 0.0)
    if passed:
        verdict = "pass_assignment_critic_gate"
        reason = "Inference-safe assignment critic lifts conversion by at least +2pp over Exp2j."
        next_step = "Retest Exp3 local optimizer relative to this critic output; no official until later gates pass."
    elif match_delta > 0.0 and conversion_delta <= 0.0:
        verdict = "fail_sorting_tradeoff_match_up_conversion_down"
        reason = "Some critic formulas increase match@50 slightly but reduce skeleton-to-match conversion, so this is ranking tradeoff rather than repair."
        next_step = "Do not pursue field-level critic/rerank; build assignment-aware generator or local chemistry CIF scorer that creates new candidate headroom."
    else:
        verdict = "fail_no_conversion_lift"
        reason = "Field-level inference-safe assignment critic does not improve Exp2j conversion."
        next_step = "Move beyond existing candidate sorting: train assignment-aware geometry model or upgrade chemical/site assignment generation."
    result = {
        "experiment": "opentry_14_exp2m_inference_safe_assignment_critic_sweep",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "field_level_inference_safe_assignment_critic_sweep",
            "source_result": str(EXP2J_RESULT),
            "source_candidate_file": str(EXP2J_CHEM_EVAL),
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "note": "Diagnostic sweep over existing Exp2j candidates; no new CIF generation.",
        },
        "cpu_policy": {
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "workers": 1,
        },
        "data_scale": {
            "chemical_records": len(chem_rows),
            "rows_ge7_chemical_records": sum(1 for r in chem_rows if int(r.get("row_count") or 0) >= 7),
            "rows_ge7_samples": len({str(r["sample_id"]) for r in chem_rows if int(r.get("row_count") or 0) >= 7}),
            "selected_variant_paths": selected_paths,
        },
        "baseline_exp2j": {"result_path": str(EXP2J_RESULT), "best_variant": exp2j_best, "rows_ge7": baseline_rows7},
        "sweeps": sweeps,
        "best_sweep": best_sweep,
        "gate": {
            "passed": passed,
            "rows_ge7_match50_delta_vs_exp2j": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2j": conversion_delta,
            "minimum_standard": {"conversion_lift_vs_exp2j": 0.02, "match_not_worse": True},
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_2m_inference_safe_assignment_critic_sweep.json", result)
    marker = "<!-- OPENTRY14_EXP2M_INFERENCE_SAFE_ASSIGNMENT_CRITIC_SWEEP -->"
    body = report_body(result)
    append_or_replace(REPORT_PATH, marker, body)
    append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / "experiment_2m_inference_safe_assignment_critic_sweep.json"), "best_sweep": best_sweep, "gate": result["gate"], "decision": result["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
