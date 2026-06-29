#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2p_exp2o_valid_skeleton_mismatch_audit"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (
    NEW_MODEL / "symcif_experiment" / "src",
    NEW_MODEL / "symcif_experiment" / "scripts",
    NEW_MODEL / "opentry_13" / "scripts",
    OUT_DIR / "scripts",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import read_jsonl, sample_id  # noqa: E402
from run_exp2j_chemical_site_order_assignment import append_or_replace, pct, write_json, write_jsonl  # noqa: E402
from run_exp2l_valid_skeleton_mismatch_audit import audit_one, mean, quantile  # noqa: E402
from run_exp3j_chgnet_after_exp2j import STRUCTURED_VAL, candidate_uid  # noqa: E402
from run_exp3n_chgnet_after_exp2n_pairchem import regenerate_selected_pairchem_tasks  # noqa: E402


EXP2O_RESULT = RESULT_DIR / "experiment_2o_expanded_pairwise_local_chemistry_assignment.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def report_body(result: dict[str, Any]) -> str:
    rates = result["rates"]
    cat = result["category_counts"]
    return f"""## opentry_14 实验 2p：Exp2o valid skeleton-hit mismatch audit

结果文件：`model/New_model/opentry_14/results/experiment_2p_exp2o_valid_skeleton_mismatch_audit.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2p_exp2o_valid_skeleton_mismatch_audit/`

- 为什么做：Exp2o 已把 rows>=7 conversion 推到 `28.633%`，但 Exp3o 证明 CHGNet local optimizer 不产生新 sample-level match。需要重新审计 Exp2o best 中 valid skeleton-hit no-match 的失败类型，决定下一步是 assignment、lattice/free-parameter residual，还是 skeleton/source retrieval。
- 核心假设：如果 Exp2o 剩余失败大量 anonymous loose match，则 species/site assignment 仍主导；如果 loose_all/default-near miss 多，则应训练连续 lattice+free-parameter residual；如果多数仍不同 basin，则需要更强 skeleton/source retrieval 或 joint generative posterior。
- 数据规模：source candidate rows `{result['data_scale']['candidate_source_rows']}`；source samples `{result['data_scale']['candidate_source_samples']}`；selected rows `{result['data_scale']['selected_rows']}`；regenerated CIF tasks `{result['data_scale']['regenerated_tasks']}`；audited records `{result['data_scale']['audited_records']}`；workers regen/audit `{result['cpu_policy']['regen_workers']}` / `{result['cpu_policy']['workers']}`。
- baseline/关系：审计对象来自 Exp2o best `{result['source']['best_variant']}`，仅用于离线 root-cause，不作为推理候选，不进入 Exp4/5/official。
- 方法变化：按 Exp2o pair-chem assignment 精确重建 CIF，用 StructureMatcher default/loose_lattice/loose_site/loose_angle/loose_all 与 anonymous matching 分类；target true CIF 只用于审计，不用于候选选择或推理特征。
- 结果分类：`{dict(cat)}`。
- 关键率：default_match `{pct(rates['default_fit_rate'])}`；loose_all_match `{pct(rates['loose_all_fit_rate'])}`；anonymous_loose_all `{pct(rates['anonymous_loose_all_rate'])}`；large_lattice_scale_mismatch `{pct(rates['large_lattice_scale_mismatch_rate'])}`。
- lattice 误差：volume_rel median `{result['lattice_error_summary']['volume_rel']['median']}`，p90 `{result['lattice_error_summary']['volume_rel']['p90']}`；max_axis_rel median `{result['lattice_error_summary']['max_axis_rel']['median']}`，p90 `{result['lattice_error_summary']['max_axis_rel']['p90']}`。
- 可信度：中等。审计用真实 CIF 和 StructureMatcher，且样本来自 Exp2o full validation top-ranked valid skeleton-hit no-match；限制是按每样本 top-N 抽样，不覆盖全部 no-match。
- 和历史实验关系：复核 Exp2l 在 Exp2o 后是否仍成立。若 species/site mismatch 仍高，则 Exp2n/2o 的 pair-chem 还没完全解决 assignment；若 large lattice/loose_all 高，则下一轮应学习 alignment-aware lattice/free-param residual。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--per-sample", type=int, default=3)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--regen-workers", type=int, default=128)
    parser.add_argument("--top-skeletons", type=int, default=10)
    parser.add_argument("--assignment-prelimit", type=int, default=64)
    parser.add_argument("--assignment-beam-width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--fmax", type=float, default=0.0)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    exp2o = read_json(EXP2O_RESULT)
    best = str(exp2o["best_variant"])
    base_path = Path(exp2o["artifacts"]["selected_variants"][best])
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl_iter(base_path):
        if (
            int(row.get("row_count") or 0) >= 7
            and str(row.get("geometry_source")) == "pairwise_local_chemistry_source_geometry"
            and bool(row.get("valid"))
            and bool(row.get("predicted_skeleton_hit"))
            and not bool(row.get("match"))
        ):
            item = dict(row)
            item["candidate_uid"] = candidate_uid(item)
            by_sid[str(item["sample_id"])].append(item)

    selected: list[dict[str, Any]] = []
    for sid in sorted(by_sid)[: int(args.max_samples)]:
        selected.extend(sorted(by_sid[sid], key=lambda r: int(r.get("rank") or 999999))[: int(args.per_sample)])
    write_jsonl(ARTIFACT_DIR / "selected_exp2o_valid_skeleton_nomatch_rows.jsonl", selected)

    tasks, regen_meta = regenerate_selected_pairchem_tasks(selected, args)
    structured_val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    audit_tasks: list[dict[str, Any]] = []
    for task in tasks:
        sid = str(task["row"]["sample_id"])
        target = structured_val.get(sid)
        if target is not None:
            audit_tasks.append({"row": task["row"], "cif": task["cif"], "target_cif_path": str(target["source_path"])})

    audited: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(audit_one, task) for task in audit_tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            audited.append(fut.result())
            if i % 200 == 0:
                print(f"[exp2p-audit] audited {i}/{len(futures)}", flush=True)
    audited.sort(key=lambda r: (str(r.get("sample_id")), int(r.get("rank") or 0), str(r.get("candidate_uid"))))
    write_jsonl(ARTIFACT_DIR / "exp2o_valid_skeleton_nomatch_audit_records.jsonl", audited)

    ok = [r for r in audited if r.get("ok")]
    cat = Counter(str(r.get("category")) for r in audited)

    def rate(pred) -> float | None:
        return sum(1 for r in ok if pred(r)) / len(ok) if ok else None

    vol = [float(r["lattice_error"]["volume_rel"]) for r in ok if r.get("lattice_error")]
    max_axis = [
        max(float(r["lattice_error"]["a_rel"]), float(r["lattice_error"]["b_rel"]), float(r["lattice_error"]["c_rel"]))
        for r in ok
        if r.get("lattice_error")
    ]
    summary = {
        "volume_rel": {"mean": mean(vol), "median": quantile(vol, 0.5), "p90": quantile(vol, 0.9), "p95": quantile(vol, 0.95)},
        "max_axis_rel": {"mean": mean(max_axis), "median": quantile(max_axis, 0.5), "p90": quantile(max_axis, 0.9), "p95": quantile(max_axis, 0.95)},
    }
    rates = {
        "default_fit_rate": rate(lambda r: bool((r.get("fit") or {}).get("default"))),
        "loose_all_fit_rate": rate(lambda r: bool((r.get("fit") or {}).get("loose_all"))),
        "anonymous_loose_all_rate": rate(lambda r: bool((r.get("anonymous_fit") or {}).get("loose_all"))),
        "large_lattice_scale_mismatch_rate": rate(lambda r: str(r.get("category")) == "large_lattice_scale_mismatch"),
    }
    if rates["anonymous_loose_all_rate"] and rates["anonymous_loose_all_rate"] > 0.20:
        verdict = "species_site_assignment_mismatch_remains_dominant"
        reason = "Anonymous loose matching remains high after Exp2o, so assignment/site identity is still the main residual bottleneck."
        next_step = "Train assignment-aware geometry model or improve pair-chem assignment generation; do not expand local optimizer."
    elif rates["loose_all_fit_rate"] and rates["loose_all_fit_rate"] > 0.20:
        verdict = "continuous_geometry_near_miss_substantial"
        reason = "Loose tolerance recovers many failures, so learned lattice+free-parameter residual alignment is plausible."
        next_step = "Train a residual posterior targeted at lattice plus row free parameters, with Exp2o assignment fixed."
    elif rates["large_lattice_scale_mismatch_rate"] and rates["large_lattice_scale_mismatch_rate"] > 0.30:
        verdict = "lattice_scale_mismatch_substantial"
        reason = "Large lattice scale mismatch remains common, but prompt forbids lattice-only repair."
        next_step = "Train joint lattice/free-parameter residual alignment; avoid lattice-only variants."
    else:
        verdict = "different_geometry_basin_dominant"
        reason = "Most audited failures do not match even under loose/anonymous checks."
        next_step = "Upgrade skeleton/source retrieval or train a stronger joint generative geometry posterior."

    result = {
        "experiment": "opentry_14_exp2p_exp2o_valid_skeleton_mismatch_audit",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "source": {"result_path": str(EXP2O_RESULT), "best_variant": best},
        "method": {
            "name": "offline_structurematcher_tolerance_and_anonymous_mismatch_audit_after_exp2o",
            "selection_rule": "rows>=7 Exp2o best pair-chem candidates with valid=true, predicted_skeleton_hit=true, match=false; diagnostic only",
            "not_inference_features": ["target CIF", "GT skeleton hit", "match", "RMSD", "StructureMatcher label"],
            "max_samples": int(args.max_samples),
            "per_sample": int(args.per_sample),
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "regen_workers": int(args.regen_workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
        },
        "data_scale": {
            "candidate_source_rows": sum(len(v) for v in by_sid.values()),
            "candidate_source_samples": len(by_sid),
            "selected_rows": len(selected),
            "regenerated_tasks": int(regen_meta.get("regenerated_tasks") or 0),
            "audited_records": len(audited),
        },
        "regeneration_meta": regen_meta,
        "category_counts": dict(cat),
        "rates": rates,
        "lattice_error_summary": summary,
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "selected_rows": str(ARTIFACT_DIR / "selected_exp2o_valid_skeleton_nomatch_rows.jsonl"),
            "audit_records": str(ARTIFACT_DIR / "exp2o_valid_skeleton_nomatch_audit_records.jsonl"),
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_2p_exp2o_valid_skeleton_mismatch_audit.json", result)
    if not args.skip_report:
        marker = "<!-- OPENTRY14_EXP2P_EXP2O_VALID_SKELETON_MISMATCH_AUDIT -->"
        body = report_body(result)
        append_or_replace(REPORT_PATH, marker, body)
        append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / "experiment_2p_exp2o_valid_skeleton_mismatch_audit.json"), "data_scale": result["data_scale"], "category_counts": result["category_counts"], "rates": result["rates"], "decision": result["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
