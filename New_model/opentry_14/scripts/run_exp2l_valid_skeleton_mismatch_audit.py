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
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2l_valid_skeleton_mismatch_audit"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Structure  # noqa: E402

from run_exp2_predicted_skeleton_renderer_site_mapping import read_jsonl, sample_id  # noqa: E402
from run_exp3j_chgnet_after_exp2j import EXP2J_RESULT, STRUCTURED_VAL, candidate_uid, regenerate_selected_chemical_tasks  # noqa: E402


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


def rel_lattice_errors(target: Structure, generated: Structure) -> dict[str, float]:
    tl = target.lattice
    gl = generated.lattice
    return {
        "a_rel": abs(float(gl.a) - float(tl.a)) / max(1.0e-9, float(tl.a)),
        "b_rel": abs(float(gl.b) - float(tl.b)) / max(1.0e-9, float(tl.b)),
        "c_rel": abs(float(gl.c) - float(tl.c)) / max(1.0e-9, float(tl.c)),
        "alpha_abs": abs(float(gl.alpha) - float(tl.alpha)),
        "beta_abs": abs(float(gl.beta) - float(tl.beta)),
        "gamma_abs": abs(float(gl.gamma) - float(tl.gamma)),
        "volume_rel": abs(float(generated.volume) - float(target.volume)) / max(1.0e-9, float(target.volume)),
    }


def audit_one(task: dict[str, Any]) -> dict[str, Any]:
    row = task["row"]
    out = {
        "candidate_uid": str(row["candidate_uid"]),
        "sample_id": str(row["sample_id"]),
        "rank": int(row.get("rank") or 0),
        "proposal_rank": int(row.get("proposal_rank") or 0),
        "assignment_rank": int(row.get("assignment_rank") or 0),
        "row_count": int(row.get("row_count") or 0),
        "input_valid": bool(row.get("valid")),
        "input_predicted_skeleton_hit": bool(row.get("predicted_skeleton_hit")),
        "input_match": bool(row.get("match")),
        "ok": False,
    }
    try:
        target = Structure.from_file(str(task["target_cif_path"]))
        generated = Structure.from_str(str(task["cif"]), fmt="cif")
        grids = {
            "default": {"ltol": 0.3, "stol": 0.5, "angle_tol": 10},
            "loose_lattice": {"ltol": 0.6, "stol": 0.5, "angle_tol": 10},
            "loose_site": {"ltol": 0.3, "stol": 1.0, "angle_tol": 10},
            "loose_angle": {"ltol": 0.3, "stol": 0.5, "angle_tol": 20},
            "loose_all": {"ltol": 0.6, "stol": 1.0, "angle_tol": 20},
        }
        fits = {}
        anon = {}
        rms = {}
        for name, cfg in grids.items():
            matcher = StructureMatcher(**cfg)
            fits[name] = bool(matcher.fit(target, generated))
            try:
                anon[name] = bool(matcher.fit_anonymous(target, generated))
            except Exception:
                anon[name] = False
            if fits[name]:
                try:
                    raw = matcher.get_rms_dist(target, generated)
                    rms[name] = float(raw[0] if isinstance(raw, (list, tuple)) else raw)
                except Exception:
                    rms[name] = None
            else:
                rms[name] = None
        lattice = rel_lattice_errors(target, generated)
        if fits["default"]:
            category = "would_match_default_unexpected"
        elif anon["default"] or anon["loose_all"]:
            category = "species_or_site_assignment_mismatch"
        elif fits["loose_lattice"] and not fits["default"]:
            category = "lattice_tolerance_limited"
        elif fits["loose_site"] and not fits["default"]:
            category = "fractional_coordinate_tolerance_limited"
        elif fits["loose_all"]:
            category = "combined_lattice_coordinate_tolerance_limited"
        elif lattice["volume_rel"] > 0.35 or max(lattice["a_rel"], lattice["b_rel"], lattice["c_rel"]) > 0.25:
            category = "large_lattice_scale_mismatch"
        else:
            category = "different_geometry_basin"
        out.update({"ok": True, "fit": fits, "anonymous_fit": anon, "rms": rms, "lattice_error": lattice, "category": category})
    except Exception as exc:  # noqa: BLE001
        out.update({"error": f"{type(exc).__name__}: {exc}", "category": "audit_failed"})
    return out


def mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def quantile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return float(vals[idx])


def report_body(result: dict[str, Any]) -> str:
    cat = result["category_counts"]
    rates = result["rates"]
    return f"""## opentry_14 实验 2l：valid skeleton-hit mismatch audit

结果文件：`model/New_model/opentry_14/results/experiment_2l_valid_skeleton_mismatch_audit.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2l_valid_skeleton_mismatch_audit/`

- 为什么做：Exp2j/3j/3k/2k 都显示 valid/formula/SG/exact 可以保持，但 skeleton-hit 仍不能转成 StructureMatcher match。需要把失败分解为 lattice tolerance、fractional coordinate basin、species/site assignment 或完全不同 geometry basin。
- 核心假设：如果大量 no-match 在 loose lattice/site/anonymous 条件下能匹配，则下一步应针对对应轴训练 repair；如果仍不能匹配，则当前 predicted skeleton/site assignment 虽命中 skeleton key，但几何 basin 已经不同。
- 数据规模：从 Exp2j best 中选择 rows>=7、chemical geometry、valid=true、predicted_skeleton_hit=true、match=false 的候选 `{result['data_scale']['selected_rows']}`；成功重建 CIF `{result['data_scale']['regenerated_tasks']}`；审计 records `{result['data_scale']['audited_records']}`；workers `{result['cpu_policy']['workers']}`。
- baseline/关系：审计对象来自 Exp2j best `h10_s10_chem25_p5` 之后的失败样本，不作为新推理候选，不进入 Exp3/official。
- 方法变化：只做离线归因，重建 CIF 后用 StructureMatcher default/loose_lattice/loose_site/loose_angle/loose_all 和 anonymous matching 检查失败类型；真值 CIF 只用于审计，不进入推理排序或候选选择。
- 结果分类：`{dict(cat)}`。
- 关键率：default_match `{pct(rates['default_fit_rate'])}`；loose_all_match `{pct(rates['loose_all_fit_rate'])}`；anonymous_loose_all `{pct(rates['anonymous_loose_all_rate'])}`；large_lattice_scale_mismatch `{pct(rates['large_lattice_scale_mismatch_rate'])}`。
- lattice 误差：volume_rel median `{result['lattice_error_summary']['volume_rel']['median']}`，p90 `{result['lattice_error_summary']['volume_rel']['p90']}`；max_axis_rel median `{result['lattice_error_summary']['max_axis_rel']['median']}`，p90 `{result['lattice_error_summary']['max_axis_rel']['p90']}`。
- 可信度：中等。使用真实 CIF 和 StructureMatcher 做离线审计；限制是默认只审计每样本前若干 valid skeleton-hit no-match 候选，不代表所有候选。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--per-sample", type=int, default=3)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--top-skeletons", type=int, default=8)
    parser.add_argument("--assignment-limit", type=int, default=8)
    parser.add_argument("--assignment-beam-width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--fmax", type=float, default=0.0)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    exp2j = read_json(EXP2J_RESULT)
    best = str(exp2j["best_variant"])
    base_path = Path(exp2j["artifacts"]["selected_variants"][best])
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl_iter(base_path):
        if (
            int(row.get("row_count") or 0) >= 7
            and str(row.get("geometry_source")) == "chemical_site_order_source_geometry"
            and bool(row.get("valid"))
            and bool(row.get("predicted_skeleton_hit"))
            and not bool(row.get("match"))
        ):
            item = dict(row)
            item["candidate_uid"] = candidate_uid(item)
            by_sid[str(item["sample_id"])].append(item)
    selected: list[dict[str, Any]] = []
    for sid in sorted(by_sid)[: int(args.max_samples)]:
        group = sorted(by_sid[sid], key=lambda r: int(r.get("rank") or 999999))[: int(args.per_sample)]
        selected.extend(group)
    write_jsonl(ARTIFACT_DIR / "selected_valid_skeleton_nomatch_rows.jsonl", selected)

    tasks, regen_meta = regenerate_selected_chemical_tasks(selected, args)
    structured_val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    audit_tasks = []
    for task in tasks:
        sid = str(task["row"]["sample_id"])
        target = structured_val.get(sid)
        if target is None:
            continue
        audit_tasks.append({"row": task["row"], "cif": task["cif"], "target_cif_path": str(target["source_path"])})

    audited: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(audit_one, task) for task in audit_tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            audited.append(fut.result())
            if i % 200 == 0:
                print(f"[exp2l-audit] audited {i}/{len(futures)}", flush=True)
    audited.sort(key=lambda r: (str(r.get("sample_id")), int(r.get("rank") or 0), str(r.get("candidate_uid"))))
    write_jsonl(ARTIFACT_DIR / "valid_skeleton_nomatch_audit_records.jsonl", audited)

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
    if rates["anonymous_loose_all_rate"] and rates["anonymous_loose_all_rate"] > 0.15:
        verdict = "species_site_assignment_mismatch_substantial"
        reason = "Many failures match anonymously under loose tolerance, so element/site assignment remains a major bottleneck."
        next_step = "Train an assignment-aware geometry model or upgrade chemical assignment; do not spend more on local optimizer."
    elif rates["loose_all_fit_rate"] and rates["loose_all_fit_rate"] > 0.15:
        verdict = "near_miss_geometry_tolerance_substantial"
        reason = "Many failures are near misses under loose StructureMatcher tolerance, so continuous lattice/coordinate residual learning is plausible."
        next_step = "Build a residual posterior targeted at lattice+coordinate tolerance rather than prototype donors."
    elif rates["large_lattice_scale_mismatch_rate"] and rates["large_lattice_scale_mismatch_rate"] > 0.30:
        verdict = "lattice_scale_mismatch_dominant"
        reason = "A large fraction of failures have major lattice scale mismatch."
        next_step = "Train a joint lattice/free-parameter residual model; avoid lattice-only repair."
    else:
        verdict = "different_geometry_basin_dominant"
        reason = "Most valid skeleton-hit no-match candidates remain unmatched even under loose/anonymous checks, indicating a different geometry basin rather than a small local correction."
        next_step = "Upgrade skeleton/site alignment or train a stronger joint generative geometry posterior; do not expand heuristic beams."

    result = {
        "experiment": "opentry_14_exp2l_valid_skeleton_mismatch_audit",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "offline_structurematcher_tolerance_and_anonymous_mismatch_audit",
            "source_result": str(EXP2J_RESULT),
            "source_variant": best,
            "selection_rule": "rows>=7 Exp2j best chemical candidates with valid=true, predicted_skeleton_hit=true, match=false; diagnostic only",
            "not_inference_features": ["target CIF", "GT skeleton hit", "match", "RMSD", "StructureMatcher label"],
            "max_samples": int(args.max_samples),
            "per_sample": int(args.per_sample),
        },
        "cpu_policy": {
            "workers": int(args.workers),
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
            "selected_rows": str(ARTIFACT_DIR / "selected_valid_skeleton_nomatch_rows.jsonl"),
            "audit_records": str(ARTIFACT_DIR / "valid_skeleton_nomatch_audit_records.jsonl"),
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_2l_valid_skeleton_mismatch_audit.json", result)
    if not args.skip_report:
        body = report_body(result)
        marker = "<!-- OPENTRY14_EXP2L_VALID_SKELETON_MISMATCH_AUDIT -->"
        append_or_replace(REPORT_PATH, marker, body)
        append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / "experiment_2l_valid_skeleton_mismatch_audit.json"), "data_scale": result["data_scale"], "category_counts": result["category_counts"], "rates": result["rates"], "decision": result["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
