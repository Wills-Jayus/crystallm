#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp3n_chgnet_after_exp2n_pairchem"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import read_jsonl, sample_id  # noqa: E402
import run_exp3j_chgnet_after_exp2j as exp3j_mod  # noqa: E402
from run_exp3j_chgnet_after_exp2j import (  # noqa: E402
    EXP3_PROPOSALS,
    LOOKUP,
    STRUCTURED_TRAIN,
    STRUCTURED_VAL,
    TRAIN_REPR,
    VAL_REPR,
    append_or_replace,
    build_train_priors,
    candidate_uid,
    chgnet_optimize_task,
    chemical_exact_cover_assignments,
    flexible_params_from_reference,
    formula_counts,
    pct,
    pp,
    render_candidate,
    source_skeleton_rows,
    write_json,
    write_jsonl,
)
from run_exp4_rows_ge7_multi_geometry_proposal import eval_sample, summarize  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


EXP2N_RESULT = RESULT_DIR / "experiment_2n_pairwise_local_chemistry_assignment.json"
PAIRCHEM_GEOMETRY_SOURCE = "pairwise_local_chemistry_source_geometry"
G_REGEN_TRAIN_REPR: dict[str, dict[str, Any]] = {}
G_REGEN_TRAIN_STRUCT: dict[str, dict[str, Any]] = {}
G_REGEN_VAL: dict[str, dict[str, Any]] = {}
G_REGEN_VAL_REPR: dict[str, dict[str, Any]] = {}
G_REGEN_PROPOSALS: dict[str, dict[str, Any]] = {}
G_REGEN_ENGINE: OrbitEngine | None = None
G_REGEN_PRIORS: dict[str, Any] = {}
G_REGEN_WANTED: dict[str, dict[str, Any]] = {}
G_REGEN_WANTED_BY_SID: dict[str, set[str]] = {}
G_REGEN_ARGS: dict[str, Any] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def init_chgnet_worker_fallback(devices: tuple[str, ...]) -> None:
    ident = mp.current_process()._identity  # type: ignore[attr-defined]
    idx = int(ident[0] - 1) if ident else 0
    requested = str(devices[idx % max(1, len(devices))])
    os.environ.setdefault("MPLCONFIGDIR", "/data/tmp/matplotlib-cache")
    from chgnet.model.dynamics import StructOptimizer

    if requested.lower() == "cpu":
        exp3j_mod._OPTIMIZER = StructOptimizer(use_device="cpu", optimizer_class="FIRE")
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = requested.split(":", 1)[-1] if requested.startswith("cuda:") else requested
    try:
        exp3j_mod._OPTIMIZER = StructOptimizer(use_device="cuda", optimizer_class="FIRE")
    except Exception as exc:  # noqa: BLE001
        print(f"[exp3n-chgnet] cuda init failed on device {requested}: {type(exc).__name__}: {exc}; falling back to CPU", flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        exp3j_mod._OPTIMIZER = StructOptimizer(use_device="cpu", optimizer_class="FIRE")


def regenerate_pairchem_sid_worker(sid: str) -> dict[str, Any]:
    engine = G_REGEN_ENGINE
    if engine is None:
        raise RuntimeError("regen worker engine is not initialized")
    args = G_REGEN_ARGS
    target = G_REGEN_VAL[sid]
    counts = {str(k): int(v) for k, v in formula_counts(target).items()}
    tasks: list[dict[str, Any]] = []
    failures = Counter()
    for proposal in G_REGEN_PROPOSALS[sid].get("proposals", [])[: int(args["top_skeletons"])]:
        source_id = str(proposal.get("source_sample_id") or "")
        source_repr = G_REGEN_TRAIN_REPR.get(source_id)
        source_struct = G_REGEN_TRAIN_STRUCT.get(source_id)
        if source_repr is None or source_struct is None:
            failures["missing_source"] += 1
            continue
        original_rows = source_skeleton_rows(engine, source_repr)
        assignments = chemical_exact_cover_assignments(
            original_rows,
            counts,
            int(target["sg"]),
            G_REGEN_PRIORS,
            limit=int(args["assignment_prelimit"]),
            beam_width=int(args["assignment_beam_width"]),
        )
        source_lattice = {k: float(source_struct["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}
        for assignment_rank, (_score, assignment, _meta) in enumerate(assignments, start=1):
            probe = {
                "sample_id": sid,
                "source_sample_id": source_id,
                "proposal_rank": int(proposal.get("rank") or 0),
                "assignment_rank": int(assignment_rank),
                "geometry_source": PAIRCHEM_GEOMETRY_SOURCE,
            }
            uid = candidate_uid(probe)
            if uid not in G_REGEN_WANTED_BY_SID.get(sid, set()):
                continue
            assigned_rows: list[dict[str, Any]] = []
            for row, element in zip(original_rows, assignment):
                item = dict(row)
                item["source_element"] = str(row.get("element"))
                item["element"] = str(element)
                assigned_rows.append(item)
            try:
                params, _fallback_count = flexible_params_from_reference(engine, assigned_rows, source_struct, neural_params=None)
                cif, _render_meta = render_candidate(
                    engine=engine,
                    target=target,
                    rows=assigned_rows,
                    option={"lattice": source_lattice, "params": params},
                    data_name=f"{sid}_exp3n_p{int(proposal.get('rank') or 0)}_a{assignment_rank}",
                )
                row = dict(G_REGEN_WANTED[uid])
                row["candidate_uid"] = uid
                tasks.append({"row": row, "cif": cif, "steps": int(args["steps"]), "fmax": float(args["fmax"])})
            except Exception as exc:  # noqa: BLE001
                failures[f"render_failed:{type(exc).__name__}"] += 1
    return {"sample_id": sid, "tasks": tasks, "failures": dict(failures)}


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def assign_ranks(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    out: list[dict[str, Any]] = []
    for sid in sorted(by_sid):
        for rank, row in enumerate(by_sid[sid][:top_k], start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    return out


def selected_task_rows(base_rows: list[dict[str, Any]], max_rank: int, per_sample: int, max_tasks: int | None) -> list[dict[str, Any]]:
    eligible = [
        r
        for r in base_rows
        if int(r.get("row_count") or 0) >= 7
        and str(r.get("geometry_source")) == PAIRCHEM_GEOMETRY_SOURCE
        and bool(r.get("formula_ok"))
        and bool(r.get("space_group_ok"))
        and bool(r.get("exact_cover_retained"))
        and int(r.get("rank") or 10**9) <= int(max_rank)
    ]
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_sid[str(row["sample_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for sid in sorted(by_sid):
        group = sorted(by_sid[sid], key=lambda r: (int(r.get("rank") or 10**9), int(r.get("proposal_rank") or 0), int(r.get("assignment_rank") or 0)))
        rows.extend(group[: int(per_sample)])
    rows.sort(key=lambda r: (str(r["sample_id"]), int(r.get("rank") or 10**9), int(r.get("proposal_rank") or 0), int(r.get("assignment_rank") or 0)))
    if max_tasks is not None:
        rows = rows[: int(max_tasks)]
    return rows


def build_broad_variants(base_rows: list[dict[str, Any]], optimized_by_uid: dict[str, dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        by_sid[str(row["sample_id"])].append(row)
    for group in by_sid.values():
        group.sort(key=lambda r: int(r.get("rank") or 10**9))
    variants: dict[str, list[dict[str, Any]]] = {}
    for variant in ("append_valid_chgnet_after_selected", "append_any_chgnet_after_selected", "replace_invalid_or_collision_if_chgnet_valid"):
        out: list[dict[str, Any]] = []
        for sid in sorted(by_sid):
            selected: list[dict[str, Any]] = []
            for row in by_sid[sid]:
                uid = str(row.get("candidate_uid") or candidate_uid(row))
                opt = optimized_by_uid.get(uid)
                is_chemical = str(row.get("geometry_source")) == PAIRCHEM_GEOMETRY_SOURCE
                unsafe_local = not bool(row.get("valid")) or bool(row.get("collision"))
                if variant == "replace_invalid_or_collision_if_chgnet_valid" and is_chemical and unsafe_local and opt is not None and bool(opt.get("valid")):
                    selected.append(opt)
                    continue
                selected.append(row)
                if opt is None:
                    continue
                if variant == "append_any_chgnet_after_selected":
                    selected.append(opt)
                elif variant == "append_valid_chgnet_after_selected" and bool(opt.get("valid")):
                    selected.append(opt)
            for row in selected[:top_k]:
                item = dict(row)
                item["selection_variant"] = variant
                out.append(item)
        variants[variant] = assign_ranks(out, top_k)
    return variants


def evaluate_candidates(candidates: list[dict[str, Any]], eval_workers: int) -> list[dict[str, Any]]:
    structured_val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    payload_by_sid: dict[str, dict[str, Any]] = {}
    for row in candidates:
        sid = str(row["sample_id"])
        target = structured_val.get(sid)
        if target is None:
            continue
        if sid not in payload_by_sid:
            counts = {str(k): int(v) for k, v in target["formula_counts"].items()}
            payload_by_sid[sid] = {
                "sample_id": sid,
                "target_cif_path": str(target["source_path"]),
                "formula_counts": counts,
                "target_atom_count": int(sum(counts.values())),
                "sg": int(target["sg"]),
                "candidates": [],
            }
        payload_by_sid[sid]["candidates"].append(row)

    evaluated: list[dict[str, Any]] = []
    payloads = list(payload_by_sid.values())
    if payloads:
        with ProcessPoolExecutor(max_workers=max(1, int(eval_workers))) as pool:
            futures = [pool.submit(eval_sample, payload) for payload in payloads]
            for i, fut in enumerate(as_completed(futures), start=1):
                evaluated.extend(fut.result())
                if i % 200 == 0:
                    print(f"[exp3n-eval] evaluated samples {i}/{len(futures)}", flush=True)
    return evaluated


def regenerate_selected_pairchem_tasks(task_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global G_REGEN_TRAIN_REPR, G_REGEN_TRAIN_STRUCT, G_REGEN_VAL, G_REGEN_VAL_REPR, G_REGEN_PROPOSALS
    global G_REGEN_ENGINE, G_REGEN_PRIORS, G_REGEN_WANTED, G_REGEN_WANTED_BY_SID, G_REGEN_ARGS
    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    train_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    train_struct_list = list(train_struct.values())
    val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in train_struct_list + list(val.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    priors = build_train_priors(train_struct_list)

    wanted = {candidate_uid(r): dict(r) for r in task_rows}
    wanted_by_sid: dict[str, set[str]] = defaultdict(set)
    for uid, row in wanted.items():
        wanted_by_sid[str(row["sample_id"])].add(uid)
    tasks: list[dict[str, Any]] = []
    failures = Counter()
    G_REGEN_TRAIN_REPR = train_repr
    G_REGEN_TRAIN_STRUCT = train_struct
    G_REGEN_VAL = val
    G_REGEN_VAL_REPR = val_repr
    G_REGEN_PROPOSALS = proposals
    G_REGEN_ENGINE = engine
    G_REGEN_PRIORS = priors
    G_REGEN_WANTED = wanted
    G_REGEN_WANTED_BY_SID = dict(wanted_by_sid)
    G_REGEN_ARGS = {
        "top_skeletons": int(args.top_skeletons),
        "assignment_prelimit": int(args.assignment_prelimit),
        "assignment_beam_width": int(args.assignment_beam_width),
        "steps": int(args.steps),
        "fmax": float(args.fmax),
    }
    sids = sorted(wanted_by_sid)
    if int(args.regen_workers) <= 1:
        for i, sid in enumerate(sids, start=1):
            res = regenerate_pairchem_sid_worker(sid)
            tasks.extend(res["tasks"])
            failures.update(res["failures"])
            if i % 200 == 0:
                print(f"[exp3n-regen] regenerated samples {i}/{len(sids)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.regen_workers))) as pool:
            futures = [pool.submit(regenerate_pairchem_sid_worker, sid) for sid in sids]
            for i, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                tasks.extend(res["tasks"])
                failures.update(res["failures"])
                if i % 200 == 0:
                    print(f"[exp3n-regen] regenerated samples {i}/{len(sids)}", flush=True)
    return tasks, {"failures": dict(failures), "wanted_tasks": len(wanted), "regenerated_tasks": len(tasks), "row_count_lookup_records": len(val_repr), "regen_workers": int(args.regen_workers)}


def report_body(result: dict[str, Any]) -> str:
    baseline = result["baseline_exp2n"]["rows_ge7"]
    best_name = result["best_variant"]
    best = result["variants"][best_name]["rows_ge7"]
    gate = result["gate"]
    diag = result["chgnet_diagnostics"]
    return f"""## opentry_14 实验 3n：CHGNet local optimizer after Exp2n pair-chem

结果文件：`model/New_model/opentry_14/results/experiment_3n_chgnet_after_exp2n_pairchem.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3n_chgnet_after_exp2n_pairchem/`

- 为什么做：Exp2n 已把 rows>=7 conversion 推过 28% target，按 prompt 必须验证 symmetry-preserving local optimizer 是否能在不破坏 SG/exact-cover 的情况下再带来至少 +2pp conversion，并改善 collision/local packing。
- 核心假设：如果 Exp2n 剩余错误主要来自局部 basin 或短距离 packing，按 inference-safe rank/per-sample top-N 选择 pair-chem candidates 后，CHGNet position-only relaxation 应在保留原候选的 append variant 中新增 rows>=7 match。
- 数据规模：base rows>=7 candidate records `{result['data_scale']['base_rows_ge7_candidate_records']}`；eligible selected rows `{result['data_scale']['optimizer_task_rows']}`；regenerated `{result['data_scale']['regenerated_tasks']}`；accepted `{diag['accepted_candidates']}`；evaluated optimized `{diag['evaluated_optimized_candidates']}`；max_rank `{result['method']['max_rank']}`；per_sample `{result['method']['per_sample']}`；CHGNet workers `{result['cpu_policy']['chgnet_workers']}`。
- baseline：Exp2n best `{result['baseline_exp2n']['best_variant']}` rows>=7 match@50 `{pct(baseline.get('match@50'))}`、conversion `{pct(baseline.get('skeleton_to_match_conversion@50'))}`、collision `{pct(baseline.get('collision_rate'))}`。Exp3 pass line requires conversion `{pct(gate['minimum_standard']['rows_ge7_conversion50'])}`。
- 方法变化：选择只使用 rank、formula_ok、space_group_ok、exact_cover_retained、row_count 和 pair-chem geometry source；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。优化后必须保持 formula/site count/SG，并改善 local proxy；append variant 保留原始 Exp2n 排序。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；collision `{pct(best.get('collision_rate'))}`；valid `{pct(best.get('valid_rate'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`。
- CHGNet 诊断：tasks `{diag['optimizer_results']}`；successful `{diag['successful_optimizations']}`；accepted `{diag['accepted_candidates']}`；optimized valid records `{diag['optimized_valid_records']}`；optimized match records `{diag['optimized_match_records']}`；mean min-distance delta `{diag.get('mean_min_distance_delta')}`；mean hard-collision delta `{diag.get('mean_hard_collision_delta')}`；mean close-pair delta `{diag.get('mean_close_pair_delta')}`。
- 可信度：中等。选择规则是推理安全的，优化后重新评估 StructureMatcher/SG/formula/site/exact；限制是只测试 Exp2n pair-chem candidates 的 rank/per-sample top-N，不覆盖所有 eligible candidates。
- 和历史实验关系：这是 Exp2n 通过 target gate 后的正式 Exp3 检验，也复核 Exp3j/3k “局部优化改善 packing 但不改善 conversion”的历史结论是否仍成立。
- gate 判定：passed=`{gate['passed']}`；conversion delta vs Exp2n `{pp(gate['rows_ge7_conversion50_delta_vs_exp2n'])}`；match@50 delta vs Exp2n `{pp(gate['rows_ge7_match50_delta_vs_exp2n'])}`；collision/local improved=`{gate['collision_or_local_packing_improved']}`；SG not worse=`{gate['sg_not_worse']}`；exact-cover not worse=`{gate['exact_cover_not_worse']}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    global ARTIFACT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result-path", default=str(EXP2N_RESULT))
    parser.add_argument("--artifact-subdir", default="exp3n_chgnet_after_exp2n_pairchem")
    parser.add_argument("--result-filename", default="experiment_3n_chgnet_after_exp2n_pairchem.json")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--chgnet-workers", type=int, default=32)
    parser.add_argument("--regen-workers", type=int, default=96)
    parser.add_argument("--eval-workers", type=int, default=112)
    parser.add_argument("--gpu-devices", default="0,1")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--fmax", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--per-sample", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--top-skeletons", type=int, default=8)
    parser.add_argument("--assignment-prelimit", type=int, default=32)
    parser.add_argument("--assignment-beam-width", type=int, default=256)
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR = OUT_DIR / "artifacts" / str(args.artifact_subdir)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    base_result_path = Path(str(args.base_result_path))
    exp2n = read_json(base_result_path)
    best_variant_2n = str(exp2n["best_variant"])
    base_path = Path(exp2n["artifacts"]["selected_variants"][best_variant_2n])
    base_rows = list(read_jsonl_iter(base_path))
    for row in base_rows:
        row["candidate_uid"] = candidate_uid(row)
    task_rows = selected_task_rows(base_rows, args.max_rank, args.per_sample, args.max_tasks)
    write_jsonl(ARTIFACT_DIR / "optimizer_task_rows.jsonl", task_rows)

    tasks, regen_meta = regenerate_selected_pairchem_tasks(task_rows, args)
    write_json(ARTIFACT_DIR / "regeneration_meta.json", regen_meta)

    devices = tuple(x.strip() for x in str(args.gpu_devices).split(",") if x.strip()) or ("0",)
    opt_results: list[dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(args.chgnet_workers)), initializer=init_chgnet_worker_fallback, initargs=(devices,)) as pool:
            futures = [pool.submit(exp3j_mod.chgnet_optimize_task, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                opt_results.append(fut.result())
                if i % 500 == 0:
                    print(f"[exp3n-chgnet] optimized {i}/{len(futures)}", flush=True)

    accepted_with_cif = []
    for result_row in opt_results:
        if not bool(result_row.get("accepted")) or not result_row.get("candidate"):
            continue
        item = dict(result_row["candidate"])
        item.update(
            {
                "pool_source": "exp3n_pairchem_chgnet",
                "geometry_variant": "chgnet_position_relax_after_exp2n_pairchem",
                "selection_rule": "CHGNet optimized Exp2n pair-chem rows>=7 candidates with formula/SG/exact retained, rank cutoff, and per-sample cap; no match/RMSD/skeleton label used for selection",
            }
        )
        accepted_with_cif.append(item)
    write_jsonl_gz(ARTIFACT_DIR / "chgnet_optimized_exp2n_pairchem_candidates_with_cif.jsonl.gz", accepted_with_cif)
    write_jsonl(ARTIFACT_DIR / "chgnet_optimizer_diagnostics.jsonl", [{k: v for k, v in r.items() if k != "candidate"} for r in opt_results])

    evaluated_optimized = evaluate_candidates(accepted_with_cif, args.eval_workers)
    write_jsonl(ARTIFACT_DIR / "evaluated_chgnet_optimized_exp2n_pairchem_candidates.jsonl", evaluated_optimized)
    optimized_by_uid = {str(r["candidate_uid"]): r for r in evaluated_optimized}

    variant_rows = build_broad_variants(base_rows, optimized_by_uid, int(args.top_k))
    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for name, rows in variant_rows.items():
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
    baseline_rows7 = exp2n["variants"][best_variant_2n]["rows_ge7"]
    conversion_min = float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0) + 0.02
    conversion_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0)
    collision_improved = bool(best_rows7.get("collision_rate") is not None and baseline_rows7.get("collision_rate") is not None and float(best_rows7["collision_rate"]) < float(baseline_rows7["collision_rate"]))
    sg_not_worse = float(best_rows7.get("sg_consistency") or 0.0) >= float(baseline_rows7.get("sg_consistency") or 0.0) - 1.0e-12
    exact_not_worse = float(best_rows7.get("exact_cover_retained") or 0.0) >= float(baseline_rows7.get("exact_cover_retained") or 0.0) - 1.0e-12
    passed = bool(float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= conversion_min and match_delta >= 0 and collision_improved and sg_not_worse and exact_not_worse)

    hard_delta: list[float] = []
    close_delta: list[float] = []
    min_delta: list[float] = []
    input_status = Counter()
    for row in opt_results:
        input_status[(bool(row.get("input_valid")), bool(row.get("input_match")), bool(row.get("input_predicted_skeleton_hit")), bool(row.get("accepted")))] += 1
        if not row.get("accepted"):
            continue
        before = row.get("before_proxy") or {}
        after = row.get("after_proxy") or {}
        hard_delta.append(float(after.get("hard_collision_count") or 0) - float(before.get("hard_collision_count") or 0))
        close_delta.append(float(after.get("close_pair_count") or 0) - float(before.get("close_pair_count") or 0))
        if before.get("min_pair_distance") is not None and after.get("min_pair_distance") is not None:
            min_delta.append(float(after["min_pair_distance"]) - float(before["min_pair_distance"]))

    diag = {
        "optimizer_results": len(opt_results),
        "successful_optimizations": sum(bool(r.get("success")) for r in opt_results),
        "accepted_candidates": len(accepted_with_cif),
        "evaluated_optimized_candidates": len(evaluated_optimized),
        "optimized_valid_records": sum(bool(r.get("valid")) for r in evaluated_optimized),
        "optimized_match_records": sum(bool(r.get("match")) for r in evaluated_optimized),
        "mean_hard_collision_delta": mean(hard_delta),
        "mean_close_pair_delta": mean(close_delta),
        "mean_min_distance_delta": mean(min_delta),
        "input_status_counts": {str(k): v for k, v in input_status.items()},
    }
    if passed:
        verdict = "pass_exp3_gate"
        reason = "CHGNet relaxation adds at least +2pp conversion over Exp2n while preserving SG/exact-cover and improving collision/local proxy."
        next_step = "Proceed to Exp4 multi-hypothesis free-parameter generator + inference-safe critic using Exp3n as the local optimizer stage."
    else:
        verdict = "fail_local_optimizer_not_conversion_limited"
        reason = "CHGNet relaxation after Exp2n does not deliver the required +2pp conversion lift, so local energy/packing is not the main remaining bottleneck."
        next_step = "Do not enter Exp4/5/official from local optimizer; return to skeleton/site/free-parameter alignment rather than expanding CHGNet."

    result = {
        "experiment": "opentry_14_exp3n_chgnet_after_exp2n_pairchem",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "chgnet_position_relax_after_exp2n_pairchem_candidates",
            "base_result": str(base_result_path),
            "base_variant": best_variant_2n,
            "selection_rule": "rows>=7 pair-chem candidates with formula_ok, space_group_ok, exact_cover_retained, rank<=max_rank and per-sample top-N",
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "max_rank": int(args.max_rank),
            "per_sample": int(args.per_sample),
            "optimizer": "CHGNet StructOptimizer/FIRE",
            "relax_cell": False,
            "steps": int(args.steps),
            "fmax": float(args.fmax),
        },
        "cpu_policy": {
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "gpu_devices": list(devices),
            "regen_workers": int(args.regen_workers),
            "chgnet_workers": int(args.chgnet_workers),
            "eval_workers": int(args.eval_workers),
        },
        "data_scale": {
            "base_rows": len(base_rows),
            "base_rows_ge7_candidate_records": len([r for r in base_rows if int(r.get("row_count") or 0) >= 7]),
            "optimizer_task_rows": len(task_rows),
            "optimizer_tasks": len(tasks),
            "regenerated_tasks": int(regen_meta.get("regenerated_tasks") or 0),
            "selected_variant_paths": selected_paths,
        },
        "regeneration_meta": regen_meta,
        "baseline_exp2n": {"result_path": str(base_result_path), "best_variant": best_variant_2n, "rows_ge7": baseline_rows7, "overall": exp2n["variants"][best_variant_2n]["overall"]},
        "chgnet_diagnostics": diag,
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "passed": passed,
            "rows_ge7_match50_delta_vs_exp2n": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2n": conversion_delta,
            "collision_or_local_packing_improved": collision_improved,
            "sg_not_worse": sg_not_worse,
            "exact_cover_not_worse": exact_not_worse,
            "minimum_standard": {"rows_ge7_conversion50": conversion_min, "conversion_lift_vs_exp2n": 0.02},
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "optimizer_task_rows": str(ARTIFACT_DIR / "optimizer_task_rows.jsonl"),
            "optimized_candidates_with_cif": str(ARTIFACT_DIR / "chgnet_optimized_exp2n_pairchem_candidates_with_cif.jsonl.gz"),
            "evaluated_optimized_candidates": str(ARTIFACT_DIR / "evaluated_chgnet_optimized_exp2n_pairchem_candidates.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    result_path = RESULT_DIR / str(args.result_filename)
    write_json(result_path, result)
    if not args.skip_report:
        body = report_body(result)
        marker = "<!-- OPENTRY14_EXP3N_CHGNET_AFTER_EXP2N_PAIRCHEM -->"
        append_or_replace(REPORT_PATH, marker, body)
        append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(result_path), "best_variant": best_variant, "gate": result["gate"], "decision": result["decision"], "diagnostics": diag}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
