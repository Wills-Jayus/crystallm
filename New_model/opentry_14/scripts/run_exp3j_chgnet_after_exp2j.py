#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

warnings.filterwarnings("ignore")

ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp3j_chgnet_after_exp2j"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from run_exp2_predicted_skeleton_renderer_site_mapping import formula_counts, read_jsonl, sample_id, source_skeleton_rows  # noqa: E402
from run_exp2j_chemical_site_order_assignment import (  # noqa: E402
    build_train_priors,
    chemical_exact_cover_assignments,
)
from run_exp4_rows_ge7_multi_geometry_proposal import eval_sample, render_candidate, summarize  # noqa: E402
from run_symcif_v4_geometry_model_eval import flexible_params_from_reference  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
EXP3_PROPOSALS = OP13 / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP2J_RESULT = RESULT_DIR / "experiment_2j_chemical_site_order_assignment.json"
BUDGETS = (1, 5, 20, 50)
HARD_COLLISION_CUTOFF = 0.50
LOCAL_CLOSE_PAIR_CUTOFF = 1.00

_OPTIMIZER: Any = None


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


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def mean(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


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


def structure_counts(structure: Structure) -> dict[str, int]:
    return {str(k): int(round(v)) for k, v in structure.composition.get_el_amt_dict().items()}


def min_pair_distance(structure: Structure) -> float | None:
    if len(structure) < 2:
        return None
    matrix = structure.distance_matrix
    vals = [float(matrix[i, j]) for i in range(len(structure)) for j in range(i + 1, len(structure))]
    return min(vals) if vals else None


def local_proxy(structure: Structure) -> dict[str, Any]:
    min_dist = min_pair_distance(structure)
    hard = close = 0
    if len(structure) >= 2:
        matrix = structure.distance_matrix
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                d = float(matrix[i, j])
                hard += int(d < HARD_COLLISION_CUTOFF)
                close += int(d < LOCAL_CLOSE_PAIR_CUTOFF)
    return {
        "min_pair_distance": min_dist,
        "hard_collision_count": hard,
        "close_pair_count": close,
        "volume_per_atom": float(structure.volume) / max(1, len(structure)),
    }


def proxy_improved(before: dict[str, Any], after: dict[str, Any]) -> bool:
    before_tuple = (
        int(before.get("hard_collision_count") or 0),
        int(before.get("close_pair_count") or 0),
        -float(before.get("min_pair_distance") or 0.0),
    )
    after_tuple = (
        int(after.get("hard_collision_count") or 0),
        int(after.get("close_pair_count") or 0),
        -float(after.get("min_pair_distance") or 0.0),
    )
    return after_tuple < before_tuple


def candidate_uid(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("sample_id") or ""),
            str(row.get("source_sample_id") or row.get("reference_sample_id") or ""),
            str(int(row.get("proposal_rank") or 0)),
            str(int(row.get("assignment_rank") or 0)),
            str(row.get("geometry_source") or ""),
        ]
    )


def init_chgnet_worker(devices: tuple[str, ...]) -> None:
    global _OPTIMIZER
    ident = mp.current_process()._identity  # type: ignore[attr-defined]
    idx = int(ident[0] - 1) if ident else 0
    device = str(devices[idx % max(1, len(devices))])
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[-1] if device.startswith("cuda:") else device
    from chgnet.model.dynamics import StructOptimizer

    _OPTIMIZER = StructOptimizer(use_device="cuda", optimizer_class="FIRE")


def chgnet_optimize_task(task: dict[str, Any]) -> dict[str, Any]:
    if _OPTIMIZER is None:
        raise RuntimeError("CHGNet optimizer not initialized")
    row = task["row"]
    cif = str(task["cif"] or "")
    started = time.time()
    out: dict[str, Any] = {
        "candidate_uid": str(row["candidate_uid"]),
        "sample_id": str(row["sample_id"]),
        "row_count": int(row.get("row_count") or 0),
        "sg": int(row.get("sg") or 0),
        "input_valid": bool(row.get("valid")),
        "input_match": bool(row.get("match")),
        "input_predicted_skeleton_hit": bool(row.get("predicted_skeleton_hit")),
        "success": False,
        "accepted": False,
        "error": None,
    }
    try:
        structure = Structure.from_str(cif, fmt="cif")
        detected_before = int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
        before_proxy = local_proxy(structure)
        result = _OPTIMIZER.relax(
            structure,
            fmax=float(task["fmax"]),
            steps=int(task["steps"]),
            relax_cell=False,
            verbose=False,
            loginterval=0,
        )
        optimized = result["final_structure"]
        detected_after = int(SpacegroupAnalyzer(optimized, symprec=0.1).get_space_group_number())
        after_proxy = local_proxy(optimized)
        same_formula = structure_counts(optimized) == structure_counts(structure)
        same_count = len(optimized) == len(structure)
        sg_ok = detected_after == int(row.get("sg") or 0)
        accepted = bool(
            detected_before == int(row.get("sg") or 0)
            and sg_ok
            and same_formula
            and same_count
            and proxy_improved(before_proxy, after_proxy)
        )
        out.update(
            {
                "success": True,
                "accepted": accepted,
                "detected_sg_before": detected_before,
                "detected_sg_after": detected_after,
                "same_formula": same_formula,
                "same_site_count": same_count,
                "before_proxy": before_proxy,
                "after_proxy": after_proxy,
                "proxy_improved": proxy_improved(before_proxy, after_proxy),
                "runtime_seconds": time.time() - started,
            }
        )
        if accepted:
            item = dict(row)
            item.update(
                {
                    "pool_source": "exp2j_chemical_chgnet",
                    "geometry_source": str(row.get("geometry_source") or "chemical_site_order_source_geometry") + "+chgnet_position_relax",
                    "geometry_variant": "chgnet_position_relax_after_exp2j",
                    "selection_rule": "CHGNet optimized only Exp2j rows>=7 chemical candidates with formula/SG/exact retained and valid=false; no match/RMSD/skeleton label used for selection",
                    "optimizer_steps": int(task["steps"]),
                    "optimizer_fmax": float(task["fmax"]),
                    "optimizer_before_min_pair_distance": before_proxy.get("min_pair_distance"),
                    "optimizer_after_min_pair_distance": after_proxy.get("min_pair_distance"),
                    "optimizer_before_hard_collision_count": before_proxy.get("hard_collision_count"),
                    "optimizer_after_hard_collision_count": after_proxy.get("hard_collision_count"),
                    "optimizer_before_close_pair_count": before_proxy.get("close_pair_count"),
                    "optimizer_after_close_pair_count": after_proxy.get("close_pair_count"),
                    "rank": None,
                    "cif": optimized.to(fmt="cif"),
                    "render_success": True,
                    "render_error": None,
                }
            )
            out["candidate"] = item
    except Exception as exc:  # noqa: BLE001
        out.update({"error": f"{type(exc).__name__}: {exc}", "runtime_seconds": time.time() - started})
    return out


def assign_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    out: list[dict[str, Any]] = []
    for sid, group in by_sid.items():
        for rank, row in enumerate(group, start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    out.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    return out


def build_variants(base_rows: list[dict[str, Any]], optimized_by_uid: dict[str, dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        by_sid[str(row["sample_id"])].append(row)
    for group in by_sid.values():
        group.sort(key=lambda r: int(r.get("rank") or 10**9))
    variants: dict[str, list[dict[str, Any]]] = {}
    for variant in ("replace_invalid_chemical_if_chgnet_valid", "append_chgnet_after_chemical_invalid", "interleave_chgnet_after_chemical_invalid"):
        rows_out: list[dict[str, Any]] = []
        for sid in sorted(by_sid):
            selected: list[dict[str, Any]] = []
            for row in by_sid[sid]:
                uid = str(row.get("candidate_uid") or candidate_uid(row))
                opt = optimized_by_uid.get(uid)
                is_target = str(row.get("geometry_source")) == "chemical_site_order_source_geometry" and not bool(row.get("valid"))
                if variant == "replace_invalid_chemical_if_chgnet_valid":
                    if is_target and opt is not None and bool(opt.get("valid")):
                        selected.append(opt)
                    else:
                        selected.append(row)
                elif variant == "append_chgnet_after_chemical_invalid":
                    selected.append(row)
                    if is_target and opt is not None and bool(opt.get("valid")):
                        selected.append(opt)
                else:
                    selected.append(row)
                    if is_target and opt is not None:
                        selected.append(opt)
            for row in selected[:top_k]:
                item = dict(row)
                item["selection_variant"] = variant
                rows_out.append(item)
        variants[variant] = assign_ranks(rows_out)
    return variants


def regenerate_selected_chemical_tasks(task_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    for sid in sorted(wanted_by_sid):
        target = val[sid]
        target_repr = val_repr[sid]
        counts = {str(k): int(v) for k, v in formula_counts(target).items()}
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_skeletons)]:
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            source_struct = train_struct.get(source_id)
            if source_repr is None or source_struct is None:
                failures["missing_source"] += 1
                continue
            original_rows = source_skeleton_rows(engine, source_repr)
            assignments = chemical_exact_cover_assignments(
                original_rows,
                counts,
                int(target["sg"]),
                priors,
                limit=int(args.assignment_limit),
                beam_width=int(args.assignment_beam_width),
            )
            source_lattice = {k: float(source_struct["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}
            for assignment_rank, (_score, assignment, _meta) in enumerate(assignments, start=1):
                probe = {
                    "sample_id": sid,
                    "source_sample_id": source_id,
                    "proposal_rank": int(proposal.get("rank") or 0),
                    "assignment_rank": int(assignment_rank),
                    "geometry_source": "chemical_site_order_source_geometry",
                }
                uid = candidate_uid(probe)
                if uid not in wanted:
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
                        data_name=f"{sid}_exp3j_p{int(proposal.get('rank') or 0)}_a{assignment_rank}",
                    )
                    row = dict(wanted[uid])
                    row["candidate_uid"] = uid
                    tasks.append({"row": row, "cif": cif, "steps": int(args.steps), "fmax": float(args.fmax)})
                except Exception as exc:  # noqa: BLE001
                    failures[f"render_failed:{type(exc).__name__}"] += 1
    return tasks, {"failures": dict(failures), "wanted_tasks": len(wanted), "regenerated_tasks": len(tasks)}


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
    baseline = result["baseline_exp2j"]["rows_ge7"]
    best_name = result["best_variant"]
    best = result["variants"][best_name]["rows_ge7"]
    gate = result["gate"]
    diag = result["chgnet_diagnostics"]
    return f"""## opentry_14 实验 3j：CHGNet local optimizer after Exp2j

结果文件：`model/New_model/opentry_14/results/experiment_3j_chgnet_after_exp2j.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3j_chgnet_after_exp2j/`

- 为什么做：Exp2j chemical/site-order assignment 已把 rows>=7 conversion 从 Exp2b `24.866%` 提到 `{pct(baseline.get('skeleton_to_match_conversion@50'))}`，超过 Exp3 +2pp 入口线；prompt 要求在 learned/constructed repair 后测试 symmetry-preserving local optimizer 是否还能再提高 conversion 并改善 local packing。
- 核心假设：如果 Exp2j 剩余失败主要是局部 collision/packing，则对 Exp2j best 中 formula/SG/exact 保持但 valid=false 的 chemical candidates 做 CHGNet position-only relaxation，应在不破坏 SG/exact-cover 的情况下再提升 conversion@50 至少 `+2pp`。
- 数据规模：Exp2j best rows>=7 candidate records `{result['data_scale']['base_rows_ge7_candidate_records']}`；selected CHGNet tasks `{result['data_scale']['optimizer_tasks']}`；regenerated CIF tasks `{result['data_scale']['regenerated_tasks']}`；accepted optimized `{diag['accepted_candidates']}`；evaluated optimized `{diag['evaluated_optimized_candidates']}`；GPU devices `{result['cpu_policy']['gpu_devices']}`；workers `{result['cpu_policy']['chgnet_workers']}`。
- baseline：Exp2j best `{result['baseline_exp2j']['best_variant']}` rows>=7 match@50 `{pct(baseline.get('match@50'))}`、conversion `{pct(baseline.get('skeleton_to_match_conversion@50'))}`、collision `{pct(baseline.get('collision_rate'))}`。Exp3j pass line requires conversion `{pct(gate['minimum_standard']['rows_ge7_conversion50'])}`。
- 方法变化：只优化 inference-safe 选择的 Exp2j chemical candidates：rows>=7、chemical_site_order_source_geometry、formula_ok、space_group_ok、exact_cover_retained、valid=false。CHGNet 不 relax cell；优化后必须保持 formula/site count/SG，并改善 local proxy，否则回退。排序和选择不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；collision `{pct(best.get('collision_rate'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`。
- CHGNet 诊断：optimized valid records `{diag['optimized_valid_records']}`；optimized match records `{diag['optimized_match_records']}`；mean min-distance delta `{diag.get('mean_min_distance_delta')}`；mean hard-collision delta `{diag.get('mean_hard_collision_delta')}`；mean close-pair delta `{diag.get('mean_close_pair_delta')}`。
- 可信度：中等。CHGNet 是真实预训练势能，优化后重新 eval StructureMatcher/SG/formula/site/exact；限制是只覆盖 Exp2j chemical invalid candidates，不覆盖 safe-pool hydrated/prototype 的原始 CIF。
- 和历史实验关系：这是 Exp3/3b 在 Exp2j 新 repair 基线后的复测，直接判断 local optimizer 是否能在新的 assignment posterior 上产生追加收益。
- gate 判定：passed=`{gate['passed']}`；conversion delta vs Exp2j `{pp(gate['rows_ge7_conversion50_delta_vs_exp2j'])}`；match@50 delta vs Exp2j `{pp(gate['rows_ge7_match50_delta_vs_exp2j'])}`；collision/local improved=`{gate['collision_or_local_packing_improved']}`；SG not worse=`{gate['sg_not_worse']}`；exact-cover not worse=`{gate['exact_cover_not_worse']}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chgnet-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=96)
    parser.add_argument("--gpu-devices", default="0,1")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--fmax", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--top-skeletons", type=int, default=8)
    parser.add_argument("--assignment-limit", type=int, default=8)
    parser.add_argument("--assignment-beam-width", type=int, default=128)
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    exp2j = read_json(EXP2J_RESULT)
    best_variant_2j = str(exp2j["best_variant"])
    base_path = Path(exp2j["artifacts"]["selected_variants"][best_variant_2j])
    base_rows = list(read_jsonl_iter(base_path))
    for row in base_rows:
        row["candidate_uid"] = candidate_uid(row)

    task_rows = [
        r
        for r in base_rows
        if int(r.get("row_count") or 0) >= 7
        and str(r.get("geometry_source")) == "chemical_site_order_source_geometry"
        and bool(r.get("formula_ok"))
        and bool(r.get("space_group_ok"))
        and bool(r.get("exact_cover_retained"))
        and not bool(r.get("valid"))
    ]
    task_rows.sort(key=lambda r: (str(r["sample_id"]), int(r.get("rank") or 10**9), int(r.get("proposal_rank") or 0), int(r.get("assignment_rank") or 0)))
    if args.max_tasks is not None:
        task_rows = task_rows[: int(args.max_tasks)]

    tasks, regen_meta = regenerate_selected_chemical_tasks(task_rows, args)
    write_jsonl(ARTIFACT_DIR / "optimizer_task_rows.jsonl", task_rows)
    write_json(ARTIFACT_DIR / "regeneration_meta.json", regen_meta)

    devices = tuple(x.strip() for x in str(args.gpu_devices).split(",") if x.strip()) or ("0",)
    opt_results: list[dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, int(args.chgnet_workers)), initializer=init_chgnet_worker, initargs=(devices,)) as pool:
            futures = [pool.submit(chgnet_optimize_task, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                opt_results.append(fut.result())
                if i % 200 == 0:
                    print(f"[exp3j-chgnet] optimized {i}/{len(futures)}", flush=True)

    accepted_with_cif = [r["candidate"] for r in opt_results if bool(r.get("accepted")) and r.get("candidate")]
    write_jsonl_gz(ARTIFACT_DIR / "chgnet_optimized_exp2j_candidates_with_cif.jsonl.gz", accepted_with_cif)
    write_jsonl(ARTIFACT_DIR / "chgnet_optimizer_diagnostics.jsonl", [{k: v for k, v in r.items() if k != "candidate"} for r in opt_results])

    structured_val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    payload_by_sid: dict[str, dict[str, Any]] = {}
    for row in accepted_with_cif:
        sid = str(row["sample_id"])
        target = structured_val.get(sid)
        if target is None:
            continue
        if sid not in payload_by_sid:
            counts = {str(k): int(v) for k, v in target["formula_counts"].items()}
            payload_by_sid[sid] = {"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": int(sum(counts.values())), "sg": int(target["sg"]), "candidates": []}
        payload_by_sid[sid]["candidates"].append(row)

    evaluated_optimized: list[dict[str, Any]] = []
    payloads = list(payload_by_sid.values())
    if payloads:
        with ProcessPoolExecutor(max_workers=max(1, int(args.eval_workers))) as pool:
            futures = [pool.submit(eval_sample, payload) for payload in payloads]
            for i, fut in enumerate(as_completed(futures), start=1):
                evaluated_optimized.extend(fut.result())
                if i % 200 == 0:
                    print(f"[exp3j-chgnet] evaluated samples {i}/{len(futures)}", flush=True)
    write_jsonl(ARTIFACT_DIR / "evaluated_chgnet_optimized_exp2j_candidates.jsonl", evaluated_optimized)
    optimized_by_uid = {str(r["candidate_uid"]): r for r in evaluated_optimized}

    variant_rows = build_variants(base_rows, optimized_by_uid, int(args.top_k))
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
    baseline_rows7 = exp2j["variants"][best_variant_2j]["rows_ge7"]
    conversion_min = float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0) + 0.02
    conversion_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0)
    collision_improved = bool(best_rows7.get("collision_rate") is not None and baseline_rows7.get("collision_rate") is not None and float(best_rows7["collision_rate"]) < float(baseline_rows7["collision_rate"]))
    sg_not_worse = float(best_rows7.get("sg_consistency") or 0.0) >= float(baseline_rows7.get("sg_consistency") or 0.0) - 1.0e-12
    exact_not_worse = float(best_rows7.get("exact_cover_retained") or 0.0) >= float(baseline_rows7.get("exact_cover_retained") or 0.0) - 1.0e-12
    passed = bool(float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= conversion_min and match_delta >= 0 and collision_improved and sg_not_worse and exact_not_worse)

    hard_delta = []
    close_delta = []
    min_delta = []
    for row in opt_results:
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
    }

    if passed:
        verdict = "pass_exp3_gate"
        reason = "CHGNet local optimizer improves Exp2j conversion by at least +2pp while improving local packing and preserving SG/exact-cover."
        next_step = "Proceed to Exp4 multi-hypothesis generator + inference-safe critic using Exp2j+Exp3j as the new base; do not enter official yet."
    else:
        verdict = "fail_diagnostic_only"
        reason = "CHGNet local optimizer after Exp2j does not meet the +2pp conversion gate, even if local packing changes."
        next_step = "Do not enter Exp4/5/official from this optimizer; the remaining bottleneck is not fixed by local relaxation."

    result = {
        "experiment": "opentry_14_exp3j_chgnet_after_exp2j",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "chgnet_position_relax_after_exp2j_chemical_assignment",
            "base_result": str(EXP2J_RESULT),
            "base_variant": best_variant_2j,
            "selection_rule": "rows>=7 Exp2j chemical candidates with formula_ok, space_group_ok, exact_cover_retained and valid=false",
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "optimizer": "CHGNet StructOptimizer/FIRE",
            "relax_cell": False,
            "steps": int(args.steps),
            "fmax": float(args.fmax),
        },
        "cpu_policy": {
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "gpu_devices": list(devices),
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
        "baseline_exp2j": {"result_path": str(EXP2J_RESULT), "best_variant": best_variant_2j, "rows_ge7": baseline_rows7, "overall": exp2j["variants"][best_variant_2j]["overall"]},
        "chgnet_diagnostics": diag,
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "passed": passed,
            "rows_ge7_match50_delta_vs_exp2j": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2j": conversion_delta,
            "collision_or_local_packing_improved": collision_improved,
            "sg_not_worse": sg_not_worse,
            "exact_cover_not_worse": exact_not_worse,
            "minimum_standard": {"rows_ge7_conversion50": conversion_min, "conversion_lift_vs_exp2j": 0.02},
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "optimizer_task_rows": str(ARTIFACT_DIR / "optimizer_task_rows.jsonl"),
            "optimized_candidates_with_cif": str(ARTIFACT_DIR / "chgnet_optimized_exp2j_candidates_with_cif.jsonl.gz"),
            "evaluated_optimized_candidates": str(ARTIFACT_DIR / "evaluated_chgnet_optimized_exp2j_candidates.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_3j_chgnet_after_exp2j.json", result)
    body = report_body(result)
    marker = "<!-- OPENTRY14_EXP3J_CHGNET_AFTER_EXP2J -->"
    append_or_replace(REPORT_PATH, marker, body)
    append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / "experiment_3j_chgnet_after_exp2j.json"), "best_variant": best_variant, "gate": result["gate"], "decision": result["decision"], "diagnostics": diag}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
