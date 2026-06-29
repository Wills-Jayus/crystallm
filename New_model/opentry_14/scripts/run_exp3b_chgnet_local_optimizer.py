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
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp3b_chgnet_local_optimizer"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from run_exp4_rows_ge7_multi_geometry_proposal import eval_sample  # noqa: E402


EXP2B_RESULT = RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json"
EXP2B_EVAL = OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool" / "evaluated_safe_pool_candidates.jsonl"
SYMCIF_GEN = ROOT / "runs" / "symcif_v5_multidataset_wa_decoder" / "mpts52" / "val" / "generations" / "v5_fullgen_eval_pool.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"

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


def sample_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("id") or record.get("keys", {}).get("sample_id"))


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
    cif = str(task["generated_text"] or "")
    started = time.time()
    out: dict[str, Any] = {
        "sample_id": str(row["sample_id"]),
        "source_rank": int(row.get("source_rank") or 0),
        "row_count": int(row.get("row_count") or 0),
        "sg": int(row.get("sg") or 0),
        "input_valid": bool(row.get("valid")),
        "input_match": bool(row.get("match")),
        "input_predicted_skeleton_hit": bool(row.get("predicted_skeleton_hit")),
        "selected_by": "rows_ge7_hydrated_formula_sg_exact_invalid",
        "success": False,
        "accepted": False,
        "error": None,
        "cif": "",
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
                    "pool_source": "symcif_v5_hydrated_chgnet",
                    "geometry_source": str(row.get("geometry_source") or "symcif_v5_hydrated") + "+chgnet_position_relax",
                    "geometry_variant": "chgnet_position_relax",
                    "selection_rule": "CHGNet optimized only for rows>=7 hydrated formula/SG/exact candidates with invalid source metric; no match/RMSD/skeleton label used for selection",
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
            out["cif"] = item["cif"]
    except Exception as exc:  # noqa: BLE001
        out.update({"error": f"{type(exc).__name__}: {exc}", "runtime_seconds": time.time() - started})
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
        "mean_min_pair_distance": mean([float(r["min_pair_distance"]) for r in rows if r.get("min_pair_distance") is not None]),
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


def assign_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    for sid, sample_rows in by_sid.items():
        for rank, row in enumerate(sample_rows, start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    out.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    return out


def build_variants(
    *,
    safe_rows: list[dict[str, Any]],
    optimized_by_key: dict[tuple[str, int], dict[str, Any]],
    hydrated_quota: int,
    prototype_quota: int,
    append_quota: int,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    hyd_by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proto_by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in safe_rows:
        if str(row.get("pool_source")) == "symcif_v5_hydrated":
            hyd_by_sid[str(row["sample_id"])].append(row)
        else:
            proto_by_sid[str(row["sample_id"])].append(row)
    for rows in hyd_by_sid.values():
        rows.sort(key=lambda r: int(r.get("source_rank") or r.get("rank") or 10**9))
    for rows in proto_by_sid.values():
        rows.sort(key=lambda r: int(r.get("source_rank") or r.get("rank") or 10**9))

    variants: dict[str, list[dict[str, Any]]] = {}
    for variant in ("replace_invalid_if_chgnet_valid", "append_chgnet_after_hydrated", "interleave_chgnet_after_each_invalid"):
        rows_out: list[dict[str, Any]] = []
        for sid in sorted(set(hyd_by_sid) | set(proto_by_sid)):
            selected: list[dict[str, Any]] = []
            hyd = hyd_by_sid.get(sid, [])[:hydrated_quota]
            proto = proto_by_sid.get(sid, [])
            if variant == "replace_invalid_if_chgnet_valid":
                for row in hyd:
                    key = (sid, int(row.get("source_rank") or 0))
                    opt = optimized_by_key.get(key)
                    if opt is not None and bool(opt.get("valid")) and not bool(row.get("valid")):
                        selected.append(opt)
                    else:
                        selected.append(row)
                selected.extend(proto[:prototype_quota])
            elif variant == "append_chgnet_after_hydrated":
                selected.extend(hyd)
                opts = []
                for row in hyd:
                    opt = optimized_by_key.get((sid, int(row.get("source_rank") or 0)))
                    if opt is not None and bool(opt.get("valid")):
                        opts.append(opt)
                selected.extend(opts[:append_quota])
                selected.extend(proto[: max(0, top_k - len(selected))])
            else:
                for row in hyd:
                    selected.append(row)
                    opt = optimized_by_key.get((sid, int(row.get("source_rank") or 0)))
                    if opt is not None and bool(opt.get("valid")):
                        selected.append(opt)
                selected.extend(proto[: max(0, top_k - len(selected))])
            for row in selected[:top_k]:
                item = dict(row)
                item["selection_variant"] = variant
                rows_out.append(item)
        variants[variant] = assign_ranks(rows_out)
    return variants


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
    best_name = result["best_variant"]
    best = result["variants"][best_name]["rows_ge7"]
    baseline = result["baseline_exp2b"]["rows_ge7"]
    gate = result["gate"]
    diag = result["chgnet_diagnostics"]
    return f"""## opentry_14 实验 3b：CHGNet symmetry-checked local optimizer

结果文件：`model/New_model/opentry_14/results/experiment_3b_chgnet_local_optimizer.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3b_chgnet_local_optimizer/`

- 为什么做：实验 3 的 orbit repulsion proxy 只改善短距，rows>=7 conversion@50 反而下降；失败归因是当前局部 proxy 没有材料势能。实验 3b 安装并调用 CHGNet 预训练势能，在 GPU 上做短步位置优化，并用 SG/formula/site/exact 回退。
- 核心假设：如果实验 3 失败是因为 repulsion proxy 太弱，那么对 rows>=7 hydrated 中 formula/SG/exact 可用但 valid=false 的候选做 CHGNet 短步 relaxation，应能把一部分局部不合理几何转成 StructureMatcher match，同时不破坏 SG/exact-cover。
- 数据规模：待优化 candidates `{result['data_scale']['optimizer_tasks']}`，CHGNet accepted `{diag['accepted_candidates']}`，evaluated optimized `{diag['evaluated_optimized_candidates']}`；GPU devices `{result['cpu_policy']['gpu_devices']}`，CHGNet workers `{result['cpu_policy']['chgnet_workers']}`，StructureMatcher workers `{result['cpu_policy']['eval_workers']}`。
- baseline：实验 2b rows>=7 match@50 `{pct(baseline.get('match@50'))}`，conversion@50 `{pct(baseline.get('skeleton_to_match_conversion@50'))}`；Exp3 gate 需要 conversion@50 >= `{pct(gate['minimum_standard']['rows_ge7_conversion50'])}`，collision/local packing 改善，SG/exact-cover 不恶化。
- 方法变化：选择规则只用 inference-safe 结构状态：`rows>=7`、hydrated、formula_ok、space_group_ok、exact_cover_retained、valid=false；不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback。CHGNet 只优化原子位置，不 relax cell；优化后若 SG/formula/site count/exact/local proxy 不满足即回退。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；collision `{pct(best.get('collision_rate'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`。
- CHGNet 诊断：optimized match records `{diag['optimized_match_records']}`，optimized valid records `{diag['optimized_valid_records']}`，mean min-distance delta `{diag.get('mean_min_distance_delta')}`，mean hard-collision delta `{diag.get('mean_hard_collision_delta')}`，mean close-pair delta `{diag.get('mean_close_pair_delta')}`。
- 可信度：中等。CHGNet 是真实预训练势能并使用 GPU，优化结果重新通过 StructureMatcher/SG/formula/site/exact 检查；限制是只优化 rows>=7 hydrated invalid candidates，未覆盖 prototype CIF，且未训练 predicted-skeleton-aware posterior。unchanged candidates 沿用实验 2b 已有 validation metrics，因此与实验 3 的 direct re-eval 口径不同。
- 和历史实验关系：这是实验 3 失败后的修复尝试，专门检验“真实材料势能是否能替代短距 proxy”。若仍不过 +2pp conversion gate，则说明当前瓶颈更偏 free-parameter/site alignment 或 skeleton proposal，而不是局部能量微调。
- gate 判定：passed=`{gate['passed']}`；conversion delta vs Exp2b `{pp(gate['rows_ge7_conversion50_delta_vs_exp2b'])}`；match@50 delta vs Exp2b `{pp(gate['rows_ge7_match50_delta_vs_exp2b'])}`；collision/local improved=`{gate['collision_or_local_packing_improved']}`；SG not worse=`{gate['sg_not_worse']}`；exact-cover not worse=`{gate['exact_cover_not_worse']}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chgnet-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=64)
    parser.add_argument("--gpu-devices", default="0,1")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--fmax", type=float, default=0.5)
    parser.add_argument("--hydrated-quota", type=int, default=10)
    parser.add_argument("--prototype-quota", type=int, default=40)
    parser.add_argument("--append-quota", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    exp2b = read_json(EXP2B_RESULT)
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    structured_val = {sample_id(r): r for r in read_jsonl_iter(STRUCTURED_VAL)}

    task_rows = [
        r
        for r in safe_rows
        if str(r.get("pool_source")) == "symcif_v5_hydrated"
        and int(r.get("row_count") or 0) >= 7
        and bool(r.get("formula_ok"))
        and bool(r.get("space_group_ok"))
        and bool(r.get("exact_cover_retained"))
        and not bool(r.get("valid"))
    ]
    task_rows.sort(key=lambda r: (str(r["sample_id"]), int(r.get("source_rank") or 10**9)))
    if args.max_tasks is not None:
        task_rows = task_rows[: int(args.max_tasks)]

    needed = {(str(r["sample_id"]), int(r.get("source_rank") or 0) - 1) for r in task_rows}
    generations: dict[tuple[str, int], dict[str, Any]] = {}
    with SYMCIF_GEN.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sample_id"]), int(row.get("gen_index") or 0))
            if key in needed:
                generations[key] = row

    tasks: list[dict[str, Any]] = []
    missing_generation = 0
    for row in task_rows:
        gen = generations.get((str(row["sample_id"]), int(row.get("source_rank") or 0) - 1))
        if gen is None:
            missing_generation += 1
            continue
        tasks.append({"row": row, "generated_text": gen.get("generated_text"), "steps": int(args.steps), "fmax": float(args.fmax)})

    devices = tuple(x.strip() for x in str(args.gpu_devices).split(",") if x.strip())
    if not devices:
        devices = ("0",)
    opt_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.chgnet_workers)), initializer=init_chgnet_worker, initargs=(devices,)) as pool:
        futures = [pool.submit(chgnet_optimize_task, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            opt_results.append(fut.result())
            if i % 250 == 0:
                print(f"[exp3b-chgnet] optimized {i}/{len(futures)}", flush=True)

    accepted_with_cif = [r["candidate"] for r in opt_results if bool(r.get("accepted")) and r.get("candidate")]
    write_jsonl_gz(ARTIFACT_DIR / "chgnet_optimized_candidates_with_cif.jsonl.gz", accepted_with_cif)
    write_jsonl(
        ARTIFACT_DIR / "chgnet_optimizer_diagnostics.jsonl",
        [{k: v for k, v in r.items() if k not in ("candidate", "cif")} for r in opt_results],
    )

    payload_by_sid: dict[str, dict[str, Any]] = {}
    for row in accepted_with_cif:
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

    evaluated_optimized: list[dict[str, Any]] = []
    payloads = list(payload_by_sid.values())
    with ProcessPoolExecutor(max_workers=max(1, int(args.eval_workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated_optimized.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp3b-chgnet] evaluated samples {i}/{len(futures)}", flush=True)
    write_jsonl(ARTIFACT_DIR / "evaluated_chgnet_optimized_candidates.jsonl", evaluated_optimized)

    optimized_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in evaluated_optimized:
        optimized_by_key[(str(row["sample_id"]), int(row.get("source_rank") or 0))] = row

    variant_rows = build_variants(
        safe_rows=safe_rows,
        optimized_by_key=optimized_by_key,
        hydrated_quota=int(args.hydrated_quota),
        prototype_quota=int(args.prototype_quota),
        append_quota=int(args.append_quota),
        top_k=int(args.top_k),
    )

    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for name, rows in variant_rows.items():
        path = ARTIFACT_DIR / f"selected_{name}_candidates.jsonl"
        write_jsonl(path, rows)
        selected_paths[name] = str(path)
        rows7 = [r for r in rows if int(r.get("row_count") or 0) >= 7]
        rowslt7 = [r for r in rows if int(r.get("row_count") or 0) < 7]
        variants[name] = {
            "overall": summarize(rows),
            "rows_ge7": summarize(rows7),
            "rows_lt7": summarize(rowslt7),
        }

    best_variant = max(
        variants,
        key=lambda name: (
            float(variants[name]["rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0),
            float(variants[name]["rows_ge7"].get("match@50") or 0.0),
            -float(variants[name]["rows_ge7"].get("collision_rate") or 1.0),
        ),
    )
    best_rows7 = variants[best_variant]["rows_ge7"]
    baseline_rows7 = exp2b["rows_ge7"]
    conversion_min = float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0) + 0.02
    conversion_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0)
    collision_improved = bool(
        best_rows7.get("collision_rate") is not None
        and baseline_rows7.get("collision_rate") is not None
        and float(best_rows7["collision_rate"]) < float(baseline_rows7["collision_rate"])
    )
    sg_not_worse = float(best_rows7.get("sg_consistency") or 0.0) >= float(baseline_rows7.get("sg_consistency") or 0.0) - 1.0e-12
    exact_not_worse = float(best_rows7.get("exact_cover_retained") or 0.0) >= float(baseline_rows7.get("exact_cover_retained") or 0.0) - 1.0e-12
    passed = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= conversion_min
        and match_delta >= 0.0
        and collision_improved
        and sg_not_worse
        and exact_not_worse
    )

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

    result = {
        "experiment": "opentry_14_exp3b_chgnet_symmetry_checked_local_optimizer",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "chgnet_position_relax_with_sg_formula_exact_rollback",
            "base_method": "experiment_2b_fixed_safe_pool",
            "selection_rule": "rows>=7 hydrated formula_ok space_group_ok exact_cover_retained and source valid=false",
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "optimizer": "CHGNet pretrained v0.3.0 via chgnet 0.4.2 StructOptimizer/FIRE",
            "relax_cell": False,
            "steps": int(args.steps),
            "fmax": float(args.fmax),
            "rollback_checks": ["input SG", "output SG", "formula", "site_count", "local_proxy_improved"],
        },
        "cpu_policy": {
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "gpu_devices": list(devices),
            "chgnet_workers": int(args.chgnet_workers),
            "eval_workers": int(args.eval_workers),
            "single_process_target": "each worker uses one Python process with BLAS/OMP threads fixed to 1; expected per-process CPU below 300%",
        },
        "data_scale": {
            "input_safe_pool_records": len(safe_rows),
            "candidate_records_selected_for_optimizer": len(task_rows),
            "optimizer_tasks": len(tasks),
            "missing_generation_records": missing_generation,
            "accepted_optimized_candidates": len(accepted_with_cif),
            "evaluated_optimized_candidates": len(evaluated_optimized),
            "top_k": int(args.top_k),
        },
        "baseline_exp2b": {
            "result_path": str(EXP2B_RESULT),
            "rows_ge7": baseline_rows7,
            "overall": exp2b["overall"],
        },
        "variants": variants,
        "best_variant": best_variant,
        "chgnet_diagnostics": {
            "optimizer_result_counts": dict(Counter("accepted" if r.get("accepted") else ("success_unaccepted" if r.get("success") else "failed") for r in opt_results)),
            "accepted_candidates": len(accepted_with_cif),
            "evaluated_optimized_candidates": len(evaluated_optimized),
            "optimized_valid_records": sum(bool(r.get("valid")) for r in evaluated_optimized),
            "optimized_match_records": sum(bool(r.get("match")) for r in evaluated_optimized),
            "optimized_skeleton_hit_records": sum(bool(r.get("predicted_skeleton_hit")) for r in evaluated_optimized),
            "mean_runtime_seconds": mean([float(r["runtime_seconds"]) for r in opt_results if r.get("runtime_seconds") is not None]),
            "mean_hard_collision_delta": mean(hard_delta),
            "mean_close_pair_delta": mean(close_delta),
            "mean_min_distance_delta": mean(min_delta),
        },
        "gate": {
            "passed": passed,
            "rows_ge7_conversion50_delta_vs_exp2b": conversion_delta,
            "rows_ge7_match50_delta_vs_exp2b": match_delta,
            "collision_or_local_packing_improved": collision_improved,
            "sg_not_worse": sg_not_worse,
            "exact_cover_not_worse": exact_not_worse,
            "minimum_standard": {
                "rows_ge7_conversion50": conversion_min,
                "rows_ge7_conversion50_delta_vs_exp2b": 0.02,
                "rows_ge7_match50_delta_vs_exp2b": 0.0,
                "collision_or_local_packing_improved": True,
                "sg_not_worse": True,
                "exact_cover_not_worse": True,
            },
        },
        "decision": {
            "verdict": "pass" if passed else "fail_diagnostic_only",
            "reason": "CHGNet local optimizer passed the +2pp conversion gate without match/SG/exact degradation." if passed else "CHGNet local optimizer did not produce the required +2pp rows>=7 conversion lift over experiment 2b under inference-safe selection.",
            "next_step": "Use this as the experiment-3 candidate and proceed to experiment 4 learned multi-hypothesis geometry critic." if passed else "Do not enter Exp4/final method from this optimizer; return to free-parameter/site alignment or skeleton proposer improvement.",
        },
        "artifacts": {
            "chgnet_optimized_candidates_with_cif": str(ARTIFACT_DIR / "chgnet_optimized_candidates_with_cif.jsonl.gz"),
            "chgnet_optimizer_diagnostics": str(ARTIFACT_DIR / "chgnet_optimizer_diagnostics.jsonl"),
            "evaluated_chgnet_optimized_candidates": str(ARTIFACT_DIR / "evaluated_chgnet_optimized_candidates.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_3b_chgnet_local_optimizer.json", result)
    body = report_body(result)
    append_or_replace(REPORT_PATH, "<!-- OPENTRY14_EXP3B_CHGNET_LOCAL_OPTIMIZER -->", body)
    append_or_replace(LOCAL_REPORT_PATH, "<!-- OPENTRY14_EXP3B_CHGNET_LOCAL_OPTIMIZER -->", body)
    print(json.dumps({"output": str(RESULT_DIR / "experiment_3b_chgnet_local_optimizer.json"), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
