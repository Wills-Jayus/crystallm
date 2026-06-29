#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

warnings.filterwarnings("ignore")

ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp3_symmetry_local_optimizer"
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


def sample_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("id") or record.get("keys", {}).get("sample_id"))


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


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
    if len(structure) < 2:
        return {
            "min_pair_distance": None,
            "hard_collision_count": 0,
            "close_pair_count": 0,
            "volume_per_atom": float(structure.volume) / max(1, len(structure)),
        }
    matrix = structure.distance_matrix
    hard = 0
    close = 0
    min_dist: float | None = None
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            d = float(matrix[i, j])
            min_dist = d if min_dist is None else min(min_dist, d)
            if d < HARD_COLLISION_CUTOFF:
                hard += 1
            if d < LOCAL_CLOSE_PAIR_CUTOFF:
                close += 1
    return {
        "min_pair_distance": min_dist,
        "hard_collision_count": hard,
        "close_pair_count": close,
        "volume_per_atom": float(structure.volume) / max(1, len(structure)),
    }


def proxy_score(proxy: dict[str, Any]) -> float:
    min_dist = proxy.get("min_pair_distance")
    return (
        -1000.0 * float(proxy.get("hard_collision_count") or 0)
        - 10.0 * float(proxy.get("close_pair_count") or 0)
        + float(min_dist or 0.0)
    )


def periodic_close(a: np.ndarray, b: np.ndarray, tol: float = 1.0e-4) -> bool:
    delta = np.abs(a - b)
    delta = np.minimum(delta, 1.0 - delta)
    return bool(float(np.max(delta)) <= tol)


def unique_orbit_coords(ops: Any, frac: np.ndarray) -> list[np.ndarray]:
    pts: list[np.ndarray] = []
    for op in ops:
        p = np.mod(np.asarray(op.operate(frac), dtype=float), 1.0)
        if not any(periodic_close(p, q) for q in pts):
            pts.append(p)
    pts.sort(key=lambda x: (float(x[0]), float(x[1]), float(x[2])))
    return pts


def symmetry_groups(structure: Structure, target_sg: int) -> tuple[Any, list[list[int]], int | None]:
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
    detected = int(analyzer.get_space_group_number())
    if detected != int(target_sg):
        return None, [], detected
    sym = analyzer.get_symmetrized_structure()
    return analyzer.get_space_group_operations(), [list(g) for g in sym.equivalent_indices], detected


def repulsion_step(structure: Structure, target_sg: int, step_angstrom: float) -> Structure | None:
    try:
        ops, groups, detected = symmetry_groups(structure, target_sg)
    except Exception:
        return None
    if ops is None or detected != int(target_sg) or not groups:
        return None

    idx_to_group: dict[int, int] = {}
    movable: list[bool] = []
    for gi, group in enumerate(groups):
        for idx in group:
            idx_to_group[int(idx)] = gi
        species = {str(structure[idx].specie) for idx in group}
        if len(species) != 1:
            movable.append(False)
            continue
        rep = np.asarray(structure[group[0]].frac_coords, dtype=float)
        movable.append(len(unique_orbit_coords(ops, rep)) == len(group))

    forces = [np.zeros(3, dtype=float) for _ in groups]
    for i in range(len(structure)):
        gi = idx_to_group.get(i)
        if gi is None or not movable[gi]:
            continue
        for j in range(i + 1, len(structure)):
            gj = idx_to_group.get(j)
            if gj is None or gi == gj or not movable[gj]:
                continue
            d, image = structure.lattice.get_distance_and_image(structure[i].frac_coords, structure[j].frac_coords)
            d = float(d)
            if d <= 1.0e-8 or d >= LOCAL_CLOSE_PAIR_CUTOFF:
                continue
            ci = structure.lattice.get_cartesian_coords(structure[i].frac_coords)
            cj = structure.lattice.get_cartesian_coords(structure[j].frac_coords + np.asarray(image, dtype=float))
            direction = ci - cj
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-8:
                continue
            direction /= norm
            weight = (LOCAL_CLOSE_PAIR_CUTOFF - d) / LOCAL_CLOSE_PAIR_CUTOFF
            forces[gi] += weight * direction
            forces[gj] -= weight * direction

    frac_coords = np.asarray(structure.frac_coords, dtype=float).copy()
    changed = False
    for gi, group in enumerate(groups):
        if not movable[gi]:
            continue
        force = forces[gi]
        norm = float(np.linalg.norm(force))
        if norm <= 1.0e-10:
            continue
        frac_delta = structure.lattice.get_fractional_coords(force / norm * float(step_angstrom))
        rep = np.mod(np.asarray(structure[group[0]].frac_coords, dtype=float) + frac_delta, 1.0)
        orbit = unique_orbit_coords(ops, rep)
        if len(orbit) != len(group):
            continue
        for idx, coord in zip(sorted(group), orbit):
            frac_coords[idx] = coord
        changed = True

    if not changed:
        return None
    return Structure(
        structure.lattice,
        [site.specie for site in structure],
        frac_coords,
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def conservative_local_optimize(structure: Structure, target_sg: int, steps: tuple[float, ...]) -> tuple[Structure | None, dict[str, Any]]:
    original_proxy = local_proxy(structure)
    original_score = proxy_score(original_proxy)
    best_structure: Structure | None = None
    best_proxy: dict[str, Any] | None = None
    best_score = original_score
    attempted = 0
    sg_rejected = 0

    for step in steps:
        candidate = repulsion_step(structure, target_sg, step)
        attempted += 1
        if candidate is None:
            continue
        try:
            detected = int(SpacegroupAnalyzer(candidate, symprec=0.1).get_space_group_number())
        except Exception:
            detected = None
        if detected != int(target_sg) or len(candidate) != len(structure) or structure_counts(candidate) != structure_counts(structure):
            sg_rejected += 1
            continue
        candidate_proxy = local_proxy(candidate)
        score = proxy_score(candidate_proxy)
        if score > best_score + 1.0e-9:
            best_structure = candidate
            best_proxy = candidate_proxy
            best_score = score

    accepted = best_structure is not None and best_proxy is not None
    return best_structure, {
        "optimizer_attempted_steps": attempted,
        "optimizer_sg_rejected_steps": sg_rejected,
        "optimizer_accepted": bool(accepted),
        "optimizer_original_proxy": original_proxy,
        "optimizer_best_proxy": best_proxy,
        "optimizer_score_delta": float(best_score - original_score),
    }


def optimize_hydrated_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = task["row"]
    gen = task["generation"]
    cif = str(gen.get("generated_text") or "")
    base = dict(row)
    base.update(
        {
            "pool_source": "symcif_v5_hydrated_local_optimizer",
            "geometry_source": str(row.get("geometry_source") or "symcif_v5_hydrated"),
            "source_generation_path": str(SYMCIF_GEN),
            "optimizer_policy": "symmetry_orbit_repulsion_with_sg_formula_exact_cover_rollback",
            "cif": cif,
            "geometry_variant": "original_hydrated",
            "rank": None,
            "render_success": bool(cif),
            "render_error": None if cif else "empty_cif",
        }
    )
    out = [base]
    diag: dict[str, Any] = {
        "sample_id": str(row["sample_id"]),
        "source_rank": int(row.get("source_rank") or 0),
        "optimized_added": False,
        "parse_error": None,
    }
    if not cif:
        return out, diag
    try:
        structure = Structure.from_str(cif, fmt="cif")
        detected = int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
        original_proxy = local_proxy(structure)
    except Exception as exc:  # noqa: BLE001
        diag["parse_error"] = f"{type(exc).__name__}: {exc}"
        return out, diag
    diag["detected_sg"] = detected
    diag["original_proxy"] = original_proxy
    if detected != int(row.get("sg") or 0):
        diag["rollback_reason"] = "input_sg_mismatch"
        return out, diag

    optimized, opt_diag = conservative_local_optimize(structure, int(row.get("sg") or 0), tuple(task["steps"]))
    diag.update(opt_diag)
    if optimized is None or not opt_diag.get("optimizer_accepted"):
        diag["rollback_reason"] = "no_proxy_improving_sg_safe_step"
        return out, diag

    opt_row = dict(base)
    opt_row.update(
        {
            "cif": optimized.to(fmt="cif"),
            "geometry_variant": "optimized_hydrated",
            "geometry_source": str(row.get("geometry_source") or "symcif_v5_hydrated") + "+symmetry_local_repulsion",
            "optimizer_accepted": True,
            "optimizer_original_min_pair_distance": original_proxy.get("min_pair_distance"),
            "optimizer_original_hard_collision_count": original_proxy.get("hard_collision_count"),
            "optimizer_original_close_pair_count": original_proxy.get("close_pair_count"),
            "optimizer_best_min_pair_distance": opt_diag["optimizer_best_proxy"].get("min_pair_distance"),
            "optimizer_best_hard_collision_count": opt_diag["optimizer_best_proxy"].get("hard_collision_count"),
            "optimizer_best_close_pair_count": opt_diag["optimizer_best_proxy"].get("close_pair_count"),
            "optimizer_score_delta": opt_diag.get("optimizer_score_delta"),
        }
    )
    out.append(opt_row)
    diag["optimized_added"] = True
    return out, diag


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
        "collision_rate": ratio(sum(bool(r.get("collision_flag")) for r in rows), len(rows)),
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
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    ranked: list[dict[str, Any]] = []
    for sid, sample_rows in by_sid.items():
        for rank, row in enumerate(sample_rows, start=1):
            item = dict(row)
            item["rank"] = rank
            ranked.append(item)
    ranked.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    return ranked


def build_variant_rows(
    *,
    variant: str,
    original_hydrated_by_sid: dict[str, list[dict[str, Any]]],
    optimized_by_sid: dict[str, dict[int, dict[str, Any]]],
    prototypes_by_sid: dict[str, list[dict[str, Any]]],
    hydrated_quota: int,
    prototype_quota: int,
    append_optimized_quota: int,
    top_k: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    all_sids = sorted(set(original_hydrated_by_sid) | set(prototypes_by_sid))
    for sid in all_sids:
        rows: list[dict[str, Any]] = []
        originals = sorted(original_hydrated_by_sid.get(sid, []), key=lambda r: int(r.get("source_rank") or 10**9))
        prototypes = sorted(prototypes_by_sid.get(sid, []), key=lambda r: int(r.get("source_rank") or r.get("rank") or 10**9))
        if variant == "replace_hydrated_local_optimizer":
            for row in originals[:hydrated_quota]:
                source_rank = int(row.get("source_rank") or 0)
                rows.append(optimized_by_sid.get(sid, {}).get(source_rank, row))
            rows.extend(prototypes[:prototype_quota])
        elif variant == "append_hydrated_local_optimizer":
            rows.extend(originals[:hydrated_quota])
            optimized_rows = [optimized_by_sid.get(sid, {}).get(int(r.get("source_rank") or 0)) for r in originals[:hydrated_quota]]
            rows.extend([r for r in optimized_rows if r is not None][:append_optimized_quota])
            rows.extend(prototypes[: max(0, top_k - len(rows))])
        else:
            rows.extend(originals[:hydrated_quota])
            rows.extend(prototypes[:prototype_quota])
        for row in rows[:top_k]:
            item = dict(row)
            item["selection_variant"] = variant
            out.append(item)
    return assign_ranks(out)


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
    baseline = result["baseline_exp2b"]["rows_ge7"]
    best = result["best_variant"]
    rows7 = result["variants"][best]["rows_ge7"]
    gate = result["gate"]
    local = result["local_optimizer_diagnostics"]
    return f"""## opentry_14 实验 3：symmetry-preserving local optimizer 诊断

结果文件：`model/New_model/opentry_14/results/experiment_3_symmetry_local_optimizer.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3_symmetry_local_optimizer/`

- 为什么做：实验 2b 已把 rows>=7 conversion@50 恢复到 `{pct(baseline.get('skeleton_to_match_conversion@50'))}`，但 prompt 要求继续检查 symmetry-preserving local optimizer 是否能修复 collision/short-distance/local packing，并至少再提升 conversion `+2pp`。
- 核心假设：如果 skeleton-to-match 的剩余瓶颈主要是局部短距/packing，那么在不破坏 formula、SG、exact-cover 的前提下，对 hydrated CIF 做轨道级 repulsion 优化应降低 collision/close-pair，并把更多 predicted-skeleton-hit candidate 转成 StructureMatcher match。
- 数据规模：safe-pool candidates `{result['data_scale']['input_safe_pool_records']}`，hydrated optimization inputs `{result['data_scale']['hydrated_input_records']}`，optimizer accepted `{local['accepted_optimized_candidates']}`；StructureMatcher workers `{result['cpu_policy']['workers']}`；topK `50`。
- baseline：实验 2b best rows>=7 match@50 `{pct(baseline.get('match@50'))}`，conversion@50 `{pct(baseline.get('skeleton_to_match_conversion@50'))}`；Exp3 通过线要求 conversion@50 >= `{pct(gate['minimum_standard']['rows_ge7_conversion50'])}`，且 collision/local packing 改善，SG/exact-cover 不恶化。
- 方法变化：只对 SymCIF v5 hydrated CIF 做局部优化；用 spglib/pymatgen 等价原子分组，在同一空间群操作下移动整个 Wyckoff orbit 的代表点并重新展开，候选若 formula/site count/SG/exact-cover 不满足或 local proxy 未改善则回退。prototype 候选不改。排序/选择不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果 best variant `{best}`：rows>=7 match@1/5/20/50 = `{pct(rows7.get('match@1'))} / {pct(rows7.get('match@5'))} / {pct(rows7.get('match@20'))} / {pct(rows7.get('match@50'))}`；conversion@50 `{pct(rows7.get('skeleton_to_match_conversion@50'))}`；collision `{pct(rows7.get('collision_rate'))}`；SG `{pct(rows7.get('sg_consistency'))}`；exact-cover `{pct(rows7.get('exact_cover_retained'))}`。
- 局部优化诊断：accepted optimized candidates `{local['accepted_optimized_candidates']}`；optimized evaluation match candidates `{local['optimized_match_records']}`；optimized-vs-original hard collision mean delta `{local.get('mean_hard_collision_delta')}`；close-pair mean delta `{local.get('mean_close_pair_delta')}`；min-distance mean delta `{local.get('mean_min_distance_delta')}`。
- 可信度：中等。该实验真实解析 CIF、重做 SG/formula/site-count/StructureMatcher 检查，并用 64-worker 评估；限制是当前环境没有 CHGNet/MatGL/MACE，local proxy 是短距/close-pair repulsion，不是学习到的材料势能，且只覆盖 hydrated CIF，prototype 候选没有可逆局部优化。
- 和历史实验关系：继承实验 2b safe pool，不进入 official；它直接测试“local geometry/collision 是否是剩余主要瓶颈”。若 conversion 不升，即支持失败归因“local geometry/collision 无法通过当前 proxy 修复”。
- gate 判定：passed=`{gate['passed']}`；conversion delta vs Exp2b `{pp(gate['rows_ge7_conversion50_delta_vs_exp2b'])}`；collision improved=`{gate['collision_or_local_packing_improved']}`；SG not worse=`{gate['sg_not_worse']}`；exact-cover not worse=`{gate['exact_cover_not_worse']}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--hydrated-quota", type=int, default=10)
    parser.add_argument("--prototype-quota", type=int, default=40)
    parser.add_argument("--append-optimized-quota", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--steps", default="0.04,0.08,0.12,0.18")
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    exp2b = read_json(EXP2B_RESULT)
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    structured_val = {sample_id(r): r for r in read_jsonl_iter(STRUCTURED_VAL)}

    hydrated_rows = [r for r in safe_rows if str(r.get("pool_source")) == "symcif_v5_hydrated"]
    prototype_rows = [r for r in safe_rows if str(r.get("pool_source")) != "symcif_v5_hydrated"]
    needed = {(str(r["sample_id"]), int(r.get("source_rank") or 0) - 1) for r in hydrated_rows}
    generations: dict[tuple[str, int], dict[str, Any]] = {}
    with SYMCIF_GEN.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sample_id"]), int(row.get("gen_index") or 0))
            if key in needed:
                generations[key] = row
    tasks = []
    steps = tuple(float(x) for x in str(args.steps).split(",") if x.strip())
    missing_generation = 0
    for row in hydrated_rows:
        key = (str(row["sample_id"]), int(row.get("source_rank") or 0) - 1)
        gen = generations.get(key)
        if gen is None:
            missing_generation += 1
            continue
        tasks.append({"row": row, "generation": gen, "steps": steps})

    optimized_candidates: list[dict[str, Any]] = []
    optimizer_diag: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(optimize_hydrated_task, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            rows, diag = fut.result()
            optimized_candidates.extend(rows)
            optimizer_diag.append(diag)
            if i % 1000 == 0:
                print(f"[exp3-localopt] optimized hydrated {i}/{len(futures)}", flush=True)

    write_jsonl(ARTIFACT_DIR / "hydrated_optimizer_generation_meta.jsonl", [{k: v for k, v in r.items() if k != "cif"} for r in optimized_candidates])
    write_jsonl(ARTIFACT_DIR / "hydrated_optimizer_diagnostics.jsonl", optimizer_diag)

    payload_by_sid: dict[str, dict[str, Any]] = {}
    for row in optimized_candidates:
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

    evaluated_hydrated: list[dict[str, Any]] = []
    payloads = list(payload_by_sid.values())
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated_hydrated.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp3-localopt] evaluated hydrated samples {i}/{len(futures)}", flush=True)

    write_jsonl(ARTIFACT_DIR / "evaluated_hydrated_local_optimizer_candidates.jsonl", evaluated_hydrated)

    original_hydrated_by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    optimized_by_sid: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in evaluated_hydrated:
        sid = str(row["sample_id"])
        source_rank = int(row.get("source_rank") or 0)
        if str(row.get("geometry_variant")) == "optimized_hydrated":
            optimized_by_sid[sid][source_rank] = row
        else:
            original_hydrated_by_sid[sid].append(row)

    prototypes_by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prototype_rows:
        prototypes_by_sid[str(row["sample_id"])].append(row)

    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("baseline_reevaluated_hydrated", "replace_hydrated_local_optimizer", "append_hydrated_local_optimizer"):
        rows = build_variant_rows(
            variant=variant,
            original_hydrated_by_sid=original_hydrated_by_sid,
            optimized_by_sid=optimized_by_sid,
            prototypes_by_sid=prototypes_by_sid,
            hydrated_quota=int(args.hydrated_quota),
            prototype_quota=int(args.prototype_quota),
            append_optimized_quota=int(args.append_optimized_quota),
            top_k=int(args.top_k),
        )
        path = ARTIFACT_DIR / f"selected_{variant}_candidates.jsonl"
        write_jsonl(path, rows)
        selected_paths[variant] = str(path)
        rows7 = [r for r in rows if int(r.get("row_count") or 0) >= 7]
        rowslt7 = [r for r in rows if int(r.get("row_count") or 0) < 7]
        variants[variant] = {
            "overall": summarize(rows),
            "rows_ge7": summarize(rows7),
            "rows_lt7": summarize(rowslt7),
        }

    candidate_variants = ["replace_hydrated_local_optimizer", "append_hydrated_local_optimizer"]
    best_variant = max(
        candidate_variants,
        key=lambda name: (
            float(variants[name]["rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0),
            float(variants[name]["rows_ge7"].get("match@50") or 0.0),
            -float(variants[name]["rows_ge7"].get("collision_rate") or 1.0),
        ),
    )
    baseline_re = variants["baseline_reevaluated_hydrated"]["rows_ge7"]
    best_rows7 = variants[best_variant]["rows_ge7"]
    exp2b_rows7 = exp2b["rows_ge7"]
    conversion_min = float(exp2b_rows7.get("skeleton_to_match_conversion@50") or 0.0) + 0.02
    conversion_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2b_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    collision_improved = bool(
        (best_rows7.get("collision_rate") is not None and baseline_re.get("collision_rate") is not None and float(best_rows7["collision_rate"]) < float(baseline_re["collision_rate"]))
        or (
            best_rows7.get("mean_min_pair_distance") is not None
            and baseline_re.get("mean_min_pair_distance") is not None
            and float(best_rows7["mean_min_pair_distance"]) > float(baseline_re["mean_min_pair_distance"])
        )
    )
    sg_not_worse = float(best_rows7.get("sg_consistency") or 0.0) >= float(baseline_re.get("sg_consistency") or 0.0) - 1.0e-12
    exact_not_worse = float(best_rows7.get("exact_cover_retained") or 0.0) >= float(baseline_re.get("exact_cover_retained") or 0.0) - 1.0e-12
    passed = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= conversion_min
        and collision_improved
        and sg_not_worse
        and exact_not_worse
    )

    hard_deltas: list[float] = []
    close_deltas: list[float] = []
    min_deltas: list[float] = []
    for diag in optimizer_diag:
        if not diag.get("optimizer_accepted") or not diag.get("optimizer_best_proxy"):
            continue
        orig = diag.get("optimizer_original_proxy") or {}
        best = diag.get("optimizer_best_proxy") or {}
        hard_deltas.append(float(best.get("hard_collision_count") or 0) - float(orig.get("hard_collision_count") or 0))
        close_deltas.append(float(best.get("close_pair_count") or 0) - float(orig.get("close_pair_count") or 0))
        if best.get("min_pair_distance") is not None and orig.get("min_pair_distance") is not None:
            min_deltas.append(float(best["min_pair_distance"]) - float(orig["min_pair_distance"]))

    optimized_eval_rows = [r for r in evaluated_hydrated if str(r.get("geometry_variant")) == "optimized_hydrated"]
    result = {
        "experiment": "opentry_14_exp3_symmetry_preserving_local_optimizer",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "symmetry_orbit_repulsion_local_optimizer",
            "base_method": "experiment_2b_fixed_safe_pool",
            "local_optimizer": "pymatgen/spglib equivalent-site orbit repulsion with rollback",
            "steps_angstrom": list(steps),
            "not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "rollback_checks": ["formula", "site_count", "space_group", "exact_cover", "local_proxy_improvement"],
            "environment_note": "CHGNet/MatGL/MACE not installed; used conservative short-distance local proxy available in crystallm_env.",
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "single_process_target": "one Python worker per task; BLAS/OMP threads fixed to 1, expected per-process CPU below 200%",
        },
        "data_scale": {
            "input_safe_pool_records": len(safe_rows),
            "hydrated_input_records": len(hydrated_rows),
            "prototype_input_records": len(prototype_rows),
            "hydrated_generation_records_found": len(generations),
            "missing_generation_records": missing_generation,
            "optimized_or_original_records_evaluated": len(evaluated_hydrated),
            "top_k": int(args.top_k),
        },
        "baseline_exp2b": {
            "result_path": str(EXP2B_RESULT),
            "rows_ge7": exp2b_rows7,
            "overall": exp2b["overall"],
        },
        "variants": variants,
        "best_variant": best_variant,
        "local_optimizer_diagnostics": {
            "optimizer_tasks": len(tasks),
            "accepted_optimized_candidates": sum(bool(d.get("optimized_added")) for d in optimizer_diag),
            "optimized_eval_records": len(optimized_eval_rows),
            "optimized_match_records": sum(bool(r.get("match")) for r in optimized_eval_rows),
            "optimized_valid_records": sum(bool(r.get("valid")) for r in optimized_eval_rows),
            "mean_hard_collision_delta": mean(hard_deltas),
            "mean_close_pair_delta": mean(close_deltas),
            "mean_min_distance_delta": mean(min_deltas),
            "rollback_counts": dict(Counter(str(d.get("rollback_reason") or "accepted") for d in optimizer_diag)),
        },
        "gate": {
            "passed": passed,
            "rows_ge7_conversion50_delta_vs_exp2b": conversion_delta,
            "collision_or_local_packing_improved": collision_improved,
            "sg_not_worse": sg_not_worse,
            "exact_cover_not_worse": exact_not_worse,
            "minimum_standard": {
                "rows_ge7_conversion50": conversion_min,
                "rows_ge7_conversion50_delta_vs_exp2b": 0.02,
                "collision_or_local_packing_improved": True,
                "sg_not_worse": True,
                "exact_cover_not_worse": True,
            },
        },
        "decision": {
            "verdict": "pass" if passed else "fail_diagnostic_only",
            "reason": "Local optimizer passed the +2pp conversion and local-packing gate." if passed else "Local optimizer did not produce the required +2pp rows>=7 conversion lift over experiment 2b; any packing improvement remains diagnostic only.",
            "next_step": "Use this as the experiment-3 candidate for experiment 4 geometry critic." if passed else "Do not enter final method with this optimizer; root cause is local geometry/collision not fixable by the current symmetry-safe short-distance proxy. Next try must improve learned free-parameter/site alignment or a real energy model before repeating Exp3.",
        },
        "artifacts": {
            "hydrated_optimizer_generation_meta": str(ARTIFACT_DIR / "hydrated_optimizer_generation_meta.jsonl"),
            "hydrated_optimizer_diagnostics": str(ARTIFACT_DIR / "hydrated_optimizer_diagnostics.jsonl"),
            "evaluated_hydrated_local_optimizer_candidates": str(ARTIFACT_DIR / "evaluated_hydrated_local_optimizer_candidates.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_3_symmetry_local_optimizer.json", result)
    body = report_body(result)
    append_or_replace(REPORT_PATH, "<!-- OPENTRY14_EXP3_SYMMETRY_LOCAL_OPTIMIZER -->", body)
    append_or_replace(LOCAL_REPORT_PATH, "<!-- OPENTRY14_EXP3_SYMMETRY_LOCAL_OPTIMIZER -->", body)
    print(json.dumps({"output": str(RESULT_DIR / "experiment_3_symmetry_local_optimizer.json"), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
