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
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2h_train_param_prior_geometry_posterior"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import (  # noqa: E402
    formula_counts,
    read_jsonl,
    sample_id,
    source_skeleton_rows,
)
from run_exp4_rows_ge7_multi_geometry_proposal import (  # noqa: E402
    assign_structural_ranks,
    eval_sample,
    render_candidate,
    summarize,
)
from run_exp2f_permutation_aware_alignment import (  # noqa: E402
    by_sid_ranked,
    enumerate_exact_cover_assignments,
    sample_sets,
    split_safe_pool,
)
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
EXP3_PROPOSALS = OP13 / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP2B_RESULT = RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json"
EXP2B_EVAL = OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool" / "evaluated_safe_pool_candidates.jsonl"
EXP2D_RESULT = RESULT_DIR / "experiment_2d_site_assignment_multi_hypothesis.json"
EXP2D_SITE_EVAL = OUT_DIR / "artifacts" / "exp2d_site_assignment_multi_hypothesis" / "evaluated_site_assignment_candidates.jsonl"
EXP2G_RESULT = RESULT_DIR / "experiment_2g_candidate_headroom_audit.json"
BUDGETS = (1, 5, 20, 50)
_ROW_POOL_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


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


def formula_l1(a: dict[str, Any], b: dict[str, Any]) -> float:
    aa = normalize_formula_counts(a)
    bb = normalize_formula_counts(b)
    keys = set(aa) | set(bb)
    return sum(abs(int(aa.get(k, 0)) - int(bb.get(k, 0))) for k in keys) / float(max(1, sum(aa.values()) + sum(bb.values())))


def circular_mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sum(math.sin(2.0 * math.pi * (float(v) % 1.0)) for v in vals)
    c = sum(math.cos(2.0 * math.pi * (float(v) % 1.0)) for v in vals)
    return (math.atan2(s, c) / (2.0 * math.pi)) % 1.0


def circular_blend(a: float, b: float, alpha: float) -> float:
    a = float(a) % 1.0
    b = float(b) % 1.0
    delta = ((b - a + 0.5) % 1.0) - 0.5
    return (a + alpha * delta) % 1.0


def deterministic_params(engine: OrbitEngine, orbit_id: str, row_index: int) -> dict[str, float]:
    orbit = engine.get_orbit_by_id(str(orbit_id))
    base = {"x": 0.173, "y": 0.271, "z": 0.389}
    return {str(symbol): (base.get(str(symbol), 0.173) + 0.037 * row_index) % 1.0 for symbol in orbit.free_symbols}


def lattice_from_record(record: dict[str, Any]) -> dict[str, float]:
    return {k: float(record["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}


def lattice_median(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"a": 8.0, "b": 8.0, "c": 8.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    return {
        k: float(statistics.median(float(r["lattice"][k]) for r in records if r.get("lattice") and k in r["lattice"]))
        for k in ("a", "b", "c", "alpha", "beta", "gamma")
    }


def build_lattice_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_sg_atom: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        sg = int(rec["sg"])
        atom_count = int(rec.get("atom_count") or sum(int(v) for v in normalize_formula_counts(rec["formula_counts"]).values()))
        by_sg[sg].append(rec)
        by_sg_atom[(sg, atom_count)].append(rec)
    return {
        "by_sg": dict(by_sg),
        "by_sg_atom": dict(by_sg_atom),
        "global_median": lattice_median(records),
        "sg_median": {sg: lattice_median(group) for sg, group in by_sg.items()},
        "sg_atom_median": {key: lattice_median(group) for key, group in by_sg_atom.items()},
    }


def nearest_records(records: list[dict[str, Any]], target_counts: dict[str, int], limit: int) -> list[dict[str, Any]]:
    scored = []
    target_atoms = sum(int(v) for v in target_counts.values())
    target_elems = set(target_counts)
    for rec in records:
        counts = normalize_formula_counts(rec["formula_counts"])
        elems = set(counts)
        jaccard = len(target_elems & elems) / max(1, len(target_elems | elems))
        atoms = sum(int(v) for v in counts.values())
        score = formula_l1(target_counts, counts) + 0.02 * abs(target_atoms - atoms) - 0.05 * jaccard
        scored.append((score, str(rec["sample_id"]), rec))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [rec for _, _, rec in scored[: max(1, int(limit))]]


def build_param_bank(engine: OrbitEngine, train_records: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    orbit_only: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    sg_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    total_rows = 0
    for rec in train_records:
        sg = int(rec["sg"])
        counts = normalize_formula_counts(rec["formula_counts"])
        for row in rec.get("wa_table") or []:
            total_rows += 1
            orbit_id = str(row.get("orbit_id") or "")
            try:
                orbit = engine.get_orbit_by_id(orbit_id)
            except Exception:
                skipped["missing_orbit"] += 1
                continue
            expected = {str(s) for s in orbit.free_symbols}
            params = {str(k): float(v) % 1.0 for k, v in dict(row.get("free_params") or {}).items()}
            if set(params) != expected:
                if expected:
                    skipped["free_param_symbol_mismatch"] += 1
                    continue
                params = {}
            item = {
                "sample_id": str(rec["sample_id"]),
                "sg": sg,
                "orbit_id": orbit_id,
                "element": str(row.get("element") or ""),
                "multiplicity": int(row.get("multiplicity") or orbit.multiplicity),
                "params": params,
                "formula_counts": counts,
                "atom_count": int(rec.get("atom_count") or sum(int(v) for v in counts.values())),
                "lattice": rec["lattice"],
            }
            exact[(sg, orbit_id, str(item["element"]))].append(item)
            orbit_only[(sg, orbit_id)].append(item)
            sg_rows[sg].append(item)
    return {
        "exact": {k: v for k, v in exact.items()},
        "orbit": {k: v for k, v in orbit_only.items()},
        "sg_rows": {k: v for k, v in sg_rows.items()},
        "data_scale": {
            "train_records": len(train_records),
            "train_wa_rows": total_rows,
            "usable_param_rows": sum(len(v) for v in exact.values()),
            "exact_keys": len(exact),
            "orbit_keys": len(orbit_only),
            "skipped": dict(skipped),
        },
    }


def row_pool(
    bank: dict[str, Any],
    row: dict[str, Any],
    sg: int,
    target_counts: dict[str, int],
    exact: bool,
    limit: int,
) -> list[dict[str, Any]]:
    counts_key = tuple(sorted((str(k), int(v)) for k, v in target_counts.items()))
    cache_key = (
        int(sg),
        str(row["orbit_id"]),
        str(row.get("element") or "") if exact else "*",
        bool(exact),
        counts_key,
        int(limit),
    )
    if cache_key in _ROW_POOL_CACHE:
        return _ROW_POOL_CACHE[cache_key]
    if exact:
        pool = list(bank["exact"].get((int(sg), str(row["orbit_id"]), str(row.get("element") or "")), []))
    else:
        pool = list(bank["orbit"].get((int(sg), str(row["orbit_id"])), []))
    ranked = nearest_records(pool, target_counts, limit) if pool else []
    _ROW_POOL_CACHE[cache_key] = ranked
    return ranked


def params_from_pool(
    *,
    engine: OrbitEngine,
    rows: list[dict[str, Any]],
    sg: int,
    target_counts: dict[str, int],
    bank: dict[str, Any],
    variant: str,
    source_params: dict[int, dict[str, float]] | None,
) -> tuple[dict[int, dict[str, float]], int, dict[str, int]]:
    params: dict[int, dict[str, float]] = {}
    fallback = 0
    stats = Counter()
    occurrence: Counter[tuple[str, str]] = Counter()
    for idx, row in enumerate(rows):
        orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
        symbols = [str(s) for s in orbit.free_symbols]
        occ_key = (str(row["orbit_id"]), str(row.get("element") or ""))
        occ = int(occurrence[occ_key])
        occurrence[occ_key] += 1
        if not symbols:
            params[idx] = {}
            stats["fixed_orbit_rows"] += 1
            continue
        exact_pool = row_pool(bank, row, sg, target_counts, exact=True, limit=24)
        orbit_pool = row_pool(bank, row, sg, target_counts, exact=False, limit=24)
        source = {str(k): float(v) % 1.0 for k, v in dict((source_params or {}).get(idx) or {}).items()}
        chosen: dict[str, float] = {}
        if variant.startswith("exact_top"):
            n = int(variant.split("_top", 1)[1].split("_", 1)[0])
            pool = exact_pool or orbit_pool
            if pool:
                chosen = dict(pool[(n - 1 + occ) % len(pool)]["params"])
                stats["exact_top_rows" if exact_pool else "orbit_top_rows"] += 1
        elif variant.startswith("orbit_top"):
            n = int(variant.split("_top", 1)[1].split("_", 1)[0])
            pool = orbit_pool or exact_pool
            if pool:
                chosen = dict(pool[(n - 1 + occ) % len(pool)]["params"])
                stats["orbit_top_rows" if orbit_pool else "exact_top_rows"] += 1
        elif variant == "exact_circular_mean":
            pool = exact_pool or orbit_pool
            if pool:
                mean = {s: circular_mean([float(p["params"][s]) for p in pool if s in p["params"]]) for s in symbols}
                diverse = dict(pool[occ % len(pool)]["params"])
                chosen = {s: circular_blend(mean[s], diverse.get(s, mean[s]), 0.35) for s in symbols}
                stats["mean_rows"] += 1
        elif variant == "blend_source_exact_mean":
            pool = exact_pool or orbit_pool
            if pool:
                mean = {s: circular_mean([float(p["params"][s]) for p in pool if s in p["params"]]) for s in symbols}
                chosen = {s: circular_blend(source.get(s, mean[s]), mean[s], 0.5) for s in symbols}
                stats["blend_rows"] += 1
        if set(chosen) != set(symbols):
            det = deterministic_params(engine, str(row["orbit_id"]), idx)
            chosen = {s: source.get(s, det.get(s, 0.173)) for s in symbols}
            fallback += 1
            stats["fallback_rows"] += 1
        params[idx] = {s: float(chosen[s]) % 1.0 for s in symbols}
    return params, fallback, dict(stats)


def build_variant_rows(
    *,
    variant: str,
    hydrated: dict[str, list[dict[str, Any]]],
    prototype: dict[str, list[dict[str, Any]]],
    siteassign: dict[str, list[dict[str, Any]]],
    posterior: dict[str, list[dict[str, Any]]],
    rows_lt7: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    quotas = {
        "h10_s10_prior25_p5": (10, 10, 25, 5),
        "h10_s5_prior30_p5": (10, 5, 30, 5),
        "prior30_h10_s5_p5": (10, 5, 30, 5),
        "h10_interleave_s10_prior25_p5": (10, 10, 25, 5),
    }
    hq, sq, rq, pq = quotas[variant]
    out: list[dict[str, Any]] = []
    for row in rows_lt7:
        item = dict(row)
        item["selection_variant"] = variant
        out.append(item)
    for sid in sorted(set(hydrated) | set(prototype) | set(siteassign) | set(posterior)):
        hyd = hydrated.get(sid, [])
        site = siteassign.get(sid, [])
        post = posterior.get(sid, [])
        proto = prototype.get(sid, [])
        selected: list[dict[str, Any]] = []
        if variant == "prior30_h10_s5_p5":
            selected.extend(post[:rq])
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(proto[:pq])
        elif variant == "h10_interleave_s10_prior25_p5":
            for i in range(max(hq, sq, rq)):
                if i < hq and i < len(hyd):
                    selected.append(hyd[i])
                if i < sq and i < len(site):
                    selected.append(site[i])
                if i < rq and i < len(post):
                    selected.append(post[i])
            selected.extend(proto[: max(0, top_k - len(selected))])
        else:
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(post[:rq])
            selected.extend(proto[:pq])
        for rank, row in enumerate(selected[:top_k], start=1):
            item = dict(row)
            item["rank"] = rank
            item["selection_variant"] = variant
            out.append(item)
    out.sort(key=lambda r: (str(r["sample_id"]), int(r.get("rank") or 10**9)))
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
    best_name = result["best_variant"]
    best = result["variants"][best_name]["rows_ge7"]
    exp2b = result["baseline_exp2b"]["rows_ge7"]
    exp2d = result["baseline_exp2d"]["rows_ge7"]
    gate = result["gate"]
    oracle = result["oracle_diagnostic"]
    return f"""## opentry_14 实验 2h：train conditional free-parameter prior posterior

结果文件：`model/New_model/opentry_14/results/{result['result_filename']}`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2h_train_param_prior_geometry_posterior/`

- 为什么做：Exp2g 证明现有候选池 all-candidate oracle 仍低于 Exp3/Exp2 gate，ranking 不是主瓶颈；需要产生新的 geometry basin。Exp2h 用 train true structures 的 row-level free-parameter 条件分布构造 posterior，而不是继续扩大 site/residual/permutation beam。
- 核心假设：如果 skeleton-hit/no-match 的主要错误是 predicted skeleton 下 row-level free parameters 落在错误 basin，则按 `(SG, orbit, element)` 和 `(SG, orbit)` 从 train split 学到的参数先验，给同一 predicted skeleton 生成 top/mean/blend 多假设，应新增 sample-level match。
- 数据规模：rows>=7 validation samples `{result['data_scale']['rows_ge7_samples']}`；train usable param rows `{result['param_bank']['usable_param_rows']}`；generated posterior candidates `{result['data_scale']['generated_prior_candidates']}`；evaluated candidates `{result['data_scale']['evaluated_prior_candidates']}`；skipped atom-count samples `{result['data_scale']['skipped_target_atom_count_gt_limit']}`；workers `{result['cpu_policy']['workers']}`。
- baseline：Exp2b rows>=7 match@50 `{pct(exp2b.get('match@50'))}`、conversion `{pct(exp2b.get('skeleton_to_match_conversion@50'))}`；Exp2d best rows>=7 match@50 `{pct(exp2d.get('match@50'))}`、conversion `{pct(exp2d.get('skeleton_to_match_conversion@50'))}`；Exp2g union oracle conversion `{pct(result['baseline_exp2g']['union_conversion_any'])}`。
- 方法变化：对 validation predicted skeleton 做 exact-cover assignment 后，不再复制单个 source row 参数，而是为每个 row 从 train 参数库取 exact-top1/top2/top3、orbit-top1、exact circular mean、source-exact blend 等 posterior variants；lattice 使用 source、SG+atom-count median、SG median 的 inference-safe train priors。排序只用 structural score，不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；valid `{pct(best.get('valid_rate'))}`；formula `{pct(best.get('formula_consistency'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`；collision `{pct(best.get('collision_rate'))}`。
- oracle 诊断：prior-only match samples `{oracle['prior_match_samples']}`，相对 Exp2b 新增 match/skelmatch samples `{oracle['new_match_samples_vs_exp2b']}` / `{oracle['new_skelmatch_samples_vs_exp2b']}`，相对 Exp2d best 新增 `{oracle['new_match_samples_vs_exp2d_best']}` / `{oracle['new_skelmatch_samples_vs_exp2d_best']}`；union upper bound match@50 `{pct(oracle['union_match50_upper_bound'])}`、conversion `{pct(oracle['union_conversion50_upper_bound'])}`。
- 可信度：中等。参数先验只来自 train split，validation 推理不使用禁用标签，所有候选真实 render/parse/SG/StructureMatcher；限制是 row-wise posterior 仍假设 rows 条件独立，可能破坏跨 row 几何相关性，且默认跳过超大 atom-count 长尾。
- 和历史实验关系：这是 Exp2e residual posterior 和 Exp2f deterministic permutation 后的实质性 generator 变体，直接回应 Exp2g “候选 headroom 不足”的结论；若仍无新增 oracle match，则当前 predicted skeleton/geometry posterior 路线进一步收窄到 skeleton proposer 或更强联合模型。
- gate 判定：minimum_passed=`{gate['minimum_passed']}`；target_passed=`{gate['target_passed']}`；exp3_line_passed=`{gate['exp3_line_passed']}`；match@50 delta vs Exp2b `{pp(gate['rows_ge7_match50_delta_vs_exp2b'])}`；conversion delta vs Exp2b `{pp(gate['rows_ge7_conversion50_delta_vs_exp2b'])}`；conversion delta vs Exp3 line `{pp(gate['rows_ge7_conversion50_delta_vs_exp3_line'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=6)
    parser.add_argument("--assignment-limit", type=int, default=2)
    parser.add_argument("--posterior-variants", type=int, default=6)
    parser.add_argument("--max-target-atoms", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    train_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    train_struct_list = list(train_struct.values())
    val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in train_struct_list + list(val.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    bank = build_param_bank(engine, train_struct_list)
    lattice_index = build_lattice_index(train_struct_list)

    exp2b = read_json(EXP2B_RESULT)
    exp2d = read_json(EXP2D_RESULT)
    exp2g = read_json(EXP2G_RESULT) if EXP2G_RESULT.exists() else {}
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    site_rows = list(read_jsonl_iter(EXP2D_SITE_EVAL))

    selected_sids = [sid for sid in sorted(proposals) if sid in val and sid in val_repr and int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    base_param_variants = [
        "exact_top1",
        "exact_top2",
        "exact_top3",
        "orbit_top1",
        "exact_circular_mean",
        "blend_source_exact_mean",
    ][: max(1, int(args.posterior_variants))]

    payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    source_stats: Counter[str] = Counter()
    for si, sid in enumerate(selected_sids, start=1):
        target = val[sid]
        target_repr = val_repr[sid]
        counts = {str(k): int(v) for k, v in formula_counts(target).items()}
        target_atom_count = int(sum(counts.values()))
        candidates: list[dict[str, Any]] = []
        if target_atom_count > int(args.max_target_atoms):
            failures["skipped_target_atom_count_gt_limit"] += 1
            payloads.append(
                {
                    "sample_id": sid,
                    "target_cif_path": str(target["source_path"]),
                    "formula_counts": counts,
                    "target_atom_count": target_atom_count,
                    "sg": int(target["sg"]),
                    "candidates": candidates,
                }
            )
            continue

        sg = int(target["sg"])
        sg_atom_lattice = lattice_index["sg_atom_median"].get((sg, target_atom_count))
        sg_lattice = lattice_index["sg_median"].get(sg) or lattice_index["global_median"]
        lattice_variants = ["source", "sg_atom_median", "sg_median"]
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_skeletons)]:
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            source_struct = train_struct.get(source_id)
            if source_repr is None or source_struct is None:
                failures["missing_source"] += 1
                continue
            original_rows = source_skeleton_rows(engine, source_repr)
            if not original_rows:
                failures["empty_source_rows"] += 1
                continue
            assignments = enumerate_exact_cover_assignments(original_rows, counts, int(args.assignment_limit))
            if not assignments:
                failures["no_exact_cover_assignment"] += 1
                continue
            source_lattice = lattice_from_record(source_struct)
            source_params = {}
            for idx, row in enumerate(original_rows):
                free = dict((source_struct.get("wa_table") or [{}] * len(original_rows))[idx].get("free_params") or {}) if idx < len(source_struct.get("wa_table") or []) else {}
                source_params[idx] = {str(k): float(v) % 1.0 for k, v in free.items()}
            for assignment_rank, (assignment_score, assignment) in enumerate(assignments, start=1):
                assigned_rows: list[dict[str, Any]] = []
                for row, element in zip(original_rows, assignment):
                    item = dict(row)
                    item["source_element"] = str(row.get("element"))
                    item["element"] = str(element)
                    assigned_rows.append(item)
                for param_rank, param_variant in enumerate(base_param_variants, start=1):
                    params, fallback_count, pstats = params_from_pool(
                        engine=engine,
                        rows=assigned_rows,
                        sg=sg,
                        target_counts=counts,
                        bank=bank,
                        variant=param_variant,
                        source_params=source_params,
                    )
                    source_stats.update(pstats)
                    for lattice_rank, lattice_variant in enumerate(lattice_variants, start=1):
                        if lattice_variant == "source":
                            lattice = source_lattice
                        elif lattice_variant == "sg_atom_median" and sg_atom_lattice is not None:
                            lattice = sg_atom_lattice
                        elif lattice_variant == "sg_atom_median":
                            continue
                        else:
                            lattice = sg_lattice
                        try:
                            option = {"lattice": lattice, "params": params}
                            cif, render_meta = render_candidate(
                                engine=engine,
                                target=target,
                                rows=assigned_rows,
                                option=option,
                                data_name=f"{sid}_prior_p{int(proposal.get('rank') or 0)}_a{assignment_rank}_v{param_rank}_l{lattice_rank}",
                            )
                            row = {
                                "sample_id": sid,
                                "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                                "proposal_rank": int(proposal.get("rank") or 0),
                                "assignment_rank": int(assignment_rank),
                                "assignment_source_preserved_atoms": int(assignment_score),
                                "posterior_param_rank": int(param_rank),
                                "posterior_lattice_rank": int(lattice_rank),
                                "geometry_rank": len(candidates) + 1,
                                "raw_generation_order": len(candidates) + 1,
                                "row_count": int(target_repr.get("row_count") or 0),
                                "sg": sg,
                                "formula_counts": counts,
                                "target_atom_count": target_atom_count,
                                "source_sample_id": source_id,
                                "proposal_source": str(proposal.get("source") or ""),
                                "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                                "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                                "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == str(target_repr.get("canonical_skeleton_key") or ""),
                                "candidate_row_count": len(assigned_rows),
                                "site_mapping_rule": "exact_cover_assignment_plus_train_param_prior",
                                "geometry_source": "train_conditional_free_parameter_prior",
                                "posterior_param_variant": param_variant,
                                "posterior_lattice_variant": lattice_variant,
                                "reference_sample_id": source_id,
                                "reference_score": float(assignment_score + max(0, 100 - int(proposal.get("rank") or 0)) + max(0, 20 - fallback_count)),
                                "param_fallback_rows": int(fallback_count),
                                "param_source_stats": pstats,
                                "cif": cif,
                                **render_meta,
                            }
                            candidates.append(row)
                            generation_meta.append({k: v for k, v in row.items() if k != "cif"})
                        except Exception as exc:  # noqa: BLE001
                            failures[f"render_failed:{type(exc).__name__}"] += 1
        payloads.append(
            {
                "sample_id": sid,
                "target_cif_path": str(target["source_path"]),
                "formula_counts": counts,
                "target_atom_count": target_atom_count,
                "sg": sg,
                "candidates": candidates,
            }
        )
        if si % 200 == 0:
            print(f"[exp2h-param-prior] rendered {si}/{len(selected_sids)}", flush=True)

    suffix = f"_{args.output_suffix}" if str(args.output_suffix).strip() else ""
    result_filename = f"experiment_2h_train_param_prior_geometry_posterior{suffix}.json"
    write_jsonl(ARTIFACT_DIR / f"generated_train_param_prior_meta{suffix}.jsonl", generation_meta)

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp2h-param-prior] evaluated {i}/{len(futures)}", flush=True)
    ranked_prior = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / f"evaluated_train_param_prior_candidates{suffix}.jsonl", ranked_prior)

    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by_sid = by_sid_ranked(assign_structural_ranks(site_rows, int(args.top_k)))
    prior_by_sid = by_sid_ranked(ranked_prior)

    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("h10_s10_prior25_p5", "h10_s5_prior30_p5", "prior30_h10_s5_p5", "h10_interleave_s10_prior25_p5"):
        rows = build_variant_rows(
            variant=variant,
            hydrated=hydrated,
            prototype=prototype,
            siteassign=site_by_sid,
            posterior=prior_by_sid,
            rows_lt7=rows_lt7,
            top_k=int(args.top_k),
        )
        path = ARTIFACT_DIR / f"selected_{variant}_candidates{suffix}.jsonl"
        write_jsonl(path, rows)
        selected_paths[variant] = str(path)
        rows7 = [r for r in rows if int(r.get("row_count") or 0) >= 7]
        rowslt7 = [r for r in rows if int(r.get("row_count") or 0) < 7]
        variants[variant] = {"overall": summarize(rows), "rows_ge7": summarize(rows7), "rows_lt7": summarize(rowslt7)}

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
    exp2d_rows7 = exp2d["variants"][exp2d["best_variant"]]["rows_ge7"]
    exp3_line = float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0) + 0.02

    min_gate = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.223
        and float(best_rows7.get("match@50") or 0.0) >= 0.17931372549019609
        and float(best_rows7.get("formula_consistency") or 0.0) >= 0.95
        and float(best_rows7.get("sg_consistency") or 0.0) >= 0.90
        and float(best_rows7.get("exact_cover_retained") or 0.0) >= 0.95
    )
    target_gate = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28
        or (
            float(best_rows7.get("match@5") or 0.0) >= float(baseline_rows7.get("match@5") or 0.0) + 0.05
            and float(best_rows7.get("match@20") or 0.0) >= float(baseline_rows7.get("match@20") or 0.0) + 0.05
        )
    )
    exp3_line_passed = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= exp3_line
    conv_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0)

    base_sets = sample_sets([r for r in safe_rows if int(r.get("row_count") or 0) >= 7])
    site_best_path = Path(exp2d["artifacts"]["selected_variants"][exp2d["best_variant"]])
    site_best_sets = sample_sets(list(read_jsonl_iter(site_best_path)))
    prior_sets = sample_sets(ranked_prior)
    union_match = base_sets["match"] | site_best_sets["match"] | prior_sets["match"]
    union_skel = base_sets["skeleton"] | site_best_sets["skeleton"] | prior_sets["skeleton"]
    union_skelmatch = base_sets["skelmatch"] | site_best_sets["skelmatch"] | prior_sets["skelmatch"]
    all_rows7_samples = base_sets["samples"] | site_best_sets["samples"] | prior_sets["samples"]
    oracle = {
        "prior_match_samples": len(prior_sets["match"]),
        "prior_skelmatch_samples": len(prior_sets["skelmatch"]),
        "new_match_samples_vs_exp2b": len(prior_sets["match"] - base_sets["match"]),
        "new_skelmatch_samples_vs_exp2b": len(prior_sets["skelmatch"] - base_sets["skelmatch"]),
        "new_match_samples_vs_exp2d_best": len(prior_sets["match"] - site_best_sets["match"]),
        "new_skelmatch_samples_vs_exp2d_best": len(prior_sets["skelmatch"] - site_best_sets["skelmatch"]),
        "union_match50_upper_bound": ratio(len(union_match), len(all_rows7_samples)),
        "union_conversion50_upper_bound": ratio(len(union_skelmatch), len(union_skel)),
    }

    if target_gate:
        verdict = "pass_target_gate"
        reason = "Train conditional free-parameter posterior reaches the Exp2 target repair gate."
        next_step = "Proceed to Exp3 local optimizer relative to this candidate, still without official/full validation until later gates pass."
    elif exp3_line_passed:
        verdict = "pass_exp3_line_but_not_exp2_target"
        reason = "Train conditional free-parameter posterior clears the Exp3 +2pp conversion line but not the Exp2 target line."
        next_step = "Retest Exp3 local optimizer only as a diagnostic; do not enter Exp4/5/official unless target/full validation gates are later met."
    elif min_gate:
        verdict = "pass_minimum_gate_but_no_exp3_headroom"
        reason = "Train conditional free-parameter posterior stays above the Exp2 minimum gate but does not clear the Exp3 conversion line."
        next_step = "Do not expand this row-wise posterior blindly; inspect oracle headroom and consider a stronger joint model or skeleton proposer upgrade."
    else:
        verdict = "fail_validation_gate"
        reason = "Train conditional free-parameter posterior does not maintain the Exp2 minimum validation gate."
        next_step = "Stop this posterior variant and return to data alignment or skeleton proposer root cause."

    result = {
        "experiment": "opentry_14_exp2h_train_conditional_free_parameter_prior_posterior",
        "result_filename": result_filename,
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "train_conditional_row_free_parameter_prior",
            "top_skeletons": int(args.top_skeletons),
            "assignment_limit": int(args.assignment_limit),
            "posterior_variants": base_param_variants,
            "lattice_variants": ["source", "sg_atom_median", "sg_median"],
            "max_target_atoms": int(args.max_target_atoms),
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "single_process_cpu_policy": "many single-thread worker processes; target each Python worker near 100% CPU and below user 200/300% per-core cap",
        },
        "param_bank": bank["data_scale"],
        "data_scale": {
            "rows_ge7_samples": len(selected_sids),
            "input_safe_pool_records": len(safe_rows),
            "input_site_assignment_records": len(site_rows),
            "generated_prior_candidates": len(generation_meta),
            "evaluated_prior_candidates": len(ranked_prior),
            "skipped_target_atom_count_gt_limit": int(failures.get("skipped_target_atom_count_gt_limit", 0)),
            "source_stats": dict(source_stats),
            "selected_variant_paths": selected_paths,
        },
        "baseline_exp2b": {"result_path": str(EXP2B_RESULT), "rows_ge7": baseline_rows7, "overall": exp2b["overall"]},
        "baseline_exp2d": {"result_path": str(EXP2D_RESULT), "best_variant": exp2d["best_variant"], "rows_ge7": exp2d_rows7, "overall": exp2d["variants"][exp2d["best_variant"]]["overall"]},
        "baseline_exp2g": {
            "result_path": str(EXP2G_RESULT),
            "union_conversion_any": (exp2g.get("union_all_candidates") or {}).get("skeleton_to_match_conversion_any"),
            "union_match_any": (exp2g.get("union_all_candidates") or {}).get("match_rate_any_fixed_denominator"),
        },
        "mapping_failures": dict(failures),
        "prior_only": {"rows_ge7": summarize([r for r in ranked_prior if int(r.get("row_count") or 0) >= 7]) if ranked_prior else {}},
        "oracle_diagnostic": oracle,
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "minimum_passed": min_gate,
            "target_passed": target_gate,
            "exp3_line_passed": exp3_line_passed,
            "passed": min_gate,
            "rows_ge7_match50_delta_vs_exp2b": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2b": conv_delta,
            "rows_ge7_match50_delta_vs_exp2d": float(best_rows7.get("match@50") or 0.0) - float(exp2d_rows7.get("match@50") or 0.0),
            "rows_ge7_conversion50_delta_vs_exp2d": float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2d_rows7.get("skeleton_to_match_conversion@50") or 0.0),
            "rows_ge7_conversion50_delta_vs_exp3_line": float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - exp3_line,
            "minimum_standard": {
                "rows_ge7_conversion50": 0.223,
                "rows_ge7_match50_allowed_lower_bound": 0.17931372549019609,
                "rows_ge7_formula_consistency": 0.95,
                "rows_ge7_sg_consistency": 0.90,
                "rows_ge7_exact_cover_retained": 0.95,
                "target_rows_ge7_conversion50": 0.28,
                "exp3_required_conversion50": exp3_line,
            },
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "generated_meta": str(ARTIFACT_DIR / f"generated_train_param_prior_meta{suffix}.jsonl"),
            "evaluated_candidates": str(ARTIFACT_DIR / f"evaluated_train_param_prior_candidates{suffix}.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / result_filename, result)
    if not args.skip_report:
        body = report_body(result)
        marker = "<!-- OPENTRY14_EXP2H_TRAIN_PARAM_PRIOR_GEOMETRY_POSTERIOR -->"
        append_or_replace(REPORT_PATH, marker, body)
        append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / result_filename), "best_variant": best_variant, "gate": result["gate"], "decision": result["decision"], "oracle": oracle}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
