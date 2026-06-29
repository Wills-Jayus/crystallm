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
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2j_chemical_site_order_assignment"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.core.periodic_table import Element  # noqa: E402

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
from run_exp2f_permutation_aware_alignment import by_sid_ranked, sample_sets, split_safe_pool  # noqa: E402
from run_symcif_v4_geometry_model_eval import flexible_params_from_reference  # noqa: E402
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
EXP2I_RESULT = RESULT_DIR / "experiment_2i_gt_assignment_beam_audit.json"
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


def element_props(symbol: str) -> dict[str, Any]:
    try:
        e = Element(str(symbol))
        return {
            "Z": int(e.Z),
            "row": int(e.row),
            "group": int(e.group) if e.group is not None else 0,
            "X": float(e.X) if e.X is not None else None,
            "is_metal": bool(e.is_metal),
            "is_transition_metal": bool(e.is_transition_metal),
            "is_metalloid": bool(e.is_metalloid),
            "block": str(getattr(e, "block", "") or ""),
        }
    except Exception:
        return {"Z": 0, "row": 0, "group": 0, "X": None, "is_metal": False, "is_transition_metal": False, "is_metalloid": False, "block": ""}


_ELEM_CACHE: dict[str, dict[str, Any]] = {}


def props(symbol: str) -> dict[str, Any]:
    symbol = str(symbol)
    if symbol not in _ELEM_CACHE:
        _ELEM_CACHE[symbol] = element_props(symbol)
    return _ELEM_CACHE[symbol]


def chemical_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if str(a) == str(b):
        return 3.0
    pa, pb = props(a), props(b)
    score = 0.0
    if pa["group"] and pb["group"] and pa["group"] == pb["group"]:
        score += 1.0
    if pa["row"] and pb["row"] and pa["row"] == pb["row"]:
        score += 0.25
    if pa["block"] and pa["block"] == pb["block"]:
        score += 0.35
    if pa["is_metal"] == pb["is_metal"]:
        score += 0.45
    if pa["is_transition_metal"] == pb["is_transition_metal"]:
        score += 0.20
    if pa["is_metalloid"] == pb["is_metalloid"]:
        score += 0.15
    if pa["X"] is not None and pb["X"] is not None:
        score -= 0.35 * min(5.0, abs(float(pa["X"]) - float(pb["X"])))
    if pa["group"] and pb["group"]:
        score -= 0.035 * abs(int(pa["group"]) - int(pb["group"]))
    if pa["row"] and pb["row"]:
        score -= 0.07 * abs(int(pa["row"]) - int(pb["row"]))
    return float(score)


def build_train_priors(train_struct: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    orbit: dict[str, Counter[str]] = defaultdict(Counter)
    sg: dict[int, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    total_rows = 0
    for rec in train_struct:
        rec_sg = int(rec["sg"])
        for row in rec.get("wa_table") or []:
            elem = str(row.get("element") or "")
            orbit_id = str(row.get("orbit_id") or "")
            if not elem or not orbit_id:
                continue
            exact[(rec_sg, orbit_id)][elem] += 1
            orbit[orbit_id][elem] += 1
            sg[rec_sg][elem] += 1
            global_counts[elem] += 1
            total_rows += 1
    return {
        "exact": dict(exact),
        "orbit": dict(orbit),
        "sg": dict(sg),
        "global": global_counts,
        "data_scale": {
            "train_records": len(train_struct),
            "train_wa_rows": total_rows,
            "exact_sg_orbit_keys": len(exact),
            "orbit_keys": len(orbit),
            "sg_keys": len(sg),
        },
    }


def logprob(counter: Counter[str] | None, element: str, vocab_size: int) -> float:
    if not counter:
        return -math.log(max(1, vocab_size))
    total = sum(counter.values())
    return math.log((float(counter.get(element, 0)) + 0.25) / (float(total) + 0.25 * max(1, vocab_size)))


def row_element_score(row: dict[str, Any], element: str, sg: int, priors: dict[str, Any], vocab_size: int) -> float:
    orbit_id = str(row.get("orbit_id") or "")
    source_element = str(row.get("element") or "")
    score = 1.40 * chemical_similarity(element, source_element)
    score += 0.38 * logprob(priors["exact"].get((int(sg), orbit_id)), element, vocab_size)
    score += 0.25 * logprob(priors["orbit"].get(orbit_id), element, vocab_size)
    score += 0.10 * logprob(priors["sg"].get(int(sg)), element, vocab_size)
    score += 0.05 * logprob(priors["global"], element, vocab_size)
    return float(score)


def chemical_exact_cover_assignments(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    sg: int,
    priors: dict[str, Any],
    limit: int,
    beam_width: int,
) -> list[tuple[float, list[str], dict[str, Any]]]:
    mults = [int(row.get("multiplicity") or 0) for row in rows]
    elements = sorted(str(e) for e, c in counts.items() if int(c) > 0)
    vocab_size = max(1, len(set(priors["global"]) | set(elements)))
    row_best = []
    row_scores: dict[tuple[int, str], float] = {}
    for idx, row in enumerate(rows):
        scores = {e: row_element_score(row, e, sg, priors, vocab_size) for e in elements}
        for e, s in scores.items():
            row_scores[(idx, e)] = float(s)
        ordered = sorted(scores.values(), reverse=True)
        gap = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
        row_best.append((idx, gap))
    order = [idx for idx, _ in sorted(row_best, key=lambda x: (-mults[x[0]], -x[1], str(rows[x[0]].get("orbit_id")), x[0]))]
    start_remaining = {str(k): int(v) for k, v in counts.items()}
    states: list[tuple[float, dict[str, int], list[str | None]]] = [(0.0, start_remaining, [None] * len(rows))]
    for idx in order:
        mult = mults[idx]
        expanded: list[tuple[float, dict[str, int], list[str | None]]] = []
        choices = sorted(elements, key=lambda e: (-row_scores.get((idx, e), -1.0e9), e))
        for score, remaining, assignment in states:
            for elem in choices:
                if int(remaining.get(elem, 0)) < mult:
                    continue
                new_remaining = dict(remaining)
                new_remaining[elem] -= mult
                new_assignment = list(assignment)
                new_assignment[idx] = elem
                expanded.append((score + row_scores.get((idx, elem), 0.0) * max(1, mult), new_remaining, new_assignment))
        expanded.sort(key=lambda x: (-x[0], tuple("" if v is None else str(v) for v in x[2])))
        states = expanded[: max(1, int(beam_width))]
        if not states:
            return []
    out: list[tuple[float, list[str], dict[str, Any]]] = []
    seen: set[tuple[str, ...]] = set()
    for score, remaining, assignment in sorted(states, key=lambda x: (-x[0], tuple(str(v) for v in x[2]))):
        if any(int(v) != 0 for v in remaining.values()):
            continue
        final = [str(x) for x in assignment]
        key = tuple(final)
        if key in seen:
            continue
        seen.add(key)
        source_preserved = sum(int(rows[i].get("multiplicity") or 0) for i, elem in enumerate(final) if str(rows[i].get("element")) == str(elem))
        out.append((float(score), final, {"source_preserved_atoms": int(source_preserved)}))
        if len(out) >= int(limit):
            break
    return out


def build_variant_rows(
    *,
    variant: str,
    hydrated: dict[str, list[dict[str, Any]]],
    prototype: dict[str, list[dict[str, Any]]],
    siteassign: dict[str, list[dict[str, Any]]],
    chemassign: dict[str, list[dict[str, Any]]],
    rows_lt7: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    quotas = {
        "h10_s10_chem25_p5": (10, 10, 25, 5),
        "h10_s5_chem30_p5": (10, 5, 30, 5),
        "chem30_h10_s5_p5": (10, 5, 30, 5),
        "h10_interleave_s10_chem25_p5": (10, 10, 25, 5),
    }
    hq, sq, cq, pq = quotas[variant]
    out: list[dict[str, Any]] = []
    for row in rows_lt7:
        item = dict(row)
        item["selection_variant"] = variant
        out.append(item)
    for sid in sorted(set(hydrated) | set(prototype) | set(siteassign) | set(chemassign)):
        hyd = hydrated.get(sid, [])
        site = siteassign.get(sid, [])
        chem = chemassign.get(sid, [])
        proto = prototype.get(sid, [])
        selected: list[dict[str, Any]] = []
        if variant == "chem30_h10_s5_p5":
            selected.extend(chem[:cq])
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(proto[:pq])
        elif variant == "h10_interleave_s10_chem25_p5":
            for i in range(max(hq, sq, cq)):
                if i < hq and i < len(hyd):
                    selected.append(hyd[i])
                if i < sq and i < len(site):
                    selected.append(site[i])
                if i < cq and i < len(chem):
                    selected.append(chem[i])
            selected.extend(proto[: max(0, top_k - len(selected))])
        else:
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(chem[:cq])
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
    oracle = result["oracle_diagnostic"]
    gate = result["gate"]
    return f"""## opentry_14 实验 2j：chemical/site-order assignment posterior

结果文件：`model/New_model/opentry_14/results/{result['result_filename']}`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2j_chemical_site_order_assignment/`

- 为什么做：Exp2i 显示 GT element assignment 大量不在 deterministic rank<=64，说明 source-preserved assignment order 与真实 site/element order 不一致；但 GT-WA 不能用于推理。本实验用 train split 的元素-轨道统计和化学相似度，构造 inference-safe site-order assignment posterior。
- 核心假设：如果 assignment order 是主要瓶颈，则按 source row 元素的化学相似度、train `(SG, orbit)->element` 先验、train `(orbit)->element` 先验生成 exact-cover assignments，应产生 Exp2b/Exp2d 没有的新 match samples，并提高 rows>=7 conversion。
- 数据规模：rows>=7 validation samples `{result['data_scale']['rows_ge7_samples']}`；train prior rows `{result['train_priors']['train_wa_rows']}`；generated chemical assignments `{result['data_scale']['generated_chemical_assignment_candidates']}`；evaluated candidates `{result['data_scale']['evaluated_chemical_assignment_candidates']}`；workers `{result['cpu_policy']['workers']}`。
- baseline：Exp2b rows>=7 match@50 `{pct(exp2b.get('match@50'))}`、conversion `{pct(exp2b.get('skeleton_to_match_conversion@50'))}`；Exp2d best match@50 `{pct(exp2d.get('match@50'))}`、conversion `{pct(exp2d.get('skeleton_to_match_conversion@50'))}`；Exp2i rank<=10 coverage `{pct(result['baseline_exp2i']['rank_le10_rate'])}`。
- 方法变化：替代 deterministic source-preserved exact-cover order；对每个 row/element 打分，特征只含元素周期表相似度、train split 的 SG/orbit/element 频率和 source-row 元素相似度。assignment beam 完成 exact-cover 后，用 source lattice/free params 渲染，并用 structural score 排序；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；valid `{pct(best.get('valid_rate'))}`；formula `{pct(best.get('formula_consistency'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`；collision `{pct(best.get('collision_rate'))}`。
- oracle 诊断：chemical-only match samples `{oracle['chemical_match_samples']}`；相对 Exp2b 新增 match/skelmatch `{oracle['new_match_samples_vs_exp2b']}` / `{oracle['new_skelmatch_samples_vs_exp2b']}`；相对 Exp2d best 新增 `{oracle['new_match_samples_vs_exp2d_best']}` / `{oracle['new_skelmatch_samples_vs_exp2d_best']}`；union upper bound match@50 `{pct(oracle['union_match50_upper_bound'])}`、conversion `{pct(oracle['union_conversion50_upper_bound'])}`。
- 可信度：中等。所有候选真实 render/parse/SG/StructureMatcher；assignment posterior 只用 train split 和元素表，validation match 只用于离线评估。限制是它仍复制 source geometry/free params，未学习 correlated geometry posterior。
- 和历史实验关系：这是 Exp2d site assignment 的非 GT、非普通 scorer 替代排序，直接回应 Exp2i 的 assignment-order root cause。
- gate 判定：minimum_passed=`{gate['minimum_passed']}`；target_passed=`{gate['target_passed']}`；exp3_line_passed=`{gate['exp3_line_passed']}`；conversion delta vs Exp2b `{pp(gate['rows_ge7_conversion50_delta_vs_exp2b'])}`；conversion delta vs Exp3 line `{pp(gate['rows_ge7_conversion50_delta_vs_exp3_line'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=8)
    parser.add_argument("--assignment-limit", type=int, default=8)
    parser.add_argument("--assignment-beam-width", type=int, default=128)
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
    priors = build_train_priors(train_struct_list)

    exp2b = read_json(EXP2B_RESULT)
    exp2d = read_json(EXP2D_RESULT)
    exp2i = read_json(EXP2I_RESULT) if EXP2I_RESULT.exists() else {}
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    site_rows = list(read_jsonl_iter(EXP2D_SITE_EVAL))

    selected_sids = [sid for sid in sorted(proposals) if sid in val and sid in val_repr and int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for si, sid in enumerate(selected_sids, start=1):
        target = val[sid]
        target_repr = val_repr[sid]
        counts = {str(k): int(v) for k, v in formula_counts(target).items()}
        target_atom_count = int(sum(counts.values()))
        candidates: list[dict[str, Any]] = []
        if target_atom_count > int(args.max_target_atoms):
            failures["skipped_target_atom_count_gt_limit"] += 1
            payloads.append({"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": target_atom_count, "sg": int(target["sg"]), "candidates": candidates})
            continue
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
            assignments = chemical_exact_cover_assignments(
                original_rows,
                counts,
                int(target["sg"]),
                priors,
                limit=int(args.assignment_limit),
                beam_width=int(args.assignment_beam_width),
            )
            if not assignments:
                failures["no_chemical_exact_cover_assignment"] += 1
                continue
            source_lattice = {k: float(source_struct["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}
            for assignment_rank, (assignment_score, assignment, assignment_meta) in enumerate(assignments, start=1):
                assigned_rows: list[dict[str, Any]] = []
                for row, element in zip(original_rows, assignment):
                    item = dict(row)
                    item["source_element"] = str(row.get("element"))
                    item["element"] = str(element)
                    assigned_rows.append(item)
                try:
                    params, fallback_count = flexible_params_from_reference(engine, assigned_rows, source_struct, neural_params=None)
                    option = {"lattice": source_lattice, "params": params}
                    cif, render_meta = render_candidate(
                        engine=engine,
                        target=target,
                        rows=assigned_rows,
                        option=option,
                        data_name=f"{sid}_chem_p{int(proposal.get('rank') or 0)}_a{assignment_rank}",
                    )
                    row = {
                        "sample_id": sid,
                        "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                        "proposal_rank": int(proposal.get("rank") or 0),
                        "assignment_rank": int(assignment_rank),
                        "assignment_score": float(assignment_score),
                        "assignment_source_preserved_atoms": int(assignment_meta.get("source_preserved_atoms") or 0),
                        "geometry_rank": len(candidates) + 1,
                        "raw_generation_order": len(candidates) + 1,
                        "row_count": int(target_repr.get("row_count") or 0),
                        "sg": int(target["sg"]),
                        "formula_counts": counts,
                        "target_atom_count": target_atom_count,
                        "source_sample_id": source_id,
                        "proposal_source": str(proposal.get("source") or ""),
                        "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                        "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                        "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == str(target_repr.get("canonical_skeleton_key") or ""),
                        "candidate_row_count": len(assigned_rows),
                        "site_mapping_rule": "chemical_site_order_exact_cover_assignment",
                        "geometry_source": "chemical_site_order_source_geometry",
                        "reference_sample_id": source_id,
                        "reference_score": float(assignment_score),
                        "param_fallback_rows": int(fallback_count),
                        "cif": cif,
                        **render_meta,
                    }
                    candidates.append(row)
                    generation_meta.append({k: v for k, v in row.items() if k != "cif"})
                except Exception as exc:  # noqa: BLE001
                    failures[f"render_failed:{type(exc).__name__}"] += 1
        payloads.append({"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": target_atom_count, "sg": int(target["sg"]), "candidates": candidates})
        if si % 200 == 0:
            print(f"[exp2j-chemassign] rendered {si}/{len(selected_sids)}", flush=True)

    suffix = f"_{args.output_suffix}" if str(args.output_suffix).strip() else ""
    result_filename = f"experiment_2j_chemical_site_order_assignment{suffix}.json"
    write_jsonl(ARTIFACT_DIR / f"generated_chemical_assignment_meta{suffix}.jsonl", generation_meta)
    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp2j-chemassign] evaluated {i}/{len(futures)}", flush=True)
    ranked_chem = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / f"evaluated_chemical_assignment_candidates{suffix}.jsonl", ranked_chem)

    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by_sid = by_sid_ranked(assign_structural_ranks(site_rows, int(args.top_k)))
    chem_by_sid = by_sid_ranked(ranked_chem)
    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("h10_s10_chem25_p5", "h10_s5_chem30_p5", "chem30_h10_s5_p5", "h10_interleave_s10_chem25_p5"):
        rows = build_variant_rows(variant=variant, hydrated=hydrated, prototype=prototype, siteassign=site_by_sid, chemassign=chem_by_sid, rows_lt7=rows_lt7, top_k=int(args.top_k))
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
    target_gate = bool(float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28)
    exp3_line_passed = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= exp3_line

    base_sets = sample_sets([r for r in safe_rows if int(r.get("row_count") or 0) >= 7])
    site_best_path = Path(exp2d["artifacts"]["selected_variants"][exp2d["best_variant"]])
    site_best_sets = sample_sets(list(read_jsonl_iter(site_best_path)))
    chem_sets = sample_sets(ranked_chem)
    all_rows7_samples = base_sets["samples"] | site_best_sets["samples"] | chem_sets["samples"]
    union_match = base_sets["match"] | site_best_sets["match"] | chem_sets["match"]
    union_skel = base_sets["skeleton"] | site_best_sets["skeleton"] | chem_sets["skeleton"]
    union_skelmatch = base_sets["skelmatch"] | site_best_sets["skelmatch"] | chem_sets["skelmatch"]
    oracle = {
        "chemical_match_samples": len(chem_sets["match"]),
        "chemical_skelmatch_samples": len(chem_sets["skelmatch"]),
        "new_match_samples_vs_exp2b": len(chem_sets["match"] - base_sets["match"]),
        "new_skelmatch_samples_vs_exp2b": len(chem_sets["skelmatch"] - base_sets["skelmatch"]),
        "new_match_samples_vs_exp2d_best": len(chem_sets["match"] - site_best_sets["match"]),
        "new_skelmatch_samples_vs_exp2d_best": len(chem_sets["skelmatch"] - site_best_sets["skelmatch"]),
        "union_match50_upper_bound": ratio(len(union_match), len(all_rows7_samples)),
        "union_conversion50_upper_bound": ratio(len(union_skelmatch), len(union_skel)),
    }

    if target_gate:
        verdict = "pass_target_gate"
        reason = "Chemical/site-order assignment posterior reaches the Exp2 target repair gate."
        next_step = "Proceed to Exp3 local optimizer relative to this candidate; keep official frozen until later gates pass."
    elif exp3_line_passed:
        verdict = "pass_exp3_line_but_not_exp2_target"
        reason = "Chemical/site-order assignment clears the Exp3 +2pp conversion line but not the Exp2 target."
        next_step = "Retest Exp3 local optimizer as the next gated step; do not enter Exp4/5/official yet."
    elif min_gate:
        verdict = "pass_minimum_gate_but_no_exp3_headroom"
        reason = "Chemical/site-order assignment remains above the Exp2 minimum gate but does not clear the Exp3 conversion line."
        next_step = "Do not expand chemistry beam blindly; inspect oracle headroom and move to a true joint skeleton/site/geometry model."
    else:
        verdict = "fail_validation_gate"
        reason = "Chemical/site-order assignment does not maintain the Exp2 minimum validation gate."
        next_step = "Stop this assignment posterior and redesign skeleton/site alignment."

    result = {
        "experiment": "opentry_14_exp2j_chemical_site_order_assignment_posterior",
        "result_filename": result_filename,
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "chemical_site_order_exact_cover_assignment",
            "top_skeletons": int(args.top_skeletons),
            "assignment_limit": int(args.assignment_limit),
            "assignment_beam_width": int(args.assignment_beam_width),
            "max_target_atoms": int(args.max_target_atoms),
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "single_process_cpu_policy": "many single-thread worker processes; target each Python worker near 100% CPU and below user 200/300% per-core cap",
        },
        "train_priors": priors["data_scale"],
        "data_scale": {
            "rows_ge7_samples": len(selected_sids),
            "input_safe_pool_records": len(safe_rows),
            "input_site_assignment_records": len(site_rows),
            "generated_chemical_assignment_candidates": len(generation_meta),
            "evaluated_chemical_assignment_candidates": len(ranked_chem),
            "skipped_target_atom_count_gt_limit": int(failures.get("skipped_target_atom_count_gt_limit", 0)),
            "selected_variant_paths": selected_paths,
        },
        "baseline_exp2b": {"result_path": str(EXP2B_RESULT), "rows_ge7": baseline_rows7, "overall": exp2b["overall"]},
        "baseline_exp2d": {"result_path": str(EXP2D_RESULT), "best_variant": exp2d["best_variant"], "rows_ge7": exp2d_rows7, "overall": exp2d["variants"][exp2d["best_variant"]]["overall"]},
        "baseline_exp2i": {
            "result_path": str(EXP2I_RESULT),
            "rank_le10_rate": (exp2i.get("assignment_rank_audit") or {}).get("rank_le10_rate"),
            "not_found_le64_rate": (exp2i.get("assignment_rank_audit") or {}).get("not_found_le64_rate"),
        },
        "mapping_failures": dict(failures),
        "chemical_assignment_only": {"rows_ge7": summarize([r for r in ranked_chem if int(r.get("row_count") or 0) >= 7]) if ranked_chem else {}},
        "oracle_diagnostic": oracle,
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "minimum_passed": min_gate,
            "target_passed": target_gate,
            "exp3_line_passed": exp3_line_passed,
            "passed": min_gate,
            "rows_ge7_match50_delta_vs_exp2b": float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0),
            "rows_ge7_conversion50_delta_vs_exp2b": float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0),
            "rows_ge7_match50_delta_vs_exp2d": float(best_rows7.get("match@50") or 0.0) - float(exp2d_rows7.get("match@50") or 0.0),
            "rows_ge7_conversion50_delta_vs_exp2d": float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2d_rows7.get("skeleton_to_match_conversion@50") or 0.0),
            "rows_ge7_conversion50_delta_vs_exp3_line": float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - exp3_line,
            "minimum_standard": {"rows_ge7_conversion50": 0.223, "rows_ge7_match50_allowed_lower_bound": 0.17931372549019609, "target_rows_ge7_conversion50": 0.28, "exp3_required_conversion50": exp3_line},
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "generated_meta": str(ARTIFACT_DIR / f"generated_chemical_assignment_meta{suffix}.jsonl"),
            "evaluated_candidates": str(ARTIFACT_DIR / f"evaluated_chemical_assignment_candidates{suffix}.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / result_filename, result)
    if not args.skip_report:
        body = report_body(result)
        marker = "<!-- OPENTRY14_EXP2J_CHEMICAL_SITE_ORDER_ASSIGNMENT -->"
        append_or_replace(REPORT_PATH, marker, body)
        append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / result_filename), "best_variant": best_variant, "gate": result["gate"], "decision": result["decision"], "oracle": oracle}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
