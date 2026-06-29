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
from statistics import median
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2s_source_retrieval_supplement"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.core import Structure  # noqa: E402
from pymatgen.core.periodic_table import Element  # noqa: E402

from run_exp2_predicted_skeleton_renderer_site_mapping import formula_counts, read_jsonl, sample_id, source_skeleton_rows  # noqa: E402
from run_exp2f_permutation_aware_alignment import by_sid_ranked, sample_sets, split_safe_pool  # noqa: E402
from run_exp2j_chemical_site_order_assignment import (  # noqa: E402
    EXP2B_EVAL,
    EXP2B_RESULT,
    EXP2D_RESULT,
    EXP2D_SITE_EVAL,
    EXP3_PROPOSALS,
    LOOKUP,
    STRUCTURED_TRAIN,
    STRUCTURED_VAL,
    TRAIN_REPR,
    VAL_REPR,
    append_or_replace,
    build_train_priors,
    build_variant_rows,
    chemical_similarity,
    logprob,
    pp,
    pct,
    read_json,
    read_jsonl_iter,
    ratio,
    write_json,
    write_jsonl,
)
from run_exp4_rows_ge7_multi_geometry_proposal import assign_structural_ranks, eval_sample, render_candidate, summarize  # noqa: E402
from run_symcif_v4_geometry_model_eval import flexible_params_from_reference  # noqa: E402
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


BUDGETS = (1, 5, 20, 50)
EXP2O_RESULT = RESULT_DIR / "experiment_2o_expanded_pairwise_local_chemistry_assignment.json"
DEFAULT_ASSIGNMENT_MODES = "balanced,prior_strong,source_light"
ANIONS = {"O", "N", "F", "Cl", "S", "Se", "Te", "Br", "I", "P", "As"}
G_TRAIN_REPR: dict[str, dict[str, Any]] = {}
G_TRAIN_STRUCT: dict[str, dict[str, Any]] = {}
G_VAL: dict[str, dict[str, Any]] = {}
G_VAL_REPR: dict[str, dict[str, Any]] = {}
G_PROPOSALS: dict[str, dict[str, Any]] = {}
G_ENGINE: OrbitEngine | None = None
G_TRAIN_PRIORS: dict[str, Any] = {}
G_PAIR_PRIORS: dict[str, Any] = {}
G_GEN_ARGS: dict[str, Any] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def elem_radius(symbol: str) -> float:
    try:
        e = Element(str(symbol))
        r = getattr(e, "covalent_radius", None) or getattr(e, "atomic_radius", None)
        if r is None:
            return 1.35
        return max(0.25, float(r))
    except Exception:
        return 1.35


def elem_category(symbol: str) -> str:
    try:
        e = Element(str(symbol))
        s = str(symbol)
        if s in ANIONS:
            return "anion"
        if bool(e.is_metal):
            return "metal"
        if bool(e.is_metalloid):
            return "metalloid"
        return "nonmetal"
    except Exception:
        return "unknown"


def pair_key(a: str, b: str) -> str:
    x, y = sorted((str(a), str(b)))
    return f"{x}|{y}"


def category_pair_key(a: str, b: str) -> str:
    x, y = sorted((elem_category(a), elem_category(b)))
    return f"{x}|{y}"


def robust_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "median": 1.0, "spread": 0.35}
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {"n": 0, "median": 1.0, "spread": 0.35}
    med = float(median(vals))
    q1 = vals[len(vals) // 4]
    q3 = vals[(3 * len(vals)) // 4]
    spread = max(0.08, 0.7413 * (q3 - q1), 0.12 * med)
    return {"n": len(vals), "median": med, "spread": spread}


def extract_pair_ratios_worker(payload: dict[str, Any]) -> dict[str, Any]:
    path = str(payload["source_path"])
    cutoff = float(payload["cutoff"])
    max_sites = int(payload["max_sites"])
    max_pairs = int(payload["max_pairs_per_structure"])
    try:
        structure = Structure.from_file(path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "pairs": {}, "cats": {}}
    if len(structure) < 2 or len(structure) > max_sites:
        return {"ok": True, "skipped": True, "pairs": {}, "cats": {}}
    pairs: dict[str, list[float]] = defaultdict(list)
    cats: dict[str, list[float]] = defaultdict(list)
    matrix = structure.distance_matrix
    observed = 0
    for i in range(len(structure)):
        ai = str(structure[i].specie.symbol)
        ri = elem_radius(ai)
        for j in range(i + 1, len(structure)):
            d = float(matrix[i, j])
            if d <= 1.0e-8 or d > cutoff:
                continue
            bj = str(structure[j].specie.symbol)
            ratio_ij = d / max(0.25, ri + elem_radius(bj))
            if not math.isfinite(ratio_ij):
                continue
            pairs[pair_key(ai, bj)].append(ratio_ij)
            cats[category_pair_key(ai, bj)].append(ratio_ij)
            observed += 1
            if observed >= max_pairs:
                break
        if observed >= max_pairs:
            break
    return {"ok": True, "skipped": False, "pairs": dict(pairs), "cats": dict(cats)}


def build_pair_priors(train_struct: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    records = train_struct[: int(args.max_train_structures)]
    payloads = [
        {
            "source_path": str(row["source_path"]),
            "cutoff": float(args.pair_cutoff),
            "max_sites": int(args.train_pair_max_sites),
            "max_pairs_per_structure": int(args.max_pairs_per_structure),
        }
        for row in records
    ]
    raw_pairs: dict[str, list[float]] = defaultdict(list)
    raw_cats: dict[str, list[float]] = defaultdict(list)
    failures = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=max(1, int(args.prior_workers))) as pool:
        futures = [pool.submit(extract_pair_ratios_worker, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if not res.get("ok"):
                failures += 1
                continue
            if res.get("skipped"):
                skipped += 1
            for k, vals in (res.get("pairs") or {}).items():
                bucket = raw_pairs[str(k)]
                if len(bucket) < int(args.max_values_per_pair):
                    bucket.extend(float(v) for v in vals[: max(0, int(args.max_values_per_pair) - len(bucket))])
            for k, vals in (res.get("cats") or {}).items():
                bucket = raw_cats[str(k)]
                if len(bucket) < int(args.max_values_per_pair):
                    bucket.extend(float(v) for v in vals[: max(0, int(args.max_values_per_pair) - len(bucket))])
            if i % 1000 == 0:
                print(f"[exp2n-priors] processed {i}/{len(payloads)}", flush=True)
    pair_stats = {k: robust_stats(v) for k, v in raw_pairs.items() if len(v) >= int(args.min_pair_count)}
    cat_stats = {k: robust_stats(v) for k, v in raw_cats.items() if len(v) >= int(args.min_category_count)}
    return {
        "pair_stats": pair_stats,
        "category_stats": cat_stats,
        "data_scale": {
            "train_records_requested": len(records),
            "train_records_failed_parse": failures,
            "train_records_skipped": skipped,
            "element_pair_keys": len(pair_stats),
            "category_pair_keys": len(cat_stats),
            "raw_element_pair_keys": len(raw_pairs),
        },
    }


def local_pair_score_from_structure(structure: Structure, priors: dict[str, Any], cutoff: float) -> dict[str, Any]:
    if len(structure) < 2:
        return {"local_pair_score": 0.0, "local_pair_observations": 0, "local_pair_penalty": 0.0, "local_pair_known_rate": None}
    pair_stats = priors["pair_stats"]
    category_stats = priors["category_stats"]
    matrix = structure.distance_matrix
    scores: list[float] = []
    penalties = 0.0
    known = 0
    for i in range(len(structure)):
        ai = str(structure[i].specie.symbol)
        ri = elem_radius(ai)
        for j in range(i + 1, len(structure)):
            d = float(matrix[i, j])
            if d <= 1.0e-8 or d > cutoff:
                continue
            bj = str(structure[j].specie.symbol)
            ratio_ij = d / max(0.25, ri + elem_radius(bj))
            stats = pair_stats.get(pair_key(ai, bj))
            if stats is None:
                stats = category_stats.get(category_pair_key(ai, bj))
            else:
                known += 1
            if stats is None:
                med = 1.0
                spread = 0.35
                prior_n = 1
            else:
                med = max(0.2, float(stats["median"]))
                spread = max(0.08, float(stats["spread"]))
                prior_n = max(1, int(stats["n"]))
            z = abs(math.log(max(1.0e-6, ratio_ij / med))) / spread
            obs_score = -min(9.0, z * z) + 0.08 * min(6.0, math.log1p(prior_n))
            if ratio_ij < 0.55:
                penalties += 8.0 * (0.55 - ratio_ij)
            if ratio_ij > 2.5:
                penalties += 0.5 * min(4.0, ratio_ij - 2.5)
            scores.append(obs_score)
    if not scores:
        return {"local_pair_score": -4.0, "local_pair_observations": 0, "local_pair_penalty": 4.0, "local_pair_known_rate": 0.0}
    return {
        "local_pair_score": float(sum(scores) / len(scores) - penalties / max(1, len(scores))),
        "local_pair_observations": len(scores),
        "local_pair_penalty": float(penalties),
        "local_pair_known_rate": float(known) / float(len(scores)),
    }


def score_rendered_cif(cif: str, priors: dict[str, Any], cutoff: float) -> dict[str, Any]:
    try:
        structure = Structure.from_str(cif, fmt="cif")
    except Exception as exc:  # noqa: BLE001
        return {
            "local_pair_score": -20.0,
            "local_pair_observations": 0,
            "local_pair_penalty": 20.0,
            "local_pair_known_rate": 0.0,
            "local_pair_parse_error": f"{type(exc).__name__}: {exc}",
        }
    out = local_pair_score_from_structure(structure, priors, cutoff)
    out["local_pair_parse_error"] = None
    return out


ASSIGNMENT_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": {"source": 1.40, "exact": 0.38, "orbit": 0.25, "sg": 0.10, "global": 0.05},
    "prior_strong": {"source": 0.45, "exact": 0.95, "orbit": 0.45, "sg": 0.18, "global": 0.05},
    "source_light": {"source": 0.85, "exact": 0.55, "orbit": 0.40, "sg": 0.20, "global": 0.06},
    "sg_orbit": {"source": 0.30, "exact": 0.45, "orbit": 0.65, "sg": 0.42, "global": 0.08},
    "chem_diverse": {"source": 0.25, "exact": 0.20, "orbit": 0.25, "sg": 0.10, "global": 0.05},
}


def weighted_row_element_score(row: dict[str, Any], element: str, sg: int, priors: dict[str, Any], vocab_size: int, mode: str) -> float:
    orbit_id = str(row.get("orbit_id") or "")
    source_element = str(row.get("element") or "")
    weights = ASSIGNMENT_MODE_WEIGHTS.get(str(mode), ASSIGNMENT_MODE_WEIGHTS["balanced"])
    score = weights["source"] * chemical_similarity(element, source_element)
    score += weights["exact"] * logprob(priors["exact"].get((int(sg), orbit_id)), element, vocab_size)
    score += weights["orbit"] * logprob(priors["orbit"].get(orbit_id), element, vocab_size)
    score += weights["sg"] * logprob(priors["sg"].get(int(sg)), element, vocab_size)
    score += weights["global"] * logprob(priors["global"], element, vocab_size)
    if mode == "chem_diverse":
        # A mild deterministic perturbation breaks ties between chemically similar choices.
        code = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{orbit_id}|{element}"))
        score += 0.07 * math.sin((code % 997) / 997.0 * 2.0 * math.pi)
    return float(score)


def exact_cover_assignments_for_mode(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    sg: int,
    priors: dict[str, Any],
    *,
    mode: str,
    limit: int,
    beam_width: int,
) -> list[tuple[float, list[str], dict[str, Any]]]:
    mults = [int(row.get("multiplicity") or 0) for row in rows]
    elements = sorted(str(e) for e, c in counts.items() if int(c) > 0)
    vocab_size = max(1, len(set(priors["global"]) | set(elements)))
    row_scores: dict[tuple[int, str], float] = {}
    row_order_terms: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        scores = {e: weighted_row_element_score(row, e, sg, priors, vocab_size, mode) for e in elements}
        for elem, score in scores.items():
            row_scores[(idx, elem)] = float(score)
        ordered = sorted(scores.values(), reverse=True)
        gap = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
        if mode in {"prior_strong", "sg_orbit"}:
            gap = -abs(gap)
        row_order_terms.append((idx, gap))
    order = [idx for idx, _gap in sorted(row_order_terms, key=lambda x: (-mults[x[0]], -x[1], str(rows[x[0]].get("orbit_id")), x[0]))]
    start_remaining = {str(k): int(v) for k, v in counts.items()}
    states: list[tuple[float, dict[str, int], list[str | None]]] = [(0.0, start_remaining, [None] * len(rows))]
    for idx in order:
        mult = mults[idx]
        choices = sorted(elements, key=lambda e: (-row_scores.get((idx, e), -1.0e9), e))
        if mode == "chem_diverse" and len(choices) > 2:
            choices = choices[:1] + list(reversed(choices[1:]))
        expanded: list[tuple[float, dict[str, int], list[str | None]]] = []
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
        out.append((float(score), final, {"source_preserved_atoms": int(source_preserved), "assignment_mode": str(mode)}))
        if len(out) >= int(limit):
            break
    return out


def assignment_hamming(a: list[str], b: list[str]) -> int:
    return sum(1 for x, y in zip(a, b) if str(x) != str(y))


def diverse_exact_cover_assignments(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    sg: int,
    priors: dict[str, Any],
    *,
    modes: list[str],
    total_limit: int,
    per_mode_limit: int,
    beam_width: int,
    min_hamming: int,
) -> list[tuple[float, list[str], dict[str, Any]]]:
    raw: list[tuple[float, list[str], dict[str, Any]]] = []
    seen: set[tuple[str, ...]] = set()
    for mode_index, mode in enumerate(modes):
        for score, assignment, meta in exact_cover_assignments_for_mode(
            rows,
            counts,
            sg,
            priors,
            mode=mode,
            limit=per_mode_limit,
            beam_width=beam_width,
        ):
            key = tuple(assignment)
            if key in seen:
                continue
            seen.add(key)
            item_meta = dict(meta)
            item_meta["assignment_mode_index"] = int(mode_index)
            raw.append((float(score) - 0.01 * mode_index, assignment, item_meta))
    raw.sort(key=lambda x: (-x[0], tuple(x[1])))
    selected: list[tuple[float, list[str], dict[str, Any]]] = []
    deferred: list[tuple[float, list[str], dict[str, Any]]] = []
    for item in raw:
        assignment = item[1]
        if not selected or all(assignment_hamming(assignment, prev[1]) >= int(min_hamming) for prev in selected):
            selected.append(item)
        else:
            deferred.append(item)
        if len(selected) >= int(total_limit):
            break
    if len(selected) < int(total_limit):
        selected_keys = {tuple(x[1]) for x in selected}
        for item in deferred:
            if tuple(item[1]) in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(tuple(item[1]))
            if len(selected) >= int(total_limit):
                break
    return selected[: int(total_limit)]


def formula_l1(a: dict[str, Any], b: dict[str, Any]) -> float:
    aa = normalize_formula_counts(a)
    bb = normalize_formula_counts(b)
    keys = set(aa) | set(bb)
    return sum(abs(int(aa.get(k, 0)) - int(bb.get(k, 0))) for k in keys) / float(max(1, sum(aa.values()) + sum(bb.values())))


def build_source_retrieval_proposals(
    *,
    train_struct: dict[str, dict[str, Any]],
    train_repr: dict[str, dict[str, Any]],
    val: dict[str, dict[str, Any]],
    val_repr: dict[str, dict[str, Any]],
    base_proposals: dict[str, dict[str, Any]],
    selected_sids: list[str],
    limit: int,
    pool_multiplier: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_sg_atom: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for sid, rec in train_struct.items():
        rep = train_repr.get(sid)
        if rep is None:
            continue
        sg = int(rec["sg"])
        counts = normalize_formula_counts(rec["formula_counts"])
        atom_count = int(rec.get("atom_count") or sum(int(v) for v in counts.values()))
        by_sg_atom[(sg, atom_count)].append(
            {
                "sample_id": sid,
                "sg": sg,
                "atom_count": atom_count,
                "formula_counts": counts,
                "canonical_skeleton_key": str(rep.get("canonical_skeleton_key") or ""),
                "row_count": int(rep.get("row_count") or 0),
            }
        )
    out: dict[str, dict[str, Any]] = {}
    stats = Counter()
    total_candidates = 0
    for sid in selected_sids:
        target = val[sid]
        target_rep = val_repr[sid]
        sg = int(target["sg"])
        counts = normalize_formula_counts(formula_counts(target))
        atom_count = int(sum(int(v) for v in counts.values()))
        existing_sources = {str(p.get("source_sample_id") or "") for p in (base_proposals.get(sid, {}).get("proposals") or [])}
        pool = by_sg_atom.get((sg, atom_count), [])
        scored: list[tuple[float, str, dict[str, Any]]] = []
        target_elems = set(counts)
        for rec in pool:
            source_id = str(rec["sample_id"])
            if source_id in existing_sources:
                continue
            elems = set(rec["formula_counts"])
            jaccard = len(target_elems & elems) / max(1, len(target_elems | elems))
            row_gap = abs(int(rec["row_count"]) - int(target_rep.get("row_count") or 0))
            score = -formula_l1(counts, rec["formula_counts"]) + 0.08 * jaccard - 0.01 * row_gap
            scored.append((float(score), source_id, rec))
        scored.sort(key=lambda x: (-x[0], x[1]))
        proposals = []
        for rank, (score, source_id, rec) in enumerate(scored[: max(1, int(limit) * int(pool_multiplier))], start=1):
            proposals.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "source": "train_sg_atom_formula_deep_retrieval_excluding_exp2o_sources",
                    "source_sample_id": source_id,
                    "source_atom_count": int(rec["atom_count"]),
                    "source_row_count": int(rec["row_count"]),
                    "skeleton_key": str(rec["canonical_skeleton_key"]),
                    "short_skeleton_key": str(rec["canonical_skeleton_key"]).replace("setting=crystalformer|", "")[:240],
                    "exact_cover_feasible": True,
                    "train_rank_before_dedup": rank,
                }
            )
            if len(proposals) >= int(limit):
                break
        if not proposals:
            stats["no_extra_source"] += 1
        total_candidates += len(proposals)
        out[sid] = {"sample_id": sid, "material_id": str(target.get("material_id") or sid.split("__")[-1]), "proposals": proposals}
    return out, {
        "selected_samples": len(selected_sids),
        "sg_atom_buckets": len(by_sg_atom),
        "retrieved_source_candidates": total_candidates,
        "mean_sources_per_sample": ratio(total_candidates, len(selected_sids)),
        "failures": dict(stats),
        "retrieval_limit": int(limit),
    }


def render_sample_payload(sid: str) -> dict[str, Any]:
    engine = G_ENGINE
    if engine is None:
        raise RuntimeError("generation worker engine is not initialized")
    args = G_GEN_ARGS
    target = G_VAL[sid]
    target_repr = G_VAL_REPR[sid]
    counts = {str(k): int(v) for k, v in formula_counts(target).items()}
    target_atom_count = int(sum(counts.values()))
    candidates: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    if target_atom_count > int(args["max_target_atoms"]):
        failures["skipped_target_atom_count_gt_limit"] += 1
        return {
            "payload": {"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": target_atom_count, "sg": int(target["sg"]), "candidates": candidates},
            "generation_meta": generation_meta,
            "failures": dict(failures),
        }
    for proposal in G_PROPOSALS[sid].get("proposals", [])[: int(args["top_skeletons"])]:
        source_id = str(proposal.get("source_sample_id") or "")
        source_repr = G_TRAIN_REPR.get(source_id)
        source_struct = G_TRAIN_STRUCT.get(source_id)
        if source_repr is None or source_struct is None:
            failures["missing_source"] += 1
            continue
        original_rows = source_skeleton_rows(engine, source_repr)
        if not original_rows:
            failures["empty_source_rows"] += 1
            continue
        assignments = diverse_exact_cover_assignments(
            original_rows,
            counts,
            int(target["sg"]),
            G_TRAIN_PRIORS,
            modes=list(args["assignment_modes"]),
            total_limit=int(args["assignment_prelimit"]),
            per_mode_limit=int(args["per_mode_limit"]),
            beam_width=int(args["assignment_beam_width"]),
            min_hamming=int(args["diversity_min_hamming"]),
        )
        if not assignments:
            failures["no_chemical_exact_cover_assignment"] += 1
            continue
        rendered_for_proposal: list[dict[str, Any]] = []
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
                    data_name=f"{sid}_pairchem_p{int(proposal.get('rank') or 0)}_a{assignment_rank}",
                )
                local = score_rendered_cif(cif, G_PAIR_PRIORS, float(args["pair_cutoff"]))
                combined_score = float(args["assignment_score_weight"]) * float(assignment_score) + float(args["local_pair_weight"]) * float(local["local_pair_score"])
                row = {
                    "sample_id": sid,
                    "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                    "proposal_rank": int(proposal.get("rank") or 0),
                    "assignment_rank": int(assignment_rank),
                    "assignment_score": float(assignment_score),
                    "assignment_source_preserved_atoms": int(assignment_meta.get("source_preserved_atoms") or 0),
                    "assignment_mode": str(assignment_meta.get("assignment_mode") or ""),
                    "assignment_mode_index": int(assignment_meta.get("assignment_mode_index") or 0),
                    "local_pair_assignment_score": float(combined_score),
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
                    "site_mapping_rule": "diverse_pairwise_local_chemistry_exact_cover_assignment",
                    "geometry_source": "diverse_pairwise_local_chemistry_source_geometry",
                    "reference_sample_id": source_id,
                    "reference_score": float(combined_score),
                    "param_fallback_rows": int(fallback_count),
                    "cif": cif,
                    **local,
                    **render_meta,
                }
                rendered_for_proposal.append(row)
            except Exception as exc:  # noqa: BLE001
                failures[f"render_failed:{type(exc).__name__}"] += 1
        rendered_for_proposal.sort(
            key=lambda r: (
                -float(r.get("local_pair_assignment_score") or -1.0e9),
                int(r.get("assignment_rank") or 999999),
            )
        )
        for row in rendered_for_proposal[: int(args["assignment_limit"])]:
            item = dict(row)
            item["geometry_rank"] = len(candidates) + 1
            item["raw_generation_order"] = len(candidates) + 1
            candidates.append(item)
            generation_meta.append({k: v for k, v in item.items() if k != "cif"})
    return {
        "payload": {"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": target_atom_count, "sg": int(target["sg"]), "candidates": candidates},
        "generation_meta": generation_meta,
        "failures": dict(failures),
    }


def generate_payloads(args: argparse.Namespace, pair_priors: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    global G_TRAIN_REPR, G_TRAIN_STRUCT, G_VAL, G_VAL_REPR, G_PROPOSALS, G_ENGINE, G_TRAIN_PRIORS, G_PAIR_PRIORS, G_GEN_ARGS
    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    train_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    train_struct_list = list(train_struct.values())
    val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    base_proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in train_struct_list + list(val.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    priors = build_train_priors(train_struct_list)

    selected_sids = [sid for sid in sorted(base_proposals) if sid in val and sid in val_repr and int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]
    proposals, retrieval_context = build_source_retrieval_proposals(
        train_struct=train_struct,
        train_repr=train_repr,
        val=val,
        val_repr=val_repr,
        base_proposals=base_proposals,
        selected_sids=selected_sids,
        limit=int(args.retrieval_sources),
        pool_multiplier=int(args.retrieval_pool_multiplier),
    )

    payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    G_TRAIN_REPR = train_repr
    G_TRAIN_STRUCT = train_struct
    G_VAL = val
    G_VAL_REPR = val_repr
    G_PROPOSALS = proposals
    G_ENGINE = engine
    G_TRAIN_PRIORS = priors
    G_PAIR_PRIORS = pair_priors
    G_GEN_ARGS = {
        "top_skeletons": int(args.top_skeletons),
        "assignment_prelimit": int(args.assignment_prelimit),
        "assignment_limit": int(args.assignment_limit),
        "assignment_beam_width": int(args.assignment_beam_width),
        "max_target_atoms": int(args.max_target_atoms),
        "pair_cutoff": float(args.pair_cutoff),
        "assignment_score_weight": float(args.assignment_score_weight),
        "local_pair_weight": float(args.local_pair_weight),
        "assignment_modes": [x.strip() for x in str(args.assignment_modes).split(",") if x.strip()],
        "per_mode_limit": int(args.per_mode_limit),
        "diversity_min_hamming": int(args.diversity_min_hamming),
    }
    if int(args.generation_workers) <= 1:
        iterator = ((sid, render_sample_payload(sid)) for sid in selected_sids)
        for si, (_sid, res) in enumerate(iterator, start=1):
            payloads.append(res["payload"])
            generation_meta.extend(res["generation_meta"])
            failures.update(res.get("failures") or {})
            if si % 200 == 0:
                print(f"[exp2s-source-retrieval] rendered {si}/{len(selected_sids)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.generation_workers))) as pool:
            futures = {pool.submit(render_sample_payload, sid): sid for sid in selected_sids}
            for si, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                payloads.append(res["payload"])
                generation_meta.extend(res["generation_meta"])
                failures.update(res.get("failures") or {})
                if si % 200 == 0:
                    print(f"[exp2s-source-retrieval] rendered {si}/{len(selected_sids)}", flush=True)
    context = {
        "train_priors": priors["data_scale"],
        "rows_ge7_samples": len(selected_sids),
        "mapping_failures": dict(failures),
        "source_retrieval": retrieval_context,
    }
    return payloads, generation_meta, context, {"val": val, "val_repr": val_repr}


def report_body(result: dict[str, Any]) -> str:
    best_name = result["best_variant"]
    best = result["variants"][best_name]["rows_ge7"]
    exp2o = result["baseline_exp2o"]["rows_ge7"]
    gate = result["gate"]
    return f"""## opentry_14 实验 2s：source retrieval supplement

结果文件：`model/New_model/opentry_14/results/{result['result_filename']}`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2s_source_retrieval_supplement/`

- 为什么做：Exp2q 证明 lattice/free-param posterior 基本不能新增 match，Exp2r 证明 assignment diversity 有少量 oracle 但 ranked gate 不提升；同时 opentry_13 proposals 平均每样本只有约 3.8 个 source，Exp2o 的 top_skeletons=10 多数已经吃满。下一步需要检查更深 train source retrieval 是否能提供新的 skeleton/source basin。
- 核心假设：如果当前瓶颈来自 source skeleton basin 覆盖不足，而不是同一 source 内 assignment/lattice 调整，那么用 formula+GT-SG+atom_count 从 train split 检索更多未被 opentry_13 proposer 选中的 source skeleton，再做 exact-cover assignment，应新增 sample-level match 或提高 Exp2o conversion。
- 数据规模：rows>=7 validation samples `{result['data_scale']['rows_ge7_samples']}`；retrieved source candidates `{result['data_scale']['source_retrieval']['retrieved_source_candidates']}`；generated retrieval assignments `{result['data_scale']['generated_pairchem_assignment_candidates']}`；evaluated retrieval candidates `{result['data_scale']['evaluated_source_retrieval_assignment_candidates']}`；input Exp2o pairchem candidates `{result['data_scale']['input_exp2o_pairchem_candidates']}`；combined candidates `{result['data_scale']['combined_pairchem_candidates']}`；train pair records `{result['pair_priors']['train_records_requested']}`；workers gen/eval/prior `{result['cpu_policy']['generation_workers']}` / `{result['cpu_policy']['eval_workers']}` / `{result['cpu_policy']['prior_workers']}`。
- baseline：Exp2o best `{result['baseline_exp2o']['best_variant']}` rows>=7 match@50 `{pct(exp2o.get('match@50'))}`、conversion `{pct(exp2o.get('skeleton_to_match_conversion@50'))}`、collision `{pct(exp2o.get('collision_rate'))}`。
- 方法变化：保留 Exp2o pairwise local-chemistry pool，并把 deep source retrieval candidates 作为 supplement 与 Exp2o candidates 合并后统一 structural rank；retrieval 只用 train split formula/SG/atom_count/row_count，不使用 validation GT skeleton。assignment 使用 `{','.join(result['method']['assignment_modes'])}` 多模式 exact-cover beam。推理不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、test true CIF 或 official feedback，也不是 RF/HGB/阈值 scorer。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；valid `{pct(best.get('valid_rate'))}`；formula `{pct(best.get('formula_consistency'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`；collision `{pct(best.get('collision_rate'))}`。
- oracle 诊断：combined pair-chem match/skelmatch samples `{result['oracle_diagnostic']['pairchem_match_samples']}` / `{result['oracle_diagnostic']['pairchem_skelmatch_samples']}`；source-retrieval-only match/skelmatch `{result['oracle_diagnostic']['source_retrieval_only_match_samples']}` / `{result['oracle_diagnostic']['source_retrieval_only_skelmatch_samples']}`；相对 Exp2o 新增 match/skelmatch `{result['oracle_diagnostic']['new_match_samples_vs_exp2o']}` / `{result['oracle_diagnostic']['new_skelmatch_samples_vs_exp2o']}`；union upper bound match@50 `{pct(result['oracle_diagnostic']['union_match50_upper_bound'])}`、conversion `{pct(result['oracle_diagnostic']['union_conversion50_upper_bound'])}`。
- 可信度：中等。候选真实 render/parse/SG/StructureMatcher 评估，retrieval 与 assignment 只用 train split 和 target composition/GT-SG；限制是仍沿用 source geometry/free params，且 source retrieval 是 heuristic nearest-neighbor 而非 learned proposer。
- 和历史实验关系：这是 Exp2n/2o/2r 之后从 assignment 内部转向 source skeleton basin 覆盖的实验；若失败，说明简单 train source retrieval 也不足，需要 learned assignment-aware/source-aware generator。
- gate 判定：passed=`{gate['passed']}`；non_regression_vs_exp2o=`{gate['non_regression_vs_exp2o_passed']}`；conversion delta vs Exp2o `{pp(gate['rows_ge7_conversion50_delta_vs_exp2o'])}`；match delta vs Exp2o `{pp(gate['rows_ge7_match50_delta_vs_exp2o'])}`；target_passed=`{gate['target_passed']}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=8)
    parser.add_argument("--assignment-prelimit", type=int, default=32)
    parser.add_argument("--assignment-limit", type=int, default=8)
    parser.add_argument("--assignment-beam-width", type=int, default=256)
    parser.add_argument("--max-target-atoms", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--generation-workers", type=int, default=96)
    parser.add_argument("--eval-workers", type=int, default=128)
    parser.add_argument("--prior-workers", type=int, default=64)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--max-train-structures", type=int, default=8000)
    parser.add_argument("--train-pair-max-sites", type=int, default=120)
    parser.add_argument("--max-pairs-per-structure", type=int, default=3000)
    parser.add_argument("--max-values-per-pair", type=int, default=6000)
    parser.add_argument("--min-pair-count", type=int, default=8)
    parser.add_argument("--min-category-count", type=int, default=25)
    parser.add_argument("--pair-cutoff", type=float, default=4.2)
    parser.add_argument("--assignment-score-weight", type=float, default=0.02)
    parser.add_argument("--local-pair-weight", type=float, default=4.0)
    parser.add_argument("--assignment-modes", default=DEFAULT_ASSIGNMENT_MODES)
    parser.add_argument("--per-mode-limit", type=int, default=24)
    parser.add_argument("--diversity-min-hamming", type=int, default=2)
    parser.add_argument("--retrieval-sources", type=int, default=12)
    parser.add_argument("--retrieval-pool-multiplier", type=int, default=3)
    args = parser.parse_args()

    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_struct_list = list(read_jsonl(STRUCTURED_TRAIN))
    pair_priors = build_pair_priors(train_struct_list, args)
    payloads, generation_meta, context, _loaded = generate_payloads(args, pair_priors)

    suffix = f"_{args.output_suffix}" if str(args.output_suffix).strip() else ""
    result_filename = f"experiment_2s_source_retrieval_supplement{suffix}.json"
    write_jsonl(ARTIFACT_DIR / f"generated_source_retrieval_assignment_meta{suffix}.jsonl", generation_meta)
    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.eval_workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp2s-source-retrieval] evaluated {i}/{len(futures)}", flush=True)
    ranked_pairchem = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / f"evaluated_source_retrieval_assignment_candidates{suffix}.jsonl", ranked_pairchem)

    exp2b = read_json(EXP2B_RESULT)
    exp2d = read_json(EXP2D_RESULT)
    exp2j = read_json(RESULT_DIR / "experiment_2j_chemical_site_order_assignment.json")
    exp2o = read_json(EXP2O_RESULT)
    exp2j_best = str(exp2j["best_variant"])
    exp2o_best = str(exp2o["best_variant"])
    exp2o_eval = list(read_jsonl_iter(Path(exp2o["artifacts"]["evaluated_candidates"])))
    combined_pairchem = assign_structural_ranks(exp2o_eval + ranked_pairchem, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / f"combined_exp2o_plus_source_retrieval_candidates{suffix}.jsonl", combined_pairchem)
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    site_rows = list(read_jsonl_iter(EXP2D_SITE_EVAL))
    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by_sid = by_sid_ranked(assign_structural_ranks(site_rows, int(args.top_k)))
    pairchem_by_sid = by_sid_ranked(combined_pairchem)
    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("h10_s10_chem25_p5", "h10_s5_chem30_p5", "chem30_h10_s5_p5", "h10_interleave_s10_chem25_p5"):
        rows = build_variant_rows(variant=variant, hydrated=hydrated, prototype=prototype, siteassign=site_by_sid, chemassign=pairchem_by_sid, rows_lt7=rows_lt7, top_k=int(args.top_k))
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
    exp2j_rows7 = exp2j["variants"][exp2j_best]["rows_ge7"]
    exp2o_rows7 = exp2o["variants"][exp2o_best]["rows_ge7"]
    conversion_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2o_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(exp2o_rows7.get("match@50") or 0.0)
    target_gate = bool(float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28)
    non_regression_gate = bool(target_gate and conversion_delta >= 0.0 and match_delta >= 0.0)
    plus2pp_gate = bool(conversion_delta >= 0.02 and match_delta >= 0.0)
    passed = bool(non_regression_gate or plus2pp_gate)

    exp2o_sets = sample_sets([r for r in exp2o_eval if int(r.get("row_count") or 0) >= 7])
    pair_sets = sample_sets([r for r in combined_pairchem if int(r.get("row_count") or 0) >= 7])
    retrieval_sets = sample_sets([r for r in ranked_pairchem if int(r.get("row_count") or 0) >= 7])
    union_samples = exp2o_sets["samples"] | pair_sets["samples"]
    union_match = exp2o_sets["match"] | pair_sets["match"]
    union_skel = exp2o_sets["skeleton"] | pair_sets["skeleton"]
    union_skelmatch = exp2o_sets["skelmatch"] | pair_sets["skelmatch"]
    oracle = {
        "pairchem_match_samples": len(pair_sets["match"]),
        "pairchem_skelmatch_samples": len(pair_sets["skelmatch"]),
        "source_retrieval_only_match_samples": len(retrieval_sets["match"]),
        "source_retrieval_only_skelmatch_samples": len(retrieval_sets["skelmatch"]),
        "new_match_samples_vs_exp2o": len(pair_sets["match"] - exp2o_sets["match"]),
        "new_skelmatch_samples_vs_exp2o": len(pair_sets["skelmatch"] - exp2o_sets["skelmatch"]),
        "union_match50_upper_bound": ratio(len(union_match), len(union_samples)),
        "union_conversion50_upper_bound": ratio(len(union_skelmatch), len(union_skel)),
    }

    if plus2pp_gate:
        verdict = "pass_plus2pp_vs_exp2o_gate"
        reason = "Diverse assignment posterior improves rows>=7 conversion by at least +2pp over Exp2o without match loss."
        next_step = "Run Exp3 local optimizer against Exp2r; keep Exp4/5/official frozen."
    elif non_regression_gate and oracle["new_match_samples_vs_exp2o"] > 0:
        verdict = "pass_non_regression_with_new_oracle"
        reason = "Diverse assignment posterior keeps the Exp2o-level target gate and creates new sample-level matches."
        next_step = "Inspect new assignments and decide whether to run Exp3 or refine diversity selection."
    elif passed:
        verdict = "pass_exp2o_non_regression"
        reason = "Diverse assignment posterior preserves Exp2o rows>=7 target metrics but does not add a +2pp lift."
        next_step = "Treat as diagnostic unless Exp3/local optimizer can exploit the added diversity."
    elif conversion_delta > 0.0:
        verdict = "partial_conversion_lift_below_gate"
        reason = "Diverse assignment posterior adds a small conversion lift over Exp2o but not enough for the +2pp gate."
        next_step = "Inspect mode-specific new matches; if coherent, refine assignment diversity rather than lattice/free-param posterior."
    else:
        verdict = "fail_no_conversion_lift"
        reason = "Diverse assignment posterior does not improve Exp2o conversion."
        next_step = "Stop heuristic assignment scoring; move to learned assignment-aware geometry generation or skeleton/source retrieval."

    result = {
        "experiment": "opentry_14_exp2s_source_retrieval_supplement",
        "result_filename": result_filename,
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "exp2o_pairchem_plus_deep_source_retrieval_supplement",
            "top_skeletons": int(args.top_skeletons),
            "assignment_prelimit": int(args.assignment_prelimit),
            "assignment_limit": int(args.assignment_limit),
            "assignment_beam_width": int(args.assignment_beam_width),
            "assignment_modes": [x.strip() for x in str(args.assignment_modes).split(",") if x.strip()],
            "per_mode_limit": int(args.per_mode_limit),
            "diversity_min_hamming": int(args.diversity_min_hamming),
            "retrieval_sources": int(args.retrieval_sources),
            "retrieval_pool_multiplier": int(args.retrieval_pool_multiplier),
            "pair_cutoff": float(args.pair_cutoff),
            "assignment_score_weight": float(args.assignment_score_weight),
            "local_pair_weight": float(args.local_pair_weight),
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
        },
        "cpu_policy": {
            "generation_workers": int(args.generation_workers),
            "eval_workers": int(args.eval_workers),
            "prior_workers": int(args.prior_workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "single_process_cpu_policy": "many single-thread worker processes; keep individual Python workers below the user 200%CPU allowance",
        },
        "pair_priors": pair_priors["data_scale"],
        "train_priors": context["train_priors"],
        "data_scale": {
            "rows_ge7_samples": int(context["rows_ge7_samples"]),
            "input_safe_pool_records": len(safe_rows),
            "input_site_assignment_records": len(site_rows),
            "generated_pairchem_assignment_candidates": len(generation_meta),
            "evaluated_source_retrieval_assignment_candidates": len(ranked_pairchem),
            "input_exp2o_pairchem_candidates": len(exp2o_eval),
            "combined_pairchem_candidates": len(combined_pairchem),
            "selected_variant_paths": selected_paths,
            "source_retrieval": context["source_retrieval"],
        },
        "mapping_failures": context["mapping_failures"],
        "source_retrieval_assignment_only": {"rows_ge7": summarize([r for r in ranked_pairchem if int(r.get("row_count") or 0) >= 7]) if ranked_pairchem else {}},
        "combined_pairchem_assignment_pool": {"rows_ge7": summarize([r for r in combined_pairchem if int(r.get("row_count") or 0) >= 7]) if combined_pairchem else {}},
        "baseline_exp2b": {"result_path": str(EXP2B_RESULT), "rows_ge7": exp2b["rows_ge7"], "overall": exp2b["overall"]},
        "baseline_exp2d": {"result_path": str(EXP2D_RESULT), "best_variant": exp2d["best_variant"], "rows_ge7": exp2d["variants"][exp2d["best_variant"]]["rows_ge7"]},
        "baseline_exp2j": {"result_path": str(RESULT_DIR / "experiment_2j_chemical_site_order_assignment.json"), "best_variant": exp2j_best, "rows_ge7": exp2j_rows7},
        "baseline_exp2o": {"result_path": str(EXP2O_RESULT), "best_variant": exp2o_best, "rows_ge7": exp2o_rows7},
        "oracle_diagnostic": oracle,
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "passed": passed,
            "target_passed": target_gate,
            "non_regression_vs_exp2o_passed": non_regression_gate,
            "plus2pp_vs_exp2o_passed": plus2pp_gate,
            "rows_ge7_match50_delta_vs_exp2o": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2o": conversion_delta,
            "minimum_standard": {"conversion_lift_vs_exp2o": 0.02, "match_not_worse": True, "target_rows_ge7_conversion50": 0.28},
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "generated_meta": str(ARTIFACT_DIR / f"generated_source_retrieval_assignment_meta{suffix}.jsonl"),
            "evaluated_candidates": str(ARTIFACT_DIR / f"evaluated_source_retrieval_assignment_candidates{suffix}.jsonl"),
            "combined_candidates": str(ARTIFACT_DIR / f"combined_exp2o_plus_source_retrieval_candidates{suffix}.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / result_filename, result)
    if not args.skip_report:
        marker = "<!-- OPENTRY14_EXP2S_SOURCE_RETRIEVAL_SUPPLEMENT -->"
        body = report_body(result)
        append_or_replace(REPORT_PATH, marker, body)
        append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / result_filename), "best_variant": best_variant, "gate": result["gate"], "decision": result["decision"], "oracle": oracle}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
