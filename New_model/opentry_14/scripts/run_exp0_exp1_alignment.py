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


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp1_predicted_skeleton_noise_pairs"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from run_symcif_v4_geometry_model_eval import flexible_params_from_reference  # noqa: E402
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
EXP0_SOURCE = OP13 / "results" / "experiment_2_predicted_skeleton_renderer_site_mapping.json"
OP13_EXP3 = OP13 / "results" / "experiment_3_rows_ge7_skeleton_proposer_validation_gate.json"
OP13_EXP4 = OP13 / "results" / "experiment_4_rows_ge7_multi_geometry_proposal.json"
OP13_EXP5 = OP13 / "results" / "experiment_5_main_ablation_and_final_boundary.json"
OP7_MP20 = NEW_MODEL / "opentry_7" / "eval" / "crystallm_a_gt_sg_mp_20_test_k20" / "summary.json"
OP7_MPTS52 = NEW_MODEL / "opentry_7" / "eval" / "crystallm_a_gt_sg_mpts_52_test_k20" / "summary.json"

DATASETS = {
    "mp20": {
        "structured_root": SYMCIF_ROOT / "data" / "structured_symcif_v4_mp20",
        "anchor_path": OP7_MP20,
    },
    "mpts52": {
        "structured_root": SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52",
        "anchor_path": OP7_MPTS52,
    },
}

EXACT_COVER_CACHE: dict[tuple[tuple[int, ...], tuple[int, ...]], bool] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def open_jsonl_gz(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8")


def sample_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("id") or record.get("keys", {}).get("sample_id"))


def formula_counts(record: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in normalize_formula_counts(record["formula_counts"]).items()}


def formula_frac(record: dict[str, Any]) -> dict[str, float]:
    counts = formula_counts(record)
    total = max(1, sum(counts.values()))
    return {k: float(v) / total for k, v in counts.items()}


def formula_l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def formula_equal(a: dict[str, int], b: dict[str, int]) -> bool:
    return dict(sorted(a.items())) == dict(sorted(b.items()))


def skeleton_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in record.get("wa_table") or []:
        rows.append(
            {
                "element": str(row.get("element") or "X"),
                "orbit_id": str(row["orbit_id"]),
                "letter": str(row.get("letter") or ""),
                "multiplicity": int(row.get("multiplicity") or row.get("declared_multiplicity") or 0),
                "site_symmetry": str(row.get("site_symmetry") or ""),
                "free_symbols": list(row.get("free_symbols") or []),
                "enumeration": row.get("enumeration"),
            }
        )
    return rows


def multiplicities(record: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(row.get("multiplicity") or row.get("declared_multiplicity") or 0) for row in record.get("wa_table") or [])


def exact_cover_feasible(mults: tuple[int, ...], counts: dict[str, int]) -> bool:
    targets = tuple(sorted((int(v) for v in counts.values()), reverse=True))
    cache_key = (tuple(sorted(int(x) for x in mults)), targets)
    if cache_key in EXACT_COVER_CACHE:
        return EXACT_COVER_CACHE[cache_key]
    if sum(mults) != sum(targets):
        EXACT_COVER_CACHE[cache_key] = False
        return False
    mm = tuple(sorted((int(x) for x in mults if int(x) > 0), reverse=True))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def rec(i: int, remaining: tuple[int, ...]) -> bool:
        if i >= len(mm):
            return all(v == 0 for v in remaining)
        m = mm[i]
        tried: set[int] = set()
        for idx, value in enumerate(remaining):
            if value < m or value in tried:
                continue
            tried.add(value)
            nxt = list(remaining)
            nxt[idx] -= m
            if rec(i + 1, tuple(sorted(nxt, reverse=True))):
                return True
        return False

    result = rec(0, targets)
    EXACT_COVER_CACHE[cache_key] = result
    return result


def assign_elements(
    *,
    target_counts: dict[str, int],
    source_rows: list[dict[str, Any]],
    source_counts: dict[str, int],
) -> tuple[list[dict[str, Any]] | None, str]:
    if formula_equal(target_counts, source_counts):
        per_element: Counter[str] = Counter()
        for row in source_rows:
            per_element[str(row["element"])] += int(row["multiplicity"])
        if dict(per_element) == target_counts:
            return [dict(row) for row in source_rows], "source_formula_exact_order"

    mults = [int(row["multiplicity"]) for row in source_rows]
    order = sorted(range(len(source_rows)), key=lambda i: (-mults[i], str(source_rows[i]["orbit_id"]), i))
    remaining = dict(target_counts)
    assigned: list[str | None] = [None] * len(source_rows)

    def rec(pos: int) -> bool:
        if pos == len(order):
            return all(int(v) == 0 for v in remaining.values())
        idx = order[pos]
        mult = mults[idx]
        preferred = str(source_rows[idx]["element"])
        choices = sorted(remaining, key=lambda e: (0 if e == preferred else 1, -remaining[e], e))
        for element in choices:
            if remaining[element] < mult:
                continue
            assigned[idx] = element
            remaining[element] -= mult
            if rec(pos + 1):
                return True
            remaining[element] += mult
            assigned[idx] = None
        return False

    if not rec(0):
        return None, "exact_cover_assignment_failed"
    out: list[dict[str, Any]] = []
    per_element = Counter()
    for row, element in zip(source_rows, assigned):
        item = dict(row)
        item["element"] = str(element)
        out.append(item)
        per_element[str(element)] += int(item["multiplicity"])
    if dict(per_element) != target_counts:
        return None, "formula_after_assignment_mismatch"
    return out, "source_preferred_exact_cover"


def build_index(train_rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    index: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for order, row in enumerate(train_rows):
        row["_train_order"] = order
        atom_count = int(row.get("atom_count") or sum(formula_counts(row).values()))
        index[(int(row["sg"]), atom_count)].append(row)
    return dict(index)


def propose_sources(target: dict[str, Any], index: dict[tuple[int, int], list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    target_counts = formula_counts(target)
    target_frac = formula_frac(target)
    target_sid = sample_id(target)
    atom_count = int(target.get("atom_count") or sum(target_counts.values()))
    pool = index.get((int(target["sg"]), atom_count), [])
    scored: list[tuple[float, int, dict[str, Any], bool]] = []
    for source in pool:
        if sample_id(source) == target_sid:
            continue
        mults = multiplicities(source)
        if not exact_cover_feasible(mults, target_counts):
            continue
        source_counts = formula_counts(source)
        exact_formula = formula_equal(target_counts, source_counts)
        source_rows7 = int(source.get("n_sites") or len(source.get("wa_table") or [])) >= 7
        score = 0.0
        score += 3.0 if source_rows7 else 0.0
        score += 2.0 if exact_formula else 0.0
        score += 0.25 if int(source.get("num_elements") or 0) == int(target.get("num_elements") or 0) else 0.0
        score += 0.15 if set(source_counts) == set(target_counts) else 0.0
        score -= formula_l1(target_frac, formula_frac(source))
        score -= 1.0e-6 * int(source.get("_train_order") or 0)
        scored.append((score, int(source.get("_train_order") or 0), source, exact_formula))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, (score, _order, source, exact_formula) in enumerate(scored, start=1):
        skey = str(source.get("canonical_skeleton_key") or "")
        if skey in seen:
            continue
        seen.add(skey)
        out.append(
            {
                "rank": len(out) + 1,
                "score": float(score),
                "source_sample_id": sample_id(source),
                "source_formula": str(source.get("formula") or ""),
                "source_row_count": int(source.get("n_sites") or len(source.get("wa_table") or [])),
                "source_skeleton_key": skey,
                "exact_formula": bool(exact_formula),
                "train_rank_before_dedup": rank,
            }
        )
        if len(out) >= int(top_k):
            break
    return out


def structure_counts(structure: Structure) -> dict[str, int]:
    return {str(k): int(round(v)) for k, v in structure.composition.get_el_amt_dict().items()}


def min_pair_distance(structure: Structure) -> float | None:
    if len(structure) < 2:
        return None
    matrix = structure.distance_matrix
    vals = [float(matrix[i, j]) for i in range(len(structure)) for j in range(i + 1, len(structure))]
    return min(vals) if vals else None


def eval_structure(cif: str, expected_counts: dict[str, int], sg: int, target_atom_count: int) -> dict[str, Any]:
    try:
        structure = Structure.from_str(cif, fmt="cif")
        legal = True
        formula_ok = structure_counts(structure) == expected_counts
        site_count_ok = len(structure) == int(target_atom_count)
        try:
            detected_sg = int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
        except Exception:
            detected_sg = None
        sg_ok = detected_sg == int(sg)
        min_dist = min_pair_distance(structure)
        collision = bool(min_dist is not None and min_dist < 0.5)
        volume_per_atom = float(structure.volume) / max(1, len(structure))
        volume_ok = 2.0 <= volume_per_atom <= 120.0
        valid = bool(legal and formula_ok and site_count_ok and sg_ok and not collision and volume_ok)
        return {
            "parse_success": True,
            "legal_cif": legal,
            "formula_ok": formula_ok,
            "site_count_ok": site_count_ok,
            "space_group_ok": sg_ok,
            "detected_sg": detected_sg,
            "min_pair_distance": min_dist,
            "collision": collision,
            "volume_per_atom": volume_per_atom,
            "volume_per_atom_ok": volume_ok,
            "valid": valid,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "parse_success": False,
            "legal_cif": False,
            "formula_ok": False,
            "site_count_ok": False,
            "space_group_ok": False,
            "detected_sg": None,
            "min_pair_distance": None,
            "collision": None,
            "volume_per_atom": None,
            "volume_per_atom_ok": False,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def target_local_stats(record: dict[str, Any]) -> dict[str, Any]:
    path = str(record.get("source_path") or "")
    try:
        structure = Structure.from_file(path)
        min_dist = min_pair_distance(structure)
        volume_per_atom = float(structure.volume) / max(1, len(structure))
        return {
            "parse_success": True,
            "min_pair_distance": min_dist,
            "collision": bool(min_dist is not None and min_dist < 0.5),
            "volume_per_atom": volume_per_atom,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "parse_success": False,
            "min_pair_distance": None,
            "collision": None,
            "volume_per_atom": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def align_target_params(engine: OrbitEngine, candidate_rows: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any]:
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    orbit_pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target.get("wa_table") or []:
        pools[(str(row.get("orbit_id")), str(row.get("element")))].append(row)
        orbit_pools[str(row.get("orbit_id"))].append(row)
    used_target_rows: set[int] = set()

    def pop_unused(pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        while pool:
            row = pool.pop(0)
            marker = id(row)
            if marker in used_target_rows:
                continue
            used_target_rows.add(marker)
            return row
        return None

    aligned_rows = 0
    exact_element_aligned_rows = 0
    orbit_only_aligned_rows = 0
    rows_requiring_params = 0
    rows_with_param_targets = 0
    values_required = 0
    values_recovered = 0
    target_params_by_row: dict[str, dict[str, float] | None] = {}
    row_alignment: list[dict[str, Any]] = []
    for idx, row in enumerate(candidate_rows):
        key = (str(row["orbit_id"]), str(row["element"]))
        ref = pop_unused(pools[key]) if pools.get(key) else None
        alignment_kind = "none"
        if ref is not None:
            exact_element_aligned_rows += 1
            alignment_kind = "orbit_and_element"
        else:
            ref = pop_unused(orbit_pools[str(row["orbit_id"])]) if orbit_pools.get(str(row["orbit_id"])) else None
            if ref is not None:
                orbit_only_aligned_rows += 1
                alignment_kind = "orbit_only_element_mismatch"
        orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
        expected = [str(x) for x in orbit.free_symbols]
        if ref is not None:
            aligned_rows += 1
        if expected:
            rows_requiring_params += 1
            values_required += len(expected)
            params = {str(k): float(v) % 1.0 for k, v in dict((ref or {}).get("free_params") or {}).items()}
            if all(symbol in params for symbol in expected):
                rows_with_param_targets += 1
                values_recovered += len(expected)
                target_params_by_row[str(idx)] = {symbol: params[symbol] for symbol in expected}
            else:
                target_params_by_row[str(idx)] = None
        else:
            target_params_by_row[str(idx)] = {}
        row_alignment.append(
            {
                "idx": idx,
                "orbit_id": str(row["orbit_id"]),
                "element": str(row["element"]),
                "free_symbols": expected,
                "row_aligned": ref is not None,
                "alignment_kind": alignment_kind,
                "target_element": None if ref is None else str(ref.get("element")),
                "free_param_target_recovered": (not expected) or target_params_by_row[str(idx)] is not None,
            }
        )
    return {
        "aligned_rows": aligned_rows,
        "exact_element_aligned_rows": exact_element_aligned_rows,
        "orbit_only_aligned_rows": orbit_only_aligned_rows,
        "candidate_rows": len(candidate_rows),
        "row_alignment_rate": float(aligned_rows) / max(1, len(candidate_rows)),
        "exact_element_row_alignment_rate": float(exact_element_aligned_rows) / max(1, len(candidate_rows)),
        "orbit_only_row_alignment_rate": float(orbit_only_aligned_rows) / max(1, len(candidate_rows)),
        "rows_requiring_params": rows_requiring_params,
        "rows_with_param_targets": rows_with_param_targets,
        "free_param_row_recovery_rate": float(rows_with_param_targets) / max(1, rows_requiring_params) if rows_requiring_params else 1.0,
        "free_param_values_required": values_required,
        "free_param_values_recovered": values_recovered,
        "free_param_value_recovery_rate": float(values_recovered) / max(1, values_required) if values_required else 1.0,
        "free_param_target_complete": rows_with_param_targets == rows_requiring_params,
        "full_row_alignment": aligned_rows == len(candidate_rows),
        "target_params_by_row": target_params_by_row,
        "row_alignment": row_alignment,
    }


def lattice_delta(initial: dict[str, Any], target: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("a", "b", "c"):
        out[f"{key}_rel_error"] = abs(float(initial[key]) - float(target[key])) / max(1.0e-6, float(target[key]))
    for key in ("alpha", "beta", "gamma"):
        out[f"{key}_abs_error"] = abs(float(initial[key]) - float(target[key]))
    if "volume" in target:
        init_volume = None
        try:
            from symcif_v4.orbit_engine import cell_volume

            init_volume = cell_volume(initial)
        except Exception:
            init_volume = None
        out["volume_rel_error"] = abs(float(init_volume) - float(target["volume"])) / max(1.0e-6, float(target["volume"])) if init_volume is not None else None
    return out


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    rows7 = [r for r in rows if int(r.get("target_row_count") or 0) >= 7]

    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        requiring = [r for r in items if int(r.get("rows_requiring_params") or 0) > 0]
        return {
            "samples": total,
            "rows_ge7_samples": len([r for r in items if int(r.get("target_row_count") or 0) >= 7]),
            "nonempty_pair_rate": float(sum(bool(r.get("pair_constructed")) for r in items)) / max(1, total),
            "initial_valid_rate": float(sum(bool(r.get("initial_quality", {}).get("valid")) for r in items)) / max(1, total),
            "initial_formula_rate": float(sum(bool(r.get("initial_quality", {}).get("formula_ok")) for r in items)) / max(1, total),
            "initial_sg_rate": float(sum(bool(r.get("initial_quality", {}).get("space_group_ok")) for r in items)) / max(1, total),
            "initial_exact_cover_rate": float(sum(bool(r.get("initial_exact_cover_retained")) for r in items)) / max(1, total),
            "initial_collision_rate": float(sum(bool(r.get("initial_quality", {}).get("collision")) for r in items)) / max(1, total),
            "target_local_stats_rate": float(sum(bool(r.get("target_local_stats", {}).get("parse_success")) for r in items)) / max(1, total),
            "skeleton_hit_rate": float(sum(bool(r.get("predicted_skeleton_hit")) for r in items)) / max(1, total),
            "full_row_alignment_rate": float(sum(bool(r.get("full_row_alignment")) for r in items)) / max(1, total),
            "free_param_target_complete_rate": float(sum(bool(r.get("free_param_target_complete")) for r in items)) / max(1, total),
            "free_param_target_complete_rate_among_requiring": float(sum(bool(r.get("free_param_target_complete")) for r in requiring)) / max(1, len(requiring)) if requiring else 1.0,
            "free_param_requiring_samples": len(requiring),
            "usable_joint_pair_rate": float(sum(bool(r.get("usable_joint_pair")) for r in items)) / max(1, total),
            "free_param_value_recovery_rate": float(sum(int(r.get("free_param_values_recovered") or 0) for r in items)) / max(1, sum(int(r.get("free_param_values_required") or 0) for r in items)),
        }

    out = block(rows)
    out["rows_ge7"] = block(rows7)
    return out


def make_exp0() -> dict[str, Any]:
    source = read_json(EXP0_SOURCE)
    exp3 = read_json(OP13_EXP3)
    exp4 = read_json(OP13_EXP4)
    exp5 = read_json(OP13_EXP5)
    selected = source["modes"]["train_prototype"]["selected_by_safe_checks"]
    overall = selected["overall"]
    rows7 = selected["rows_ge7"]
    gate = {
        "passed": bool(
            rows7.get("valid_rate", 0.0) >= 0.96
            and rows7.get("formula_consistency", 0.0) >= 0.99
            and rows7.get("sg_consistency", 0.0) >= 0.98
            and rows7.get("exact_cover_retained", 0.0) >= 1.0
        ),
        "criteria": {
            "rows_ge7_valid_min": 0.96,
            "rows_ge7_formula_min": 0.99,
            "rows_ge7_sg_min": 0.98,
            "rows_ge7_exact_cover_required": 1.0,
        },
    }
    result = {
        "experiment": "opentry_14_exp0_frontend_and_baseline_freeze",
        "time": now_iso(),
        "method": {
            "frontend": "opentry_13 exp2 selected_train_prototype renderer/site mapping",
            "main_anchor": "CrystaLLM-a GT-SG",
            "not_used": ["match labels", "RMSD", "official feedback", "GT-WA", "GT-skeleton", "RF/HGB/rerank"],
        },
        "anchors": {
            "mp20_crystallm_a_gt_sg": read_json(OP7_MP20),
            "mpts52_crystallm_a_gt_sg": read_json(OP7_MPTS52),
        },
        "opentry_13_context": {
            "rows_ge7_skeleton_hit_at50": exp3["rows_ge7"].get("top50_skeleton_hit_coverage"),
            "rows_ge7_hydrated_match_at50": exp3["rows_ge7"].get("top50_hydrated_match_coverage"),
            "rows_ge7_proposal_to_match_conversion_at50": exp3["rows_ge7"].get("top50_proposal_skeleton_to_hydrated_match_conversion"),
            "multi_geometry_rows_ge7_match_at50": exp4["rows_ge7"].get("match@50"),
            "multi_geometry_rows_ge7_conversion_at50": exp4["rows_ge7"].get("skeleton_to_match_conversion@50"),
            "lattice_only_repair_gate_passed": exp5["final_judgment"].get("predicted_skeleton_lattice_repair_gate_passed"),
        },
        "data_scale": source["data_scale"],
        "selected_train_prototype": {
            "overall": overall,
            "rows_ge7": rows7,
            "selection": selected["selection"],
        },
        "gate": gate,
        "decision": {
            "verdict": "pass" if gate["passed"] else "fail_frontend_gate",
            "next_step": "enter experiment_1 data alignment" if gate["passed"] else "repair renderer/site mapping before geometry repair",
        },
        "source": str(EXP0_SOURCE),
    }
    write_json(RESULT_DIR / "experiment_0_frontend_and_baseline_freeze.json", result)
    return result


def run_exp1(args: argparse.Namespace) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dataset_results: dict[str, Any] = {}
    all_pair_rows_for_gate: list[dict[str, Any]] = []
    if int(args.num_shards) > 1:
        pair_path = ARTIFACT_DIR / "shards" / f"predicted_skeleton_noise_geometry_pairs_shard{int(args.shard_index):03d}_of_{int(args.num_shards):03d}.jsonl.gz"
        result_path = RESULT_DIR / f"experiment_1_predicted_skeleton_noise_geometry_pairs_shard{int(args.shard_index):03d}_of_{int(args.num_shards):03d}.json"
    else:
        pair_path = ARTIFACT_DIR / "predicted_skeleton_noise_geometry_pairs.jsonl.gz"
        result_path = RESULT_DIR / "experiment_1_predicted_skeleton_noise_geometry_pairs.json"
    with open_jsonl_gz(pair_path) as out_f:
        for dataset_name, cfg in DATASETS.items():
            root = Path(cfg["structured_root"])
            train_rows = read_jsonl(root / "train.jsonl")
            if args.max_samples is not None:
                train_rows = train_rows[: int(args.max_samples)]
            if int(args.num_shards) > 1:
                train_rows = [row for idx, row in enumerate(train_rows) if idx % int(args.num_shards) == int(args.shard_index)]
            sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in train_rows}
            engine = OrbitEngine(LOOKUP, sg_symbols)
            index = build_index(train_rows)
            by_sid = {sample_id(r): r for r in train_rows}
            pair_rows: list[dict[str, Any]] = []
            failures: Counter[str] = Counter()
            source_counts: Counter[str] = Counter()
            for i, target in enumerate(train_rows, start=1):
                if args.sleep_every and i % int(args.sleep_every) == 0:
                    time.sleep(float(args.sleep_seconds))
                if i % 2000 == 0:
                    print(f"[exp1] {dataset_name}: processed {i}/{len(train_rows)}", flush=True)
                sid = sample_id(target)
                target_counts = formula_counts(target)
                target_atom_count = int(target.get("atom_count") or sum(target_counts.values()))
                proposals = propose_sources(target, index, top_k=max(1, int(args.top_k)))
                if not proposals:
                    failures["no_exact_cover_source"] += 1
                    row = {
                        "dataset": dataset_name,
                        "sample_id": sid,
                        "target_row_count": int(target.get("n_sites") or len(target.get("wa_table") or [])),
                        "pair_constructed": False,
                        "failure_reason": "no_exact_cover_source",
                    }
                    pair_rows.append(row)
                    out_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                proposal = proposals[0]
                source = by_sid.get(str(proposal["source_sample_id"]))
                if source is None:
                    failures["missing_source"] += 1
                    continue
                rows, mapping_rule = assign_elements(
                    target_counts=target_counts,
                    source_rows=skeleton_rows(source),
                    source_counts=formula_counts(source),
                )
                if rows is None:
                    failures[mapping_rule] += 1
                    continue
                try:
                    params, fallback_count = flexible_params_from_reference(engine, rows, source, neural_params=None)
                    expanded = int(engine.expanded_atom_count(rows, params))
                    lattice = {k: float(source["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}
                    cif = engine.render_cif_from_wa_table(
                        rows,
                        lattice=lattice,
                        free_params_by_row=params,
                        formula_counts=target_counts,
                        sg=int(target["sg"]),
                        sg_symbol=str(target.get("sg_symbol") or ""),
                        data_name=f"{dataset_name}_{sid}_exp1_rank1",
                    )
                except Exception as exc:  # noqa: BLE001
                    failures["render_or_params_error"] += 1
                    row = {
                        "dataset": dataset_name,
                        "sample_id": sid,
                        "target_row_count": int(target.get("n_sites") or len(target.get("wa_table") or [])),
                        "pair_constructed": False,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                    pair_rows.append(row)
                    out_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                initial_quality = eval_structure(cif, target_counts, int(target["sg"]), target_atom_count)
                target_stats = target_local_stats(target)
                alignment = align_target_params(engine, rows, target)
                initial_min = initial_quality.get("min_pair_distance")
                target_min = target_stats.get("min_pair_distance")
                initial_vpa = initial_quality.get("volume_per_atom")
                target_vpa = target_stats.get("volume_per_atom")
                source_counts["rows_ge7_source" if int(source.get("n_sites") or 0) >= 7 else "rows_lt7_source"] += 1
                usable_joint_pair = bool(
                    initial_quality.get("parse_success")
                    and target_stats.get("parse_success")
                    and alignment["free_param_target_complete"]
                    and target.get("lattice")
                )
                row = {
                    "dataset": dataset_name,
                    "sample_id": sid,
                    "material_id": str(target.get("material_id") or ""),
                    "sg": int(target["sg"]),
                    "formula_counts": target_counts,
                    "target_atom_count": target_atom_count,
                    "target_row_count": int(target.get("n_sites") or len(target.get("wa_table") or [])),
                    "source_sample_id": str(proposal["source_sample_id"]),
                    "source_row_count": int(proposal["source_row_count"]),
                    "proposal_rank": int(proposal["rank"]),
                    "proposal_score": float(proposal["score"]),
                    "mapping_rule": mapping_rule,
                    "pair_constructed": True,
                    "predicted_skeleton_key": str(proposal["source_skeleton_key"]),
                    "target_skeleton_key": str(target.get("canonical_skeleton_key") or ""),
                    "predicted_skeleton_hit": str(proposal["source_skeleton_key"]) == str(target.get("canonical_skeleton_key") or ""),
                    "candidate_row_count": len(rows),
                    "initial_lattice": lattice,
                    "target_lattice": {k: float(target["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")},
                    "lattice_delta": lattice_delta(lattice, target["lattice"]),
                    "initial_row_params": {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in params.items()},
                    "target_params_by_row": alignment["target_params_by_row"],
                    "param_fallback_rows": int(fallback_count),
                    "initial_exact_cover_retained": expanded == target_atom_count,
                    "initial_quality": initial_quality,
                    "target_local_stats": target_stats,
                    "local_deltas": {
                        "min_pair_distance_delta": float(initial_min) - float(target_min) if initial_min is not None and target_min is not None else None,
                        "volume_per_atom_delta": float(initial_vpa) - float(target_vpa) if initial_vpa is not None and target_vpa is not None else None,
                    },
                    "aligned_rows": alignment["aligned_rows"],
                    "candidate_rows": alignment["candidate_rows"],
                    "row_alignment_rate": alignment["row_alignment_rate"],
                    "rows_requiring_params": alignment["rows_requiring_params"],
                    "rows_with_param_targets": alignment["rows_with_param_targets"],
                    "free_param_row_recovery_rate": alignment["free_param_row_recovery_rate"],
                    "free_param_values_required": alignment["free_param_values_required"],
                    "free_param_values_recovered": alignment["free_param_values_recovered"],
                    "free_param_value_recovery_rate": alignment["free_param_value_recovery_rate"],
                    "free_param_target_complete": alignment["free_param_target_complete"],
                    "full_row_alignment": alignment["full_row_alignment"],
                    "row_alignment": alignment["row_alignment"] if args.keep_row_alignment else None,
                    "usable_joint_pair": usable_joint_pair,
                    "not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback"],
                }
                pair_rows.append(row)
                out_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            summary = summarize_pairs(pair_rows)
            summary["failures"] = dict(failures)
            summary["source_counts"] = dict(source_counts)
            summary["train_records"] = len(train_rows)
            dataset_results[dataset_name] = summary
            all_pair_rows_for_gate.extend(pair_rows)
    all_summary = summarize_pairs(all_pair_rows_for_gate)
    rows7 = all_summary.get("rows_ge7", {})
    gate_pass = bool(
        all_summary.get("nonempty_pair_rate", 0.0) >= 0.80
        and rows7.get("nonempty_pair_rate", 0.0) >= 0.70
        and all_summary.get("target_local_stats_rate", 0.0) >= 0.80
        and all_summary.get("free_param_target_complete_rate_among_requiring", 0.0) >= 0.70
        and rows7.get("free_param_target_complete_rate_among_requiring", 0.0) >= 0.70
        and all_summary.get("free_param_value_recovery_rate", 0.0) >= 0.70
        and rows7.get("free_param_value_recovery_rate", 0.0) >= 0.70
    )
    result = {
        "experiment": "opentry_14_exp1_predicted_skeleton_noise_geometry_pairs",
        "time": now_iso(),
        "method": {
            "datasets": ["mp20", "mpts52"],
            "split": "train",
            "proposer": "train-side exact-cover skeleton proposer using composition/formula + GT-SG + train prototypes; self-source excluded",
            "renderer": "selected_train_prototype-style source lattice/free-params with exact-cover element remapping, rank-1 source for pair construction",
            "target_recovery": "train true lattice, target row free parameters when candidate row aligns by orbit_id+element, target local min-distance/volume/collision stats",
            "not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
        },
        "data_scale": {
            "total_records": len(all_pair_rows_for_gate),
            "pair_artifact": str(pair_path),
            "top_k_sources_scored": int(args.top_k),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
        "datasets": dataset_results,
        "overall": all_summary,
        "gate": {
            "passed": gate_pass,
            "minimum_standard": {
                "overall_nonempty_pair_rate": 0.80,
                "rows_ge7_nonempty_pair_rate": 0.70,
                "target_local_stats_rate": 0.80,
                "overall_free_param_complete_among_requiring": 0.70,
                "rows_ge7_free_param_complete_among_requiring": 0.70,
                "overall_free_param_value_recovery": 0.70,
                "rows_ge7_free_param_value_recovery": 0.70,
            },
        },
        "decision": {
            "verdict": "pass" if gate_pass else "fail_data_alignment_gate",
            "failure_category": None if gate_pass else "free-parameter alignment failed",
            "next_step": "train joint lattice + free-parameter repair head" if gate_pass else "repair data alignment before training; do not enter experiment 2",
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(result_path, result)
    return result


def aggregate_exp1_shards(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    shard_dir = ARTIFACT_DIR / "shards"
    merged_path = ARTIFACT_DIR / "predicted_skeleton_noise_geometry_pairs_merged_sharded.jsonl.gz"
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASETS}
    failures_by_dataset: dict[str, Counter[str]] = {name: Counter() for name in DATASETS}
    source_counts_by_dataset: dict[str, Counter[str]] = {name: Counter() for name in DATASETS}
    all_rows: list[dict[str, Any]] = []
    shard_paths = [
        shard_dir / f"predicted_skeleton_noise_geometry_pairs_shard{idx:03d}_of_{int(args.aggregate_shards):03d}.jsonl.gz"
        for idx in range(int(args.aggregate_shards))
    ]
    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing shard artifacts: {missing[:5]}")
    with open_jsonl_gz(merged_path) as out_f:
        for path in shard_paths:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    dataset = str(row.get("dataset") or "unknown")
                    if dataset not in rows_by_dataset:
                        rows_by_dataset[dataset] = []
                        failures_by_dataset[dataset] = Counter()
                        source_counts_by_dataset[dataset] = Counter()
                    rows_by_dataset[dataset].append(row)
                    all_rows.append(row)
                    if not bool(row.get("pair_constructed")):
                        failures_by_dataset[dataset][str(row.get("failure_reason") or "unknown")] += 1
                    elif int(row.get("source_row_count") or 0) >= 7:
                        source_counts_by_dataset[dataset]["rows_ge7_source"] += 1
                    else:
                        source_counts_by_dataset[dataset]["rows_lt7_source"] += 1
                    out_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    dataset_results: dict[str, Any] = {}
    for dataset_name, rows in rows_by_dataset.items():
        summary = summarize_pairs(rows)
        train_path = DATASETS.get(dataset_name, {}).get("structured_root")
        train_records = None
        if train_path:
            train_records = sum(1 for _ in (Path(train_path) / "train.jsonl").open("r", encoding="utf-8"))
        summary["failures"] = dict(failures_by_dataset[dataset_name])
        summary["source_counts"] = dict(source_counts_by_dataset[dataset_name])
        summary["train_records"] = train_records if train_records is not None else len(rows)
        dataset_results[dataset_name] = summary

    all_summary = summarize_pairs(all_rows)
    rows7 = all_summary.get("rows_ge7", {})
    gate_pass = bool(
        all_summary.get("nonempty_pair_rate", 0.0) >= 0.80
        and rows7.get("nonempty_pair_rate", 0.0) >= 0.70
        and all_summary.get("target_local_stats_rate", 0.0) >= 0.80
        and all_summary.get("free_param_target_complete_rate_among_requiring", 0.0) >= 0.70
        and rows7.get("free_param_target_complete_rate_among_requiring", 0.0) >= 0.70
        and all_summary.get("free_param_value_recovery_rate", 0.0) >= 0.70
        and rows7.get("free_param_value_recovery_rate", 0.0) >= 0.70
    )
    result = {
        "experiment": "opentry_14_exp1_predicted_skeleton_noise_geometry_pairs",
        "time": now_iso(),
        "method": {
            "datasets": ["mp20", "mpts52"],
            "split": "train",
            "proposer": "train-side exact-cover skeleton proposer using composition/formula + GT-SG + train prototypes; self-source excluded",
            "renderer": "selected_train_prototype-style source lattice/free-params with exact-cover element remapping, rank-1 source for pair construction",
            "target_recovery": "train true lattice, target row free parameters when candidate row aligns by orbit_id+element, target local min-distance/volume/collision stats",
            "not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
        },
        "data_scale": {
            "total_records": len(all_rows),
            "pair_artifact": str(merged_path),
            "top_k_sources_scored": int(args.top_k),
            "num_shards": int(args.aggregate_shards),
            "shard_artifacts": [str(p) for p in shard_paths],
        },
        "datasets": dataset_results,
        "overall": all_summary,
        "gate": {
            "passed": gate_pass,
            "minimum_standard": {
                "overall_nonempty_pair_rate": 0.80,
                "rows_ge7_nonempty_pair_rate": 0.70,
                "target_local_stats_rate": 0.80,
                "overall_free_param_complete_among_requiring": 0.70,
                "rows_ge7_free_param_complete_among_requiring": 0.70,
                "overall_free_param_value_recovery": 0.70,
                "rows_ge7_free_param_value_recovery": 0.70,
            },
        },
        "decision": {
            "verdict": "pass" if gate_pass else "fail_data_alignment_gate",
            "failure_category": None if gate_pass else "free-parameter alignment failed",
            "next_step": "train joint lattice + free-parameter repair head" if gate_pass else "repair data alignment before training; do not enter experiment 2",
        },
        "runtime_seconds": time.time() - started,
    }
    out_path = RESULT_DIR / "experiment_1_predicted_skeleton_noise_geometry_pairs_sharded.json"
    write_json(out_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--keep-row-alignment", action="store_true")
    parser.add_argument("--sleep-every", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-exp0", action="store_true")
    parser.add_argument("--aggregate-shards", type=int, default=0)
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if int(args.aggregate_shards) > 0:
        exp1 = aggregate_exp1_shards(args)
        print(
            json.dumps(
                {
                    "aggregate_exp1": exp1["gate"],
                    "outputs": {
                        "exp1": str(RESULT_DIR / "experiment_1_predicted_skeleton_noise_geometry_pairs_sharded.json"),
                        "pairs": exp1["data_scale"]["pair_artifact"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= int(args.shard_index) < int(args.num_shards)):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    exp0 = read_json(RESULT_DIR / "experiment_0_frontend_and_baseline_freeze.json") if args.skip_exp0 else make_exp0()
    exp1 = run_exp1(args) if exp0["gate"]["passed"] else None
    print(
        json.dumps(
            {
                "exp0": exp0["gate"],
                "exp1": None if exp1 is None else exp1["gate"],
                "outputs": {
                    "exp0": str(RESULT_DIR / "experiment_0_frontend_and_baseline_freeze.json"),
                    "exp1": str(RESULT_DIR / "experiment_1_predicted_skeleton_noise_geometry_pairs.json"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
