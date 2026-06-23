#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pymatgen.symmetry.groups import SpaceGroup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_symcif_v2_constrained import classify_failure, record_assignment_signature, record_skeleton_signature  # noqa: E402
from run_generation_eval import load_test_cases, split_concat_records  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def system_from_sg(sg_number: int | None) -> str:
    if sg_number is None:
        return "unknown"
    try:
        return str(SpaceGroup.from_int_number(int(sg_number)).crystal_system)
    except Exception:
        return "unknown"


def group_by_sample(rows: list[dict[str, Any]], n: int) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["gen_index"]) < n:
            grouped[int(row["sample_index"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda r: int(r["gen_index"]))
    return grouped


def angle_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.mean(values), "std": std, "min": min(values), "max": max(values)}


def close_frac(values: list[float], target: float) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if abs(value - target) < 1e-4) / len(values)


def summarize_angles(gens: list[dict[str, Any]], systems: dict[int, str], lookup: WyckoffLookup, n: int) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    failures = Counter()
    for row in gens:
        if int(row["gen_index"]) >= n:
            continue
        system = systems.get(int(row["sample_index"]), "unknown")
        try:
            rec = parse_symcif_v2_text(row.get("generated_text") or "", lookup)
        except Exception as exc:  # noqa: BLE001
            failures[f"{type(exc).__name__}: {str(exc).split(':')[0]}"] += 1
            continue
        values[system]["alpha"].append(float(rec.lattice.alpha))
        values[system]["beta"].append(float(rec.lattice.beta))
        values[system]["gamma"].append(float(rec.lattice.gamma))
    out: dict[str, Any] = {}
    for system, axis_values in sorted(values.items()):
        alpha = axis_values["alpha"]
        beta = axis_values["beta"]
        gamma = axis_values["gamma"]
        out[system] = {
            "count": len(beta),
            "alpha": angle_stats(alpha),
            "beta": angle_stats(beta),
            "gamma": angle_stats(gamma),
            "alpha_eq_90_frac": close_frac(alpha, 90.0),
            "beta_eq_100_frac": close_frac(beta, 100.0),
            "gamma_eq_90_frac": close_frac(gamma, 90.0),
        }
    return {"by_system": out, "parse_failures": dict(failures)}


def summarize_per_system(
    metrics: list[dict[str, Any]],
    systems: dict[int, str],
    total_cases: int,
    n: int,
) -> dict[str, Any]:
    grouped = group_by_sample(metrics, n)
    out: dict[str, dict[str, Any]] = {}
    for sample_index in range(total_cases):
        system = systems.get(sample_index, "unknown")
        rows = grouped.get(sample_index, [])
        bucket = out.setdefault(system, {"samples": 0, "matched": 0, "match_at_1": 0})
        bucket["samples"] += 1
        if rows and rows[0].get("match_ok"):
            bucket["match_at_1"] += 1
        if any(row.get("match_ok") for row in rows):
            bucket["matched"] += 1
    for bucket in out.values():
        samples = int(bucket["samples"])
        bucket["match@1"] = bucket["match_at_1"] / samples if samples else math.nan
        bucket["match@5"] = bucket["matched"] / samples if samples else math.nan
    return dict(sorted(out.items()))


def summarize_conditional(metrics: list[dict[str, Any]], n: int) -> dict[str, Any]:
    rows = [row for row in metrics if int(row["gen_index"]) < n and row.get("formula_ok") and row.get("space_group_ok")]
    matched = [row for row in rows if row.get("match_ok")]
    return {
        "formula_sg_ok_generations": len(rows),
        "matched_generations": len(matched),
        "conditional_match_rate": len(matched) / len(rows) if rows else math.nan,
    }


def summarize_skeleton_diversity(gens: list[dict[str, Any]], lookup: WyckoffLookup, total_cases: int, n: int) -> dict[str, Any]:
    grouped = group_by_sample(gens, n)
    skeleton_counts: list[int] = []
    assignment_counts: list[int] = []
    for sample_index in range(total_cases):
        skeletons: set[str] = set()
        assignments: set[str] = set()
        for row in grouped.get(sample_index, []):
            if row.get("skeleton_signature"):
                skeletons.add(str(row["skeleton_signature"]))
                assignments.add(str(row.get("element_wyckoff_assignment_signature") or row["skeleton_signature"]))
                continue
            try:
                rec = parse_symcif_v2_text(row.get("generated_text") or "", lookup)
            except Exception:
                continue
            skeletons.add(record_skeleton_signature(rec))
            assignments.add(record_assignment_signature(rec))
        skeleton_counts.append(len(skeletons))
        assignment_counts.append(len(assignments))
    return {
        "unique_skeleton_per_sample_mean": statistics.mean(skeleton_counts) if skeleton_counts else math.nan,
        "unique_skeleton_per_sample_median": statistics.median(skeleton_counts) if skeleton_counts else math.nan,
        "unique_assignment_per_sample_mean": statistics.mean(assignment_counts) if assignment_counts else math.nan,
        "unique_assignment_per_sample_median": statistics.median(assignment_counts) if assignment_counts else math.nan,
        "single_skeleton_samples": sum(1 for value in skeleton_counts if value <= 1),
    }


def summarize_hit_distribution(metrics: list[dict[str, Any]], total_cases: int, n: int) -> dict[str, int]:
    grouped = group_by_sample(metrics, n)
    counter = Counter()
    for sample_index in range(total_cases):
        hit_rank = None
        for row in grouped.get(sample_index, []):
            if row.get("match_ok"):
                hit_rank = int(row["gen_index"]) + 1
                break
        counter[str(hit_rank) if hit_rank is not None else "no_match"] += 1
    return dict(counter)


def summarize_unmatched_root_cause(
    metrics: list[dict[str, Any]],
    gens: list[dict[str, Any]],
    gt_records: list[Any],
    lookup: WyckoffLookup,
    n: int,
) -> dict[str, int]:
    gen_by_key = {(int(row["sample_index"]), int(row["gen_index"])): row for row in gens}
    counter = Counter()
    for metric in metrics:
        if int(metric["gen_index"]) >= n:
            continue
        if metric.get("match_ok") or not (metric.get("formula_ok") and metric.get("space_group_ok")):
            continue
        key = (int(metric["sample_index"]), int(metric["gen_index"]))
        gen = gen_by_key.get(key, {})
        label = classify_failure(metric, gen.get("generated_text") or "", gt_records[int(metric["sample_index"])], lookup)
        counter[label] += 1
    return dict(counter.most_common())


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SymCIF-v2 match@5 evaluation outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lookup = WyckoffLookup.from_json(PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    cases = load_test_cases(args.test_limit, modes=tuple(args.modes))
    systems = {case.index: system_from_sg(case.target_sg_number) for case in cases}
    gt_records = [
        parse_symcif_v2_text(text, lookup)
        for text in split_concat_records(PROJECT_ROOT / "data" / "symcif_v2" / "test.txt")[: len(cases)]
    ]
    summary_rows = load_summary(args.run_dir / "summary.csv")
    summary_by_mode_n = {(row["mode"], int(row["n"])): row for row in summary_rows}

    result: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "n": args.n,
        "test_samples": len(cases),
        "summary": {mode: summary_by_mode_n.get((mode, args.n), {}) for mode in args.modes},
        "modes": {},
    }
    for mode in args.modes:
        metrics = load_jsonl(args.run_dir / "metrics" / f"{mode}_per_generation_metrics.jsonl")
        gens = load_jsonl(args.run_dir / "generations" / f"{mode}.jsonl")
        mode_result: dict[str, Any] = {
            "per_system": summarize_per_system(metrics, systems, len(cases), args.n),
            "conditional": summarize_conditional(metrics, args.n),
            "hit_distribution": summarize_hit_distribution(metrics, len(cases), args.n),
        }
        if mode.startswith("symcif_v2"):
            mode_result["angle_sanity"] = summarize_angles(gens, systems, lookup, args.n)
            mode_result["skeleton_diversity"] = summarize_skeleton_diversity(gens, lookup, len(cases), args.n)
            mode_result["unmatched_root_cause_formula_sg_ok"] = summarize_unmatched_root_cause(
                metrics,
                gens,
                gt_records,
                lookup,
                args.n,
            )
        result["modes"][mode] = mode_result

    out_path = args.out or args.run_dir / "analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[analyze] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
