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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import cif_dict, load_test_cases  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402
from symcif_v2.to_cif import render_standard_cif_v2  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_by_sample(rows: list[dict[str, Any]], n: int) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["gen_index"]) < n:
            grouped[int(row["sample_index"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda r: int(r["gen_index"]))
    return grouped


def ratio(rows: list[dict[str, Any]], key: str, denom: int) -> float:
    return sum(1 for row in rows if row.get(key)) / denom if denom else math.nan


def counter_signature(counter: Counter[tuple[Any, ...]]) -> str:
    return "|".join(
        "{}x{}".format(",".join(str(item) for item in key), count)
        for key, count in sorted(counter.items())
    )


def direct_cif_site_pattern(text: str) -> str | None:
    block = cif_dict(text)
    if not block:
        return None
    species = block.get("_atom_site_type_symbol")
    mults = block.get("_atom_site_symmetry_multiplicity")
    if species is None or mults is None:
        return None
    if not isinstance(species, list):
        species = [species]
    if not isinstance(mults, list):
        mults = [mults]
    counts: Counter[tuple[str, int]] = Counter()
    for elem, mult in zip(species, mults, strict=False):
        try:
            counts[(str(elem).strip("'\""), int(float(str(mult).strip("'\""))))] += 1
        except Exception:
            continue
    return counter_signature(counts) if counts else None


def symcif_v2_skeleton(text: str, lookup: WyckoffLookup) -> tuple[str | None, str | None, str | None]:
    try:
        rec = parse_symcif_v2_text(text, lookup)
    except Exception:
        return None, None, None
    skeleton = Counter((str(site.letter), int(site.multiplicity)) for site in rec.sites)
    assignment = Counter((str(site.element), str(site.letter), int(site.multiplicity)) for site in rec.sites)
    return counter_signature(skeleton), counter_signature(assignment), None


def generation_signature(mode: str, row: dict[str, Any], lookup: WyckoffLookup) -> tuple[str | None, str | None]:
    if row.get("skeleton_signature"):
        return str(row.get("skeleton_signature")), str(row.get("element_wyckoff_assignment_signature") or row.get("skeleton_signature"))
    text = row.get("generated_text") or ""
    if mode.startswith("symcif_v2"):
        skeleton, assignment, _ = symcif_v2_skeleton(text, lookup)
        return skeleton, assignment
    pattern = direct_cif_site_pattern(text)
    return pattern, pattern


def write_standard_cifs(out_dir: Path, mode: str, gens: list[dict[str, Any]], lookup: WyckoffLookup, n: int) -> int:
    dst = out_dir / "standard_cifs" / mode
    dst.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in gens:
        if int(row["gen_index"]) >= n:
            continue
        text = row.get("generated_text") or ""
        if not text.strip():
            continue
        try:
            if mode.startswith("symcif_v2"):
                rec = parse_symcif_v2_text(text, lookup)
                cif = render_standard_cif_v2(rec, symprec=0.1, lookup=lookup)
            else:
                cif = text
            path = dst / f"{int(row['sample_index']):04d}_{row['sample_id']}_g{int(row['gen_index']):02d}.cif"
            path.write_text(cif, encoding="utf-8")
            written += 1
        except Exception:
            continue
    return written


def summarize_mode(
    mode: str,
    metrics: list[dict[str, Any]],
    gens: list[dict[str, Any]],
    lookup: WyckoffLookup,
    *,
    total_cases: int,
    n: int,
) -> dict[str, Any]:
    subset = [row for row in metrics if int(row["gen_index"]) < n]
    denom = total_cases * n
    by_sample = group_by_sample(subset, n)
    n1 = 0
    n5 = 0
    best_rms: list[float] = []
    for sample_index in range(total_cases):
        rows = by_sample.get(sample_index, [])
        first = rows[0] if rows else None
        if first and first.get("match_ok"):
            n1 += 1
        rms = [float(row["rms"]) for row in rows if row.get("match_ok") and row.get("rms") is not None]
        if rms:
            n5 += 1
            best_rms.append(min(rms))

    gen_by_sample = group_by_sample(gens, n)
    unique_skeleton_counts: list[int] = []
    unique_assignment_counts: list[int] = []
    for sample_index in range(total_cases):
        skeletons: set[str] = set()
        assignments: set[str] = set()
        for row in gen_by_sample.get(sample_index, []):
            skeleton, assignment = generation_signature(mode, row, lookup)
            if skeleton:
                skeletons.add(skeleton)
            if assignment:
                assignments.add(assignment)
        unique_skeleton_counts.append(len(skeletons))
        unique_assignment_counts.append(len(assignments))

    bond_sum = sum(float(row["bond_length_score"]) for row in subset if row.get("bond_length_score") is not None)
    gen_times = [float(row["generation_time_seconds"]) for row in subset if row.get("generation_time_seconds") is not None]
    row: dict[str, Any] = {
        "mode": mode,
        "n": n,
        "num_attempts": denom,
        "raw_generation_success": ratio(subset, "raw_generation_success", denom),
        "parse_success": ratio(subset, "parse_success", denom),
        "to_cif_success": ratio(subset, "symcif_to_cif_success", denom),
        "readable": ratio(subset, "pymatgen_readable", denom),
        "formula_ok": ratio(subset, "formula_ok", denom),
        "sg_ok": ratio(subset, "space_group_ok", denom),
        "multiplicity_ok": ratio(subset, "multiplicity_ok", denom),
        "valid": ratio(subset, "valid", denom),
        "eval_timeout": ratio(subset, "eval_timeout", denom),
        "bond_score": bond_sum / denom if denom else math.nan,
        "match@1": n1 / total_cases if total_cases else math.nan,
        "match@5": n5 / total_cases if total_cases else math.nan,
        "RMSE": float(np.mean(best_rms)) if best_rms else math.nan,
        "matched_samples_for_RMSE": len(best_rms),
        "unique_skeleton_per_sample_mean": float(statistics.mean(unique_skeleton_counts)) if unique_skeleton_counts else math.nan,
        "unique_skeleton_per_sample_median": float(statistics.median(unique_skeleton_counts)) if unique_skeleton_counts else math.nan,
        "unique_assignment_per_sample_mean": float(statistics.mean(unique_assignment_counts)) if unique_assignment_counts else math.nan,
        "unique_assignment_per_sample_median": float(statistics.median(unique_assignment_counts)) if unique_assignment_counts else math.nan,
        "average_generation_time": float(np.mean(gen_times)) if gen_times else math.nan,
    }
    return row


def fmt_pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "N/A"
    if math.isnan(v):
        return "N/A"
    return f"{v * 100:.2f}%"


def fmt_num(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "N/A"
    if math.isnan(v):
        return "N/A"
    return f"{v:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write match@1/match@5 summary for SymCIF-v2 constrained-v2 evaluation.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_runs" / "symcif_v2_constrained_v2_match1_match5",
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline_minprompt", "symcif_v2_constrained_old", "symcif_v2_constrained_new"],
    )
    parser.add_argument("--date-tag", default="20260520_symcif_v2_v2_match5")
    args = parser.parse_args()

    lookup = WyckoffLookup.from_json(PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    cases = load_test_cases(None, modes=tuple(args.modes))
    rows: list[dict[str, Any]] = []
    std_counts: dict[str, int] = {}
    for mode in args.modes:
        metrics = load_jsonl(args.out_dir / "metrics" / f"{mode}_per_generation_metrics.jsonl")
        gens = load_jsonl(args.out_dir / "generations" / f"{mode}.jsonl")
        rows.append(summarize_mode(mode, metrics, gens, lookup, total_cases=len(cases), n=args.n))
        std_counts[mode] = write_standard_cifs(args.out_dir, mode, gens, lookup, args.n)

    summary_json = args.out_dir / "summary.json"
    summary_csv = args.out_dir / "summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_dir = PROJECT_ROOT / "Log_GPT"
    round_dir = report_dir / "round_20260520_05_symcif_v2_v2_match5"
    report_dir.mkdir(parents=True, exist_ok=True)
    round_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# SymCIF-v2 Constrained-v2 Match@1/Match@5 Eval Report",
        "",
        f"日期：{args.date_tag}",
        "",
        "## 1. 设置",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| test split | {len(cases)} samples |",
        f"| n | {args.n} |",
        "| seeds | 1337-1341 |",
        "| max_new_tokens | 2048 |",
        "| max_match_sites | 96 |",
        "| max_eval_sites | 96 |",
        f"| output dir | `{args.out_dir}` |",
        "",
        "## 2. 汇总结果",
        "",
        "| mode | parse | to_cif | readable | formula_ok | sg_ok | valid | bond_score | match@1 | match@5 | RMSE | uniq_skel_mean | gen_time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report_lines.append(
            "| {mode} | {parse} | {tocif} | {readable} | {formula} | {sg} | {valid} | {bond} | {m1} | {m5} | {rmse} | {uniq} | {time} |".format(
                mode=row["mode"],
                parse=fmt_pct(row["parse_success"]),
                tocif=fmt_pct(row["to_cif_success"]),
                readable=fmt_pct(row["readable"]),
                formula=fmt_pct(row["formula_ok"]),
                sg=fmt_pct(row["sg_ok"]),
                valid=fmt_pct(row["valid"]),
                bond=fmt_num(row["bond_score"]),
                m1=fmt_pct(row["match@1"]),
                m5=fmt_pct(row["match@5"]),
                rmse=fmt_num(row["RMSE"]),
                uniq=fmt_num(row["unique_skeleton_per_sample_mean"]),
                time=fmt_num(row["average_generation_time"]),
            )
        )
    report_lines.extend(
        [
            "",
            "## 3. standard_cifs 输出",
            "",
            "| mode | written CIFs |",
            "| --- | ---: |",
        ]
    )
    for mode, count in std_counts.items():
        report_lines.append(f"| {mode} | {count} |")
    report_lines.extend(
        [
            "",
            "## 4. 输出文件",
            "",
            f"- summary CSV: `{summary_csv}`",
            f"- summary JSON: `{summary_json}`",
            f"- metrics: `{args.out_dir / 'metrics'}`",
            f"- generations: `{args.out_dir / 'generations'}`",
            f"- standard_cifs: `{args.out_dir / 'standard_cifs'}`",
            "",
        ]
    )
    text = "\n".join(report_lines)
    top_report = report_dir / "symcif_v2_constrained_v2_eval_report.md"
    round_report = round_dir / "symcif_v2_constrained_v2_eval_report.md"
    top_report.write_text(text, encoding="utf-8")
    round_report.write_text(text, encoding="utf-8")
    print(f"[summary] wrote {summary_csv}")
    print(f"[summary] wrote {summary_json}")
    print(f"[summary] wrote {top_report}")
    print(f"[summary] wrote {round_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

