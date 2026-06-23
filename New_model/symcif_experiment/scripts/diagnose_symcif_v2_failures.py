#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystallm import CIFTokenizer  # type: ignore  # noqa: E402
from run_generation_eval import load_model, load_test_cases, split_concat_records  # noqa: E402
from sample_symcif_v2_constrained import lattice_system, parse_target_counts, v2_prompt_prefix  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402
from symcif_v2_diagnostic_utils import (  # noqa: E402
    assignment_signature,
    free_coord_errors,
    group_by_sample,
    lattice_error,
    load_jsonl,
    max_cell_error,
    rank_skeleton_candidates,
    record_assignment_counter,
    record_assignment_signature,
    record_skeleton_counter,
    record_skeleton_signature,
    search_legal_skeleton_candidates,
    skeleton_from_record,
    skeleton_signature,
)


def bool_rate(values: list[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else math.nan


def mean_or_nan(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def median_or_nan(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def classify_bucket(metric: dict[str, Any], pred_record: Any | None, gt_record: Any, skeleton_ok: bool, assignment_ok: bool) -> str:
    if metric.get("eval_timeout") or metric.get("match_skipped_reason") == "too_many_sites" or metric.get("conversion_skipped_reason"):
        return "structurematcher_skip_or_timeout"
    if (
        not metric.get("parse_success")
        or not metric.get("symcif_to_cif_success")
        or not metric.get("pymatgen_readable")
        or not metric.get("space_group_ok")
        or pred_record is None
    ):
        return "readable_or_sg_render_issue"
    if not skeleton_ok:
        return "skeleton_wrong"
    if not assignment_ok:
        return "assignment_wrong"
    max_len_rel, max_angle_abs, vol_rel = max_cell_error(gt_record, pred_record)
    if max_len_rel > 0.15 or max_angle_abs > 8.0 or vol_rel > 0.35:
        return "skeleton_assignment_correct_but_cell_bad"
    return "skeleton_assignment_correct_but_free_coord_bad"


def rank_worker(
    *,
    worker_id: int,
    cases_payload: list[dict[str, Any]],
    gt_texts: list[str],
    out_path: str,
    model_dir: str,
    lookup_json: str,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    beam_size: int,
    candidate_limit: int,
    max_sites: int,
    max_expansions_per_beam: int,
    compile_model: bool,
) -> None:
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    tokenizer = CIFTokenizer()
    model = load_model(Path(model_dir), device=device, dtype=dtype, compile_model=compile_model)
    lookup = WyckoffLookup.from_json(lookup_json)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for local_i, case in enumerate(cases_payload, start=1):
            try:
                gt_record = parse_symcif_v2_text(gt_texts[int(case["index"])], lookup)
                gt_skeleton = skeleton_from_record(gt_record, lookup)
                gt_skel_sig = skeleton_signature(gt_skeleton)
                gt_assign_sig = assignment_signature(gt_skeleton)
                prefix = v2_prompt_prefix(case["prompt"])
                candidates = search_legal_skeleton_candidates(
                    model,
                    tokenizer,
                    prefix,
                    parse_target_counts(case["target_formula"]),
                    int(case["target_sg_number"]),
                    lookup,
                    device=device,
                    dtype=dtype,
                    temperature=temperature,
                    top_k=top_k,
                    beam_size=beam_size,
                    candidate_limit=candidate_limit,
                    max_sites=max_sites,
                    max_expansions_per_beam=max_expansions_per_beam,
                )
                ranked = rank_skeleton_candidates(model, tokenizer, prefix, candidates, device=device, dtype=dtype)
                skel_rank = None
                assign_rank = None
                for idx, beam in enumerate(ranked, start=1):
                    if skel_rank is None and skeleton_signature(beam.skeleton) == gt_skel_sig:
                        skel_rank = idx
                    if assign_rank is None and assignment_signature(beam.skeleton) == gt_assign_sig:
                        assign_rank = idx
                    if skel_rank is not None and assign_rank is not None:
                        break
                row = {
                    "sample_index": case["index"],
                    "sample_id": case["sample_id"],
                    "candidate_count": len(ranked),
                    "gt_skeleton_signature": gt_skel_sig,
                    "gt_assignment_signature": gt_assign_sig,
                    "gt_skeleton_rank": skel_rank,
                    "gt_assignment_rank": assign_rank,
                    "missing_gt_skeleton": skel_rank is None,
                    "missing_gt_assignment": assign_rank is None,
                    "gt_skeleton_in_top5": skel_rank is not None and skel_rank <= 5,
                    "gt_assignment_in_top5": assign_rank is not None and assign_rank <= 5,
                    "top_skeleton_signature": skeleton_signature(ranked[0].skeleton) if ranked else None,
                    "top_assignment_signature": assignment_signature(ranked[0].skeleton) if ranked else None,
                    "top_normalized_logprob": ranked[0].normalized_logprob if ranked else None,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "sample_index": case["index"],
                    "sample_id": case["sample_id"],
                    "candidate_count": 0,
                    "gt_skeleton_rank": None,
                    "gt_assignment_rank": None,
                    "missing_gt_skeleton": True,
                    "missing_gt_assignment": True,
                    "gt_skeleton_in_top5": False,
                    "gt_assignment_in_top5": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            if local_i % 10 == 0 or local_i == len(cases_payload):
                print(f"[rank-diagnosis:{device}:worker{worker_id}] {local_i}/{len(cases_payload)} prompts done", flush=True)


def compute_gt_ranks(args: argparse.Namespace, cases: list[Any]) -> Path:
    out_path = args.out_dir / "gt_skeleton_logprob_rank.jsonl"
    expected = len(cases)
    if out_path.exists() and not args.overwrite_rank:
        existing = sum(1 for _ in out_path.open(encoding="utf-8"))
        if existing == expected:
            print(f"[diagnosis] found complete GT rank file -> {out_path}")
            return out_path
    gt_texts = split_concat_records(PROJECT_ROOT / "data" / "symcif_v2" / "test.txt")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()] or ["cpu"]
    chunks: list[list[Any]] = [[] for _ in devices]
    for i, case in enumerate(cases):
        chunks[i % len(devices)].append(case)
    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    worker_paths: list[Path] = []
    for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
        payload = [
            {
                "index": c.index,
                "sample_id": c.sample_id,
                "prompt": c.prompts["symcif_v2_constrained"],
                "target_formula": c.target_formula,
                "target_sg_number": c.target_sg_number,
            }
            for c in chunk
        ]
        worker_path = args.out_dir / f"gt_skeleton_logprob_rank.worker{worker_id}.jsonl"
        worker_paths.append(worker_path)
        proc = ctx.Process(
            target=rank_worker,
            kwargs={
                "worker_id": worker_id,
                "cases_payload": payload,
                "gt_texts": gt_texts,
                "out_path": str(worker_path),
                "model_dir": str(args.model_dir),
                "lookup_json": str(args.lookup_json),
                "device": device,
                "dtype": args.dtype,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "beam_size": args.rank_beam_size,
                "candidate_limit": args.rank_candidate_limit,
                "max_sites": args.rank_max_sites,
                "max_expansions_per_beam": args.rank_max_expansions_per_beam,
                "compile_model": args.compile,
            },
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"rank diagnosis worker failed with exit code {proc.exitcode}")
    rows: list[dict[str, Any]] = []
    for path in worker_paths:
        rows.extend(load_jsonl(path))
    rows.sort(key=lambda r: int(r["sample_index"]))
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return out_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_ranks(rank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rank_rows)
    ranks = [int(row["gt_skeleton_rank"]) for row in rank_rows if row.get("gt_skeleton_rank") not in (None, "")]
    assign_ranks = [int(row["gt_assignment_rank"]) for row in rank_rows if row.get("gt_assignment_rank") not in (None, "")]
    return {
        "rank_total_samples": total,
        "gt_skeleton_missing_rate": sum(1 for row in rank_rows if row.get("missing_gt_skeleton")) / total if total else math.nan,
        "gt_assignment_missing_rate": sum(1 for row in rank_rows if row.get("missing_gt_assignment")) / total if total else math.nan,
        "gt_skeleton_rank_le1": sum(1 for r in ranks if r <= 1) / total if total else math.nan,
        "gt_skeleton_rank_le5": sum(1 for r in ranks if r <= 5) / total if total else math.nan,
        "gt_skeleton_rank_le10": sum(1 for r in ranks if r <= 10) / total if total else math.nan,
        "gt_skeleton_rank_le50": sum(1 for r in ranks if r <= 50) / total if total else math.nan,
        "gt_skeleton_rank_median": median_or_nan([float(r) for r in ranks]),
        "gt_assignment_rank_le5": sum(1 for r in assign_ranks if r <= 5) / total if total else math.nan,
        "gt_assignment_rank_median": median_or_nan([float(r) for r in assign_ranks]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose SymCIF-v2 failures at skeleton/assignment/cell/coord levels.")
    parser.add_argument("--eval-run-dir", type=Path, default=PROJECT_ROOT / "eval_runs" / "symcif_v2_fixed_cell_match5_20260521")
    parser.add_argument("--mode", default="symcif_v2_constrained_fixed_cell")
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_runs" / "generation_eval_input_ablation_t1_topk10_n20_20260520",
    )
    parser.add_argument("--baseline-mode", default="baseline_minprompt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_runs" / "symcif_v2_diagnosis_20260521")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--compute-rank", action="store_true")
    parser.add_argument("--overwrite-rank", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "runs" / "exp_symcif_v2")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rank-beam-size", type=int, default=64)
    parser.add_argument("--rank-candidate-limit", type=int, default=128)
    parser.add_argument("--rank-max-sites", type=int, default=64)
    parser.add_argument("--rank-max-expansions-per-beam", type=int, default=24)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lookup = WyckoffLookup.from_json(args.lookup_json)
    cases = load_test_cases(args.test_limit, modes=("symcif_v2_constrained",))
    gt_records = [parse_symcif_v2_text(text, lookup) for text in split_concat_records(PROJECT_ROOT / "data" / "symcif_v2" / "test.txt")]

    metrics = load_jsonl(args.eval_run_dir / "metrics" / f"{args.mode}_per_generation_metrics.jsonl")
    gens = load_jsonl(args.eval_run_dir / "generations" / f"{args.mode}.jsonl")
    baseline_metrics = load_jsonl(args.baseline_run_dir / "metrics" / f"{args.baseline_mode}_per_generation_metrics.jsonl")
    allowed_sample_indices = {int(case.index) for case in cases}
    metrics = [row for row in metrics if int(row["sample_index"]) in allowed_sample_indices]
    gens = [row for row in gens if int(row["sample_index"]) in allowed_sample_indices]
    baseline_metrics = [row for row in baseline_metrics if int(row["sample_index"]) in allowed_sample_indices]
    gen_by_key = {(int(row["sample_index"]), int(row["gen_index"])): row for row in gens}
    metric_by_sample = group_by_sample(metrics, args.n)
    baseline_by_sample = group_by_sample(baseline_metrics, args.n)

    per_gen: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    free_coord_values: list[float] = []
    bucket_counter: Counter[str] = Counter()
    conditional_skeleton_rows = 0
    conditional_skeleton_matches = 0
    conditional_assignment_rows = 0
    conditional_assignment_matches = 0
    for metric in metrics:
        if int(metric["gen_index"]) >= args.n:
            continue
        gt = gt_records[int(metric["sample_index"])]
        gen = gen_by_key.get((int(metric["sample_index"]), int(metric["gen_index"])), {})
        pred = None
        parse_error = None
        try:
            pred = parse_symcif_v2_text(gen.get("generated_text") or "", lookup)
        except Exception as exc:  # noqa: BLE001
            parse_error = f"{type(exc).__name__}: {exc}"
        gt_skel_sig = record_skeleton_signature(gt)
        gt_assign_sig = record_assignment_signature(gt)
        pred_skel_sig = record_skeleton_signature(pred) if pred is not None else None
        pred_assign_sig = record_assignment_signature(pred) if pred is not None else None
        skeleton_ok = pred is not None and record_skeleton_counter(pred) == record_skeleton_counter(gt)
        assignment_ok = pred is not None and record_assignment_counter(pred) == record_assignment_counter(gt)
        if skeleton_ok:
            conditional_skeleton_rows += 1
            if metric.get("match_ok"):
                conditional_skeleton_matches += 1
        if assignment_ok:
            conditional_assignment_rows += 1
            if metric.get("match_ok"):
                conditional_assignment_matches += 1
        bucket = "matched" if metric.get("match_ok") else classify_bucket(metric, pred, gt, skeleton_ok, assignment_ok)
        if not metric.get("match_ok"):
            bucket_counter[bucket] += 1
        cell_err = {}
        coord_mae = None
        coord_count = 0
        if pred is not None and metric.get("formula_ok") and metric.get("space_group_ok") and skeleton_ok and assignment_ok:
            cell_err = lattice_error(gt, pred)
            system = lattice_system(int(gt.sg_number))
            cell_rows.append(
                {
                    "sample_index": metric["sample_index"],
                    "sample_id": metric["sample_id"],
                    "gen_index": metric["gen_index"],
                    "crystal_system": system,
                    **cell_err,
                }
            )
            coord_mae, coord_count = free_coord_errors(gt, pred)
            if coord_mae is not None:
                free_coord_values.append(float(coord_mae))
        per_gen.append(
            {
                "sample_index": metric["sample_index"],
                "sample_id": metric["sample_id"],
                "gen_index": metric["gen_index"],
                "match_ok": bool(metric.get("match_ok")),
                "formula_ok": bool(metric.get("formula_ok")),
                "sg_ok": bool(metric.get("space_group_ok")),
                "readable": bool(metric.get("pymatgen_readable")),
                "valid": bool(metric.get("valid")),
                "skeleton_correct": bool(skeleton_ok),
                "assignment_correct": bool(assignment_ok),
                "gt_skeleton_signature": gt_skel_sig,
                "pred_skeleton_signature": pred_skel_sig,
                "gt_assignment_signature": gt_assign_sig,
                "pred_assignment_signature": pred_assign_sig,
                "unmatched_bucket": bucket,
                "parse_error": parse_error,
                "free_coord_mae": coord_mae,
                "free_coord_count": coord_count,
                **cell_err,
            }
        )

    per_sample: list[dict[str, Any]] = []
    for case in cases:
        rows = metric_by_sample.get(case.index, [])
        base_rows = baseline_by_sample.get(case.index, [])
        diag_rows = [row for row in per_gen if int(row["sample_index"]) == case.index]
        first = diag_rows[0] if diag_rows else {}
        per_sample.append(
            {
                "sample_index": case.index,
                "sample_id": case.sample_id,
                "crystal_system": lattice_system(int(case.target_sg_number or 1)),
                "baseline_match@1": bool(base_rows and base_rows[0].get("match_ok")),
                "baseline_match@5": any(row.get("match_ok") for row in base_rows),
                "match@1": bool(rows and rows[0].get("match_ok")),
                "match@5": any(row.get("match_ok") for row in rows),
                "skeleton@1": bool(first.get("skeleton_correct")),
                "skeleton@5": any(row.get("skeleton_correct") for row in diag_rows),
                "assignment@1": bool(first.get("assignment_correct")),
                "assignment@5": any(row.get("assignment_correct") for row in diag_rows),
                "gt_skeleton_signature": first.get("gt_skeleton_signature"),
                "gt_assignment_signature": first.get("gt_assignment_signature"),
            }
        )

    rank_rows: list[dict[str, Any]] = []
    if args.compute_rank:
        rank_path = compute_gt_ranks(args, cases)
        rank_rows = load_jsonl(rank_path)
        rank_by_sample = {int(row["sample_index"]): row for row in rank_rows}
        for row in per_sample:
            rank = rank_by_sample.get(int(row["sample_index"]), {})
            row.update(
                {
                    "gt_skeleton_rank": rank.get("gt_skeleton_rank"),
                    "gt_assignment_rank": rank.get("gt_assignment_rank"),
                    "missing_gt_skeleton": rank.get("missing_gt_skeleton"),
                    "missing_gt_assignment": rank.get("missing_gt_assignment"),
                    "rank_candidate_count": rank.get("candidate_count"),
                }
            )

    write_csv(args.out_dir / "per_generation_diagnosis.csv", per_gen)
    write_csv(args.out_dir / "per_sample_diagnosis.csv", per_sample)

    denom_samples = len(cases)
    summary: dict[str, Any] = {
        "mode": args.mode,
        "baseline_mode": args.baseline_mode,
        "samples": denom_samples,
        "n": args.n,
        "match@1": bool_rate([bool(row["match@1"]) for row in per_sample]),
        "match@5": bool_rate([bool(row["match@5"]) for row in per_sample]),
        "baseline_match@1": bool_rate([bool(row["baseline_match@1"]) for row in per_sample]),
        "baseline_match@5": bool_rate([bool(row["baseline_match@5"]) for row in per_sample]),
        "skeleton@1": bool_rate([bool(row["skeleton@1"]) for row in per_sample]),
        "skeleton@5": bool_rate([bool(row["skeleton@5"]) for row in per_sample]),
        "assignment@1": bool_rate([bool(row["assignment@1"]) for row in per_sample]),
        "assignment@5": bool_rate([bool(row["assignment@5"]) for row in per_sample]),
        "conditional_match_given_skeleton_correct": conditional_skeleton_matches / conditional_skeleton_rows
        if conditional_skeleton_rows
        else math.nan,
        "conditional_match_given_assignment_correct": conditional_assignment_matches / conditional_assignment_rows
        if conditional_assignment_rows
        else math.nan,
        "conditional_skeleton_generation_rows": conditional_skeleton_rows,
        "conditional_assignment_generation_rows": conditional_assignment_rows,
        "free_coord_mae_mean": mean_or_nan(free_coord_values),
        "free_coord_mae_median": median_or_nan(free_coord_values),
    }
    for key in ("a_abs_error", "b_abs_error", "c_abs_error", "alpha_abs_error", "beta_abs_error", "gamma_abs_error", "volume_rel_error"):
        vals = [float(row[key]) for row in cell_rows if row.get(key) is not None]
        summary[key + "_mean"] = mean_or_nan(vals)
        summary[key + "_median"] = median_or_nan(vals)
    for bucket, count in sorted(bucket_counter.items()):
        summary[f"bucket_{bucket}"] = count
        summary[f"bucket_{bucket}_rate_of_unmatched"] = count / sum(bucket_counter.values()) if bucket_counter else math.nan

    per_system_rows: list[dict[str, Any]] = []
    for system in sorted({row["crystal_system"] for row in per_sample}):
        sample_rows = [row for row in per_sample if row["crystal_system"] == system]
        system_cell = [row for row in cell_rows if row["crystal_system"] == system]
        sys_row = {
            "group": f"system:{system}",
            "samples": len(sample_rows),
            "match@5": bool_rate([bool(row["match@5"]) for row in sample_rows]),
            "skeleton@5": bool_rate([bool(row["skeleton@5"]) for row in sample_rows]),
            "assignment@5": bool_rate([bool(row["assignment@5"]) for row in sample_rows]),
        }
        for key in ("a_abs_error", "b_abs_error", "c_abs_error", "alpha_abs_error", "beta_abs_error", "gamma_abs_error", "volume_rel_error"):
            vals = [float(row[key]) for row in system_cell if row.get(key) is not None]
            sys_row[key + "_mean"] = mean_or_nan(vals)
        per_system_rows.append(sys_row)
    summary_rows = [{"group": "overall", **summary}, *per_system_rows]
    if rank_rows:
        rank_summary = summarize_ranks(rank_rows)
        summary_rows.append({"group": "gt_skeleton_logprob_rank", **rank_summary})
        summary.update(rank_summary)

    (args.out_dir / "diagnosis_summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.out_dir / "diagnosis_summary.csv", summary_rows)
    write_report(args, summary, summary_rows, bucket_counter, rank_rows)
    return 0


def fmt_pct(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "N/A"
    if math.isnan(v):
        return "N/A"
    return f"{v * 100:.2f}%"


def fmt_num(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return "N/A"
    if math.isnan(v):
        return "N/A"
    return f"{v:.{digits}f}"


def write_report(
    args: argparse.Namespace,
    summary: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    bucket_counter: Counter[str],
    rank_rows: list[dict[str, Any]],
) -> None:
    rank_summary = next((row for row in summary_rows if row.get("group") == "gt_skeleton_logprob_rank"), {})
    lines = [
        "# SymCIF-v2 Failure Diagnosis Report",
        "",
        "日期：2026-05-21",
        "",
        "## 输入",
        "",
        f"- baseline run: `{args.baseline_run_dir}`",
        f"- diagnosed run: `{args.eval_run_dir}`",
        f"- mode: `{args.mode}`",
        f"- output dir: `{args.out_dir}`",
        "",
        "## 样本级指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| baseline match@1 | {fmt_pct(summary.get('baseline_match@1'))} |",
        f"| baseline match@5 | {fmt_pct(summary.get('baseline_match@5'))} |",
        f"| fixed_cell match@1 | {fmt_pct(summary.get('match@1'))} |",
        f"| fixed_cell match@5 | {fmt_pct(summary.get('match@5'))} |",
        f"| skeleton@1 | {fmt_pct(summary.get('skeleton@1'))} |",
        f"| skeleton@5 | {fmt_pct(summary.get('skeleton@5'))} |",
        f"| assignment@1 | {fmt_pct(summary.get('assignment@1'))} |",
        f"| assignment@5 | {fmt_pct(summary.get('assignment@5'))} |",
        f"| conditional match given skeleton_correct | {fmt_pct(summary.get('conditional_match_given_skeleton_correct'))} |",
        f"| conditional match given assignment_correct | {fmt_pct(summary.get('conditional_match_given_assignment_correct'))} |",
        "",
        "## Cell / Coord 误差",
        "",
        "| 指标 | mean | median |",
        "| --- | ---: | ---: |",
        f"| a abs error | {fmt_num(summary.get('a_abs_error_mean'))} | {fmt_num(summary.get('a_abs_error_median'))} |",
        f"| b abs error | {fmt_num(summary.get('b_abs_error_mean'))} | {fmt_num(summary.get('b_abs_error_median'))} |",
        f"| c abs error | {fmt_num(summary.get('c_abs_error_mean'))} | {fmt_num(summary.get('c_abs_error_median'))} |",
        f"| alpha abs error | {fmt_num(summary.get('alpha_abs_error_mean'))} | {fmt_num(summary.get('alpha_abs_error_median'))} |",
        f"| beta abs error | {fmt_num(summary.get('beta_abs_error_mean'))} | {fmt_num(summary.get('beta_abs_error_median'))} |",
        f"| gamma abs error | {fmt_num(summary.get('gamma_abs_error_mean'))} | {fmt_num(summary.get('gamma_abs_error_median'))} |",
        f"| volume rel error | {fmt_num(summary.get('volume_rel_error_mean'))} | {fmt_num(summary.get('volume_rel_error_median'))} |",
        f"| free coord MAE | {fmt_num(summary.get('free_coord_mae_mean'))} | {fmt_num(summary.get('free_coord_mae_median'))} |",
        "",
        "## Unmatched 分桶",
        "",
        "| bucket | generations |",
        "| --- | ---: |",
    ]
    for bucket, count in bucket_counter.most_common():
        lines.append(f"| {bucket} | {count} |")
    lines.extend(
        [
            "",
            "## GT Skeleton Logprob Rank",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
        ]
    )
    if rank_rows:
        lines.extend(
            [
                f"| samples | {rank_summary.get('rank_total_samples', 0)} |",
                f"| missing_gt_skeleton | {fmt_pct(rank_summary.get('gt_skeleton_missing_rate'))} |",
                f"| skeleton rank <= 1 | {fmt_pct(rank_summary.get('gt_skeleton_rank_le1'))} |",
                f"| skeleton rank <= 5 | {fmt_pct(rank_summary.get('gt_skeleton_rank_le5'))} |",
                f"| skeleton rank <= 10 | {fmt_pct(rank_summary.get('gt_skeleton_rank_le10'))} |",
                f"| skeleton rank <= 50 | {fmt_pct(rank_summary.get('gt_skeleton_rank_le50'))} |",
                f"| skeleton rank median | {fmt_num(rank_summary.get('gt_skeleton_rank_median'), 1)} |",
                f"| assignment rank <= 5 | {fmt_pct(rank_summary.get('gt_assignment_rank_le5'))} |",
                f"| assignment rank median | {fmt_num(rank_summary.get('gt_assignment_rank_median'), 1)} |",
            ]
        )
    else:
        lines.append("| status | not computed in this run |")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- `diagnosis_summary.csv`: `{args.out_dir / 'diagnosis_summary.csv'}`",
            f"- `diagnosis_summary.json`: `{args.out_dir / 'diagnosis_summary.json'}`",
            f"- `per_sample_diagnosis.csv`: `{args.out_dir / 'per_sample_diagnosis.csv'}`",
            f"- `per_generation_diagnosis.csv`: `{args.out_dir / 'per_generation_diagnosis.csv'}`",
            "",
            "## 诊断说明",
            "",
            "- skeleton_signature 只比较 Wyckoff multiplicity + letter，忽略元素与顺序。",
            "- assignment_signature 比较 element + multiplicity + letter，忽略行顺序。",
            "- GT skeleton rank 使用合法 skeleton beam candidate pool + 完整 skeleton 文本 length-normalized LM logprob 排序；这是离线 oracle 诊断，不参与正式生成。",
        ]
    )
    local_dir = args.out_dir / "Log_GPT"
    global_dir = PROJECT_ROOT / "Log_GPT"
    round_dir = global_dir / "round_20260521_03_diagnosis_oracle_rank"
    for directory in (local_dir, global_dir, round_dir):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "symcif_v2_diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[diagnosis] wrote report -> {global_dir / 'symcif_v2_diagnosis_report.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
