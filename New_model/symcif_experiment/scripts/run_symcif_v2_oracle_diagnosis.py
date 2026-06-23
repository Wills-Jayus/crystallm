#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystallm import CIFTokenizer  # type: ignore  # noqa: E402
from run_generation_eval import extract_generated_record, load_model, load_test_cases, split_concat_records  # noqa: E402
from sample_symcif_v2_constrained import ConstraintStats, sample_cell_tail, sample_unit_coord, v2_prompt_prefix  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402
from symcif_v2.to_cif import render_standard_cif_v2  # noqa: E402
from symcif_v2_diagnostic_utils import (  # noqa: E402
    free_coord_errors,
    group_by_sample,
    lattice_error,
    load_jsonl,
    skeleton_from_record,
)


ORACLE_A_MODE = "symcif_v2_oracle_gt_skeleton"
ORACLE_B_MODE = "symcif_v2_oracle_gt_skeleton_gt_cell"


def append_gt_cell(record: Any) -> str:
    lat = record.lattice
    return (
        f"\n_cell_length_a {float(lat.a):.4f}\n"
        f"_cell_length_b {float(lat.b):.4f}\n"
        f"_cell_length_c {float(lat.c):.4f}\n"
        f"_cell_angle_alpha {float(lat.alpha):.4f}\n"
        f"_cell_angle_beta {float(lat.beta):.4f}\n"
        f"_cell_angle_gamma {float(lat.gamma):.4f}\n"
        f"_cell_volume {float(lat.volume):.4f}\n"
    )


def sample_oracle_text(
    model: Any,
    tokenizer: CIFTokenizer,
    prefix: str,
    gt_record: Any,
    lookup: WyckoffLookup,
    seed: int,
    *,
    use_gt_cell: bool,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
) -> tuple[str, ConstraintStats]:
    stats = ConstraintStats()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    text = prefix
    skeleton = skeleton_from_record(gt_record, lookup)
    for site_index, site in enumerate(skeleton, start=1):
        tpl = site.template
        row_text = text + f"{site_index} {site.element} {tpl.letter} "
        for axis, free in enumerate(tpl.free_mask):
            if axis:
                row_text += " "
            if free:
                _coord, row_text = sample_unit_coord(
                    model,
                    tokenizer,
                    row_text,
                    generator=gen,
                    device=device,
                    dtype=dtype,
                    temperature=temperature,
                    top_k=top_k,
                    stats=stats,
                )
            else:
                row_text += "FIXED"
        row_text += "\n"
        text = row_text
    stats.formula_closure_success = True
    text = text.rstrip() + "\n"
    if use_gt_cell:
        text += append_gt_cell(gt_record)
    else:
        text += sample_cell_tail(
            model,
            tokenizer,
            text,
            int(gt_record.sg_number),
            generator=gen,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
            stats=stats,
        )
    return text, stats


def sample_case(
    model: Any,
    tokenizer: CIFTokenizer,
    lookup: WyckoffLookup,
    case: dict[str, Any],
    gt_text: str,
    seeds: list[int],
    *,
    mode: str,
    use_gt_cell: bool,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
) -> list[dict[str, Any]]:
    start = time.monotonic()
    gt_record = parse_symcif_v2_text(gt_text, lookup)
    prefix = v2_prompt_prefix(case["prompt"])
    out: list[dict[str, Any]] = []
    for gen_index, seed in enumerate(seeds):
        gen_start = time.monotonic()
        text, stats = sample_oracle_text(
            model,
            tokenizer,
            prefix,
            gt_record,
            lookup,
            seed,
            use_gt_cell=use_gt_cell,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
        )
        record_text = extract_generated_record(text, mode)
        out.append(
            {
                "mode": mode,
                "sample_index": case["index"],
                "sample_id": case["sample_id"],
                "gen_index": gen_index,
                "seed": int(seed),
                "raw_generation_success": bool(record_text.strip()),
                "generated_text": record_text,
                "error": None,
                "generation_time_seconds": time.monotonic() - gen_start,
                "case_total_generation_time_seconds": time.monotonic() - start,
                "formula_closure_success": bool(stats.formula_closure_success),
                "mask_rejected_tokens": int(stats.mask_rejected_tokens),
                "resample_count": int(stats.resample_count),
                "cell_generation_failure_reason": stats.cell_generation_failure_reason,
                "oracle_uses_gt_skeleton": True,
                "oracle_uses_gt_assignment": True,
                "oracle_uses_gt_cell": bool(use_gt_cell),
            }
        )
    return out


def worker_main(
    *,
    worker_id: int,
    cases_payload: list[dict[str, Any]],
    gt_texts: list[str],
    seeds: list[int],
    out_paths: dict[str, str],
    std_dirs: dict[str, str],
    model_dir: str,
    lookup_json: str,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    compile_model: bool,
) -> None:
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    tokenizer = CIFTokenizer()
    model = load_model(Path(model_dir), device=device, dtype=dtype, compile_model=compile_model)
    lookup = WyckoffLookup.from_json(lookup_json)
    files = {mode: Path(path).open("w", encoding="utf-8") for mode, path in out_paths.items()}
    try:
        for mode, path in std_dirs.items():
            Path(path).mkdir(parents=True, exist_ok=True)
        for local_i, case in enumerate(cases_payload, start=1):
            gt_text = gt_texts[int(case["index"])]
            for mode, use_gt_cell in ((ORACLE_A_MODE, False), (ORACLE_B_MODE, True)):
                try:
                    records = sample_case(
                        model,
                        tokenizer,
                        lookup,
                        case,
                        gt_text,
                        seeds,
                        mode=mode,
                        use_gt_cell=use_gt_cell,
                        device=device,
                        dtype=dtype,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    for rec in records:
                        try:
                            parsed = parse_symcif_v2_text(rec["generated_text"], lookup)
                            cif = render_standard_cif_v2(parsed, symprec=0.1, lookup=lookup)
                            (
                                Path(std_dirs[mode])
                                / f"{case['index']:04d}_{case['sample_id']}_g{int(rec['gen_index']):02d}.cif"
                            ).write_text(cif, encoding="utf-8")
                        except Exception:
                            pass
                        files[mode].write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
                except Exception as exc:  # noqa: BLE001
                    for gen_index, seed in enumerate(seeds):
                        files[mode].write(
                            json.dumps(
                                {
                                    "mode": mode,
                                    "sample_index": case["index"],
                                    "sample_id": case["sample_id"],
                                    "gen_index": gen_index,
                                    "seed": int(seed),
                                    "raw_generation_success": False,
                                    "generated_text": "",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(),
                                    "generation_time_seconds": None,
                                    "formula_closure_success": False,
                                    "oracle_uses_gt_skeleton": True,
                                    "oracle_uses_gt_assignment": True,
                                    "oracle_uses_gt_cell": bool(use_gt_cell),
                                },
                                ensure_ascii=True,
                                sort_keys=True,
                            )
                            + "\n"
                        )
            if local_i % 10 == 0 or local_i == len(cases_payload):
                print(f"[oracle:{device}:worker{worker_id}] {local_i}/{len(cases_payload)} prompts done", flush=True)
    finally:
        for f in files.values():
            f.close()


def generate(args: argparse.Namespace) -> None:
    cases = load_test_cases(args.test_limit, modes=("symcif_v2_constrained",))
    gt_texts = split_concat_records(PROJECT_ROOT / "data" / "symcif_v2" / "test.txt")
    seeds = [args.seed + i for i in range(args.num_gens)]
    generation_dir = args.out_dir / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    expected = len(cases) * len(seeds)
    if not args.overwrite:
        complete = True
        for mode in (ORACLE_A_MODE, ORACLE_B_MODE):
            path = generation_dir / f"{mode}.jsonl"
            complete = complete and path.exists() and sum(1 for _ in path.open(encoding="utf-8")) == expected
        if complete:
            print(f"[oracle] found complete generation files in {generation_dir}, skipping")
            return

    devices = [item.strip() for item in args.devices.split(",") if item.strip()] or ["cpu"]
    chunks: list[list[Any]] = [[] for _ in devices]
    for i, case in enumerate(cases):
        chunks[i % len(devices)].append(case)

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    worker_paths: dict[str, list[Path]] = {ORACLE_A_MODE: [], ORACLE_B_MODE: []}
    for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
        payload = [
            {
                "index": c.index,
                "sample_id": c.sample_id,
                "prompt": c.prompts["symcif_v2_constrained"],
                "target_formula": c.target_formula,
                "target_sg_number": c.target_sg_number,
                "target_sg_symbol": c.target_sg_symbol,
            }
            for c in chunk
        ]
        out_paths: dict[str, str] = {}
        std_dirs: dict[str, str] = {}
        for mode in (ORACLE_A_MODE, ORACLE_B_MODE):
            worker_path = generation_dir / f"{mode}.worker{worker_id}.jsonl"
            worker_paths[mode].append(worker_path)
            out_paths[mode] = str(worker_path)
            std_dirs[mode] = str(args.out_dir / "standard_cifs" / mode / f"worker{worker_id}")
        proc = ctx.Process(
            target=worker_main,
            kwargs={
                "worker_id": worker_id,
                "cases_payload": payload,
                "gt_texts": gt_texts,
                "seeds": seeds,
                "out_paths": out_paths,
                "std_dirs": std_dirs,
                "model_dir": str(args.model_dir),
                "lookup_json": str(args.lookup_json),
                "device": device,
                "dtype": args.dtype,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "compile_model": args.compile,
            },
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"oracle worker failed with exit code {proc.exitcode}")

    for mode in (ORACLE_A_MODE, ORACLE_B_MODE):
        records: list[dict[str, Any]] = []
        for path in worker_paths[mode]:
            with path.open(encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f if line.strip())
        records.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
        with (generation_dir / f"{mode}.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
        print(f"[oracle:{mode}] wrote {len(records)} records", flush=True)

    metadata = {
        "modes": [ORACLE_A_MODE, ORACLE_B_MODE],
        "test_samples": len(cases),
        "num_gens": len(seeds),
        "seeds": seeds,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "devices": devices,
        "oracle_warning": "GT skeleton/assignment/cell are used only for diagnosis; not formal generation.",
    }
    (args.out_dir / "oracle_generation_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def aggregate_mode(metrics: list[dict[str, Any]], n: int, total_cases: int) -> dict[str, Any]:
    subset = [row for row in metrics if int(row["gen_index"]) < n]
    denom = total_cases * n
    grouped = group_by_sample(subset, n)
    n1 = 0
    n5 = 0
    rms_values: list[float] = []
    for sample_index in range(total_cases):
        rows = grouped.get(sample_index, [])
        if rows and rows[0].get("match_ok"):
            n1 += 1
        rms = [float(row["rms"]) for row in rows if row.get("match_ok") and row.get("rms") is not None]
        if rms:
            n5 += 1
            rms_values.append(min(rms))
    return {
        "match@1": n1 / total_cases if total_cases else math.nan,
        "match@5": n5 / total_cases if total_cases else math.nan,
        "RMSE": statistics.mean(rms_values) if rms_values else math.nan,
        "formula_ok": sum(1 for row in subset if row.get("formula_ok")) / denom if denom else math.nan,
        "sg_ok": sum(1 for row in subset if row.get("space_group_ok")) / denom if denom else math.nan,
        "readable": sum(1 for row in subset if row.get("pymatgen_readable")) / denom if denom else math.nan,
        "valid": sum(1 for row in subset if row.get("valid")) / denom if denom else math.nan,
        "eval_timeout": sum(1 for row in subset if row.get("eval_timeout")) / denom if denom else math.nan,
    }


def summarize(args: argparse.Namespace) -> None:
    cases = load_test_cases(args.test_limit, modes=(ORACLE_A_MODE, ORACLE_B_MODE))
    lookup = WyckoffLookup.from_json(args.lookup_json)
    gt_records = [parse_symcif_v2_text(text, lookup) for text in split_concat_records(PROJECT_ROOT / "data" / "symcif_v2" / "test.txt")]
    rows: list[dict[str, Any]] = []
    for mode in (ORACLE_A_MODE, ORACLE_B_MODE):
        metrics = load_jsonl(args.out_dir / "metrics" / f"{mode}_per_generation_metrics.jsonl")
        summary = {"mode": mode}
        summary.update(aggregate_mode(metrics, args.num_gens, len(cases)))
        rows.append(summary)

    cell_rows: list[dict[str, Any]] = []
    gens = load_jsonl(args.out_dir / "generations" / f"{ORACLE_A_MODE}.jsonl")
    for rec in gens:
        if int(rec["gen_index"]) >= args.num_gens:
            continue
        try:
            pred = parse_symcif_v2_text(rec["generated_text"], lookup)
            gt = gt_records[int(rec["sample_index"])]
            err = lattice_error(gt, pred)
            coord_mae, coord_count = free_coord_errors(gt, pred)
            cell_rows.append({**err, "free_coord_mae": coord_mae, "free_coord_count": coord_count})
        except Exception:
            continue
    cell_summary: dict[str, Any] = {"mode": "oracle_c_gt_skeleton_generated_cell_stats"}
    for key in ("a_abs_error", "b_abs_error", "c_abs_error", "alpha_abs_error", "beta_abs_error", "gamma_abs_error", "volume_rel_error", "free_coord_mae"):
        vals = [float(row[key]) for row in cell_rows if row.get(key) is not None]
        cell_summary[key + "_mean"] = statistics.mean(vals) if vals else math.nan
        cell_summary[key + "_median"] = statistics.median(vals) if vals else math.nan
    cell_summary["cell_stats_generations"] = len(cell_rows)
    rows.append(cell_summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "oracle_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out_dir / "oracle_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_report(args, rows)


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


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    report_lines = [
        "# SymCIF-v2 Oracle Diagnosis Report",
        "",
        "日期：2026-05-21",
        "",
        "> Oracle 诊断使用 GT skeleton / assignment / cell，只用于定位瓶颈，不作为正式生成结果。",
        "",
        "## 设置",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| output dir | `{args.out_dir}` |",
        f"| n | {args.num_gens} |",
        "| seeds | 1337-1341 |",
        f"| temperature/top_k | {args.temperature} / {args.top_k} |",
        "",
        "## Match 结果",
        "",
        "| oracle | match@1 | match@5 | RMSE | formula_ok | sg_ok | readable | valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if not str(row.get("mode", "")).startswith("symcif_v2_oracle"):
            continue
        report_lines.append(
            "| {mode} | {m1} | {m5} | {rmse} | {formula} | {sg} | {readable} | {valid} |".format(
                mode=row["mode"],
                m1=fmt_pct(row.get("match@1")),
                m5=fmt_pct(row.get("match@5")),
                rmse=fmt_num(row.get("RMSE")),
                formula=fmt_pct(row.get("formula_ok")),
                sg=fmt_pct(row.get("sg_ok")),
                readable=fmt_pct(row.get("readable")),
                valid=fmt_pct(row.get("valid")),
            )
        )
    cell = next((row for row in rows if row.get("mode") == "oracle_c_gt_skeleton_generated_cell_stats"), {})
    report_lines.extend(
        [
            "",
            "## Cell / Coord 诊断",
            "",
            "| 指标 | mean | median |",
            "| --- | ---: | ---: |",
            f"| a abs error | {fmt_num(cell.get('a_abs_error_mean'))} | {fmt_num(cell.get('a_abs_error_median'))} |",
            f"| b abs error | {fmt_num(cell.get('b_abs_error_mean'))} | {fmt_num(cell.get('b_abs_error_median'))} |",
            f"| c abs error | {fmt_num(cell.get('c_abs_error_mean'))} | {fmt_num(cell.get('c_abs_error_median'))} |",
            f"| alpha abs error | {fmt_num(cell.get('alpha_abs_error_mean'))} | {fmt_num(cell.get('alpha_abs_error_median'))} |",
            f"| beta abs error | {fmt_num(cell.get('beta_abs_error_mean'))} | {fmt_num(cell.get('beta_abs_error_median'))} |",
            f"| gamma abs error | {fmt_num(cell.get('gamma_abs_error_mean'))} | {fmt_num(cell.get('gamma_abs_error_median'))} |",
            f"| volume rel error | {fmt_num(cell.get('volume_rel_error_mean'))} | {fmt_num(cell.get('volume_rel_error_median'))} |",
            f"| free coord MAE | {fmt_num(cell.get('free_coord_mae_mean'))} | {fmt_num(cell.get('free_coord_mae_median'))} |",
            "",
            "## 结论口径",
            "",
            "- Oracle A = GT skeleton + GT assignment + generated coord/cell。",
            "- Oracle B = GT skeleton + GT assignment + GT cell + generated free coords。",
            "- Oracle C = GT skeleton + GT assignment 下，统计 generated cell 与 GT cell 的误差。",
            "- 若 Oracle A 仍低而 Oracle B 明显高，cell 是主拖累；若 Oracle B 仍低，free coordinate 是主拖累；若 Oracle A 高，正式生成主瓶颈主要是 skeleton/assignment。",
        ]
    )
    local_dir = args.out_dir / "Log_GPT"
    global_dir = PROJECT_ROOT / "Log_GPT"
    round_dir = global_dir / "round_20260521_03_diagnosis_oracle_rank"
    for directory in (local_dir, global_dir, round_dir):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "symcif_v2_oracle_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[oracle] wrote report -> {global_dir / 'symcif_v2_oracle_report.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SymCIF-v2 oracle diagnosis generation and summary.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_runs" / "symcif_v2_oracle_diagnosis_20260521",
    )
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "runs" / "exp_symcif_v2")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-gens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        generate(args)
    if (args.out_dir / "metrics" / f"{ORACLE_A_MODE}_per_generation_metrics.jsonl").exists():
        summarize(args)
    else:
        print("[oracle] metrics not found yet; run run_generation_eval.py --skip-generation, then rerun with --summarize-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
