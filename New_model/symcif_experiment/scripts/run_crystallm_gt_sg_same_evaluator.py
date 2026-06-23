#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Composition

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTODL_ROOT = PROJECT_ROOT.parents[2]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystallm import get_atomic_props_block_for_formula  # noqa: E402
from run_generation_eval import (  # noqa: E402
    aggregate_metrics,
    evaluate_mode_with_hard_timeouts,
    extract_generated_record,
    generate_batch_mode_aware,
    load_model,
    load_generation_records,
)


def load_cif_tokenizer(code_root: Path):
    tokenizer_path = code_root / "crystallm" / "_tokenizer.py"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"missing CrystaLLM tokenizer: {tokenizer_path}")
    module_name = f"_crystallm_tokenizer_{abs(hash(tokenizer_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, tokenizer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load tokenizer module from {tokenizer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CIFTokenizer()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def formula_id(formula: str) -> str:
    comp = Composition(formula)
    return comp.formula.replace(" ", "")


def prompt_for_record(record: dict[str, Any]) -> str:
    fid = formula_id(str(record["formula"]))
    atom_block = get_atomic_props_block_for_formula(fid).strip()
    sg = str(record.get("sg_symbol") or "").strip().strip("'\"")
    if not sg:
        sg = f"SG{int(record['sg'])}"
    parts = [f"data_{fid}"]
    if atom_block:
        parts.append(atom_block)
    parts.append(f"_symmetry_space_group_name_H-M {sg}")
    return "\n".join(parts).rstrip() + "\n"


def case_payload(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        out.append(
            {
                "index": index,
                "sample_id": row["sample_id"],
                "source_path": str(row["source_path"]),
                "target_formula": row["formula"],
                "target_sg_number": int(row["sg"]),
                "target_sg_symbol": row.get("sg_symbol"),
            }
        )
    return out


def split_even(items: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(max(1, int(n)))]
    for i, item in enumerate(items):
        buckets[i % len(buckets)].append(item)
    return buckets


def generation_worker_seed_batches(
    *,
    model_dir: str,
    cases_payload: list[dict[str, Any]],
    seeds: list[int],
    out_path: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    compile_model: bool,
    worker_id: int,
    seed_batch_size: int,
    tokenizer_code_root: str,
) -> None:
    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    model = load_model(Path(model_dir), device=device, dtype=dtype, compile_model=compile_model)
    tokenizer = load_cif_tokenizer(Path(tokenizer_code_root))
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer.token_to_id)))
    if len(tokenizer.token_to_id) != vocab_size:
        raise ValueError(
            f"tokenizer vocab size {len(tokenizer.token_to_id)} does not match checkpoint vocab size {vocab_size}"
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    seed_batch_size = max(1, int(seed_batch_size))
    with out.open("w", encoding="utf-8") as f:
        for local_i, case in enumerate(cases_payload, start=1):
            prompt = case["prompt"]
            for offset in range(0, len(seeds), seed_batch_size):
                seed_chunk = seeds[offset : offset + seed_batch_size]
                try:
                    texts = generate_batch_mode_aware(
                        model,
                        tokenizer,
                        prompt,
                        seed_chunk,
                        mode="baseline",
                        device=device,
                        dtype=dtype,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    for local_gen_index, (seed, text) in enumerate(zip(seed_chunk, texts)):
                        gen_index = int(offset + local_gen_index)
                        record_text = extract_generated_record(text, "baseline")
                        f.write(
                            json.dumps(
                                {
                                    "mode": "baseline",
                                    "sample_index": case["index"],
                                    "sample_id": case["sample_id"],
                                    "gen_index": gen_index,
                                    "seed": int(seed),
                                    "raw_generation_success": bool(record_text.strip()),
                                    "generated_text": record_text,
                                    "error": None,
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    tb = traceback.format_exc()
                    for local_gen_index, seed in enumerate(seed_chunk):
                        gen_index = int(offset + local_gen_index)
                        f.write(
                            json.dumps(
                                {
                                    "mode": "baseline",
                                    "sample_index": case["index"],
                                    "sample_id": case["sample_id"],
                                    "gen_index": gen_index,
                                    "seed": int(seed),
                                    "raw_generation_success": False,
                                    "generated_text": "",
                                    "error": err,
                                    "traceback": tb,
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
            if local_i % 25 == 0 or local_i == len(cases_payload):
                print(f"[generate:baseline:worker{worker_id}] {local_i}/{len(cases_payload)} prompts done", flush=True)


def load_or_generate(args: argparse.Namespace, records: list[dict[str, Any]], seeds: list[int]) -> Path:
    generation_dir = args.out_dir / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    merged_path = generation_dir / "baseline.jsonl"
    expected = len(records) * len(seeds)
    existing_rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if not args.overwrite_generation:
        existing_paths = [merged_path] if merged_path.exists() else []
        existing_paths.extend(sorted(generation_dir.glob("baseline.worker*.jsonl")))
        for path in existing_paths:
            try:
                for row in read_jsonl(path):
                    key = (int(row["sample_index"]), int(row["gen_index"]))
                    if 0 <= key[0] < len(records) and 0 <= key[1] < len(seeds):
                        existing_rows_by_key.setdefault(key, row)
            except Exception as exc:  # noqa: BLE001
                print(f"[generate] ignoring partial/corrupt generation file {path}: {exc}", flush=True)
    if merged_path.exists() and not args.overwrite_generation:
        existing = len(existing_rows_by_key)
        if existing == expected:
            print(f"[generate] found complete {merged_path}, skipping", flush=True)
            if sum(1 for _ in merged_path.open(encoding="utf-8")) != expected:
                rows = sorted(existing_rows_by_key.values(), key=lambda row: (int(row["sample_index"]), int(row["gen_index"])))
                write_jsonl(merged_path, rows)
            return merged_path
    if args.skip_generation:
        raise FileNotFoundError(f"--skip-generation set but complete generation file is missing: {merged_path}")

    complete_sample_indices = {
        index
        for index in range(len(records))
        if all((index, gen_index) in existing_rows_by_key for gen_index in range(len(seeds)))
    }
    cases = [
        {
            "index": index,
            "sample_id": row["sample_id"],
            "prompt": prompt_for_record(row),
        }
        for index, row in enumerate(records)
        if index not in complete_sample_indices
    ]
    if complete_sample_indices:
        print(
            f"[generate] reusing complete generations for {len(complete_sample_indices)}/{len(records)} prompts",
            flush=True,
        )
    if not cases:
        rows = sorted(existing_rows_by_key.values(), key=lambda row: (int(row["sample_index"]), int(row["gen_index"])))
        write_jsonl(merged_path, rows)
        return merged_path
    prompts_dir = args.out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        (prompts_dir / f"{case['sample_id']}.txt").write_text(str(case["prompt"]), encoding="utf-8")

    devices = [item.strip() for item in str(args.devices).split(",") if item.strip()]
    if not devices:
        devices = ["cuda:0"]
    if int(args.workers_per_device) > 1:
        devices = [device for device in devices for _ in range(int(args.workers_per_device))]
    chunks = split_even(cases, len(devices))
    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    worker_paths: list[Path] = []
    resume_tag = f"{os.getpid()}.{int(time.time())}"
    for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
        if not chunk:
            continue
        worker_path = generation_dir / f"baseline.worker{worker_id}.jsonl"
        if not args.overwrite_generation and worker_path.exists():
            worker_path = generation_dir / f"baseline.resume.{resume_tag}.worker{worker_id}.jsonl"
        worker_paths.append(worker_path)
        proc = ctx.Process(
            target=generation_worker_seed_batches,
            kwargs={
                "model_dir": str(args.model_dir),
                "cases_payload": chunk,
                "seeds": seeds,
                "out_path": str(worker_path),
                "device": str(device),
                "dtype": str(args.dtype),
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_k": int(args.top_k),
                "compile_model": bool(args.compile),
                "worker_id": int(worker_id),
                "seed_batch_size": int(args.seed_batch_size),
                "tokenizer_code_root": str(args.crystallm_code_root),
            },
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"generation worker failed with exit code {proc.exitcode}")

    rows_by_key = dict(existing_rows_by_key)
    for worker_path in worker_paths:
        for row in read_jsonl(worker_path):
            key = (int(row["sample_index"]), int(row["gen_index"]))
            rows_by_key[key] = row
    rows: list[dict[str, Any]] = list(rows_by_key.values())
    rows.sort(key=lambda row: (int(row["sample_index"]), int(row["gen_index"])))
    write_jsonl(merged_path, rows)
    if len(rows) != expected:
        raise RuntimeError(f"generation count mismatch: got {len(rows)}, expected {expected}")
    return merged_path


def strict_valid(row: dict[str, Any]) -> bool:
    bond = row.get("bond_length_score")
    try:
        bond_ok = bond is not None and float(bond) >= 1.0
    except Exception:
        bond_ok = False
    return bool(
        row.get("pymatgen_readable")
        and row.get("formula_ok")
        and row.get("space_group_ok")
        and bond_ok
        and row.get("valid")
    )


def aggregate_for_k(metrics: list[dict[str, Any]], *, k: int, total_cases: int) -> dict[str, Any]:
    base = aggregate_metrics(metrics, n=int(k), total_cases=total_cases)
    subset = [m for m in metrics if int(m.get("gen_index", 0)) < int(k)]
    denom = max(1, total_cases * int(k))
    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in subset:
        by_sample.setdefault(int(row["sample_index"]), []).append(row)
    any_strict = sum(1 for rows in by_sample.values() if any(strict_valid(row) for row in rows))
    any_readable = sum(1 for rows in by_sample.values() if any(bool(row.get("pymatgen_readable")) for row in rows))
    any_formula = sum(1 for rows in by_sample.values() if any(bool(row.get("formula_ok")) for row in rows))
    any_sg = sum(1 for rows in by_sample.values() if any(bool(row.get("space_group_ok")) for row in rows))
    any_valid = sum(1 for rows in by_sample.values() if any(bool(row.get("valid")) for row in rows))
    return {
        "k": int(k),
        "samples": int(total_cases),
        "attempts": int(total_cases) * int(k),
        "match_rate": base["match_rate_n1"] if int(k) == 1 else base["match_rate_n20"],
        "RMSE": base["RMSE"],
        "matched_samples_for_RMSE": base["matched_samples_for_RMSE"],
        "raw_generation_success_attempt_rate": base["raw_generation_success"],
        "parse_success_attempt_rate": base["parse_success"],
        "pymatgen_readable_attempt_rate": base["pymatgen_readable"],
        "formula_ok_attempt_rate": base["formula_ok"],
        "space_group_ok_attempt_rate": base["space_group_ok"],
        "valid_attempt_rate": base["valid"],
        "strict_valid_attempt_rate": sum(1 for row in subset if strict_valid(row)) / denom,
        "readable_any_sample_rate": any_readable / max(1, total_cases),
        "formula_ok_any_sample_rate": any_formula / max(1, total_cases),
        "SG_ok_any_sample_rate": any_sg / max(1, total_cases),
        "valid_any_sample_rate": any_valid / max(1, total_cases),
        "strict_valid_any_sample_rate": any_strict / max(1, total_cases),
        "eval_timeout_attempt_rate": base["eval_timeout"],
        "bond_length_score_attempt_mean": base["bond_length_score"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrystaLLM + GT-SG baseline with SymCIF evaluator.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--num-gens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--seed-batch-size", type=int, default=5)
    parser.add_argument(
        "--crystallm-code-root",
        type=Path,
        default=AUTODL_ROOT / "model" / "scp_task" / "CrystaLLM",
        help="CrystaLLM source tree whose tokenizer matches the benchmark checkpoint.",
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--overwrite-generation", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--eval-workers", type=int, default=64)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--match-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-match-sites", type=int, default=300)
    parser.add_argument("--max-eval-sites", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.data_root / "test.jsonl")
    if args.test_limit is not None:
        records = records[: max(0, int(args.test_limit))]
    seeds = [int(args.seed) + i for i in range(int(args.num_gens))]
    meta = {
        "data_root": str(args.data_root),
        "model_dir": str(args.model_dir),
        "samples": len(records),
        "num_gens": int(args.num_gens),
        "seeds": seeds,
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "devices": str(args.devices),
        "workers_per_device": int(args.workers_per_device),
        "crystallm_code_root": str(args.crystallm_code_root),
        "dtype": str(args.dtype),
        "prompt_rule": "data_<composition> + CrystaLLM atom properties + oracle GT _symmetry_space_group_name_H-M",
        "evaluator": "run_generation_eval.py:evaluate_mode_with_hard_timeouts",
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generation_path = load_or_generate(args, records, seeds)

    grouped = load_generation_records(generation_path)
    eval_args = argparse.Namespace(
        eval_workers=int(args.eval_workers),
        bond_timeout_seconds=float(args.bond_timeout_seconds),
        parse_timeout_seconds=float(args.parse_timeout_seconds),
        sg_timeout_seconds=float(args.sg_timeout_seconds),
        valid_timeout_seconds=float(args.valid_timeout_seconds),
        match_timeout_seconds=float(args.match_timeout_seconds),
        sample_timeout_seconds=float(args.sample_timeout_seconds),
        max_match_sites=int(args.max_match_sites),
        max_eval_sites=int(args.max_eval_sites),
    )
    metrics = evaluate_mode_with_hard_timeouts(
        mode="baseline",
        case_payload=case_payload(records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    metrics.sort(key=lambda row: (int(row["sample_index"]), int(row["gen_index"])))
    write_jsonl(args.out_dir / "metrics" / "baseline_per_generation_metrics.jsonl", metrics)
    summary_rows = [aggregate_for_k(metrics, k=k, total_cases=len(records)) for k in (1, 5, int(args.num_gens))]
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "summary.csv", summary_rows)
    print(json.dumps(summary_rows, indent=2, sort_keys=True, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
