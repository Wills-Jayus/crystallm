#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
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
from run_generation_eval import extract_generated_record, load_model, load_test_cases  # noqa: E402
from sample_symcif_v2_constrained import parse_target_counts, v2_prompt_prefix  # noqa: E402
from sample_symcif_v2_skeleton_beam import sample_from_skeleton  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402
from symcif_v2.to_cif import render_standard_cif_v2  # noqa: E402
from symcif_v2_diagnostic_utils import (  # noqa: E402
    assignment_signature,
    rank_skeleton_candidates,
    search_legal_skeleton_candidates,
    skeleton_signature,
)


def sample_one_case(
    model: Any,
    tokenizer: CIFTokenizer,
    lookup: WyckoffLookup,
    case: dict[str, Any],
    seeds: list[int],
    *,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    beam_size: int,
    candidate_limit: int,
    top_skeletons: int,
    max_sites: int,
    max_expansions_per_beam: int,
    mode_name: str,
) -> list[dict[str, Any]]:
    start_time = time.monotonic()
    sg_number = int(case["target_sg_number"])
    target_counts = parse_target_counts(case["target_formula"])
    prefix = v2_prompt_prefix(case["prompt"])
    candidates = search_legal_skeleton_candidates(
        model,
        tokenizer,
        prefix,
        target_counts,
        sg_number,
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
    if not candidates:
        raise ValueError("skeleton_rank_found_no_complete_skeleton")
    ranked = rank_skeleton_candidates(
        model,
        tokenizer,
        prefix,
        candidates,
        device=device,
        dtype=dtype,
    )[:top_skeletons]
    if not ranked:
        raise ValueError("skeleton_rank_found_no_ranked_skeleton")

    records: list[dict[str, Any]] = []
    for gen_index, seed in enumerate(seeds):
        beam = ranked[gen_index % len(ranked)]
        gen_start = time.monotonic()
        text, stats = sample_from_skeleton(
            model,
            tokenizer,
            prefix,
            beam.skeleton,
            sg_number,
            seed,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
        )
        record_text = extract_generated_record(text, mode_name)
        records.append(
            {
                "mode": mode_name,
                "sample_index": case["index"],
                "sample_id": case["sample_id"],
                "gen_index": gen_index,
                "seed": int(seed),
                "raw_generation_success": bool(record_text.strip()),
                "generated_text": record_text,
                "error": None,
                "generation_time_seconds": time.monotonic() - gen_start,
                "case_total_generation_time_seconds": time.monotonic() - start_time,
                "formula_closure_success": bool(stats.formula_closure_success),
                "mask_rejected_tokens": int(stats.mask_rejected_tokens),
                "resample_count": int(stats.resample_count),
                "cell_generation_failure_reason": stats.cell_generation_failure_reason,
                "skeleton_rank": int(gen_index % len(ranked)) + 1,
                "skeleton_rank_num_complete": len(candidates),
                "skeleton_rank_partial_logprob": float(beam.partial_logprob),
                "skeleton_rank_full_logprob": float(beam.full_logprob) if beam.full_logprob is not None else None,
                "skeleton_rank_normalized_logprob": (
                    float(beam.normalized_logprob) if beam.normalized_logprob is not None else None
                ),
                "skeleton_signature": skeleton_signature(beam.skeleton),
                "element_wyckoff_assignment_signature": assignment_signature(beam.skeleton),
            }
        )
    return records


def worker_main(
    *,
    worker_id: int,
    cases_payload: list[dict[str, Any]],
    seeds: list[int],
    out_path: str,
    std_dir: str,
    model_dir: str,
    lookup_json: str,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    beam_size: int,
    candidate_limit: int,
    top_skeletons: int,
    max_sites: int,
    max_expansions_per_beam: int,
    compile_model: bool,
    mode_name: str,
) -> None:
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    tokenizer = CIFTokenizer()
    model = load_model(Path(model_dir), device=device, dtype=dtype, compile_model=compile_model)
    lookup = WyckoffLookup.from_json(lookup_json)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    std_path = Path(std_dir)
    std_path.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for local_i, case in enumerate(cases_payload, start=1):
            try:
                records = sample_one_case(
                    model,
                    tokenizer,
                    lookup,
                    case,
                    seeds,
                    device=device,
                    dtype=dtype,
                    temperature=temperature,
                    top_k=top_k,
                    beam_size=beam_size,
                    candidate_limit=candidate_limit,
                    top_skeletons=top_skeletons,
                    max_sites=max_sites,
                    max_expansions_per_beam=max_expansions_per_beam,
                    mode_name=mode_name,
                )
                for rec in records:
                    try:
                        parsed = parse_symcif_v2_text(rec["generated_text"], lookup)
                        cif = render_standard_cif_v2(parsed, symprec=0.1, lookup=lookup)
                        (std_path / f"{case['index']:04d}_{case['sample_id']}_g{int(rec['gen_index']):02d}.cif").write_text(
                            cif,
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                    f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
            except Exception as exc:  # noqa: BLE001
                for gen_index, seed in enumerate(seeds):
                    f.write(
                        json.dumps(
                            {
                                "mode": mode_name,
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
                                "mask_rejected_tokens": None,
                                "resample_count": None,
                                "cell_generation_failure_reason": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if local_i % 5 == 0 or local_i == len(cases_payload):
                print(f"[skeleton_rank:{device}:worker{worker_id}] {local_i}/{len(cases_payload)} prompts done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SymCIF-v2 candidates with legal skeleton search and full LM ranking.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_runs" / "symcif_v2_skeleton_rank_match5_20260521",
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
    parser.add_argument("--beam-size", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, default=128)
    parser.add_argument("--top-skeletons", type=int, default=5)
    parser.add_argument("--max-sites", type=int, default=64)
    parser.add_argument("--max-expansions-per-beam", type=int, default=24)
    parser.add_argument("--mode-name", default="symcif_v2_constrained_skeleton_rank")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases = load_test_cases(args.test_limit, modes=("symcif_v2_constrained",))
    seeds = [args.seed + i for i in range(args.num_gens)]
    generation_dir = args.out_dir / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    merged_path = generation_dir / f"{args.mode_name}.jsonl"
    expected = len(cases) * len(seeds)
    if merged_path.exists() and not args.overwrite:
        existing = sum(1 for _ in merged_path.open(encoding="utf-8"))
        if existing == expected:
            print(f"[{args.mode_name}] found complete {merged_path}, skipping")
            return 0

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
                "target_sg_symbol": c.target_sg_symbol,
            }
            for c in chunk
        ]
        worker_path = generation_dir / f"{args.mode_name}.worker{worker_id}.jsonl"
        worker_paths.append(worker_path)
        proc = ctx.Process(
            target=worker_main,
            kwargs={
                "worker_id": worker_id,
                "cases_payload": payload,
                "seeds": seeds,
                "out_path": str(worker_path),
                "std_dir": str(args.out_dir / "standard_cifs" / args.mode_name / f"worker{worker_id}"),
                "model_dir": str(args.model_dir),
                "lookup_json": str(args.lookup_json),
                "device": device,
                "dtype": args.dtype,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "beam_size": args.beam_size,
                "candidate_limit": args.candidate_limit,
                "top_skeletons": args.top_skeletons,
                "max_sites": args.max_sites,
                "max_expansions_per_beam": args.max_expansions_per_beam,
                "compile_model": args.compile,
                "mode_name": args.mode_name,
            },
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"skeleton rank worker failed with exit code {proc.exitcode}")

    records: list[dict[str, Any]] = []
    for worker_path in worker_paths:
        with worker_path.open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    records.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    with merged_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
    metadata = {
        "mode": args.mode_name,
        "test_samples": len(cases),
        "num_gens": len(seeds),
        "seeds": seeds,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "beam_size": args.beam_size,
        "candidate_limit": args.candidate_limit,
        "top_skeletons": args.top_skeletons,
        "max_sites": args.max_sites,
        "max_expansions_per_beam": args.max_expansions_per_beam,
        "devices": devices,
        "model_dir": str(args.model_dir),
        "lookup_json": str(args.lookup_json),
    }
    (args.out_dir / f"{args.mode_name}_generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[{args.mode_name}] wrote {len(records)} records -> {merged_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
