#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
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
from sample_symcif_v2_constrained import (  # noqa: E402
    ConstraintStats,
    can_close_all,
    encode_prefix,
    is_fixed_template,
    parse_target_counts,
    sample_cell_tail,
    sample_unit_coord,
    valid_templates_for_sg,
    v2_prompt_prefix,
)
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402
from symcif_v2.to_cif import render_standard_cif_v2  # noqa: E402


@dataclass(frozen=True)
class SkeletonSite:
    element: str
    template: Any


@dataclass
class BeamState:
    remaining: dict[str, int]
    used_fixed: frozenset[str]
    skeleton: tuple[SkeletonSite, ...]
    score_text: str
    logprob: float


def counter_signature(counter: Counter[tuple[Any, ...]]) -> str:
    return "|".join(
        "{}x{}".format(",".join(str(item) for item in key), count)
        for key, count in sorted(counter.items())
    )


def skeleton_signature(skeleton: tuple[SkeletonSite, ...]) -> str:
    return counter_signature(Counter((site.template.letter, int(site.template.multiplicity)) for site in skeleton))


def assignment_signature(skeleton: tuple[SkeletonSite, ...]) -> str:
    return counter_signature(
        Counter((site.element, site.template.letter, int(site.template.multiplicity)) for site in skeleton)
    )


def placeholder_coord_text(template: Any) -> str:
    tokens = ["0.5000" if bool(free) else "FIXED" for free in template.free_mask]
    return " ".join(tokens)


@torch.no_grad()
def allowed_token_logprobs(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    allowed_tokens: list[str],
    *,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
) -> dict[str, float]:
    token_to_id = tokenizer.token_to_id
    token_ids = {token: token_to_id[token] for token in allowed_tokens if token in token_to_id}
    if not token_ids:
        return {}
    idx = encode_prefix(tokenizer, text, device)
    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size :]
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else torch.no_grad()
    with ctx:
        logits, _ = model(idx_cond)
    logits = logits[0, -1, :].float() / max(temperature, 1e-6)
    allowed_ids = list(token_ids.values())
    if top_k and top_k > 0 and len(allowed_ids) > top_k:
        allowed_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=device)
        vals = logits[allowed_tensor]
        kth = torch.topk(vals, k=min(top_k, vals.numel())).values[-1]
        allowed_ids = [token_id for token_id in allowed_ids if float(logits[token_id]) >= float(kth)]
    allowed_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=device)
    scores = torch.log_softmax(logits[allowed_tensor], dim=-1)
    id_to_score = {int(token_id): float(score) for token_id, score in zip(allowed_ids, scores.tolist(), strict=True)}
    return {token: id_to_score[token_id] for token, token_id in token_ids.items() if token_id in id_to_score}


def valid_actions(
    remaining: dict[str, int],
    used_fixed: frozenset[str],
    templates: tuple[Any, ...],
    tokenizer: CIFTokenizer,
) -> dict[str, list[Any]]:
    valid_by_element: dict[str, list[Any]] = {}
    for element, count in sorted(remaining.items()):
        if count <= 0 or element not in tokenizer.token_to_id:
            continue
        candidates: list[Any] = []
        for tpl in templates:
            mult = int(tpl.multiplicity)
            if mult > count:
                continue
            fixed = is_fixed_template(tpl)
            if fixed and tpl.letter in used_fixed:
                continue
            trial = dict(remaining)
            trial[element] -= mult
            if trial[element] == 0:
                trial.pop(element)
            trial_used = frozenset((*used_fixed, tpl.letter)) if fixed else used_fixed
            if can_close_all(trial, trial_used, templates):
                candidates.append(tpl)
        if candidates:
            valid_by_element[element] = candidates
    return valid_by_element


def expand_beam(
    model: Any,
    tokenizer: CIFTokenizer,
    beam: BeamState,
    templates: tuple[Any, ...],
    *,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    max_expansions_per_beam: int,
) -> list[BeamState]:
    site_index = len(beam.skeleton) + 1
    row_prefix = beam.score_text + f"{site_index} "
    valid_by_element = valid_actions(beam.remaining, beam.used_fixed, templates, tokenizer)
    if not valid_by_element:
        return []
    element_logprobs = allowed_token_logprobs(
        model,
        tokenizer,
        row_prefix,
        list(valid_by_element),
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
    )
    expansions: list[tuple[float, str, Any]] = []
    for element, elem_lp in sorted(element_logprobs.items(), key=lambda item: item[1], reverse=True):
        letter_options = [tpl.letter for tpl in valid_by_element[element]]
        letter_logprobs = allowed_token_logprobs(
            model,
            tokenizer,
            row_prefix + element + " ",
            letter_options,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
        )
        template_by_letter = {tpl.letter: tpl for tpl in valid_by_element[element]}
        for letter, letter_lp in letter_logprobs.items():
            expansions.append((beam.logprob + elem_lp + letter_lp, element, template_by_letter[letter]))
    expansions.sort(key=lambda item: item[0], reverse=True)
    out: list[BeamState] = []
    for new_score, element, tpl in expansions[:max_expansions_per_beam]:
        new_remaining = dict(beam.remaining)
        new_remaining[element] -= int(tpl.multiplicity)
        if new_remaining[element] == 0:
            new_remaining.pop(element)
        new_used = frozenset((*beam.used_fixed, tpl.letter)) if is_fixed_template(tpl) else beam.used_fixed
        score_text = row_prefix + f"{element} {tpl.letter} {placeholder_coord_text(tpl)}\n"
        out.append(
            BeamState(
                remaining=new_remaining,
                used_fixed=new_used,
                skeleton=(*beam.skeleton, SkeletonSite(element=element, template=tpl)),
                score_text=score_text,
                logprob=new_score,
            )
        )
    return out


def skeleton_beam_search(
    model: Any,
    tokenizer: CIFTokenizer,
    prefix: str,
    remaining: dict[str, int],
    templates: tuple[Any, ...],
    *,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    beam_size: int,
    top_skeletons: int,
    max_sites: int,
    max_expansions_per_beam: int,
) -> list[BeamState]:
    beams = [BeamState(remaining=dict(remaining), used_fixed=frozenset(), skeleton=tuple(), score_text=prefix, logprob=0.0)]
    completed: list[BeamState] = []
    for _site_depth in range(max_sites):
        next_beams: list[BeamState] = []
        for beam in beams:
            if not beam.remaining:
                completed.append(beam)
                continue
            next_beams.extend(
                expand_beam(
                    model,
                    tokenizer,
                    beam,
                    templates,
                    device=device,
                    dtype=dtype,
                    temperature=temperature,
                    top_k=top_k,
                    max_expansions_per_beam=max_expansions_per_beam,
                )
            )
        if not next_beams:
            break
        next_beams.sort(key=lambda item: item.logprob / max(1, len(item.skeleton)), reverse=True)
        beams = next_beams[:beam_size]
        completed.extend([beam for beam in beams if not beam.remaining])
        unique: dict[str, BeamState] = {}
        for beam in completed:
            if beam.remaining:
                continue
            sig = assignment_signature(beam.skeleton)
            old = unique.get(sig)
            if old is None or beam.logprob > old.logprob:
                unique[sig] = beam
        if len(unique) >= top_skeletons and all(not beam.remaining for beam in beams[: min(len(beams), top_skeletons)]):
            break
    unique_completed: dict[str, BeamState] = {}
    for beam in completed + [beam for beam in beams if not beam.remaining]:
        sig = assignment_signature(beam.skeleton)
        old = unique_completed.get(sig)
        if old is None or beam.logprob > old.logprob:
            unique_completed[sig] = beam
    ranked = sorted(unique_completed.values(), key=lambda item: item.logprob / max(1, len(item.skeleton)), reverse=True)
    return ranked[:top_skeletons]


def sample_from_skeleton(
    model: Any,
    tokenizer: CIFTokenizer,
    prefix: str,
    skeleton: tuple[SkeletonSite, ...],
    sg_number: int,
    seed: int,
    *,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
) -> tuple[str, ConstraintStats]:
    stats = ConstraintStats()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    text = prefix
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
    text += sample_cell_tail(
        model,
        tokenizer,
        text,
        sg_number,
        generator=gen,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
    )
    return text, stats


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
    top_skeletons: int,
    max_sites: int,
    max_expansions_per_beam: int,
    mode_name: str,
) -> list[dict[str, Any]]:
    start_time = time.monotonic()
    sg_number = int(case["target_sg_number"])
    target_counts = parse_target_counts(case["target_formula"])
    templates = tuple(valid_templates_for_sg(lookup, sg_number, tokenizer))
    if not templates:
        raise ValueError(f"no tokenizable Wyckoff letters for SG={sg_number}")
    prefix = v2_prompt_prefix(case["prompt"])
    skeletons = skeleton_beam_search(
        model,
        tokenizer,
        prefix,
        target_counts,
        templates,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        beam_size=beam_size,
        top_skeletons=top_skeletons,
        max_sites=max_sites,
        max_expansions_per_beam=max_expansions_per_beam,
    )
    if not skeletons:
        raise ValueError("skeleton_beam_found_no_complete_skeleton")

    records: list[dict[str, Any]] = []
    for gen_index, seed in enumerate(seeds):
        beam = skeletons[gen_index % len(skeletons)]
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
                "skeleton_beam_rank": int(gen_index % len(skeletons)) + 1,
                "skeleton_beam_num_complete": len(skeletons),
                "skeleton_beam_logprob": float(beam.logprob),
                "skeleton_beam_avg_logprob": float(beam.logprob / max(1, len(beam.skeleton))),
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
                    rec = {
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
                    }
                    f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
            if local_i % 5 == 0 or local_i == len(cases_payload):
                print(f"[skeleton_beam:{device}:worker{worker_id}] {local_i}/{len(cases_payload)} prompts done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SymCIF-v2 candidates with constrained skeleton beam search.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_runs" / "symcif_v2_skeleton_beam_match5_20260521")
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
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--top-skeletons", type=int, default=5)
    parser.add_argument("--max-sites", type=int, default=64)
    parser.add_argument("--max-expansions-per-beam", type=int, default=16)
    parser.add_argument("--mode-name", default="symcif_v2_constrained_skeleton_beam")
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
            raise RuntimeError(f"skeleton beam worker failed with exit code {proc.exitcode}")

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
