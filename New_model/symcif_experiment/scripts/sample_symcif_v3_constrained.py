#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystallm import CIFTokenizer  # type: ignore  # noqa: E402
from run_generation_eval import extract_generated_record, load_model, load_test_cases, strip_prompt_comment_lines  # noqa: E402
from sample_symcif_v2_constrained import (  # noqa: E402
    ConstraintStats,
    is_fixed_template,
    parse_target_counts,
    sample_allowed_token,
    sample_cell_tail,
    sample_unit_coord,
    valid_templates_for_sg,
)
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v3.parse import parse_symcif_v3_text  # noqa: E402
from symcif_v3.to_cif import render_standard_cif_v3  # noqa: E402


SKELETON_HEADER = """loop_
_wyckoff_site_index
_wyckoff_site_multiplicity
_wyckoff_site_letter
_wyckoff_site_symmetry
_wyckoff_site_enumeration
"""

ASSIGNMENT_HEADER = """loop_
_wyckoff_site_index
_wyckoff_site_element
"""

COORD_HEADER = """loop_
_wyckoff_site_index
_wyckoff_free_x
_wyckoff_free_y
_wyckoff_free_z
"""


@dataclass(frozen=True)
class SkeletonSite:
    index: int
    template: Any


def v3_prompt_prefix(prompt: str) -> str:
    base = strip_prompt_comment_lines(prompt).rstrip()
    return base + "\n_cell_formula_units_Z 1\n\n" + SKELETON_HEADER


def mult_assignable(counts_key: tuple[tuple[str, int], ...], mults_key: tuple[int, ...]) -> bool:
    counts0 = dict(counts_key)
    mults = tuple(sorted((int(v) for v in mults_key), reverse=True))

    @lru_cache(maxsize=None)
    def rec(i: int, state: tuple[tuple[str, int], ...]) -> bool:
        if i >= len(mults):
            return all(v == 0 for _, v in state)
        counts = dict(state)
        mult = mults[i]
        for element, count in list(counts.items()):
            if count < mult:
                continue
            nxt = dict(counts)
            nxt[element] -= mult
            if rec(i + 1, tuple(sorted((k, v) for k, v in nxt.items() if v > 0))):
                return True
        return False

    return rec(0, tuple(sorted((k, int(v)) for k, v in counts0.items() if int(v) > 0)))


def can_complete_skeleton(
    target_counts: dict[str, int],
    selected_mults: tuple[int, ...],
    remaining_total: int,
    used_fixed: frozenset[str],
    templates: tuple[Any, ...],
    max_sites_left: int,
) -> bool:
    counts_key = tuple(sorted((k, int(v)) for k, v in target_counts.items()))

    @lru_cache(maxsize=None)
    def rec(mults_key: tuple[int, ...], rem: int, used_key: tuple[str, ...], sites_left: int) -> bool:
        if rem == 0:
            return mult_assignable(counts_key, tuple(sorted(mults_key)))
        if rem < 0 or sites_left <= 0:
            return False
        used = set(used_key)
        for tpl in templates:
            mult = int(tpl.multiplicity)
            if mult > rem:
                continue
            fixed = is_fixed_template(tpl)
            if fixed and tpl.letter in used:
                continue
            next_used = tuple(sorted((*used, tpl.letter))) if fixed else used_key
            if rec(tuple(sorted((*mults_key, mult))), rem - mult, next_used, sites_left - 1):
                return True
        return False

    return rec(tuple(sorted(selected_mults)), int(remaining_total), tuple(sorted(used_fixed)), int(max_sites_left))


def choose_next_template(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    target_counts: dict[str, int],
    selected: list[SkeletonSite],
    used_fixed: set[str],
    templates: tuple[Any, ...],
    completion_templates: tuple[Any, ...] | None = None,
    *,
    generator: torch.Generator,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    max_sites: int,
    stats: ConstraintStats,
) -> tuple[Any, str]:
    completion_templates = completion_templates or templates
    selected_mults = tuple(int(site.template.multiplicity) for site in selected)
    selected_total = sum(selected_mults)
    target_total = sum(int(v) for v in target_counts.values())
    candidates: list[Any] = []
    for tpl in templates:
        mult = int(tpl.multiplicity)
        if selected_total + mult > target_total:
            continue
        fixed = is_fixed_template(tpl)
        if fixed and tpl.letter in used_fixed:
            continue
        next_used = frozenset((*used_fixed, tpl.letter)) if fixed else frozenset(used_fixed)
        if can_complete_skeleton(
            target_counts,
            tuple((*selected_mults, mult)),
            target_total - selected_total - mult,
            next_used,
            completion_templates,
            max_sites - len(selected) - 1,
        ):
            candidates.append(tpl)
    if not candidates:
        raise ValueError("no skeleton candidate can close formula")
    candidates = sorted(candidates, key=lambda tpl: (int(tpl.multiplicity), str(tpl.letter)))
    by_letter = {tpl.letter: tpl for tpl in candidates}
    letter_options = list(by_letter)
    letter = sample_allowed_token(
        model,
        tokenizer,
        text,
        letter_options,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
    )
    return by_letter[letter], text + letter


def sample_multiplicity(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    valid_mults: list[int],
    *,
    generator: torch.Generator,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    stats: ConstraintStats,
) -> tuple[int, str]:
    valid = sorted(set(str(int(m)) for m in valid_mults))
    first_options = sorted({value[0] for value in valid})
    first = sample_allowed_token(
        model,
        tokenizer,
        text,
        first_options,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
    )
    text += first
    tail_options: list[str] = []
    if first in valid:
        tail_options.append(" ")
    tail_options.extend(sorted({value[1] for value in valid if len(value) == 2 and value[0] == first}))
    if not tail_options:
        raise ValueError(f"no valid multiplicity tail for first token {first}, valid={valid}")
    tail = sample_allowed_token(
        model,
        tokenizer,
        text,
        tail_options,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
    )
    if tail == " ":
        return int(first), text + " "
    value = first + tail
    if value not in valid:
        raise ValueError(f"sampled invalid multiplicity {value}, valid={valid}")
    return int(value), text + tail + " "


def assignment_can_close(counts: dict[str, int], remaining_mults: tuple[int, ...]) -> bool:
    return mult_assignable(tuple(sorted((k, int(v)) for k, v in counts.items() if int(v) > 0)), remaining_mults)


def choose_assignment_element(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    remaining_counts: dict[str, int],
    remaining_sites: list[SkeletonSite],
    *,
    generator: torch.Generator,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    stats: ConstraintStats,
) -> tuple[str, str]:
    site = remaining_sites[0]
    mult = int(site.template.multiplicity)
    future_mults = tuple(int(s.template.multiplicity) for s in remaining_sites[1:])
    options: list[str] = []
    for element, count in sorted(remaining_counts.items()):
        if count < mult or element not in tokenizer.token_to_id:
            continue
        trial = dict(remaining_counts)
        trial[element] -= mult
        if trial[element] == 0:
            trial.pop(element)
        if assignment_can_close(trial, future_mults):
            options.append(element)
    if not options:
        raise ValueError(f"no assignment option for site mult={mult} remaining={remaining_counts}")
    element = sample_allowed_token(
        model,
        tokenizer,
        text,
        options,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
    )
    return element, text + element


def constrained_sample_one(
    model: Any,
    tokenizer: CIFTokenizer,
    lookup: WyckoffLookup,
    case: dict[str, Any],
    seed: int,
    *,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    max_sites: int,
    mode_name: str,
) -> dict[str, Any]:
    start_time = time.monotonic()
    stats = ConstraintStats()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    target_counts = parse_target_counts(case["target_formula"])
    target_total = sum(int(v) for v in target_counts.values())
    sg_number = int(case["target_sg_number"])
    templates = tuple(valid_templates_for_sg(lookup, sg_number, tokenizer))
    if not templates:
        raise ValueError(f"no tokenizable Wyckoff letters for SG={sg_number}")

    text = v3_prompt_prefix(case["prompt"])
    selected: list[SkeletonSite] = []
    used_fixed: set[str] = set()
    selected_total = 0
    while selected_total < target_total:
        if len(selected) >= max_sites:
            raise ValueError(f"max constrained sites exceeded: {max_sites}")
        site_index = len(selected) + 1
        row_text = text + f"{site_index} "
        # Multiplicity is deterministic once a template is selected; letter is sampled
        # from model logits under the legal, formula-closable skeleton mask.
        candidate_prefixes = []
        candidate_by_mult: dict[int, list[Any]] = {}
        for tpl in templates:
            candidate_by_mult.setdefault(int(tpl.multiplicity), []).append(tpl)
        valid_mults = [
            mult
            for mult in sorted(candidate_by_mult)
            if any(
                can_complete_skeleton(
                    target_counts,
                    tuple([*(int(s.template.multiplicity) for s in selected), mult]),
                    target_total - selected_total - mult,
                    frozenset((*used_fixed, tpl.letter)) if is_fixed_template(tpl) else frozenset(used_fixed),
                    templates,
                    max_sites - len(selected) - 1,
                )
                for tpl in candidate_by_mult[mult]
                if selected_total + mult <= target_total and not (is_fixed_template(tpl) and tpl.letter in used_fixed)
            )
        ]
        if not valid_mults:
            raise ValueError("no valid multiplicity for next skeleton site")
        mult, row_text = sample_multiplicity(
            model,
            tokenizer,
            row_text,
            valid_mults,
            generator=gen,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
            stats=stats,
        )
        filtered_templates = tuple(tpl for tpl in templates if int(tpl.multiplicity) == mult)
        tpl, after_letter = choose_next_template(
            model,
            tokenizer,
            row_text,
            target_counts,
            selected,
            used_fixed,
            filtered_templates,
            completion_templates=templates,
            generator=gen,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
            max_sites=max_sites,
            stats=stats,
        )
        enum = str(tpl.enumeration) if tpl.enumeration is not None else "UNKNOWN"
        text = after_letter + f" {tpl.site_symmetry or 'UNKNOWN'} {enum}\n"
        selected.append(SkeletonSite(index=site_index, template=tpl))
        selected_total += int(tpl.multiplicity)
        if is_fixed_template(tpl):
            used_fixed.add(tpl.letter)

    text = text.rstrip() + "\n\n" + ASSIGNMENT_HEADER
    remaining_counts = dict(target_counts)
    assignment: dict[int, str] = {}
    for offset, site in enumerate(selected):
        row_text = text + f"{site.index} "
        element, after_element = choose_assignment_element(
            model,
            tokenizer,
            row_text,
            remaining_counts,
            selected[offset:],
            generator=gen,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
            stats=stats,
        )
        text = after_element + "\n"
        assignment[site.index] = element
        remaining_counts[element] -= int(site.template.multiplicity)
        if remaining_counts[element] == 0:
            remaining_counts.pop(element)

    stats.formula_closure_success = not remaining_counts
    text = text.rstrip() + "\n\n" + COORD_HEADER
    for site in selected:
        row_text = text + f"{site.index} "
        tpl = site.template
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
        text = row_text + "\n"

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
    record_text = extract_generated_record(text, mode_name)
    return {
        "mode": mode_name,
        "sample_index": case["index"],
        "sample_id": case["sample_id"],
        "seed": int(seed),
        "raw_generation_success": bool(record_text.strip()),
        "generated_text": record_text,
        "error": None,
        "generation_time_seconds": time.monotonic() - start_time,
        "formula_closure_success": bool(stats.formula_closure_success),
        "mask_rejected_tokens": int(stats.mask_rejected_tokens),
        "resample_count": int(stats.resample_count),
        "cell_generation_failure_reason": stats.cell_generation_failure_reason,
    }


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
    max_sites: int,
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
            for gen_index, seed in enumerate(seeds):
                try:
                    rec = constrained_sample_one(
                        model,
                        tokenizer,
                        lookup,
                        case,
                        seed,
                        device=device,
                        dtype=dtype,
                        temperature=temperature,
                        top_k=top_k,
                        max_sites=max_sites,
                        mode_name=mode_name,
                    )
                    rec["gen_index"] = gen_index
                    try:
                        parsed = parse_symcif_v3_text(rec["generated_text"], lookup)
                        cif = render_standard_cif_v3(parsed, symprec=0.1)
                        (std_path / f"{case['index']:04d}_{case['sample_id']}_g{gen_index:02d}.cif").write_text(
                            cif,
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                except Exception as exc:  # noqa: BLE001
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
            if local_i % 10 == 0 or local_i == len(cases_payload):
                print(f"[v3:{device}:worker{worker_id}] {local_i}/{len(cases_payload)} prompts done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SymCIF-v3 with staged CrystalFormer-like constrained decoding.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_runs" / "symcif_v3_cf_order_t1_topk10_n5_20260521")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "runs" / "exp_symcif_v3_cf_order")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-gens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--max-sites", type=int, default=96)
    parser.add_argument("--mode-name", default="symcif_v3_constrained")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases = load_test_cases(args.test_limit, modes=("symcif_v3_constrained",))
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
                "prompt": c.prompts["symcif_v3_constrained"],
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
                "max_sites": args.max_sites,
                "compile_model": args.compile,
                "mode_name": args.mode_name,
            },
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"v3 constrained worker failed with exit code {proc.exitcode}")

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
