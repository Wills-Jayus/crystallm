#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from pymatgen.core import Composition, Lattice
from pymatgen.symmetry.groups import SpaceGroup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystallm import CIFTokenizer  # type: ignore  # noqa: E402
from run_generation_eval import extract_generated_record, load_model, load_test_cases, strip_prompt_comment_lines  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif.models import LatticeParameters  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402
from symcif_v2.to_cif import render_standard_cif_v2  # noqa: E402


SITE_HEADER = """loop_
_wyckoff_site_index
_wyckoff_site_element
_wyckoff_site_letter
_wyckoff_free_x
_wyckoff_free_y
_wyckoff_free_z
"""


@dataclass
class ConstraintStats:
    mask_rejected_tokens: int = 0
    resample_count: int = 0
    formula_closure_success: bool = False
    mask_rejection_reasons: Counter[str] = field(default_factory=Counter)


def parse_target_counts(formula: str) -> dict[str, int]:
    raw = Composition(formula).as_dict()
    counts: dict[str, int] = {}
    for element, value in raw.items():
        rounded = int(round(float(value)))
        if abs(float(value) - rounded) > 1e-6:
            raise ValueError(f"non-integer formula count: {formula}")
        counts[str(element)] = rounded
    return counts


def v2_prompt_prefix(prompt: str) -> str:
    base = strip_prompt_comment_lines(prompt).rstrip()
    return base + "\n\n" + SITE_HEADER


def lattice_system(sg_number: int) -> str:
    try:
        return str(SpaceGroup.from_int_number(int(sg_number)).crystal_system)
    except Exception:
        return "triclinic"


def project_lattice(sg_number: int, a: float, b: float, c: float, beta: float) -> LatticeParameters:
    system = lattice_system(sg_number)
    a = max(1.0, min(60.0, float(a)))
    b = max(1.0, min(60.0, float(b)))
    c = max(1.0, min(80.0, float(c)))
    beta = max(60.0, min(130.0, float(beta)))
    if system == "cubic":
        b = c = a
        alpha = beta_ang = gamma = 90.0
    elif system == "tetragonal":
        b = a
        alpha = beta_ang = gamma = 90.0
    elif system == "orthorhombic":
        alpha = beta_ang = gamma = 90.0
    elif system in {"hexagonal", "trigonal"}:
        b = a
        alpha = beta_ang = 90.0
        gamma = 120.0
    elif system == "monoclinic":
        alpha = gamma = 90.0
        beta_ang = beta
    else:
        alpha = 90.0
        beta_ang = beta
        gamma = 90.0
    try:
        volume = float(Lattice.from_parameters(a, b, c, alpha, beta_ang, gamma).volume)
    except Exception:
        volume = max(1.0, a * b * c)
    return LatticeParameters(a=a, b=b, c=c, alpha=alpha, beta=beta_ang, gamma=gamma, volume=volume)


def encode_prefix(tokenizer: CIFTokenizer, text: str, device: str) -> torch.Tensor:
    token_to_id = tokenizer.token_to_id
    unk_id = token_to_id.get("<unk>")
    ids = [token_to_id[token] for token in tokenizer.tokenize_cif(text) if unk_id is None or token_to_id[token] != unk_id]
    if not ids:
        raise ValueError("empty encoded prefix")
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def sample_allowed_token(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    allowed_tokens: list[str],
    *,
    generator: torch.Generator,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    stats: ConstraintStats,
    reason: str = "generic",
    bias_by_token: dict[str, float] | None = None,
) -> str:
    token_to_id = tokenizer.token_to_id
    allowed_ids = [token_to_id[token] for token in allowed_tokens if token in token_to_id]
    if not allowed_ids:
        raise ValueError(f"no allowed tokenizer ids for {allowed_tokens}")
    idx = encode_prefix(tokenizer, text, device)
    idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size :]
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else torch.no_grad()
    with ctx:
        logits, _ = model(idx_cond)
    logits = logits[0, -1, :].float()
    rejected = max(0, int(logits.numel()) - len(set(allowed_ids)))
    stats.mask_rejected_tokens += rejected
    stats.mask_rejection_reasons[reason] += rejected
    mask = torch.full_like(logits, -float("inf"))
    mask[allowed_ids] = logits[allowed_ids] / max(temperature, 1e-6)
    if bias_by_token:
        for token, bias in bias_by_token.items():
            token_id = token_to_id.get(token)
            if token_id is not None and token_id in allowed_ids:
                mask[token_id] += float(bias)
    if top_k and top_k > 0 and len(allowed_ids) > top_k:
        allowed_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=device)
        vals = mask[allowed_tensor]
        kth = torch.topk(vals, k=min(top_k, vals.numel())).values[-1]
        mask[mask < kth] = -float("inf")
    probs = torch.softmax(mask, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        stats.resample_count += 1
        return allowed_tokens[0]
    next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
    return tokenizer.id_to_token[next_id]


def valid_templates_for_sg(lookup: WyckoffLookup, sg_number: int, tokenizer: CIFTokenizer) -> list[Any]:
    token_to_id = tokenizer.token_to_id
    templates = [tpl for (sg, _), tpl in lookup.templates.items() if int(sg) == int(sg_number)]
    templates = [tpl for tpl in templates if tpl.letter in token_to_id and int(tpl.multiplicity) > 0]
    return sorted(templates, key=lambda t: (int(t.multiplicity), str(t.letter)))


def is_fixed_template(template: Any) -> bool:
    return not any(bool(v) for v in template.free_mask)


def can_close_all(remaining: dict[str, int], used_fixed: frozenset[str], templates: tuple[Any, ...]) -> bool:
    key_counts = tuple(sorted((k, int(v)) for k, v in remaining.items() if int(v) > 0))

    @lru_cache(maxsize=None)
    def rec(counts_key: tuple[tuple[str, int], ...], used_key: tuple[str, ...]) -> bool:
        counts = dict(counts_key)
        if not counts:
            return True
        element = next(iter(counts))
        count = counts[element]
        used = set(used_key)
        for tpl in templates:
            mult = int(tpl.multiplicity)
            if mult > count:
                continue
            fixed = is_fixed_template(tpl)
            if fixed and tpl.letter in used:
                continue
            new_counts = dict(counts)
            left = count - mult
            if left:
                new_counts[element] = left
            else:
                new_counts.pop(element, None)
            new_used = tuple(sorted((*used, tpl.letter))) if fixed else used_key
            if rec(tuple(sorted(new_counts.items())), new_used):
                return True
        return False

    return rec(key_counts, tuple(sorted(used_fixed)))


def choose_element_and_letter(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    remaining: dict[str, int],
    used_fixed: set[str],
    templates: tuple[Any, ...],
    *,
    generator: torch.Generator,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    stats: ConstraintStats,
) -> tuple[str, Any, str]:
    element_options: list[str] = []
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
            trial_used = frozenset((*used_fixed, tpl.letter)) if fixed else frozenset(used_fixed)
            if can_close_all(trial, trial_used, templates):
                candidates.append(tpl)
        if candidates:
            element_options.append(element)
            valid_by_element[element] = candidates
    if not element_options:
        raise ValueError(f"no formula-closure element options for remaining={remaining}")

    element_bias: dict[str, float] = {}
    for elem in element_options:
        count = max(1, int(remaining[elem]))
        n_candidates = max(1, len(valid_by_element[elem]))
        # Hard-to-close elements get a modest positive bias but are still sampled
        # through the model distribution.
        element_bias[elem] = 0.15 * (1.0 / count) + 0.10 * (1.0 / n_candidates)
    element = sample_allowed_token(
        model,
        tokenizer,
        text,
        element_options,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
        reason="element_formula_mask",
        bias_by_token=element_bias,
    )
    text_after_element = text + element + " "
    letter_options = [tpl.letter for tpl in valid_by_element[element]]
    letter_bias: dict[str, float] = {}
    for tpl in valid_by_element[element]:
        mult = max(1, int(tpl.multiplicity))
        free_dims = sum(1 for item in tpl.free_mask if item)
        # Prefer special positions/smaller multiplicities/earlier letters without
        # removing high-probability alternatives.
        letter_rank = max(0, ord(str(tpl.letter)[0]) - ord("a"))
        letter_bias[tpl.letter] = 0.18 * (1.0 / mult) + 0.10 * (3 - free_dims) - 0.005 * letter_rank
    letter = sample_allowed_token(
        model,
        tokenizer,
        text_after_element,
        letter_options,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
        reason="wyckoff_formula_mask",
        bias_by_token=letter_bias,
    )
    template = next(tpl for tpl in valid_by_element[element] if tpl.letter == letter)
    return element, template, text_after_element + letter + " "


def sample_unit_coord(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    *,
    generator: torch.Generator,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    stats: ConstraintStats,
) -> tuple[str, str]:
    out = "0."
    text = text + out
    digits = [str(i) for i in range(10)]
    for _ in range(4):
        d = sample_allowed_token(
            model,
            tokenizer,
            text,
            digits,
            generator=generator,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
            stats=stats,
            reason="coord_digit_mask",
        )
        out += d
        text += d
    return out, text


def sample_length(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    *,
    generator: torch.Generator,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    stats: ConstraintStats,
) -> tuple[float, str]:
    first = sample_allowed_token(
        model,
        tokenizer,
        text,
        [str(i) for i in range(1, 10)],
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
        reason="cell_length_first_digit_mask",
    )
    text += first
    second_or_dot = sample_allowed_token(
        model,
        tokenizer,
        text,
        [".", *[str(i) for i in range(10)]],
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
        reason="cell_length_second_digit_mask",
    )
    if second_or_dot == ".":
        num = first + "."
        text += "."
    else:
        num = first + second_or_dot + "."
        text += second_or_dot + "."
    for _ in range(4):
        d = sample_allowed_token(
            model,
            tokenizer,
            text,
            [str(i) for i in range(10)],
            generator=generator,
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
            stats=stats,
            reason="cell_length_decimal_digit_mask",
        )
        num += d
        text += d
    return float(num), text


def sample_cell_tail(
    model: Any,
    tokenizer: CIFTokenizer,
    text: str,
    sg_number: int,
    *,
    generator: torch.Generator,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    stats: ConstraintStats,
) -> str:
    text += "\n_cell_length_a "
    a, text = sample_length(
        model,
        tokenizer,
        text,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
    )
    text += "\n_cell_length_b "
    b, text = sample_length(
        model,
        tokenizer,
        text,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
    )
    text += "\n_cell_length_c "
    c, text = sample_length(
        model,
        tokenizer,
        text,
        generator=generator,
        device=device,
        dtype=dtype,
        temperature=temperature,
        top_k=top_k,
        stats=stats,
    )
    beta = 100.0
    lat = project_lattice(sg_number, a, b, c, beta)
    return (
        f"\n_cell_length_a {lat.a:.4f}\n"
        f"_cell_length_b {lat.b:.4f}\n"
        f"_cell_length_c {lat.c:.4f}\n"
        f"_cell_angle_alpha {lat.alpha:.4f}\n"
        f"_cell_angle_beta {lat.beta:.4f}\n"
        f"_cell_angle_gamma {lat.gamma:.4f}\n"
        f"_cell_volume {lat.volume:.4f}\n"
    )


def constrained_sample_one(
    model: Any,
    tokenizer: CIFTokenizer,
    lookup: WyckoffLookup,
    case: dict[str, Any],
    seed: int,
    *,
    device: str,
    dtype: str,
    temp_discrete: float,
    topk_discrete: int,
    temp_coord: float,
    topk_coord: int,
    temp_cell: float,
    topk_cell: int,
    max_sites: int,
) -> dict[str, Any]:
    start_time = time.monotonic()
    stats = ConstraintStats()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    target_counts = parse_target_counts(case["target_formula"])
    remaining = dict(target_counts)
    sg_number = int(case["target_sg_number"])
    templates = tuple(valid_templates_for_sg(lookup, sg_number, tokenizer))
    if not templates:
        raise ValueError(f"no tokenizable Wyckoff letters for SG={sg_number}")

    text = v2_prompt_prefix(case["prompt"])
    used_fixed: set[str] = set()
    site_index = 1
    remaining_trace: list[dict[str, int]] = [dict(sorted(remaining.items()))]
    selected_sites: list[dict[str, Any]] = []
    while any(value > 0 for value in remaining.values()):
        if site_index > max_sites:
            raise ValueError(f"max constrained sites exceeded: {max_sites}")
        row_prefix = text + f"{site_index} "
        element, template, after_letter = choose_element_and_letter(
            model,
            tokenizer,
            row_prefix,
            remaining,
            used_fixed,
            templates,
            generator=gen,
            device=device,
            dtype=dtype,
            temperature=temp_discrete,
            top_k=topk_discrete,
            stats=stats,
        )
        row_text = after_letter
        for axis, free in enumerate(template.free_mask):
            if axis:
                row_text += " "
            if free:
                coord, row_text = sample_unit_coord(
                    model,
                    tokenizer,
                    row_text,
                    generator=gen,
                    device=device,
                    dtype=dtype,
                    temperature=temp_coord,
                    top_k=topk_coord,
                    stats=stats,
                )
            else:
                row_text += "FIXED"
        row_text += "\n"
        text = row_text
        remaining[element] -= int(template.multiplicity)
        if remaining[element] == 0:
            remaining.pop(element)
        if is_fixed_template(template):
            used_fixed.add(template.letter)
        selected_sites.append(
            {
                "index": site_index,
                "element": element,
                "wyckoff_letter": template.letter,
                "multiplicity": int(template.multiplicity),
                "free_mask": list(template.free_mask),
                "fixed": bool(is_fixed_template(template)),
            }
        )
        remaining_trace.append(dict(sorted(remaining.items())))
        site_index += 1

    stats.formula_closure_success = not remaining
    text = text.rstrip() + "\n"
    text += sample_cell_tail(
        model,
        tokenizer,
        text,
        sg_number,
        generator=gen,
        device=device,
        dtype=dtype,
        temperature=temp_cell,
        top_k=topk_cell,
        stats=stats,
    )
    record_text = extract_generated_record(text, "symcif_v2_constrained")
    skeleton_signature = "|".join(
        f"{site['wyckoff_letter']}:{site['multiplicity']}" for site in sorted(selected_sites, key=lambda s: (s["wyckoff_letter"], s["multiplicity"]))
    )
    assignment_signature = "|".join(
        f"{site['element']}:{site['wyckoff_letter']}:{site['multiplicity']}"
        for site in sorted(selected_sites, key=lambda s: (s["element"], s["wyckoff_letter"], s["multiplicity"]))
    )
    return {
        "mode": "symcif_v2_constrained_new",
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
        "mask_rejection_reasons": dict(stats.mask_rejection_reasons),
        "remaining_counts_trace": remaining_trace,
        "selected_wyckoff_sites": selected_sites,
        "skeleton_signature": skeleton_signature,
        "element_wyckoff_assignment_signature": assignment_signature,
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
    temp_discrete: float,
    topk_discrete: int,
    temp_coord: float,
    topk_coord: int,
    temp_cell: float,
    topk_cell: int,
    max_sites: int,
    skeleton_resample_attempts: int,
    compile_model: bool,
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
            seen_skeletons: set[str] = set()
            for gen_index, seed in enumerate(seeds):
                try:
                    rec = None
                    duplicate_attempts = 0
                    for attempt in range(max(1, skeleton_resample_attempts + 1)):
                        trial_seed = int(seed) + attempt * 1000003 + int(case["index"]) * 17
                        trial = constrained_sample_one(
                            model,
                            tokenizer,
                            lookup,
                            case,
                            trial_seed,
                            device=device,
                            dtype=dtype,
                            temp_discrete=temp_discrete,
                            topk_discrete=topk_discrete,
                            temp_coord=temp_coord,
                            topk_coord=topk_coord,
                            temp_cell=temp_cell,
                            topk_cell=topk_cell,
                            max_sites=max_sites,
                        )
                        signature = str(trial.get("skeleton_signature") or "")
                        duplicate = bool(signature and signature in seen_skeletons)
                        trial["skeleton_duplicate_before_accept"] = duplicate
                        trial["skeleton_resample_attempts"] = attempt
                        if not duplicate or attempt >= skeleton_resample_attempts:
                            rec = trial
                            duplicate_attempts = attempt
                            break
                    if rec is None:
                        raise ValueError("skeleton diversity sampler returned no record")
                    rec["diversity_resample_count"] = duplicate_attempts
                    if rec.get("skeleton_signature"):
                        seen_skeletons.add(str(rec["skeleton_signature"]))
                    rec["gen_index"] = gen_index
                    try:
                        parsed = parse_symcif_v2_text(rec["generated_text"], lookup)
                        cif = render_standard_cif_v2(parsed, symprec=0.1, lookup=lookup)
                        final_cif_path = std_path / f"{case['index']:04d}_{case['sample_id']}_g{gen_index:02d}.cif"
                        final_cif_path.write_text(
                            cif,
                            encoding="utf-8",
                        )
                        rec["final_standard_cif_path"] = str(final_cif_path)
                    except Exception:
                        pass
                except Exception as exc:  # noqa: BLE001
                    rec = {
                        "mode": "symcif_v2_constrained_new",
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
                        "skeleton_resample_attempts": 0,
                        "diversity_resample_count": 0,
                    }
                f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
            if local_i % 10 == 0 or local_i == len(cases_payload):
                print(f"[constrained_v2:{device}:worker{worker_id}] {local_i}/{len(cases_payload)} prompts done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SymCIF-v2 with improved formula/Wyckoff constrained decoding.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_runs" / "symcif_v2_constrained_v2_match1_match5")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "runs" / "exp_symcif_v2")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--temp-discrete", type=float, default=1.0)
    parser.add_argument("--topk-discrete", type=int, default=10)
    parser.add_argument("--temp-coord", type=float, default=0.7)
    parser.add_argument("--topk-coord", type=int, default=5)
    parser.add_argument("--temp-cell", type=float, default=0.5)
    parser.add_argument("--topk-cell", type=int, default=5)
    parser.add_argument("--num-gens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--max-sites", type=int, default=96)
    parser.add_argument("--skeleton-resample-attempts", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases = load_test_cases(args.test_limit, modes=("symcif_v2_constrained",))
    seeds = [args.seed + i for i in range(args.num_gens)]
    generation_dir = args.out_dir / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    merged_path = generation_dir / "symcif_v2_constrained_new.jsonl"
    expected = len(cases) * len(seeds)
    if merged_path.exists() and not args.overwrite:
        existing = sum(1 for _ in merged_path.open(encoding="utf-8"))
        if existing == expected:
            print(f"[constrained_v2] found complete {merged_path}, skipping")
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
        worker_path = generation_dir / f"symcif_v2_constrained_new.worker{worker_id}.jsonl"
        worker_paths.append(worker_path)
        proc = ctx.Process(
            target=worker_main,
            kwargs={
                "worker_id": worker_id,
                "cases_payload": payload,
                "seeds": seeds,
                "out_path": str(worker_path),
                "std_dir": str(args.out_dir / "standard_cifs" / f"worker{worker_id}"),
                "model_dir": str(args.model_dir),
                "lookup_json": str(args.lookup_json),
                "device": device,
                "dtype": args.dtype,
                "temp_discrete": args.temp_discrete,
                "topk_discrete": args.topk_discrete,
                "temp_coord": args.temp_coord,
                "topk_coord": args.topk_coord,
                "temp_cell": args.temp_cell,
                "topk_cell": args.topk_cell,
                "max_sites": args.max_sites,
                "skeleton_resample_attempts": args.skeleton_resample_attempts,
                "compile_model": args.compile,
            },
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"constrained worker failed with exit code {proc.exitcode}")

    records: list[dict[str, Any]] = []
    for worker_path in worker_paths:
        with worker_path.open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    records.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    with merged_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
    metadata = {
        "mode": "symcif_v2_constrained_new",
        "test_samples": len(cases),
        "num_gens": len(seeds),
        "seeds": seeds,
        "temp_discrete": args.temp_discrete,
        "topk_discrete": args.topk_discrete,
        "temp_coord": args.temp_coord,
        "topk_coord": args.topk_coord,
        "temp_cell": args.temp_cell,
        "topk_cell": args.topk_cell,
        "skeleton_resample_attempts": args.skeleton_resample_attempts,
        "devices": devices,
        "model_dir": str(args.model_dir),
        "lookup_json": str(args.lookup_json),
    }
    (args.out_dir / "constrained_generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[constrained_v2] wrote {len(records)} records -> {merged_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
