#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
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
from sample_symcif_v2_constrained import (  # noqa: E402
    can_close_all,
    encode_prefix,
    is_fixed_template,
    valid_templates_for_sg,
)
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402


@dataclass(frozen=True)
class SkeletonSite:
    element: str
    template: Any


@dataclass
class SkeletonBeamState:
    remaining: dict[str, int]
    used_fixed: frozenset[str]
    skeleton: tuple[SkeletonSite, ...]
    score_text: str
    partial_logprob: float
    full_logprob: float | None = None
    normalized_logprob: float | None = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_by_sample(rows: list[dict[str, Any]], n: int | None = None) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if n is not None and int(row["gen_index"]) >= n:
            continue
        grouped[int(row["sample_index"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda r: int(r["gen_index"]))
    return grouped


def counter_signature(counter: Counter[tuple[Any, ...]]) -> str:
    return "|".join(
        "{}x{}".format(",".join(str(item) for item in key), count)
        for key, count in sorted(counter.items(), key=lambda item: tuple(str(x) for x in item[0]))
    )


def record_skeleton_counter(record: Any) -> Counter[tuple[int, str]]:
    return Counter((int(site.multiplicity), str(site.letter)) for site in record.sites)


def record_assignment_counter(record: Any) -> Counter[tuple[str, int, str]]:
    return Counter((str(site.element), int(site.multiplicity), str(site.letter)) for site in record.sites)


def record_skeleton_signature(record: Any) -> str:
    return counter_signature(record_skeleton_counter(record))


def record_assignment_signature(record: Any) -> str:
    return counter_signature(record_assignment_counter(record))


def skeleton_counter(skeleton: tuple[SkeletonSite, ...]) -> Counter[tuple[int, str]]:
    return Counter((int(site.template.multiplicity), str(site.template.letter)) for site in skeleton)


def assignment_counter(skeleton: tuple[SkeletonSite, ...]) -> Counter[tuple[str, int, str]]:
    return Counter((str(site.element), int(site.template.multiplicity), str(site.template.letter)) for site in skeleton)


def skeleton_signature(skeleton: tuple[SkeletonSite, ...]) -> str:
    return counter_signature(skeleton_counter(skeleton))


def assignment_signature(skeleton: tuple[SkeletonSite, ...]) -> str:
    return counter_signature(assignment_counter(skeleton))


def skeleton_from_record(record: Any, lookup: WyckoffLookup) -> tuple[SkeletonSite, ...]:
    sites: list[SkeletonSite] = []
    for site in record.sites:
        sites.append(SkeletonSite(element=str(site.element), template=lookup.get(int(record.sg_number), str(site.letter))))
    return tuple(sites)


def placeholder_coord_text(template: Any) -> str:
    return " ".join("0.5000" if bool(free) else "FIXED" for free in template.free_mask)


def skeleton_rows_text(prefix: str, skeleton: tuple[SkeletonSite, ...]) -> str:
    text = prefix
    for index, site in enumerate(skeleton, start=1):
        text += f"{index} {site.element} {site.template.letter} {placeholder_coord_text(site.template)}\n"
    return text


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
            valid_by_element[element] = sorted(candidates, key=lambda t: (int(t.multiplicity), str(t.letter)))
    return valid_by_element


def expand_skeleton_beam(
    model: Any,
    tokenizer: CIFTokenizer,
    beam: SkeletonBeamState,
    templates: tuple[Any, ...],
    *,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    max_expansions_per_beam: int,
) -> list[SkeletonBeamState]:
    site_index = len(beam.skeleton) + 1
    row_prefix = beam.score_text + f"{site_index} "
    valid_by_element = valid_actions(beam.remaining, beam.used_fixed, templates, tokenizer)
    if not valid_by_element:
        return []
    element_lps = allowed_token_logprobs(
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
    for element, element_lp in sorted(element_lps.items(), key=lambda item: item[1], reverse=True):
        letter_lps = allowed_token_logprobs(
            model,
            tokenizer,
            row_prefix + element + " ",
            [tpl.letter for tpl in valid_by_element[element]],
            device=device,
            dtype=dtype,
            temperature=temperature,
            top_k=top_k,
        )
        by_letter = {tpl.letter: tpl for tpl in valid_by_element[element]}
        for letter, letter_lp in letter_lps.items():
            expansions.append((beam.partial_logprob + element_lp + letter_lp, element, by_letter[letter]))
    expansions.sort(key=lambda item: item[0], reverse=True)
    out: list[SkeletonBeamState] = []
    for new_score, element, tpl in expansions[:max_expansions_per_beam]:
        remaining = dict(beam.remaining)
        remaining[element] -= int(tpl.multiplicity)
        if remaining[element] == 0:
            remaining.pop(element)
        used = frozenset((*beam.used_fixed, tpl.letter)) if is_fixed_template(tpl) else beam.used_fixed
        row = row_prefix + f"{element} {tpl.letter} {placeholder_coord_text(tpl)}\n"
        out.append(
            SkeletonBeamState(
                remaining=remaining,
                used_fixed=used,
                skeleton=(*beam.skeleton, SkeletonSite(element=element, template=tpl)),
                score_text=row,
                partial_logprob=new_score,
            )
        )
    return out


def search_legal_skeleton_candidates(
    model: Any,
    tokenizer: CIFTokenizer,
    prefix: str,
    target_counts: dict[str, int],
    sg_number: int,
    lookup: WyckoffLookup,
    *,
    device: str,
    dtype: str,
    temperature: float,
    top_k: int,
    beam_size: int,
    candidate_limit: int,
    max_sites: int,
    max_expansions_per_beam: int,
) -> list[SkeletonBeamState]:
    templates = tuple(valid_templates_for_sg(lookup, int(sg_number), tokenizer))
    if not templates:
        return []
    beams = [
        SkeletonBeamState(
            remaining=dict(target_counts),
            used_fixed=frozenset(),
            skeleton=tuple(),
            score_text=prefix,
            partial_logprob=0.0,
        )
    ]
    completed: dict[str, SkeletonBeamState] = {}
    for _depth in range(max_sites):
        next_beams: list[SkeletonBeamState] = []
        for beam in beams:
            if not beam.remaining:
                old = completed.get(assignment_signature(beam.skeleton))
                if old is None or beam.partial_logprob > old.partial_logprob:
                    completed[assignment_signature(beam.skeleton)] = beam
                continue
            next_beams.extend(
                expand_skeleton_beam(
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
        for beam in next_beams:
            if not beam.remaining:
                old = completed.get(assignment_signature(beam.skeleton))
                if old is None or beam.partial_logprob > old.partial_logprob:
                    completed[assignment_signature(beam.skeleton)] = beam
        next_beams.sort(key=lambda item: item.partial_logprob / max(1, len(item.skeleton)), reverse=True)
        beams = next_beams[:beam_size]
        if len(completed) >= candidate_limit and all(not beam.remaining for beam in beams[: min(len(beams), 5)]):
            break
    ranked = sorted(completed.values(), key=lambda item: item.partial_logprob / max(1, len(item.skeleton)), reverse=True)
    return ranked[:candidate_limit]


def _encode_ids(tokenizer: CIFTokenizer, text: str) -> list[int]:
    token_to_id = tokenizer.token_to_id
    unk_id = token_to_id.get("<unk>")
    ids: list[int] = []
    for token in tokenizer.tokenize_cif(text):
        if token not in token_to_id:
            continue
        token_id = token_to_id[token]
        if unk_id is not None and token_id == unk_id:
            continue
        ids.append(token_id)
    return ids


@torch.no_grad()
def score_text_continuation_logprob(
    model: Any,
    tokenizer: CIFTokenizer,
    prefix: str,
    full_text: str,
    *,
    device: str,
    dtype: str,
) -> tuple[float, float, int]:
    ids = _encode_ids(tokenizer, full_text)
    prefix_len = len(_encode_ids(tokenizer, prefix))
    if len(ids) <= prefix_len:
        return 0.0, math.nan, 0
    start_pos = max(1, prefix_len)
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else torch.no_grad()
    block_size = int(model.config.block_size)
    if len(ids) <= block_size:
        idx = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        next_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        with ctx:
            # CrystaLLM's GPT returns only the final-position logits when targets=None.
            # Supplying targets forces full-sequence logits, which are required for
            # scoring every skeleton token after the prompt.
            logits, _ = model(idx, next_ids)
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        positions = torch.arange(start_pos, len(ids), device=device)
        target = torch.tensor([ids[pos] for pos in range(start_pos, len(ids))], dtype=torch.long, device=device)
        values = logp[positions - 1, target]
        raw = float(values.sum().item())
        count = int(values.numel())
        return raw, raw / max(1, count), count

    raw = 0.0
    count = 0
    for pos in range(start_pos, len(ids)):
        ctx_ids = ids[max(0, pos - block_size) : pos]
        idx = torch.tensor(ctx_ids, dtype=torch.long, device=device).unsqueeze(0)
        with ctx:
            logits, _ = model(idx)
        logp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
        raw += float(logp[ids[pos]].item())
        count += 1
    return raw, raw / max(1, count), count


def rank_skeleton_candidates(
    model: Any,
    tokenizer: CIFTokenizer,
    prefix: str,
    candidates: list[SkeletonBeamState],
    *,
    device: str,
    dtype: str,
) -> list[SkeletonBeamState]:
    scored: list[SkeletonBeamState] = []
    for beam in candidates:
        raw, norm, _count = score_text_continuation_logprob(
            model,
            tokenizer,
            prefix,
            skeleton_rows_text(prefix, beam.skeleton),
            device=device,
            dtype=dtype,
        )
        beam.full_logprob = raw
        beam.normalized_logprob = norm
        scored.append(beam)
    scored.sort(key=lambda item: item.normalized_logprob if item.normalized_logprob is not None else -float("inf"), reverse=True)
    return scored


def lattice_error(gt: Any, pred: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        values[f"{key}_abs_error"] = abs(float(getattr(pred.lattice, key)) - float(getattr(gt.lattice, key)))
    values["volume_rel_error"] = abs(float(pred.lattice.volume) - float(gt.lattice.volume)) / max(abs(float(gt.lattice.volume)), 1e-6)
    return values


def max_cell_error(gt: Any, pred: Any) -> tuple[float, float, float]:
    length_rel = [
        abs(float(getattr(pred.lattice, key)) - float(getattr(gt.lattice, key))) / max(abs(float(getattr(gt.lattice, key))), 1e-6)
        for key in ("a", "b", "c")
    ]
    angle_abs = [
        abs(float(getattr(pred.lattice, key)) - float(getattr(gt.lattice, key)))
        for key in ("alpha", "beta", "gamma")
    ]
    volume_rel = abs(float(pred.lattice.volume) - float(gt.lattice.volume)) / max(abs(float(gt.lattice.volume)), 1e-6)
    return max(length_rel), max(angle_abs), volume_rel


def fractional_delta(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 1.0
    return min(d, 1.0 - d)


def free_coord_errors(gt: Any, pred: Any) -> tuple[float | None, int]:
    gt_groups: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    pred_groups: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for site in gt.sites:
        gt_groups[(str(site.element), str(site.letter), int(site.multiplicity))].append(site)
    for site in pred.sites:
        pred_groups[(str(site.element), str(site.letter), int(site.multiplicity))].append(site)
    diffs: list[float] = []
    for key, gt_sites in gt_groups.items():
        pred_sites = pred_groups.get(key, [])
        if len(gt_sites) != len(pred_sites):
            continue
        gt_sites = sorted(gt_sites, key=lambda s: int(s.index))
        pred_sites = sorted(pred_sites, key=lambda s: int(s.index))
        for gsite, psite in zip(gt_sites, pred_sites, strict=True):
            for axis, free in enumerate(gsite.free_mask):
                if bool(free) and axis < len(psite.representative_coord):
                    diffs.append(fractional_delta(gsite.representative_coord[axis], psite.representative_coord[axis]))
    if not diffs:
        return None, 0
    return sum(diffs) / len(diffs), len(diffs)
