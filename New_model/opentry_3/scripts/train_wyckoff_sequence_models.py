#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise SystemExit(f"refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_records is not None and len(rows) >= int(max_records):
                    break
    return rows


def write_json(path: Path, row: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pct(value: float) -> float:
    return float(value) * 100.0


def hit_at(keys: list[str], target: str, k: int) -> bool:
    return str(target) in set(str(x) for x in keys[: int(k)])


def seq_key_from_orbits(orbits: list[str]) -> str:
    return "|".join(orbits)


def seq_key_from_pairs(pairs: list[tuple[str, str]]) -> str:
    return "|".join(f"{orbit}:{element}" for element, orbit in pairs)


@dataclass(frozen=True)
class OrbitInfo:
    orbit_id: str
    sg: int
    multiplicity: int
    letter: str
    sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class Vocab:
    elements: list[str]
    sgs: list[int]
    orbits: list[str]
    pairs: list[tuple[str, str]]
    element_to_id: dict[str, int]
    sg_to_id: dict[int, int]
    orbit_to_id: dict[str, int]
    pair_to_id: dict[tuple[str, str], int]
    orbit_info: dict[str, OrbitInfo]
    sg_orbits: dict[int, list[str]]

    @property
    def start_orbit_id(self) -> int:
        return len(self.orbits) + 1

    @property
    def eos_orbit_id(self) -> int:
        return len(self.orbits)

    @property
    def start_pair_id(self) -> int:
        return len(self.pairs) + 1

    @property
    def eos_pair_id(self) -> int:
        return len(self.pairs)


def orbit_sort_tuple(site: dict[str, Any]) -> tuple[Any, ...]:
    enum = site.get("enumeration")
    enum_key = "" if enum is None else str(enum)
    return (
        int(site["multiplicity"]),
        str(site["letter"]),
        enum_key,
        str(site.get("site_symmetry") or ""),
        str(site["orbit_id"]),
    )


def build_vocab(records: list[dict[str, Any]]) -> Vocab:
    elements = sorted({str(e) for record in records for e in record["formula_counts"].keys()})
    sgs = sorted({int(record["sg"]) for record in records})
    orbit_info: dict[str, OrbitInfo] = {}
    sg_orbits_raw: defaultdict[int, set[str]] = defaultdict(set)
    pairs_set: set[tuple[str, str]] = set()
    for record in records:
        sg = int(record["sg"])
        for site in record["wa_sequence"]:
            orbit_id = str(site["orbit_id"])
            sort_key = orbit_sort_tuple(site)
            if orbit_id not in orbit_info:
                orbit_info[orbit_id] = OrbitInfo(
                    orbit_id=orbit_id,
                    sg=sg,
                    multiplicity=int(site["multiplicity"]),
                    letter=str(site["letter"]),
                    sort_key=sort_key,
                )
            sg_orbits_raw[sg].add(orbit_id)
            pairs_set.add((str(site["element"]), orbit_id))
    orbits = sorted(orbit_info, key=lambda oid: (orbit_info[oid].sg, orbit_info[oid].sort_key))
    orbit_to_id = {orbit: i for i, orbit in enumerate(orbits)}
    pairs = sorted(pairs_set, key=lambda pair: (pair[0], orbit_to_id.get(pair[1], 10**9)))
    sg_orbits = {
        sg: sorted(items, key=lambda oid: orbit_info[oid].sort_key)
        for sg, items in sg_orbits_raw.items()
    }
    return Vocab(
        elements=elements,
        sgs=sgs,
        orbits=orbits,
        pairs=pairs,
        element_to_id={e: i for i, e in enumerate(elements)},
        sg_to_id={sg: i + 1 for i, sg in enumerate(sgs)},
        orbit_to_id=orbit_to_id,
        pair_to_id={pair: i for i, pair in enumerate(pairs)},
        orbit_info=orbit_info,
        sg_orbits=sg_orbits,
    )


def vocab_to_jsonable(vocab: Vocab) -> dict[str, Any]:
    return {
        "elements": vocab.elements,
        "sgs": vocab.sgs,
        "orbits": vocab.orbits,
        "pairs": [[e, o] for e, o in vocab.pairs],
        "orbit_info": {
            oid: {
                "sg": info.sg,
                "multiplicity": info.multiplicity,
                "letter": info.letter,
                "sort_key": list(info.sort_key),
            }
            for oid, info in vocab.orbit_info.items()
        },
        "sg_orbits": {str(k): v for k, v in vocab.sg_orbits.items()},
    }


def vocab_from_jsonable(raw: dict[str, Any]) -> Vocab:
    elements = [str(x) for x in raw["elements"]]
    sgs = [int(x) for x in raw["sgs"]]
    orbits = [str(x) for x in raw["orbits"]]
    pairs = [(str(e), str(o)) for e, o in raw["pairs"]]
    orbit_info = {
        str(oid): OrbitInfo(
            orbit_id=str(oid),
            sg=int(item["sg"]),
            multiplicity=int(item["multiplicity"]),
            letter=str(item["letter"]),
            sort_key=tuple(item["sort_key"]),
        )
        for oid, item in raw["orbit_info"].items()
    }
    sg_orbits = {int(k): [str(x) for x in v] for k, v in raw["sg_orbits"].items()}
    return Vocab(
        elements=elements,
        sgs=sgs,
        orbits=orbits,
        pairs=pairs,
        element_to_id={e: i for i, e in enumerate(elements)},
        sg_to_id={sg: i + 1 for i, sg in enumerate(sgs)},
        orbit_to_id={orbit: i for i, orbit in enumerate(orbits)},
        pair_to_id={pair: i for i, pair in enumerate(pairs)},
        orbit_info=orbit_info,
        sg_orbits=sg_orbits,
    )


def build_decoder_priors(records: list[dict[str, Any]], vocab: Vocab, alpha: float = 0.25) -> dict[str, Any]:
    element_counts: Counter[str] = Counter()
    orbit_counts: Counter[str] = Counter()
    element_orbit_counts: Counter[str] = Counter()
    sg_element_orbit_counts: Counter[str] = Counter()
    sg_orbit_counts: Counter[str] = Counter()
    for record in records:
        sg = int(record["sg"])
        for site in record["wa_sequence"]:
            element = str(site["element"])
            orbit = str(site["orbit_id"])
            element_counts[element] += 1
            orbit_counts[orbit] += 1
            element_orbit_counts[f"{element}@{orbit}"] += 1
            sg_orbit_counts[f"{sg}@{orbit}"] += 1
            sg_element_orbit_counts[f"{sg}@{element}@{orbit}"] += 1
    return {
        "alpha": float(alpha),
        "num_elements": len(vocab.elements),
        "element_counts": dict(element_counts),
        "orbit_counts": dict(orbit_counts),
        "element_orbit_counts": dict(element_orbit_counts),
        "sg_orbit_counts": dict(sg_orbit_counts),
        "sg_element_orbit_counts": dict(sg_element_orbit_counts),
    }


def prior_assignment_logprob(element: str, orbit_id: str, sg: int, priors: dict[str, Any]) -> float:
    alpha = float(priors.get("alpha", 0.25))
    n_elem = max(1, int(priors.get("num_elements", 1)))
    eo = int(priors.get("element_orbit_counts", {}).get(f"{element}@{orbit_id}", 0))
    o = int(priors.get("orbit_counts", {}).get(orbit_id, 0))
    sgeo = int(priors.get("sg_element_orbit_counts", {}).get(f"{int(sg)}@{element}@{orbit_id}", 0))
    sgo = int(priors.get("sg_orbit_counts", {}).get(f"{int(sg)}@{orbit_id}", 0))
    global_term = math.log((eo + alpha) / (o + alpha * n_elem))
    sg_term = math.log((sgeo + alpha) / (sgo + alpha * n_elem)) if sgo > 0 else global_term
    return 0.65 * sg_term + 0.35 * global_term


def formula_vec(record: dict[str, Any], vocab: Vocab) -> torch.Tensor:
    vec = torch.zeros((len(vocab.elements),), dtype=torch.float32)
    total = max(1, int(record["atom_count"]))
    for element, count in record["formula_counts"].items():
        idx = vocab.element_to_id.get(str(element))
        if idx is not None:
            vec[idx] = float(int(count)) / float(total)
    return vec


def remaining_vec(remaining: dict[str, int], total: int, vocab: Vocab) -> torch.Tensor:
    vec = torch.zeros((len(vocab.elements),), dtype=torch.float32)
    denom = max(1, int(total))
    for element, count in remaining.items():
        idx = vocab.element_to_id.get(str(element))
        if idx is not None:
            vec[idx] = float(int(count)) / float(denom)
    return vec


def record_weight(record: dict[str, Any], complex_weight: float) -> float:
    weight = 1.0
    if bool(record.get("complex_flag")):
        weight *= float(complex_weight)
    if int(record.get("row_count", 0)) >= 7:
        weight *= 1.5
    return float(weight)


def make_skeleton_states(records: list[dict[str, Any]], vocab: Vocab, complex_weight: float) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for rec in records:
        seq = [str(x["orbit_id"]) for x in rec["skeleton_sequence"]]
        if any(oid not in vocab.orbit_to_id for oid in seq):
            continue
        total = int(rec["atom_count"])
        remaining_total = total
        last = vocab.start_orbit_id
        fvec = formula_vec(rec, vocab)
        weight = record_weight(rec, complex_weight)
        for step, orbit_id in enumerate(seq):
            target = vocab.orbit_to_id[orbit_id]
            states.append(
                {
                    "sg_id": vocab.sg_to_id.get(int(rec["sg"]), 0),
                    "sg": int(rec["sg"]),
                    "formula_vec": fvec,
                    "remaining_total": float(remaining_total) / max(1.0, float(total)),
                    "step": float(step) / 64.0,
                    "last": last,
                    "target": target,
                    "weight": weight,
                }
            )
            remaining_total -= int(vocab.orbit_info[orbit_id].multiplicity)
            last = target
        states.append(
            {
                "sg_id": vocab.sg_to_id.get(int(rec["sg"]), 0),
                "sg": int(rec["sg"]),
                "formula_vec": fvec,
                "remaining_total": 0.0,
                "step": float(len(seq)) / 64.0,
                "last": last,
                "target": vocab.eos_orbit_id,
                "weight": weight,
            }
        )
    return states


def make_direct_wa_states(records: list[dict[str, Any]], vocab: Vocab, complex_weight: float) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for rec in records:
        total = int(rec["atom_count"])
        remaining = {str(k): int(v) for k, v in rec["formula_counts"].items()}
        last = vocab.start_pair_id
        fvec = formula_vec(rec, vocab)
        weight = record_weight(rec, complex_weight)
        ok = True
        for step, site in enumerate(rec["wa_sequence"]):
            pair = (str(site["element"]), str(site["orbit_id"]))
            if pair not in vocab.pair_to_id:
                ok = False
                break
            states.append(
                {
                    "sg_id": vocab.sg_to_id.get(int(rec["sg"]), 0),
                    "sg": int(rec["sg"]),
                    "formula_vec": fvec,
                    "remaining_vec": remaining_vec(remaining, total, vocab),
                    "remaining_total": float(sum(remaining.values())) / max(1.0, float(total)),
                    "step": float(step) / 64.0,
                    "last": last,
                    "target": vocab.pair_to_id[pair],
                    "weight": weight,
                }
            )
            remaining[pair[0]] = int(remaining[pair[0]]) - int(site["multiplicity"])
            last = vocab.pair_to_id[pair]
        if not ok:
            continue
        states.append(
            {
                "sg_id": vocab.sg_to_id.get(int(rec["sg"]), 0),
                "sg": int(rec["sg"]),
                "formula_vec": fvec,
                "remaining_vec": remaining_vec(remaining, total, vocab),
                "remaining_total": 0.0,
                "step": float(len(rec["wa_sequence"])) / 64.0,
                "last": last,
                "target": vocab.eos_pair_id,
                "weight": weight,
            }
        )
    return states


def make_assignment_states(records: list[dict[str, Any]], vocab: Vocab, complex_weight: float) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for rec in records:
        total = int(rec["atom_count"])
        remaining = {str(k): int(v) for k, v in rec["formula_counts"].items()}
        fvec = formula_vec(rec, vocab)
        weight = record_weight(rec, complex_weight)
        for step, site in enumerate(rec["wa_sequence"]):
            orbit_id = str(site["orbit_id"])
            element = str(site["element"])
            if orbit_id not in vocab.orbit_to_id or element not in vocab.element_to_id:
                continue
            states.append(
                {
                    "sg_id": vocab.sg_to_id.get(int(rec["sg"]), 0),
                    "sg": int(rec["sg"]),
                    "formula_vec": fvec,
                    "remaining_vec": remaining_vec(remaining, total, vocab),
                    "remaining_total": float(sum(remaining.values())) / max(1.0, float(total)),
                    "step": float(step) / 64.0,
                    "orbit": vocab.orbit_to_id[orbit_id],
                    "multiplicity": float(site["multiplicity"]) / 64.0,
                    "target": vocab.element_to_id[element],
                    "weight": weight,
                }
            )
            remaining[element] = int(remaining[element]) - int(site["multiplicity"])
    return states


class SkeletonNet(nn.Module):
    def __init__(self, num_sgs: int, formula_dim: int, num_orbit_tokens: int, hidden: int) -> None:
        super().__init__()
        emb = max(32, hidden // 4)
        self.sg_emb = nn.Embedding(num_sgs + 1, emb)
        self.last_emb = nn.Embedding(num_orbit_tokens, emb)
        self.net = nn.Sequential(
            nn.LayerNorm(formula_dim + emb * 2 + 2),
            nn.Linear(formula_dim + emb * 2 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_orbit_tokens - 1),
        )

    def forward(self, sg_id: torch.Tensor, formula: torch.Tensor, last: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.sg_emb(sg_id), self.last_emb(last), formula, numeric], dim=-1)
        return self.net(x)


class DirectWANet(nn.Module):
    def __init__(self, num_sgs: int, formula_dim: int, num_pair_tokens: int, hidden: int) -> None:
        super().__init__()
        emb = max(32, hidden // 4)
        self.sg_emb = nn.Embedding(num_sgs + 1, emb)
        self.last_emb = nn.Embedding(num_pair_tokens, emb)
        self.net = nn.Sequential(
            nn.LayerNorm(formula_dim * 2 + emb * 2 + 2),
            nn.Linear(formula_dim * 2 + emb * 2 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_pair_tokens - 1),
        )

    def forward(
        self,
        sg_id: torch.Tensor,
        formula: torch.Tensor,
        remaining: torch.Tensor,
        last: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([self.sg_emb(sg_id), self.last_emb(last), formula, remaining, numeric], dim=-1)
        return self.net(x)


class AssignmentNet(nn.Module):
    def __init__(self, num_sgs: int, formula_dim: int, num_orbits: int, num_elements: int, hidden: int) -> None:
        super().__init__()
        emb = max(32, hidden // 4)
        self.sg_emb = nn.Embedding(num_sgs + 1, emb)
        self.orbit_emb = nn.Embedding(num_orbits, emb)
        self.net = nn.Sequential(
            nn.LayerNorm(formula_dim * 2 + emb * 2 + 3),
            nn.Linear(formula_dim * 2 + emb * 2 + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_elements),
        )

    def forward(
        self,
        sg_id: torch.Tensor,
        formula: torch.Tensor,
        remaining: torch.Tensor,
        orbit: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([self.sg_emb(sg_id), self.orbit_emb(orbit), formula, remaining, numeric], dim=-1)
        return self.net(x)


class WyckoffModels(nn.Module):
    def __init__(self, vocab: Vocab, hidden: int) -> None:
        super().__init__()
        formula_dim = len(vocab.elements)
        self.skeleton = SkeletonNet(
            num_sgs=len(vocab.sgs),
            formula_dim=formula_dim,
            num_orbit_tokens=len(vocab.orbits) + 2,
            hidden=hidden,
        )
        self.direct_wa = DirectWANet(
            num_sgs=len(vocab.sgs),
            formula_dim=formula_dim,
            num_pair_tokens=len(vocab.pairs) + 2,
            hidden=hidden,
        )
        self.assignment = AssignmentNet(
            num_sgs=len(vocab.sgs),
            formula_dim=formula_dim,
            num_orbits=len(vocab.orbits),
            num_elements=len(vocab.elements),
            hidden=hidden,
        )


def batch_indexes(n: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def tensor_batch(states: list[dict[str, Any]], indexes: list[int], key: str, device: torch.device) -> torch.Tensor:
    values = [states[i][key] for i in indexes]
    if values and isinstance(values[0], torch.Tensor):
        return torch.stack(values).to(device)
    dtype = torch.long if key in {"sg_id", "last", "target", "orbit"} else torch.float32
    return torch.tensor(values, dtype=dtype, device=device)


def weighted_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def train_epoch(
    model: WyckoffModels,
    optimizer: torch.optim.Optimizer,
    skeleton_states: list[dict[str, Any]],
    assignment_states: list[dict[str, Any]],
    direct_states: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.train()
    rng = random.Random(seed)
    totals = {"skeleton_loss": 0.0, "assignment_loss": 0.0, "direct_wa_loss": 0.0}
    counts = {"skeleton_loss": 0, "assignment_loss": 0, "direct_wa_loss": 0}

    for indexes in batch_indexes(len(skeleton_states), batch_size, rng):
        optimizer.zero_grad(set_to_none=True)
        sg = tensor_batch(skeleton_states, indexes, "sg_id", device)
        formula = tensor_batch(skeleton_states, indexes, "formula_vec", device)
        last = tensor_batch(skeleton_states, indexes, "last", device)
        numeric = torch.stack(
            [
                tensor_batch(skeleton_states, indexes, "remaining_total", device),
                tensor_batch(skeleton_states, indexes, "step", device),
            ],
            dim=-1,
        )
        target = tensor_batch(skeleton_states, indexes, "target", device)
        weight = tensor_batch(skeleton_states, indexes, "weight", device)
        loss = weighted_ce(model.skeleton(sg, formula, last, numeric), target, weight)
        loss.backward()
        optimizer.step()
        totals["skeleton_loss"] += float(loss.detach().cpu())
        counts["skeleton_loss"] += 1

    for indexes in batch_indexes(len(assignment_states), batch_size, rng):
        optimizer.zero_grad(set_to_none=True)
        sg = tensor_batch(assignment_states, indexes, "sg_id", device)
        formula = tensor_batch(assignment_states, indexes, "formula_vec", device)
        remaining = tensor_batch(assignment_states, indexes, "remaining_vec", device)
        orbit = tensor_batch(assignment_states, indexes, "orbit", device)
        numeric = torch.stack(
            [
                tensor_batch(assignment_states, indexes, "remaining_total", device),
                tensor_batch(assignment_states, indexes, "step", device),
                tensor_batch(assignment_states, indexes, "multiplicity", device),
            ],
            dim=-1,
        )
        target = tensor_batch(assignment_states, indexes, "target", device)
        weight = tensor_batch(assignment_states, indexes, "weight", device)
        loss = weighted_ce(model.assignment(sg, formula, remaining, orbit, numeric), target, weight)
        loss.backward()
        optimizer.step()
        totals["assignment_loss"] += float(loss.detach().cpu())
        counts["assignment_loss"] += 1

    for indexes in batch_indexes(len(direct_states), batch_size, rng):
        optimizer.zero_grad(set_to_none=True)
        sg = tensor_batch(direct_states, indexes, "sg_id", device)
        formula = tensor_batch(direct_states, indexes, "formula_vec", device)
        remaining = tensor_batch(direct_states, indexes, "remaining_vec", device)
        last = tensor_batch(direct_states, indexes, "last", device)
        numeric = torch.stack(
            [
                tensor_batch(direct_states, indexes, "remaining_total", device),
                tensor_batch(direct_states, indexes, "step", device),
            ],
            dim=-1,
        )
        target = tensor_batch(direct_states, indexes, "target", device)
        weight = tensor_batch(direct_states, indexes, "weight", device)
        loss = weighted_ce(model.direct_wa(sg, formula, remaining, last, numeric), target, weight)
        loss.backward()
        optimizer.step()
        totals["direct_wa_loss"] += float(loss.detach().cpu())
        counts["direct_wa_loss"] += 1

    return {k: totals[k] / max(1, counts[k]) for k in totals}


@torch.no_grad()
def legal_skeleton_actions(vocab: Vocab, sg: int, remaining_total: int) -> list[int]:
    if remaining_total == 0:
        return [vocab.eos_orbit_id]
    actions: list[int] = []
    for orbit_id in vocab.sg_orbits.get(int(sg), []):
        info = vocab.orbit_info[orbit_id]
        if int(info.multiplicity) <= int(remaining_total):
            actions.append(vocab.orbit_to_id[orbit_id])
    return actions


@torch.no_grad()
def beam_skeleton(
    model: WyckoffModels,
    record: dict[str, Any],
    vocab: Vocab,
    device: torch.device,
    *,
    beam_size: int,
    branch: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    sg = int(record["sg"])
    total = int(record["atom_count"])
    sg_id = torch.tensor([vocab.sg_to_id.get(sg, 0)], dtype=torch.long, device=device)
    fvec = formula_vec(record, vocab).unsqueeze(0).to(device)
    beams = [{"orbits": [], "last": vocab.start_orbit_id, "remaining": total, "score": 0.0}]
    complete: dict[str, dict[str, Any]] = {}
    model.eval()
    for step in range(max_steps):
        next_beams: list[dict[str, Any]] = []
        for beam in beams:
            legal = legal_skeleton_actions(vocab, sg, int(beam["remaining"]))
            if not legal:
                continue
            last = torch.tensor([int(beam["last"])], dtype=torch.long, device=device)
            numeric = torch.tensor(
                [[float(beam["remaining"]) / max(1.0, float(total)), float(step) / 64.0]],
                dtype=torch.float32,
                device=device,
            )
            logits = model.skeleton(sg_id, fvec, last, numeric).squeeze(0)
            legal_tensor = torch.tensor(legal, dtype=torch.long, device=device)
            legal_scores = F.log_softmax(logits[legal_tensor], dim=-1)
            top_n = min(len(legal), int(branch))
            values, indexes = torch.topk(legal_scores, k=top_n)
            for value, local_idx in zip(values.detach().cpu().tolist(), indexes.detach().cpu().tolist()):
                action = legal[int(local_idx)]
                score = float(beam["score"]) + float(value)
                if action == vocab.eos_orbit_id:
                    key = seq_key_from_orbits(list(beam["orbits"]))
                    if key and key not in complete:
                        complete[key] = {
                            "skeleton_key": key,
                            "orbits": list(beam["orbits"]),
                            "score": score,
                            "composition_exact": int(beam["remaining"]) == 0,
                        }
                    continue
                orbit_id = vocab.orbits[action]
                info = vocab.orbit_info[orbit_id]
                if int(info.multiplicity) > int(beam["remaining"]):
                    continue
                next_beams.append(
                    {
                        "orbits": [*beam["orbits"], orbit_id],
                        "last": action,
                        "remaining": int(beam["remaining"]) - int(info.multiplicity),
                        "score": score,
                    }
                )
        next_beams.sort(key=lambda item: item["score"] / max(1, len(item["orbits"])), reverse=True)
        beams = next_beams[: int(beam_size)]
        if len(complete) >= int(beam_size) and not beams:
            break
        if not beams:
            break
    ranked = sorted(complete.values(), key=lambda item: item["score"] / max(1, len(item["orbits"])), reverse=True)
    return ranked[: int(beam_size)]


@torch.no_grad()
def legal_direct_actions(vocab: Vocab, sg: int, remaining: dict[str, int]) -> list[int]:
    if sum(int(v) for v in remaining.values()) == 0:
        return [vocab.eos_pair_id]
    actions: list[int] = []
    for orbit_id in vocab.sg_orbits.get(int(sg), []):
        info = vocab.orbit_info[orbit_id]
        for element, count in remaining.items():
            if int(count) >= int(info.multiplicity):
                idx = vocab.pair_to_id.get((str(element), orbit_id))
                if idx is not None:
                    actions.append(idx)
    return actions


@torch.no_grad()
def beam_direct_wa(
    model: WyckoffModels,
    record: dict[str, Any],
    vocab: Vocab,
    device: torch.device,
    *,
    beam_size: int,
    branch: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    sg = int(record["sg"])
    total = int(record["atom_count"])
    sg_id = torch.tensor([vocab.sg_to_id.get(sg, 0)], dtype=torch.long, device=device)
    fvec = formula_vec(record, vocab).unsqueeze(0).to(device)
    beams = [
        {
            "pairs": [],
            "last": vocab.start_pair_id,
            "remaining": {str(k): int(v) for k, v in record["formula_counts"].items()},
            "score": 0.0,
        }
    ]
    complete: dict[str, dict[str, Any]] = {}
    model.eval()
    for step in range(max_steps):
        next_beams: list[dict[str, Any]] = []
        for beam in beams:
            remaining = {str(k): int(v) for k, v in beam["remaining"].items()}
            legal = legal_direct_actions(vocab, sg, remaining)
            if not legal:
                continue
            last = torch.tensor([int(beam["last"])], dtype=torch.long, device=device)
            rvec = remaining_vec(remaining, total, vocab).unsqueeze(0).to(device)
            numeric = torch.tensor(
                [[float(sum(remaining.values())) / max(1.0, float(total)), float(step) / 64.0]],
                dtype=torch.float32,
                device=device,
            )
            logits = model.direct_wa(sg_id, fvec, rvec, last, numeric).squeeze(0)
            legal_tensor = torch.tensor(legal, dtype=torch.long, device=device)
            legal_scores = F.log_softmax(logits[legal_tensor], dim=-1)
            top_n = min(len(legal), int(branch))
            values, indexes = torch.topk(legal_scores, k=top_n)
            for value, local_idx in zip(values.detach().cpu().tolist(), indexes.detach().cpu().tolist()):
                action = legal[int(local_idx)]
                score = float(beam["score"]) + float(value)
                if action == vocab.eos_pair_id:
                    key = seq_key_from_pairs(list(beam["pairs"]))
                    if key and key not in complete:
                        complete[key] = {
                            "wa_key": key,
                            "pairs": list(beam["pairs"]),
                            "score": score,
                            "source": "direct_c",
                            "composition_exact": sum(remaining.values()) == 0,
                        }
                    continue
                element, orbit_id = vocab.pairs[action]
                mult = int(vocab.orbit_info[orbit_id].multiplicity)
                if int(remaining.get(element, 0)) < mult:
                    continue
                nxt = dict(remaining)
                nxt[element] = int(nxt[element]) - mult
                next_beams.append(
                    {
                        "pairs": [*beam["pairs"], (element, orbit_id)],
                        "last": action,
                        "remaining": nxt,
                        "score": score,
                    }
                )
        next_beams.sort(key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)
        beams = next_beams[: int(beam_size)]
        if not beams:
            break
    ranked = sorted(complete.values(), key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)
    return ranked[: int(beam_size)]


@torch.no_grad()
def beam_assign_skeleton(
    model: WyckoffModels,
    record: dict[str, Any],
    skeleton: dict[str, Any],
    vocab: Vocab,
    device: torch.device,
    *,
    beam_size: int,
    branch: int,
) -> list[dict[str, Any]]:
    orbit_ids = [str(x) for x in skeleton["orbits"]]
    if sum(vocab.orbit_info[oid].multiplicity for oid in orbit_ids if oid in vocab.orbit_info) != int(record["atom_count"]):
        return []
    orbit_ids = sorted(orbit_ids, key=lambda oid: vocab.orbit_info[oid].sort_key)
    total = int(record["atom_count"])
    sg = int(record["sg"])
    sg_id = torch.tensor([vocab.sg_to_id.get(sg, 0)], dtype=torch.long, device=device)
    fvec = formula_vec(record, vocab).unsqueeze(0).to(device)
    beams = [
        {
            "pairs": [],
            "remaining": {str(k): int(v) for k, v in record["formula_counts"].items()},
            "score": float(skeleton["score"]),
        }
    ]
    for step, orbit_id in enumerate(orbit_ids):
        info = vocab.orbit_info[orbit_id]
        next_beams: list[dict[str, Any]] = []
        for beam in beams:
            remaining = {str(k): int(v) for k, v in beam["remaining"].items()}
            legal_elements = [e for e, count in sorted(remaining.items()) if int(count) >= int(info.multiplicity)]
            if not legal_elements:
                continue
            orbit = torch.tensor([vocab.orbit_to_id[orbit_id]], dtype=torch.long, device=device)
            rvec = remaining_vec(remaining, total, vocab).unsqueeze(0).to(device)
            numeric = torch.tensor(
                [
                    [
                        float(sum(remaining.values())) / max(1.0, float(total)),
                        float(step) / 64.0,
                        float(info.multiplicity) / 64.0,
                    ]
                ],
                dtype=torch.float32,
                device=device,
            )
            logits = model.assignment(sg_id, fvec, rvec, orbit, numeric).squeeze(0)
            legal_ids = [vocab.element_to_id[e] for e in legal_elements if e in vocab.element_to_id]
            if not legal_ids:
                continue
            legal_tensor = torch.tensor(legal_ids, dtype=torch.long, device=device)
            legal_scores = F.log_softmax(logits[legal_tensor], dim=-1)
            top_n = min(len(legal_ids), int(branch))
            values, indexes = torch.topk(legal_scores, k=top_n)
            for value, local_idx in zip(values.detach().cpu().tolist(), indexes.detach().cpu().tolist()):
                element = vocab.elements[legal_ids[int(local_idx)]]
                mult = int(info.multiplicity)
                nxt = dict(remaining)
                nxt[element] = int(nxt[element]) - mult
                next_beams.append(
                    {
                        "pairs": [*beam["pairs"], (element, orbit_id)],
                        "remaining": nxt,
                        "score": float(beam["score"]) + float(value),
                    }
                )
        next_beams.sort(key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)
        beams = next_beams[: int(beam_size)]
        if not beams:
            break
    complete: dict[str, dict[str, Any]] = {}
    for beam in beams:
        if sum(int(v) for v in beam["remaining"].values()) != 0:
            continue
        key = seq_key_from_pairs(list(beam["pairs"]))
        complete.setdefault(
            key,
            {
                "wa_key": key,
                "pairs": list(beam["pairs"]),
                "score": float(beam["score"]),
                "source": "ab",
                "composition_exact": True,
            },
        )
    return sorted(complete.values(), key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)[: int(beam_size)]


def beam_assign_skeleton_prior(
    record: dict[str, Any],
    skeleton: dict[str, Any],
    vocab: Vocab,
    priors: dict[str, Any],
    *,
    beam_size: int,
    branch: int,
) -> list[dict[str, Any]]:
    orbit_ids = [str(x) for x in skeleton["orbits"]]
    if any(oid not in vocab.orbit_info for oid in orbit_ids):
        return []
    if sum(vocab.orbit_info[oid].multiplicity for oid in orbit_ids) != int(record["atom_count"]):
        return []
    orbit_ids = sorted(orbit_ids, key=lambda oid: vocab.orbit_info[oid].sort_key)
    sg = int(record["sg"])
    beams = [
        {
            "pairs": [],
            "remaining": {str(k): int(v) for k, v in record["formula_counts"].items()},
            "score": float(skeleton["score"]),
        }
    ]
    for orbit_id in orbit_ids:
        info = vocab.orbit_info[orbit_id]
        next_beams: list[dict[str, Any]] = []
        for beam in beams:
            remaining = {str(k): int(v) for k, v in beam["remaining"].items()}
            scored: list[tuple[float, str]] = []
            for element, count in sorted(remaining.items()):
                if int(count) >= int(info.multiplicity):
                    scored.append((prior_assignment_logprob(element, orbit_id, sg, priors), element))
            scored.sort(key=lambda item: item[0], reverse=True)
            for value, element in scored[: max(1, int(branch))]:
                nxt = dict(remaining)
                nxt[element] = int(nxt[element]) - int(info.multiplicity)
                next_beams.append(
                    {
                        "pairs": [*beam["pairs"], (element, orbit_id)],
                        "remaining": nxt,
                        "score": float(beam["score"]) + float(value),
                    }
                )
        next_beams.sort(key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)
        beams = next_beams[: int(beam_size)]
        if not beams:
            break
    complete: dict[str, dict[str, Any]] = {}
    for beam in beams:
        if sum(int(v) for v in beam["remaining"].values()) != 0:
            continue
        key = seq_key_from_pairs(list(beam["pairs"]))
        complete.setdefault(
            key,
            {
                "wa_key": key,
                "pairs": list(beam["pairs"]),
                "score": float(beam["score"]),
                "source": "prior_assignment_b",
                "composition_exact": True,
            },
        )
    return sorted(complete.values(), key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)[: int(beam_size)]


def summarize_hits(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": len(rows)}
    for metric in ("skeleton", "wa_ab", "wa_prior", "wa_c", "wa_union"):
        for k in ks:
            out[f"{metric}@{k}"] = sum(int(row[f"{metric}_hit@{k}"]) for row in rows) / max(1, len(rows))
    out["unique_skeleton_mean"] = sum(int(row["unique_skeleton"]) for row in rows) / max(1, len(rows))
    out["unique_wa_ab_mean"] = sum(int(row["unique_wa_ab"]) for row in rows) / max(1, len(rows))
    out["unique_wa_prior_mean"] = sum(int(row["unique_wa_prior"]) for row in rows) / max(1, len(rows))
    out["unique_wa_c_mean"] = sum(int(row["unique_wa_c"]) for row in rows) / max(1, len(rows))
    out["unique_wa_union_mean"] = sum(int(row["unique_wa_union"]) for row in rows) / max(1, len(rows))
    out["candidate_nonempty_rate"] = sum(int(row["unique_wa_union"] > 0) for row in rows) / max(1, len(rows))
    out["composition_exact_rate"] = sum(int(row["composition_exact_any"]) for row in rows) / max(1, len(rows))
    return out


@torch.no_grad()
def evaluate_generators(
    model: WyckoffModels,
    records: list[dict[str, Any]],
    vocab: Vocab,
    device: torch.device,
    *,
    skeleton_beam: int,
    wa_beam: int,
    branch: int,
    assignment_branch: int,
    assignment_per_skeleton: int,
    max_steps: int,
    decoder_priors: dict[str, Any],
    use_learned_ab: bool,
    use_direct_c: bool,
    out_candidates: Path | None = None,
) -> dict[str, Any]:
    ks = (1, 5, 20, 50)
    per_sample: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        skels = beam_skeleton(
            model,
            record,
            vocab,
            device,
            beam_size=skeleton_beam,
            branch=branch,
            max_steps=max_steps,
        )
        skel_keys = [str(x["skeleton_key"]) for x in skels]
        wa_ab: dict[str, dict[str, Any]] = {}
        per_skel_beam = max(1, int(assignment_per_skeleton)) if assignment_per_skeleton > 0 else max(1, wa_beam // max(1, min(len(skels), skeleton_beam)))
        if use_learned_ab:
            for skel in skels[: min(len(skels), skeleton_beam)]:
                for cand in beam_assign_skeleton(
                    model,
                    record,
                    skel,
                    vocab,
                    device,
                    beam_size=per_skel_beam,
                    branch=assignment_branch,
                ):
                    wa_ab.setdefault(str(cand["wa_key"]), cand)
        wa_prior: dict[str, dict[str, Any]] = {}
        for skel in skels[: min(len(skels), skeleton_beam)]:
            for cand in beam_assign_skeleton_prior(
                record,
                skel,
                vocab,
                decoder_priors,
                beam_size=per_skel_beam,
                branch=assignment_branch,
            ):
                wa_prior.setdefault(str(cand["wa_key"]), cand)
        ab_ranked = sorted(wa_ab.values(), key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)[:wa_beam]
        prior_ranked = sorted(wa_prior.values(), key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)[:wa_beam]
        c_ranked = (
            beam_direct_wa(
                model,
                record,
                vocab,
                device,
                beam_size=wa_beam,
                branch=branch,
                max_steps=max_steps,
            )
            if use_direct_c
            else []
        )
        union: dict[str, dict[str, Any]] = {}
        for cand in ab_ranked:
            union.setdefault(str(cand["wa_key"]), cand)
        for cand in prior_ranked:
            union.setdefault(str(cand["wa_key"]), cand)
        for cand in c_ranked:
            if str(cand["wa_key"]) not in union:
                union[str(cand["wa_key"])] = cand
        union_ranked = sorted(union.values(), key=lambda item: item["score"] / max(1, len(item["pairs"])), reverse=True)[:wa_beam]
        ab_keys = [str(x["wa_key"]) for x in ab_ranked]
        prior_keys = [str(x["wa_key"]) for x in prior_ranked]
        c_keys = [str(x["wa_key"]) for x in c_ranked]
        union_keys = [str(x["wa_key"]) for x in union_ranked]
        row = {
            "index": idx,
            "sample_id": record["keys"].get("sample_id"),
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "row_count": int(record["row_count"]),
            "atom_count": int(record["atom_count"]),
            "complex_flag": bool(record.get("complex_flag")),
            "target_skeleton_key": record["canonical_skeleton_key"],
            "target_wa_key": record["canonical_wa_key"],
            "unique_skeleton": len(set(skel_keys)),
            "unique_wa_ab": len(set(ab_keys)),
            "unique_wa_prior": len(set(prior_keys)),
            "unique_wa_c": len(set(c_keys)),
            "unique_wa_union": len(set(union_keys)),
            "composition_exact_any": any(bool(x.get("composition_exact")) for x in union_ranked),
        }
        for k in ks:
            row[f"skeleton_hit@{k}"] = hit_at(skel_keys, str(record["canonical_skeleton_key"]), k)
            row[f"wa_ab_hit@{k}"] = hit_at(ab_keys, str(record["canonical_wa_key"]), k)
            row[f"wa_prior_hit@{k}"] = hit_at(prior_keys, str(record["canonical_wa_key"]), k)
            row[f"wa_c_hit@{k}"] = hit_at(c_keys, str(record["canonical_wa_key"]), k)
            row[f"wa_union_hit@{k}"] = hit_at(union_keys, str(record["canonical_wa_key"]), k)
        per_sample.append(row)
        if out_candidates is not None:
            candidate_rows.append(
                {
                    "sample_id": record["keys"].get("sample_id"),
                    "skeleton_candidates": skels[:50],
                    "wa_ab_candidates": ab_ranked[:50],
                    "wa_prior_candidates": prior_ranked[:50],
                    "wa_c_candidates": c_ranked[:50],
                    "wa_union_candidates": union_ranked[:50],
                }
            )
    if out_candidates is not None:
        write_jsonl(out_candidates, candidate_rows)
    full = summarize_hits(per_sample, ks)
    rows_ge_7 = [x for x in per_sample if int(x["row_count"]) >= 7]
    atoms_ge_12 = [x for x in per_sample if int(x["atom_count"]) >= 12]
    return {
        "full": full,
        "rows_ge_7": summarize_hits(rows_ge_7, ks),
        "atoms_ge_12": summarize_hits(atoms_ge_12, ks),
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train opentry_3 Wyckoff skeleton/W-A sequence models.")
    parser.add_argument("--data-dir", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52")
    parser.add_argument("--run-dir", type=Path, default=OPENTRY_ROOT / "runs" / "e02_wyckoff_seq_smoke")
    parser.add_argument("--out-dir", type=Path, default=OPENTRY_ROOT / "reports" / "e02_wyckoff_seq_smoke")
    parser.add_argument("--max-train-records", type=int, default=8192)
    parser.add_argument("--max-val-records", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--complex-weight", type=float, default=2.0)
    parser.add_argument("--skeleton-beam", type=int, default=50)
    parser.add_argument("--wa-beam", type=int, default=50)
    parser.add_argument("--branch", type=int, default=16)
    parser.add_argument("--assignment-branch", type=int, default=4)
    parser.add_argument("--assignment-per-skeleton", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--disable-learned-ab", action="store_true")
    parser.add_argument("--disable-direct-c", action="store_true")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = ensure_under_opentry(args.run_dir)
    out_dir = ensure_under_opentry(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_records = read_jsonl(args.data_dir / "train.jsonl", args.max_train_records)
    val_records = read_jsonl(args.data_dir / "val.jsonl", args.max_val_records)
    vocab = build_vocab(train_records)
    decoder_priors = build_decoder_priors(train_records, vocab)
    write_json(run_dir / "vocab.json", vocab_to_jsonable(vocab))
    write_json(run_dir / "decoder_priors_train_only.json", decoder_priors)

    skeleton_states = make_skeleton_states(train_records, vocab, args.complex_weight)
    assignment_states = make_assignment_states(train_records, vocab, args.complex_weight)
    direct_states = make_direct_wa_states(train_records, vocab, args.complex_weight)
    model = WyckoffModels(vocab, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    summary: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "vocab": {
            "elements": len(vocab.elements),
            "sgs": len(vocab.sgs),
            "orbits": len(vocab.orbits),
            "pairs": len(vocab.pairs),
        },
        "states": {
            "skeleton": len(skeleton_states),
            "assignment": len(assignment_states),
            "direct_wa": len(direct_states),
        },
        "epochs": [],
    }
    write_json(out_dir / "state_summary.json", summary)
    best_score = -float("inf")
    best_epoch = 0
    start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        losses = train_epoch(
            model,
            optimizer,
            skeleton_states,
            assignment_states,
            direct_states,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + epoch,
        )
        eval_result: dict[str, Any] | None = None
        if args.eval_every_epoch:
            eval_result = evaluate_generators(
                model,
                val_records,
                vocab,
                device,
                skeleton_beam=args.skeleton_beam,
                wa_beam=args.wa_beam,
                branch=args.branch,
                assignment_branch=args.assignment_branch,
                assignment_per_skeleton=args.assignment_per_skeleton,
                max_steps=args.max_steps,
                decoder_priors=decoder_priors,
                use_learned_ab=not args.disable_learned_ab,
                use_direct_c=not args.disable_direct_c,
                out_candidates=None,
            )
            score = float(eval_result["full"]["wa_union@50"])
        else:
            score = -float(losses["skeleton_loss"] + losses["assignment_loss"] + losses["direct_wa_loss"])
        epoch_row = {
            "epoch": epoch,
            "losses": losses,
            "eval": None if eval_result is None else {k: v for k, v in eval_result.items() if k != "per_sample"},
            "elapsed_s": time.time() - start,
        }
        summary["epochs"].append(epoch_row)
        write_json(out_dir / "training_summary_running.json", summary)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab_to_jsonable(vocab),
                    "config": summary["config"],
                    "epoch": epoch,
                    "score": score,
                },
                run_dir / "best.pt",
            )

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    final_eval = evaluate_generators(
        model,
        val_records,
        vocab,
        device,
        skeleton_beam=args.skeleton_beam,
        wa_beam=args.wa_beam,
        branch=args.branch,
        assignment_branch=args.assignment_branch,
        assignment_per_skeleton=args.assignment_per_skeleton,
        max_steps=args.max_steps,
        decoder_priors=decoder_priors,
        use_learned_ab=not args.disable_learned_ab,
        use_direct_c=not args.disable_direct_c,
        out_candidates=out_dir / "val_candidates.jsonl",
    )
    write_jsonl(out_dir / "val_per_sample.jsonl", final_eval["per_sample"])
    summary["best_epoch"] = best_epoch
    summary["best_score_wa_union@50"] = best_score
    summary["final_eval"] = {k: v for k, v in final_eval.items() if k != "per_sample"}
    summary["elapsed_s_total"] = time.time() - start
    write_json(out_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
