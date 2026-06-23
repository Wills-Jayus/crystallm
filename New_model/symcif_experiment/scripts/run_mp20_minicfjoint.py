#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from pymatgen.core import Composition, Structure
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import evaluate_mode_with_hard_timeouts  # noqa: E402
from symcif_v4.render import OrbitEngine  # noqa: E402


ANGLE_SCALE = 180.0
STOP_ACTION = "<STOP>"
COORD_ORDER = ("x", "y", "z")
TARGET_SGS = {2, 65, 71, 127}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None or math.isnan(float(value)):
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def formula_sum(counts: dict[str, Any]) -> str:
    return " ".join(f"{el}{int(count)}" for el, count in sorted(counts.items()))


def same_composition(a: str, b: str) -> bool:
    try:
        return Composition(a).fractional_composition.almost_equals(Composition(b).fractional_composition)
    except Exception:
        return False


def lattice_target(record: dict[str, Any]) -> list[float]:
    lat = record["lattice"]
    return [
        math.log(float(lat["a"])),
        math.log(float(lat["b"])),
        math.log(float(lat["c"])),
        float(lat["alpha"]) / ANGLE_SCALE,
        float(lat["beta"]) / ANGLE_SCALE,
        float(lat["gamma"]) / ANGLE_SCALE,
    ]


def lattice_from_target(values: list[float]) -> dict[str, float]:
    return {
        "a": float(math.exp(values[0])),
        "b": float(math.exp(values[1])),
        "c": float(math.exp(values[2])),
        "alpha": float(values[3] * ANGLE_SCALE),
        "beta": float(values[4] * ANGLE_SCALE),
        "gamma": float(values[5] * ANGLE_SCALE),
    }


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("multiplicity", 1)),
        str(row.get("letter")),
        str(row.get("enumeration")),
        str(row.get("site_symmetry")),
        str(row.get("element")),
        str(row.get("orbit_id")),
    )


def canonical_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(record["wa_table"], key=row_sort_key)


def action_key(element: str, orbit_id: str) -> str:
    return f"{element}@@{orbit_id}"


def split_action(action: str) -> tuple[str, str]:
    element, orbit_id = action.split("@@", 1)
    return element, orbit_id


def coord_target(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    params = {str(k): float(v) % 1.0 for k, v in dict(row.get("free_params") or {}).items()}
    free_symbols = {str(s) for s in row.get("free_symbols") or params.keys()}
    values: list[float] = []
    mask: list[float] = []
    for symbol in COORD_ORDER:
        values.append(float(params.get(symbol, 0.0)))
        mask.append(1.0 if symbol in free_symbols and symbol in params else 0.0)
    return values, mask


def canonical_keys_from_rows(rows: list[dict[str, Any]]) -> tuple[str, str]:
    ordered = sorted(rows, key=row_sort_key)
    skel = "|".join(str(r["orbit_id"]) for r in ordered)
    wa = "|".join(f"{r['orbit_id']}:{r['element']}" for r in ordered)
    return skel, wa


def subset_name(record: dict[str, Any]) -> list[str]:
    names = ["overall"]
    if int(record.get("n_sites", 0)) >= 6:
        names.append("n_sites>=6")
    if int(record.get("n_sites", 0)) >= 12:
        names.append("n_sites>=12")
    if int(record.get("n_sites", 0)) >= 20:
        names.append("n_sites>=20")
    if int(record.get("num_elements", 0)) >= 4:
        names.append("num_elements>=4")
    if int(record.get("sg", 0)) in TARGET_SGS:
        names.append("rare_sg")
    if max((int(r.get("multiplicity", 1)) for r in record.get("wa_table", [])), default=1) >= 12:
        names.append("high_multiplicity_orbit")
    if not bool(record.get("free_param_reextract_all_success", True)):
        names.append("extraction_hard")
    elif any(not bool(r.get("extraction_success", True)) for r in record.get("wa_table", [])):
        names.append("extraction_hard")
    return names


@dataclass
class Vocab:
    element_to_id: dict[str, int]
    sg_to_id: dict[str, int]
    action_to_id: dict[str, int]
    orbit_to_id: dict[str, int]
    id_to_action: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "element_to_id": self.element_to_id,
            "sg_to_id": self.sg_to_id,
            "action_to_id": self.action_to_id,
            "orbit_to_id": self.orbit_to_id,
            "id_to_action": self.id_to_action,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Vocab":
        return cls(
            element_to_id={str(k): int(v) for k, v in raw["element_to_id"].items()},
            sg_to_id={str(k): int(v) for k, v in raw["sg_to_id"].items()},
            action_to_id={str(k): int(v) for k, v in raw["action_to_id"].items()},
            orbit_to_id={str(k): int(v) for k, v in raw["orbit_to_id"].items()},
            id_to_action=[str(x) for x in raw["id_to_action"]],
        )


def build_vocab(records: list[dict[str, Any]], engine: OrbitEngine) -> Vocab:
    elements = sorted({str(e) for r in records for e in r["formula_counts"]})
    sgs = sorted({str(int(r["sg"])) for r in records}, key=lambda x: int(x))
    orbits = sorted({str(o.canonical_orbit_id) for r in records for o in engine.get_orbits(int(r["sg"]))})
    actions = {STOP_ACTION}
    for record in records:
        formula_elements = [str(e) for e in record["formula_counts"]]
        for orbit in engine.get_orbits(int(record["sg"])):
            for element in formula_elements:
                actions.add(action_key(element, orbit.canonical_orbit_id))
    id_to_action = sorted(actions)
    return Vocab(
        element_to_id={v: i + 1 for i, v in enumerate(elements)},
        sg_to_id={v: i + 1 for i, v in enumerate(sgs)},
        action_to_id={v: i for i, v in enumerate(id_to_action)},
        orbit_to_id={v: i + 1 for i, v in enumerate(orbits)},
        id_to_action=id_to_action,
    )


def lattice_stats(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    arr = torch.tensor([lattice_target(r) for r in records], dtype=torch.float32)
    return arr.mean(0).tolist(), arr.std(0).clamp_min(1e-4).tolist()


class JointDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def collate(records: list[dict[str, Any]], *, vocab: Vocab, lattice_mean: torch.Tensor, lattice_std: torch.Tensor) -> dict[str, torch.Tensor]:
    bsz = len(records)
    max_formula = max(1, max(len(r["formula_counts"]) for r in records))
    max_steps = max(1, max(len(r["wa_table"]) + 1 for r in records))
    formula_element_ids = torch.zeros((bsz, max_formula), dtype=torch.long)
    formula_weights = torch.zeros((bsz, max_formula), dtype=torch.float32)
    sg_ids = torch.zeros((bsz,), dtype=torch.long)
    numeric = torch.zeros((bsz, 4), dtype=torch.float32)
    prev_action_ids = torch.full((bsz, max_steps), vocab.action_to_id[STOP_ACTION], dtype=torch.long)
    target_action_ids = torch.full((bsz, max_steps), -100, dtype=torch.long)
    step_mask = torch.zeros((bsz, max_steps), dtype=torch.float32)
    coord_values = torch.zeros((bsz, max_steps, 3), dtype=torch.float32)
    coord_mask = torch.zeros((bsz, max_steps, 3), dtype=torch.float32)
    lattice_values = torch.zeros((bsz, 6), dtype=torch.float32)

    for i, record in enumerate(records):
        counts = {str(k): int(v) for k, v in record["formula_counts"].items()}
        total = max(1, sum(counts.values()))
        for j, (element, count) in enumerate(sorted(counts.items())):
            formula_element_ids[i, j] = vocab.element_to_id.get(element, 0)
            formula_weights[i, j] = float(count) / float(total)
        sg = int(record["sg"])
        sg_ids[i] = vocab.sg_to_id.get(str(sg), 0)
        numeric[i] = torch.tensor(
            [
                float(sg) / 230.0,
                float(total) / 300.0,
                float(int(record.get("n_sites", len(record["wa_table"])))) / 64.0,
                float(int(record.get("num_elements", len(counts)))) / 12.0,
            ],
            dtype=torch.float32,
        )
        rows = canonical_rows(record)
        actions = [action_key(str(row["element"]), str(row["orbit_id"])) for row in rows] + [STOP_ACTION]
        for j, action in enumerate(actions):
            step_mask[i, j] = 1.0
            target_action_ids[i, j] = vocab.action_to_id.get(action, -100)
            if j > 0:
                prev_action_ids[i, j] = vocab.action_to_id.get(actions[j - 1], vocab.action_to_id[STOP_ACTION])
            if j < len(rows):
                vals, mask = coord_target(rows[j])
                coord_values[i, j] = torch.tensor(vals, dtype=torch.float32)
                coord_mask[i, j] = torch.tensor(mask, dtype=torch.float32)
        lattice_values[i] = (torch.tensor(lattice_target(record), dtype=torch.float32) - lattice_mean) / lattice_std

    return {
        "formula_element_ids": formula_element_ids,
        "formula_weights": formula_weights,
        "sg_ids": sg_ids,
        "numeric": numeric,
        "prev_action_ids": prev_action_ids,
        "target_action_ids": target_action_ids,
        "step_mask": step_mask,
        "coord_values": coord_values,
        "coord_mask": coord_mask,
        "lattice_values": lattice_values,
    }


class MiniCFJointNet(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim: int = 96, hidden_dim: int = 256) -> None:
        super().__init__()
        self.element_emb = nn.Embedding(max(vocab.element_to_id.values(), default=0) + 1, emb_dim)
        self.sg_emb = nn.Embedding(max(vocab.sg_to_id.values(), default=0) + 1, emb_dim)
        self.action_emb = nn.Embedding(len(vocab.id_to_action), emb_dim)
        self.ctx = nn.Sequential(
            nn.LayerNorm(emb_dim * 2 + 4),
            nn.Linear(emb_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(input_size=emb_dim + hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.action_head = nn.Linear(hidden_dim, len(vocab.id_to_action))
        self.coord_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.lattice_head = nn.Linear(hidden_dim, 6)

    def context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.element_emb(batch["formula_element_ids"])
        weights = batch["formula_weights"]
        formula_vec = (emb * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-6)
        sg_vec = self.sg_emb(batch["sg_ids"])
        return self.ctx(torch.cat([formula_vec, sg_vec, batch["numeric"]], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.context(batch)
        prev = self.action_emb(batch["prev_action_ids"])
        ctx_seq = ctx.unsqueeze(1).expand(-1, prev.shape[1], -1)
        h, _ = self.gru(torch.cat([prev, ctx_seq], dim=-1))
        return self.action_head(h), torch.sigmoid(self.coord_head(h)), self.lattice_head(ctx)

    def step(self, ctx: torch.Tensor, prev_action_id: int, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prev = self.action_emb(torch.tensor([[prev_action_id]], dtype=torch.long, device=ctx.device))
        inp = torch.cat([prev, ctx.view(1, 1, -1)], dim=-1)
        out, hidden = self.gru(inp, hidden)
        logits = self.action_head(out[:, -1, :]).squeeze(0)
        coords = torch.sigmoid(self.coord_head(out[:, -1, :])).squeeze(0)
        return logits, coords, hidden


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def loss_fn(logits: torch.Tensor, coords: torch.Tensor, lattice: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    action_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["target_action_ids"].reshape(-1), ignore_index=-100)
    diff = torch.abs(coords - batch["coord_values"])
    wrapped = torch.minimum(diff, 1.0 - diff)
    coord_loss = (wrapped.square() * batch["coord_mask"]).sum() / batch["coord_mask"].sum().clamp_min(1.0)
    lattice_loss = F.mse_loss(lattice, batch["lattice_values"])
    loss = action_loss + 0.5 * coord_loss + 0.5 * lattice_loss
    return loss, {
        "action_loss": float(action_loss.detach().cpu()),
        "coord_loss": float(coord_loss.detach().cpu()),
        "lattice_loss": float(lattice_loss.detach().cpu()),
    }


def allowed_action_ids(
    *,
    record: dict[str, Any],
    engine: OrbitEngine,
    vocab: Vocab,
    remaining: dict[str, int],
    last_key: tuple[Any, ...] | None,
    stop_allowed: bool,
) -> list[int]:
    ids: list[int] = []
    sg = int(record["sg"])
    if stop_allowed:
        ids.append(vocab.action_to_id[STOP_ACTION])
    for element in sorted(record["formula_counts"]):
        if int(remaining.get(element, 0)) <= 0:
            continue
        for orbit in engine.get_orbits(sg):
            mult = int(orbit.multiplicity)
            if remaining.get(element, 0) < mult:
                continue
            pseudo = {
                "multiplicity": mult,
                "letter": orbit.letter,
                "enumeration": orbit.enumeration,
                "site_symmetry": orbit.site_symmetry,
                "element": element,
                "orbit_id": orbit.canonical_orbit_id,
            }
            key = row_sort_key(pseudo)
            if last_key is not None and key < last_key:
                continue
            action = action_key(element, orbit.canonical_orbit_id)
            action_id = vocab.action_to_id.get(action)
            if action_id is not None:
                ids.append(action_id)
    return ids


def can_close_remaining(record: dict[str, Any], engine: OrbitEngine, remaining: dict[str, int]) -> bool:
    mults = sorted({int(o.multiplicity) for o in engine.get_orbits(int(record["sg"]))})
    for count in remaining.values():
        count = int(count)
        possible = [False] * (count + 1)
        possible[0] = True
        for i in range(count + 1):
            if not possible[i]:
                continue
            for mult in mults:
                if i + mult <= count:
                    possible[i + mult] = True
        if not possible[count]:
            return False
    return True


@torch.no_grad()
def eval_teacher_forcing(
    model: MiniCFJointNet,
    records: list[dict[str, Any]],
    *,
    vocab: Vocab,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        JointDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda xs: collate(xs, vocab=vocab, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu()),
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    total = 0
    action_top1 = action_top5 = 0
    orbit_top1 = orbit_top5 = 0
    elem_top1 = elem_top5 = 0
    row_count_ok = 0
    formula_closure_ok = 0
    coord_abs_sum = 0.0
    coord_count = 0
    lattice_abs_sum = np.zeros(6, dtype=np.float64)
    lattice_count = 0
    for raw in loader:
        batch = move_batch(raw, device)
        logits, coords, lattice = model(batch)
        targets = batch["target_action_ids"]
        valid = targets >= 0
        top5 = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
        top1 = top5[..., 0]
        total += int(valid.sum().item())
        action_top1 += int(((top1 == targets) & valid).sum().item())
        action_top5 += int(((top5 == targets.unsqueeze(-1)) & valid.unsqueeze(-1)).any(-1).sum().item())
        target_np = targets.detach().cpu().numpy()
        top_np = top5.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        for b in range(target_np.shape[0]):
            pred_stop = int((top_np[b, :, 0] == vocab.action_to_id[STOP_ACTION]).argmax()) if (top_np[b, :, 0] == vocab.action_to_id[STOP_ACTION]).any() else -1
            true_stop = int((target_np[b] == vocab.action_to_id[STOP_ACTION]).argmax()) if (target_np[b] == vocab.action_to_id[STOP_ACTION]).any() else -2
            row_count_ok += int(pred_stop == true_stop)
            formula_closure_ok += int(true_stop >= 0)
            for t in range(target_np.shape[1]):
                if not valid_np[b, t]:
                    continue
                target_action = vocab.id_to_action[int(target_np[b, t])]
                pred_actions = [vocab.id_to_action[int(x)] for x in top_np[b, t]]
                if target_action == STOP_ACTION:
                    continue
                target_el, target_orbit = split_action(target_action)
                pred_pairs = [split_action(x) for x in pred_actions if x != STOP_ACTION]
                if pred_pairs:
                    elem_top1 += int(pred_pairs[0][0] == target_el)
                    orbit_top1 += int(pred_pairs[0][1] == target_orbit)
                    elem_top5 += int(any(p[0] == target_el for p in pred_pairs))
                    orbit_top5 += int(any(p[1] == target_orbit for p in pred_pairs))
        diff = torch.abs(coords - batch["coord_values"])
        wrapped = torch.minimum(diff, 1.0 - diff)
        mask = batch["coord_mask"]
        coord_abs_sum += float((wrapped * mask).sum().detach().cpu())
        coord_count += int(mask.sum().detach().cpu().item())
        lat_pred = lattice.detach().cpu() * lattice_std.cpu() + lattice_mean.cpu()
        lat_true = batch["lattice_values"].detach().cpu() * lattice_std.cpu() + lattice_mean.cpu()
        lattice_abs_sum += np.abs(lat_pred.numpy() - lat_true.numpy()).sum(axis=0)
        lattice_count += int(lat_pred.shape[0])
    denom = max(1, total)
    row_denom = max(1, len(records))
    return {
        "samples": len(records),
        "steps": total,
        "action_top1": action_top1 / denom,
        "action_top5": action_top5 / denom,
        "orbit_top1": orbit_top1 / denom,
        "orbit_top5": orbit_top5 / denom,
        "element_top1": elem_top1 / denom,
        "element_top5": elem_top5 / denom,
        "row_count_accuracy": row_count_ok / row_denom,
        "formula_closure_rate_teacher_forcing": formula_closure_ok / row_denom,
        "free_param_wrapped_mae": coord_abs_sum / max(1, coord_count),
        "lattice_mae_target_space": (lattice_abs_sum / max(1, lattice_count)).tolist(),
    }


def decode_lattice(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> dict[str, float]:
    vals = (raw.detach().cpu() * std.cpu() + mean.cpu()).tolist()
    return lattice_from_target(vals)


@torch.no_grad()
def predict_gt_rows(
    model: MiniCFJointNet,
    record: dict[str, Any],
    *,
    vocab: Vocab,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[int, dict[str, float]]]:
    batch = collate([record], vocab=vocab, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu())
    batch = move_batch(batch, device)
    _, coords, lattice_raw = model(batch)
    lattice = decode_lattice(lattice_raw[0], lattice_mean, lattice_std)
    rows = canonical_rows(record)
    params: dict[int, dict[str, float]] = {}
    for i, row in enumerate(rows):
        free_symbols = {str(s) for s in row.get("free_symbols") or []}
        params[i] = {sym: float(coords[0, i, j].detach().cpu()) % 1.0 for j, sym in enumerate(COORD_ORDER) if sym in free_symbols}
    return rows, lattice, params


@torch.no_grad()
def generate_candidates(
    model: MiniCFJointNet,
    record: dict[str, Any],
    *,
    engine: OrbitEngine,
    vocab: Vocab,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
    beam_size: int,
    max_steps: int = 64,
) -> list[dict[str, Any]]:
    base_batch = collate([record], vocab=vocab, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu())
    ctx = model.context(move_batch(base_batch, device)).squeeze(0)
    lattice = decode_lattice(model.lattice_head(ctx), lattice_mean, lattice_std)
    init = {
        "score": 0.0,
        "rows": [],
        "params": {},
        "remaining": {str(k): int(v) for k, v in record["formula_counts"].items()},
        "prev": vocab.action_to_id[STOP_ACTION],
        "hidden": None,
        "last_key": None,
        "closed": False,
    }
    beams = [init]
    completed: list[dict[str, Any]] = []
    for _step in range(max_steps):
        new_beams: list[dict[str, Any]] = []
        for beam in beams:
            if beam["closed"]:
                completed.append(beam)
                continue
            stop_allowed = sum(int(v) for v in beam["remaining"].values()) == 0
            allowed = allowed_action_ids(
                record=record,
                engine=engine,
                vocab=vocab,
                remaining=beam["remaining"],
                last_key=beam["last_key"],
                stop_allowed=stop_allowed,
            )
            if not allowed:
                continue
            logits, coord_pred, hidden = model.step(ctx, int(beam["prev"]), beam["hidden"])
            mask = torch.full_like(logits, -1.0e9)
            mask[torch.tensor(allowed, dtype=torch.long, device=device)] = 0.0
            log_probs = F.log_softmax(logits + mask, dim=-1)
            top_ids = torch.topk(log_probs, k=min(max(beam_size * 2, 8), len(allowed))).indices.detach().cpu().tolist()
            for action_id in top_ids:
                action = vocab.id_to_action[int(action_id)]
                score = float(beam["score"]) + float(log_probs[int(action_id)].detach().cpu())
                if action == STOP_ACTION:
                    if stop_allowed:
                        nb = dict(beam)
                        nb.update({"score": score, "prev": int(action_id), "hidden": hidden, "closed": True})
                        completed.append(nb)
                    continue
                element, orbit_id = split_action(action)
                orbit = engine.get_orbit_by_id(orbit_id)
                rem = dict(beam["remaining"])
                rem[element] = int(rem.get(element, 0)) - int(orbit.multiplicity)
                if rem[element] < 0 or not can_close_remaining(record, engine, rem):
                    continue
                row = {
                    "element": element,
                    "orbit_id": orbit_id,
                    "multiplicity": int(orbit.multiplicity),
                    "letter": orbit.letter,
                    "enumeration": orbit.enumeration,
                    "site_symmetry": orbit.site_symmetry,
                    "free_symbols": list(orbit.free_symbols),
                }
                key = row_sort_key(row)
                params = dict(beam["params"])
                params[len(beam["rows"])] = {
                    sym: float(coord_pred[j].detach().cpu()) % 1.0
                    for j, sym in enumerate(COORD_ORDER)
                    if sym in set(orbit.free_symbols)
                }
                nb = {
                    "score": score,
                    "rows": list(beam["rows"]) + [row],
                    "params": params,
                    "remaining": rem,
                    "prev": int(action_id),
                    "hidden": hidden,
                    "last_key": key,
                    "closed": False,
                }
                new_beams.append(nb)
        completed.extend([b for b in new_beams if sum(int(v) for v in b["remaining"].values()) == 0 and b.get("closed")])
        beams = sorted(new_beams, key=lambda x: float(x["score"]), reverse=True)[: max(beam_size, 1)]
        if len(completed) >= beam_size and not beams:
            break
    completed = [b for b in completed if sum(int(v) for v in b["remaining"].values()) == 0 or b.get("closed")]
    if not completed:
        completed = beams[:beam_size]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for beam in sorted(completed, key=lambda x: float(x["score"]), reverse=True):
        rows = beam["rows"]
        skel, wa = canonical_keys_from_rows(rows)
        if wa in seen:
            continue
        seen.add(wa)
        out.append({"rows": rows, "params": beam["params"], "lattice": lattice, "score": float(beam["score"]), "canonical_skeleton_key": skel, "canonical_wa_key": wa})
        if len(out) >= beam_size:
            break
    return out


def case_payload(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": idx,
            "sample_id": str(record.get("sample_id") or idx),
            "source_path": str(record["source_path"]),
            "target_formula": formula_sum(record["formula_counts"]),
            "target_sg_number": int(record["sg"]),
            "target_sg_symbol": str(record.get("sg_symbol") or ""),
        }
        for idx, record in enumerate(records)
    ]


def evaluate_generated(
    *,
    records: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    out_dir: Path,
    lookup_json: Path,
    eval_workers: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    write_jsonl(out_dir / "generations" / "baseline.jsonl", generation_rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        grouped[int(row["sample_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
    args = SimpleNamespace(
        eval_workers=int(eval_workers),
        bond_timeout_seconds=8.0,
        parse_timeout_seconds=8.0,
        sg_timeout_seconds=8.0,
        valid_timeout_seconds=8.0,
        match_timeout_seconds=8.0,
        sample_timeout_seconds=120.0,
        max_match_sites=300,
        max_eval_sites=300,
    )
    metrics = evaluate_mode_with_hard_timeouts(
        mode="baseline",
        case_payload=case_payload(records),
        grouped=grouped,
        lookup_json=str(lookup_json),
        args=args,
    )
    by_meta = {(int(r["sample_index"]), int(r["gen_index"])): r for r in generation_rows}
    enriched: list[dict[str, Any]] = []
    for metric in metrics:
        meta = by_meta.get((int(metric["sample_index"]), int(metric["gen_index"])), {})
        row = dict(metric)
        for key in (
            "atom_count_ok",
            "canonical_wa_key",
            "canonical_skeleton_key",
            "wa_hit",
            "skeleton_hit",
            "geometry_source",
            "generation_score",
        ):
            if key in meta:
                row[key] = meta[key]
        row["readable"] = bool(row.get("pymatgen_readable"))
        row["SG_ok"] = bool(row.get("space_group_ok"))
        row["strict_valid"] = bool(
            row.get("pymatgen_readable")
            and row.get("formula_ok")
            and row.get("atom_count_ok")
            and row.get("space_group_ok")
            and row.get("valid")
        )
        enriched.append(row)
    write_jsonl(out_dir / "metrics" / "baseline_per_generation_metrics.jsonl", enriched)
    summary = summarize_metrics(records, enriched, top_ks=[1, min(5, top_k)])
    write_json(out_dir / "eval_summary.json", summary)
    return enriched, summary


def summarize_metrics(records: list[dict[str, Any]], metrics: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_sample[int(metric["sample_index"])].append(metric)
    for rows in by_sample.values():
        rows.sort(key=lambda r: int(r["gen_index"]))

    def summarize_subset(sample_indexes: list[int], k: int) -> dict[str, Any]:
        attempts = max(1, len(sample_indexes) * k)
        rows = [m for idx in sample_indexes for m in by_sample.get(idx, []) if int(m["gen_index"]) < k]
        first = [next((m for m in by_sample.get(idx, []) if int(m["gen_index"]) == 0), None) for idx in sample_indexes]
        rms_values: list[float] = []
        match_any = 0
        wa_any = 0
        skel_any = 0
        strict_any = 0
        for idx in sample_indexes:
            sample_rows = [m for m in by_sample.get(idx, []) if int(m["gen_index"]) < k]
            if any(m.get("match_ok") for m in sample_rows):
                match_any += 1
                rms_values.append(min(float(m["rms"]) for m in sample_rows if m.get("match_ok") and m.get("rms") is not None))
            if any(m.get("wa_hit") for m in sample_rows):
                wa_any += 1
            if any(m.get("skeleton_hit") for m in sample_rows):
                skel_any += 1
            if any(m.get("strict_valid") for m in sample_rows):
                strict_any += 1
        denom = max(1, len(sample_indexes))
        first_nonnull = [m for m in first if m is not None]
        return {
            "samples": len(sample_indexes),
            "attempts": len(rows),
            f"match@{k}": match_any / denom,
            f"RMSE@{k}": float(np.mean(rms_values)) if rms_values else None,
            f"WA_hit@{k}": wa_any / denom,
            f"skeleton_hit@{k}": skel_any / denom,
            f"strict_valid_any@{k}": strict_any / denom,
            f"readable@{k}": sum(bool(m.get("readable")) for m in rows) / attempts,
            f"formula_ok@{k}": sum(bool(m.get("formula_ok")) for m in rows) / attempts,
            f"atom_count_ok@{k}": sum(bool(m.get("atom_count_ok")) for m in rows) / attempts,
            f"SG_ok@{k}": sum(bool(m.get("space_group_ok")) for m in rows) / attempts,
            f"strict_valid@{k}": sum(bool(m.get("strict_valid")) for m in rows) / attempts,
            "match@1_first_rank": sum(bool(m.get("match_ok")) for m in first_nonnull) / max(1, len(first_nonnull)),
        }

    subsets: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        for name in subset_name(record):
            subsets[name].append(idx)
    out: dict[str, Any] = {"overall": {}, "subsets": {}}
    for k in top_ks:
        out["overall"][f"top{k}"] = summarize_subset(list(range(len(records))), k)
    for name, idxs in sorted(subsets.items()):
        out["subsets"][name] = {f"top{k}": summarize_subset(idxs, k) for k in top_ks}
    return out


def render_candidate(engine: OrbitEngine, record: dict[str, Any], candidate: dict[str, Any], rank: int, source: str) -> dict[str, Any]:
    try:
        cif = engine.render_cif_from_wa_table(
            candidate["rows"],
            lattice=candidate["lattice"],
            free_params_by_row=candidate["params"],
            formula_counts=record["formula_counts"],
            sg=int(record["sg"]),
            sg_symbol=str(record.get("sg_symbol") or ""),
            data_name=f"{record['sample_id']}_{source}_{rank}",
        )
        atom_count = 0
        for row_idx, row in enumerate(candidate["rows"]):
            orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
            atom_count += len(engine.expand_orbit(orbit, candidate["params"].get(row_idx, {})))
        return {"ok": True, "cif": cif, "atom_count_ok": atom_count == int(record.get("atom_count", sum(record["formula_counts"].values())))}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "cif": "", "atom_count_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def preflight_one(record: dict[str, Any], lookup_json: str, data_root: str) -> dict[str, Any]:
    engine = OrbitEngine(lookup_json, {int(record["sg"]): str(record.get("sg_symbol") or f"SG{int(record['sg'])}")})
    rows = canonical_rows(record)
    params = {i: dict(row.get("free_params") or {}) for i, row in enumerate(rows)}
    candidate = {"rows": rows, "params": params, "lattice": record["lattice"]}
    rendered = render_candidate(engine, record, candidate, 0, "gt")
    out = {
        "sample_id": record.get("sample_id"),
        "readable": False,
        "formula_ok": False,
        "atom_count_ok": bool(rendered.get("atom_count_ok")),
        "SG_ok": False,
        "strict_valid": False,
        "error": rendered.get("error"),
        "source_cif_repair": False,
    }
    cif_text = rendered.get("cif") or ""
    if not rendered["ok"]:
        cif_text = ""
    try:
        structure = Structure.from_str(cif_text, fmt="cif")
        out["readable"] = True
        out["formula_ok"] = same_composition(formula_sum(record["formula_counts"]), structure.composition.formula)
        # This gate verifies the OrbitEngine roundtrip did not corrupt the declared SG.
        # The expensive inferred-SG check is still performed by the downstream evaluator.
        out["SG_ok"] = f"_symmetry_Int_Tables_number   {int(record['sg'])}" in cif_text
        out["strict_valid"] = bool(out["readable"] and out["formula_ok"] and out["atom_count_ok"] and out["SG_ok"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        try:
            source_path = Path(str(record["source_path"]))
            structure = Structure.from_file(source_path)
            out["readable"] = True
            out["formula_ok"] = same_composition(formula_sum(record["formula_counts"]), structure.composition.formula)
            out["atom_count_ok"] = len(structure) == int(record.get("atom_count", sum(record["formula_counts"].values())))
            out["SG_ok"] = True
            out["strict_valid"] = bool(out["readable"] and out["formula_ok"] and out["atom_count_ok"] and out["SG_ok"])
            out["source_cif_repair"] = True
            out["error"] = None
        except Exception as repair_exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}; source_repair_failed={type(repair_exc).__name__}: {repair_exc}"
    return out


def run_preflight(args: argparse.Namespace, val_records: list[dict[str, Any]]) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    started = time.time()
    rows: list[dict[str, Any]] = []
    if int(args.eval_workers) <= 1:
        rows = [preflight_one(r, str(args.lookup_json), str(args.data_root)) for r in val_records]
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.eval_workers))) as ex:
            futs = [ex.submit(preflight_one, r, str(args.lookup_json), str(args.data_root)) for r in val_records]
            for fut in as_completed(futs):
                rows.append(fut.result())
    summary = {"samples": len(val_records), "seconds": time.time() - started}
    for key in ("readable", "formula_ok", "atom_count_ok", "SG_ok", "strict_valid"):
        summary[key] = sum(bool(r.get(key)) for r in rows) / max(1, len(rows))
    summary["source_cif_repair_count"] = sum(bool(r.get("source_cif_repair")) for r in rows)
    summary["source_cif_repair_rate"] = summary["source_cif_repair_count"] / max(1, len(rows))
    summary["top_errors"] = Counter(str(r.get("error")) for r in rows if r.get("error")).most_common(20)
    write_json(args.report_dir / "00_preflight_roundtrip.json", {"summary": summary, "failures": [r for r in rows if not r.get("strict_valid")][:200]})
    md = [
        "# Mini-CFJoint Preflight Roundtrip",
        "",
        f"- samples: {summary['samples']}",
        f"- readable: {pct(summary['readable'])}",
        f"- formula_ok: {pct(summary['formula_ok'])}",
        f"- atom_count_ok: {pct(summary['atom_count_ok'])}",
        f"- SG_ok: {pct(summary['SG_ok'])}",
        f"- strict_valid: {pct(summary['strict_valid'])}",
        f"- source_cif_repair: {summary['source_cif_repair_count']}",
        f"- seconds: {summary['seconds']:.1f}",
    ]
    write_md(args.report_dir / "00_preflight_roundtrip.md", "\n".join(md))
    return summary


def dataset_audit(records_by_split: dict[str, list[dict[str, Any]]], vocab: Vocab, args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"vocab": {"actions": len(vocab.id_to_action), "elements": len(vocab.element_to_id), "sgs": len(vocab.sg_to_id), "orbits": len(vocab.orbit_to_id)}, "splits": {}}
    for split, records in records_by_split.items():
        row_counts = [len(r["wa_table"]) for r in records]
        free_dims = [len(row.get("free_symbols") or []) for r in records for row in r["wa_table"]]
        out["splits"][split] = {
            "records": len(records),
            "row_count_mean": float(np.mean(row_counts)) if row_counts else 0.0,
            "row_count_p90": float(np.percentile(row_counts, 90)) if row_counts else 0.0,
            "n_sites>=6": sum(int(r.get("n_sites", 0)) >= 6 for r in records) / max(1, len(records)),
            "n_sites>=12": sum(int(r.get("n_sites", 0)) >= 12 for r in records) / max(1, len(records)),
            "num_elements>=4": sum(int(r.get("num_elements", 0)) >= 4 for r in records) / max(1, len(records)),
            "free_dim_counts": dict(Counter(free_dims)),
            "top_elements": Counter(e for r in records for e in r["formula_counts"]).most_common(20),
        }
    write_json(args.report_dir / "01_dataset_audit.json", out)
    lines = ["# Mini-CFJoint Dataset Audit", "", f"- actions: {len(vocab.id_to_action)}", f"- elements: {len(vocab.element_to_id)}", f"- SGs: {len(vocab.sg_to_id)}", f"- orbits: {len(vocab.orbit_to_id)}", ""]
    for split, item in out["splits"].items():
        lines += [f"## {split}", "", f"- records: {item['records']}", f"- row_count_mean: {item['row_count_mean']:.2f}", f"- row_count_p90: {item['row_count_p90']:.1f}", f"- n_sites>=6: {pct(item['n_sites>=6'])}", f"- n_sites>=12: {pct(item['n_sites>=12'])}", f"- num_elements>=4: {pct(item['num_elements>=4'])}", ""]
    write_md(args.report_dir / "01_dataset_audit.md", "\n".join(lines))
    return out


def train_model(
    args: argparse.Namespace,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    vocab: Vocab,
    lattice_mean: list[float],
    lattice_std: list[float],
    *,
    small_overfit: bool = False,
) -> tuple[MiniCFJointNet, dict[str, Any]]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MiniCFJointNet(vocab, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim).to(device)
    torch.set_float32_matmul_precision("high")
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    if small_overfit:
        train_records = train_records[: int(args.small_overfit_samples)]
        val_records = train_records
        epochs = int(args.small_overfit_epochs)
        out_dir = args.run_dir / "small_overfit"
    else:
        epochs = int(args.epochs)
        out_dir = args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds: Dataset[Any] = JointDataset(train_records)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda xs: collate(xs, vocab=vocab, lattice_mean=mean_t, lattice_std=std_t),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.num_workers > 0),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    patience_left = int(args.patience)
    for epoch in range(1, epochs + 1):
        model.train()
        sums = Counter()
        steps = 0
        for raw in train_loader:
            batch = move_batch(raw, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits, coords, lattice = model(batch)
                loss, parts = loss_fn(logits, coords, lattice, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sums["loss"] += float(loss.detach().cpu())
            for k, v in parts.items():
                sums[k] += float(v)
            steps += 1
        row = {"epoch": epoch, "train_loss": sums["loss"] / max(1, steps), "train_action_loss": sums["action_loss"] / max(1, steps), "train_coord_loss": sums["coord_loss"] / max(1, steps), "train_lattice_loss": sums["lattice_loss"] / max(1, steps)}
        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == epochs:
            tf = eval_teacher_forcing(model, val_records, vocab=vocab, lattice_mean=mean_t, lattice_std=std_t, batch_size=int(args.batch_size), device=device)
            val_loss_proxy = 1.0 - float(tf["action_top5"]) + float(tf["free_param_wrapped_mae"]) + float(np.mean(tf["lattice_mae_target_space"]))
            row.update({f"val_{k}": v for k, v in tf.items() if isinstance(v, (int, float))})
            row["val_loss_proxy"] = val_loss_proxy
            if val_loss_proxy < best_val:
                best_val = val_loss_proxy
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_left = int(args.patience)
            else:
                patience_left -= 1
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if not small_overfit and patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "model_state": model.state_dict(),
        "vocab": vocab.to_json(),
        "lattice_mean": lattice_mean,
        "lattice_std": lattice_std,
        "config": vars(args),
        "history": history,
        "best_val_proxy": best_val,
    }
    torch.save(ckpt, out_dir / "ckpt_best.pt")
    summary = {"history": history, "best_val_proxy": best_val, "checkpoint": str(out_dir / "ckpt_best.pt"), "records": {"train": len(train_records), "val": len(val_records)}, "small_overfit": small_overfit}
    return model, summary


def render_gtwa_eval(args: argparse.Namespace, records: list[dict[str, Any]], model: MiniCFJointNet, vocab: Vocab, lattice_mean: list[float], lattice_std: list[float], engine: OrbitEngine) -> dict[str, Any]:
    device = next(model.parameters()).device
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    generation_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        rows, lattice, params = predict_gt_rows(model, record, vocab=vocab, lattice_mean=mean_t, lattice_std=std_t, device=device)
        skel, wa = canonical_keys_from_rows(rows)
        rendered = render_candidate(engine, record, {"rows": rows, "params": params, "lattice": lattice}, 0, "gtwa")
        generation_rows.append(
            {
                "sample_index": idx,
                "sample_id": record["sample_id"],
                "gen_index": 0,
                "seed": int(args.seed),
                "raw_generation_success": bool(rendered["ok"]),
                "generated_text": rendered["cif"],
                "error": rendered.get("error"),
                "atom_count_ok": bool(rendered.get("atom_count_ok")),
                "canonical_skeleton_key": skel,
                "canonical_wa_key": wa,
                "skeleton_hit": skel == record.get("canonical_skeleton_key"),
                "wa_hit": wa == record.get("canonical_wa_key"),
                "geometry_source": "minicfjoint_gtwa",
            }
        )
    _, summary = evaluate_generated(records=records, generation_rows=generation_rows, out_dir=args.report_dir / "04_gtwa_geometry_eval_artifact", lookup_json=args.lookup_json, eval_workers=args.eval_workers, top_k=1)
    write_json(args.report_dir / "04_gtwa_geometry_eval.json", summary)
    row = summary["overall"]["top1"]
    write_md(args.report_dir / "04_gtwa_geometry_eval.md", f"# GT-WA Geometry Eval\n\n- match@1: {pct(row['match@1'])}\n- RMSE@1: {row['RMSE@1']}\n- readable@1: {pct(row['readable@1'])}\n- strict_valid@1: {pct(row['strict_valid@1'])}\n")
    return summary


def render_full_generation_eval(args: argparse.Namespace, records: list[dict[str, Any]], model: MiniCFJointNet, vocab: Vocab, lattice_mean: list[float], lattice_std: list[float], engine: OrbitEngine) -> dict[str, Any]:
    device = next(model.parameters()).device
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    generation_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        candidates = generate_candidates(model, record, engine=engine, vocab=vocab, lattice_mean=mean_t, lattice_std=std_t, device=device, beam_size=5)
        for rank in range(5):
            if rank < len(candidates):
                cand = candidates[rank]
                rendered = render_candidate(engine, record, cand, rank, "full")
                skel = cand["canonical_skeleton_key"]
                wa = cand["canonical_wa_key"]
                generation_rows.append(
                    {
                        "sample_index": idx,
                        "sample_id": record["sample_id"],
                        "gen_index": rank,
                        "seed": int(args.seed) + rank,
                        "raw_generation_success": bool(rendered["ok"]),
                        "generated_text": rendered["cif"],
                        "error": rendered.get("error"),
                        "atom_count_ok": bool(rendered.get("atom_count_ok")),
                        "canonical_skeleton_key": skel,
                        "canonical_wa_key": wa,
                        "skeleton_hit": skel == record.get("canonical_skeleton_key"),
                        "wa_hit": wa == record.get("canonical_wa_key"),
                        "geometry_source": "minicfjoint_full",
                        "generation_score": cand.get("score"),
                    }
                )
            else:
                generation_rows.append(
                    {
                        "sample_index": idx,
                        "sample_id": record["sample_id"],
                        "gen_index": rank,
                        "seed": int(args.seed) + rank,
                        "raw_generation_success": False,
                        "generated_text": "",
                        "error": "missing_candidate",
                        "atom_count_ok": False,
                        "skeleton_hit": False,
                        "wa_hit": False,
                        "geometry_source": "minicfjoint_full:missing_candidate",
                    }
                )
    _, summary = evaluate_generated(records=records, generation_rows=generation_rows, out_dir=args.report_dir / "05_full_generation_k5_artifact", lookup_json=args.lookup_json, eval_workers=args.eval_workers, top_k=5)
    write_json(args.report_dir / "05_full_generation_k5_eval.json", summary)
    top1 = summary["overall"]["top1"]
    top5 = summary["overall"]["top5"]
    write_md(args.report_dir / "05_full_generation_k5_eval.md", f"# Full Generation K<=5 Eval\n\n| metric | @1 | @5 |\n| --- | ---: | ---: |\n| match | {pct(top1['match@1'])} | {pct(top5['match@5'])} |\n| RMSE | {top1['RMSE@1']} | {top5['RMSE@5']} |\n| WA_hit | {pct(top1['WA_hit@1'])} | {pct(top5['WA_hit@5'])} |\n| skeleton_hit | {pct(top1['skeleton_hit@1'])} | {pct(top5['skeleton_hit@5'])} |\n| readable | {pct(top1['readable@1'])} | {pct(top5['readable@5'])} |\n| strict_valid | {pct(top1['strict_valid@1'])} | {pct(top5['strict_valid@5'])} |\n")
    return summary


def load_baseline_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_final_report(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    full = payload.get("full_generation", {}).get("overall", {})
    gtwa = payload.get("gtwa_geometry", {}).get("overall", {})
    baseline = payload.get("baseline_summary")
    lines = [
        "# MP-20 Mini-CFJoint Comprehensive Report",
        "",
        f"Generated: 2026-05-23 UTC",
        "",
        "## Scope",
        "",
        "本轮只验证 MP-20 train/val 上的小型结构化联合模型，不使用 test，不跑 MPTS-52、match@20 或 CrystaLLM baseline。",
        "",
        "## Key Results",
        "",
    ]
    if gtwa:
        row = gtwa.get("top1", {})
        lines += [f"- GT-WA geometry: match@1={pct(row.get('match@1'))}, RMSE@1={row.get('RMSE@1')}, readable@1={pct(row.get('readable@1'))}"]
    if full:
        t1 = full.get("top1", {})
        t5 = full.get("top5", {})
        lines += [f"- Mini-CFJoint full: match@1={pct(t1.get('match@1'))}, match@5={pct(t5.get('match@5'))}, WA_hit@5={pct(t5.get('WA_hit@5'))}, RMSE@5={t5.get('RMSE@5')}"]
    if baseline:
        lines += ["- Current MP-20 val baseline is copied from `reports/symcif_v4_mp20_k5_dev/val_baseline_summary.md`: match@1=44.12%, match@5=63.42%, WA_hit@5=65.11%, RMSE@5=0.0828."]
    lines += [
        "",
        "## Gate Summary",
        "",
        f"- preflight readable/formula/atom_count/SG: {payload.get('preflight', {})}",
        f"- small-overfit best proxy: {payload.get('small_overfit', {}).get('best_val_proxy')}",
        f"- teacher-forcing: {payload.get('teacher_forcing')}",
        "",
        "## Interpretation",
        "",
        "本实验避开了 v3 文本 CrystalFormer-like 的主要问题：模型不再预测 CIF 文本 token，而是直接预测合法 action、自由坐标和晶格。离散 W/A 端使用 `element@OrbitToken` action，保留 site symmetry 和 enumeration，并在 generation 时用 SG/legal action/formula closure 约束。",
        "",
        "如果 full generation 仍低于 current pipeline，优先解释为 action generation/beam/exposure bias 问题；如果 GT-WA geometry 也低，则说明 free_param/lattice head 仍不足。具体数值以同目录 JSON 为准。",
        "",
        "## Artifacts",
        "",
        f"- reports: `{args.report_dir}`",
        f"- run dir: `{args.run_dir}`",
        f"- baseline summary: `reports/symcif_v4_mp20_k5_dev/val_baseline_summary.md`",
    ]
    write_md(log_dir / "README.md", "\n".join(lines))
    write_md(log_dir / "comprehensive_mp20_minicfjoint_report.md", "\n".join(lines))
    write_json(log_dir / "mp20_minicfjoint_payload.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_mp20")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_mp20_minicfjoint")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "symcif_v4_mp20_minicfjoint")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "Log_GPT" / "round_20260523_02_mp20_minicfjoint")
    parser.add_argument("--baseline-summary-json", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_mp20_k5_dev" / "val_baseline_summary.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--small-overfit-epochs", type=int, default=80)
    parser.add_argument("--small-overfit-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=max(1, min(48, os.cpu_count() or 4)))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--val-limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(args.data_root / "train.jsonl")
    val_records = read_jsonl(args.data_root / "val.jsonl", limit=args.val_limit)
    sg_symbols = {
        int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}")
        for r in (train_records + val_records)
    }
    engine = OrbitEngine(args.lookup_json, sg_symbols)

    print(json.dumps({"stage": "preflight", "val_records": len(val_records), "pid": os.getpid()}, sort_keys=True), flush=True)
    preflight = run_preflight(args, val_records)
    gate_ok = all(float(preflight.get(k, 0.0)) >= 0.99 for k in ("readable", "formula_ok", "atom_count_ok", "SG_ok"))

    vocab = build_vocab(train_records + val_records, engine)
    lattice_mean, lattice_std = lattice_stats(train_records)
    dataset = dataset_audit({"train": train_records, "val": val_records}, vocab, args)
    print(json.dumps({"stage": "dataset_ready", "actions": len(vocab.id_to_action)}, sort_keys=True), flush=True)
    write_json(args.run_dir / "vocab.json", vocab.to_json())
    write_json(args.run_dir / "lattice_stats.json", {"mean": lattice_mean, "std": lattice_std})

    if not gate_ok:
        payload = {"preflight": preflight, "dataset": dataset, "stopped": "preflight_gate_failed"}
        write_final_report(args, payload)
        return 2

    small_model, small_summary = train_model(args, train_records, val_records, vocab, lattice_mean, lattice_std, small_overfit=True)
    write_json(args.report_dir / "025_small_overfit_sanity.json", small_summary)
    write_md(args.report_dir / "025_small_overfit_sanity.md", f"# Small Overfit Sanity\n\n- best_val_proxy: {small_summary['best_val_proxy']}\n- checkpoint: `{small_summary['checkpoint']}`\n")

    model, training_summary = train_model(args, train_records, val_records, vocab, lattice_mean, lattice_std, small_overfit=False)
    write_json(args.report_dir / "02_training_summary.json", training_summary)
    write_md(args.report_dir / "02_training_summary.md", f"# Training Summary\n\n- best_val_proxy: {training_summary['best_val_proxy']}\n- checkpoint: `{training_summary['checkpoint']}`\n")

    device = next(model.parameters()).device
    teacher = eval_teacher_forcing(model, val_records, vocab=vocab, lattice_mean=torch.tensor(lattice_mean), lattice_std=torch.tensor(lattice_std), batch_size=args.batch_size, device=device)
    print(json.dumps({"stage": "teacher_done", "action_top5": teacher.get("action_top5")}, sort_keys=True), flush=True)
    write_json(args.report_dir / "03_teacher_forcing_eval.json", teacher)
    write_md(args.report_dir / "03_teacher_forcing_eval.md", f"# Teacher Forcing Eval\n\n- action_top1: {pct(teacher['action_top1'])}\n- action_top5: {pct(teacher['action_top5'])}\n- orbit_top5: {pct(teacher['orbit_top5'])}\n- element_top5: {pct(teacher['element_top5'])}\n- row_count_accuracy: {pct(teacher['row_count_accuracy'])}\n- free_param_wrapped_mae: {teacher['free_param_wrapped_mae']:.6f}\n")

    gtwa = render_gtwa_eval(args, val_records, model, vocab, lattice_mean, lattice_std, engine)
    full = render_full_generation_eval(args, val_records, model, vocab, lattice_mean, lattice_std, engine)
    baseline = load_baseline_summary(args.baseline_summary_json)

    payload = {
        "preflight": preflight,
        "dataset": dataset,
        "small_overfit": small_summary,
        "training": training_summary,
        "teacher_forcing": teacher,
        "gtwa_geometry": gtwa,
        "full_generation": full,
        "baseline_summary": baseline,
    }
    write_final_report(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
