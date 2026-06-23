#!/usr/bin/env python3
"""Fixed-order joint smoke with heteroscedastic free-param/lattice sampling.

This script keeps the E8009-E8016 symbolic decoder and no-ranking generation
contract, but changes the geometry objective from point-estimate CE/L1 to a
mean+scale distribution trained by NLL. Candidate order remains gen_index only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_eval_fixed_order_joint_smoke import (  # noqa: E402
    COORD_ORDER,
    NONE_ELEMENT,
    ROOT,
    SEEDS,
    STOP_ORBIT,
    SYMCIF_PROJECT,
    JointV2Dataset,
    MiniCFJointV2Net,
    OrbitEngine,
    append_text,
    build_v2_vocab,
    can_close_remaining_ordered,
    candidate_key_for,
    canonical_keys_from_rows,
    canonical_rows,
    case_payload_from_clean_records,
    choose_from_logits,
    collate_v2,
    decode_lattice,
    evaluate_mode_with_hard_timeouts,
    formula_tensors,
    lattice_stats,
    load_clean_records,
    move_batch,
    pct,
    render_candidate,
    safe_rel,
    select_train_curriculum,
    summarize_fixed_order,
    under_root,
    write_json,
    write_jsonl,
    write_md,
)
from run_mp20_minicfjoint_v2 import Vocab  # noqa: E402


MODE = "baseline_opentry5_fixed_order_geom_sampler_smoke"
ANGLE_SCALE = 180.0
_AFFINE_OP_CACHE: dict[str, tuple[list[list[float]], list[float]]] = {}


def parse_linear_expr(expr: str) -> tuple[list[float], float]:
    """Parse Wyckoff expressions such as x-y+1/3 into coeffs plus offset."""
    text = str(expr).replace(" ", "")
    if not text:
        raise ValueError("empty coordinate expression")
    if text[0] not in "+-":
        text = "+" + text
    coeff = [0.0, 0.0, 0.0]
    offset = 0.0
    pos = 0
    for match in re.finditer(r"([+-])([^+-]+)", text):
        if match.start() != pos:
            raise ValueError(f"cannot parse coordinate expression {expr!r}")
        pos = match.end()
        sign = -1.0 if match.group(1) == "-" else 1.0
        term = match.group(2).replace("*", "")
        vars_in_term = [v for v in COORD_ORDER if v in term]
        if vars_in_term:
            if len(vars_in_term) != 1:
                raise ValueError(f"unsupported multi-variable term {term!r} in {expr!r}")
            var = vars_in_term[0]
            coef_text = term.replace(var, "")
            coef = float(Fraction(coef_text)) if coef_text else 1.0
            coeff[COORD_ORDER.index(var)] += sign * coef
        else:
            offset += sign * float(Fraction(term))
    if pos != len(text):
        raise ValueError(f"cannot parse coordinate expression tail {expr!r}")
    return coeff, offset


def affine_for_operation(operation: str) -> tuple[list[list[float]], list[float]]:
    key = str(operation)
    cached = _AFFINE_OP_CACHE.get(key)
    if cached is not None:
        return cached
    parts = [p.strip() for p in key.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 3-part Wyckoff operation, got {operation!r}")
    matrix: list[list[float]] = []
    offset: list[float] = []
    for part in parts:
        coeff, bias = parse_linear_expr(part)
        matrix.append(coeff)
        offset.append(float(bias))
    _AFFINE_OP_CACHE[key] = (matrix, offset)
    return matrix, offset


def positive_scale(raw: torch.Tensor, floor: float, ceiling: float) -> torch.Tensor:
    scale = F.softplus(raw) + float(floor)
    return torch.clamp(scale, max=float(ceiling))


class MiniCFJointV2GeomSamplerNet(MiniCFJointV2Net):
    def __init__(self, vocab: Any, emb_dim: int = 128, hidden_dim: int = 384) -> None:
        super().__init__(vocab, emb_dim=emb_dim, hidden_dim=hidden_dim)
        self.prev_coord_proj = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, emb_dim),
            nn.GELU(),
        )
        step_in = emb_dim * 6 + hidden_dim + 4
        self.step_proj = nn.Sequential(
            nn.LayerNorm(step_in),
            nn.Linear(step_in, hidden_dim),
            nn.GELU(),
        )
        coord_in = hidden_dim * 2 + emb_dim * 5 + 4
        self.coord_head = nn.Sequential(
            nn.LayerNorm(coord_in),
            nn.Linear(coord_in, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.coord_scale_head = nn.Sequential(
            nn.LayerNorm(coord_in),
            nn.Linear(coord_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.lattice_scale_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )

    def coord_params(self, coord_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.coord_head(coord_input)), self.coord_scale_head(coord_input)

    def lattice_params(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lattice_head(ctx), self.lattice_scale_head(ctx)

    def prev_coord_embs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "prev_representative_coord_values" in batch:
            prev_coords = batch["prev_representative_coord_values"]
        else:
            shape = (*batch["prev_orbit_ids"].shape, 3)
            prev_coords = torch.zeros(shape, dtype=torch.float32, device=batch["prev_orbit_ids"].device)
        return self.prev_coord_proj(prev_coords)

    def make_step_inputs(self, batch: dict[str, torch.Tensor], ctx: torch.Tensor) -> torch.Tensor:
        _bsz, steps = batch["prev_orbit_ids"].shape
        remaining_vec = self.remaining_vecs(batch)
        prev_orbit = self.orbit_emb(batch["prev_orbit_ids"])
        prev_element = self.element_emb(batch["prev_element_ids"])
        prev_coord = self.prev_coord_embs(batch)
        step_emb = self.step_emb(batch["step_ids"].clamp(0, 64))
        prev_row = prev_orbit + prev_element + prev_coord
        chosen_summary = torch.cumsum(prev_row, dim=1)
        denom = torch.arange(1, steps + 1, device=prev_row.device, dtype=prev_row.dtype).view(1, steps, 1)
        chosen_summary = chosen_summary / denom
        ctx_seq = ctx.unsqueeze(1).expand(-1, steps, -1)
        return self.step_proj(torch.cat([ctx_seq, remaining_vec, prev_orbit, prev_element, prev_coord, step_emb, chosen_summary, batch["step_features"]], dim=-1))

    def forward_distribution(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.context(batch)
        remaining_vec = self.remaining_vecs(batch)
        h, _ = self.gru(self.make_step_inputs(batch, ctx))
        orbit_logits = self.orbit_head(h)
        target_orbit_for_cond = batch["target_orbit_ids"].clamp_min(0)
        orbit_cond = self.orbit_emb(target_orbit_for_cond)
        element_logits = self.element_head(torch.cat([h, orbit_cond], dim=-1))
        target_element_for_cond = batch["target_element_ids"].clamp_min(0)
        element_cond = self.element_emb(target_element_for_cond)
        prev_coord = self.prev_coord_embs(batch)
        ctx_seq = ctx.unsqueeze(1).expand(-1, h.shape[1], -1)
        step_emb = self.step_emb(batch["step_ids"].clamp(0, 64))
        coord_input = torch.cat([h, ctx_seq, remaining_vec, orbit_cond, element_cond, prev_coord, step_emb, batch["step_features"]], dim=-1)
        coord_mu, coord_scale_raw = self.coord_params(coord_input)
        lattice_mu, lattice_scale_raw = self.lattice_params(ctx)
        return orbit_logits, element_logits, coord_mu, coord_scale_raw, lattice_mu, lattice_scale_raw


def representative_values(row: dict[str, Any], engine: OrbitEngine) -> list[float]:
    orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
    params = {str(k): float(v) % 1.0 for k, v in dict(row.get("free_params") or {}).items()}
    free_symbols = set(str(s) for s in orbit.free_symbols)
    values: list[float] = []
    for axis, symbol in enumerate(COORD_ORDER):
        if symbol in free_symbols and symbol in params:
            values.append(float(params[symbol]) % 1.0)
            continue
        fixed = orbit.fixed_values[axis] if axis < len(orbit.fixed_values) else None
        if fixed is None:
            values.append(float(params.get(symbol, 0.0)) % 1.0)
        else:
            values.append(float(fixed) % 1.0)
    return values


def expanded_affine_payload(row: dict[str, Any], engine: OrbitEngine) -> list[tuple[list[list[float]], list[float], list[float]]]:
    orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
    rep = representative_values(row, engine)
    payload: list[tuple[list[list[float]], list[float], list[float]]] = []
    seen: set[tuple[int, int, int]] = set()
    for operation in orbit.symmetry_ops:
        matrix, offset = affine_for_operation(str(operation))
        coord = [
            float(sum(matrix[axis][j] * rep[j] for j in range(3)) + offset[axis]) % 1.0
            for axis in range(3)
        ]
        key = tuple(int(round(v / 1.0e-5)) for v in coord)
        if key in seen:
            continue
        seen.add(key)
        payload.append((matrix, offset, coord))
    return payload


def collate_geom_sampler(
    records: list[dict[str, Any]],
    *,
    vocab: Any,
    engine: OrbitEngine,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    include_expanded: bool = False,
) -> dict[str, torch.Tensor]:
    batch = collate_v2(records, vocab=vocab, engine=engine, lattice_mean=lattice_mean, lattice_std=lattice_std)
    bsz, max_steps = batch["target_orbit_ids"].shape
    expanded_payloads: list[list[list[tuple[list[list[float]], list[float], list[float]]]]] = []
    max_atoms_per_row = 1
    if include_expanded:
        for record in records:
            rows = canonical_rows(record)[:max_steps]
            record_payload: list[list[tuple[list[list[float]], list[float], list[float]]]] = []
            for row in rows:
                payload = expanded_affine_payload(row, engine)
                max_atoms_per_row = max(max_atoms_per_row, len(payload))
                record_payload.append(payload)
            expanded_payloads.append(record_payload)
    rep_values = torch.zeros((bsz, max_steps, 3), dtype=torch.float32)
    prev_rep_values = torch.zeros((bsz, max_steps, 3), dtype=torch.float32)
    row_pair_mask = torch.zeros((bsz, max_steps), dtype=torch.float32)
    complex_mask = torch.zeros((bsz,), dtype=torch.float32)
    expanded_matrix = torch.zeros((bsz, max_steps, max_atoms_per_row, 3, 3), dtype=torch.float32)
    expanded_offset = torch.zeros((bsz, max_steps, max_atoms_per_row, 3), dtype=torch.float32)
    expanded_target = torch.zeros((bsz, max_steps, max_atoms_per_row, 3), dtype=torch.float32)
    expanded_mask = torch.zeros((bsz, max_steps, max_atoms_per_row), dtype=torch.float32)
    for i, record in enumerate(records):
        rows = canonical_rows(record)
        complex_mask[i] = 1.0 if int(record.get("n_sites", len(rows))) >= 7 else 0.0
        for step, row in enumerate(rows[:max_steps]):
            rep = torch.tensor(representative_values(row, engine), dtype=torch.float32)
            rep_values[i, step] = rep
            if step + 1 < max_steps:
                prev_rep_values[i, step + 1] = rep
            row_pair_mask[i, step] = 1.0
            if include_expanded:
                for atom_idx, (matrix, offset, coord) in enumerate(expanded_payloads[i][step]):
                    expanded_matrix[i, step, atom_idx] = torch.tensor(matrix, dtype=torch.float32)
                    expanded_offset[i, step, atom_idx] = torch.tensor(offset, dtype=torch.float32)
                    expanded_target[i, step, atom_idx] = torch.tensor(coord, dtype=torch.float32)
                    expanded_mask[i, step, atom_idx] = 1.0
    batch["representative_coord_values"] = rep_values
    batch["prev_representative_coord_values"] = prev_rep_values
    batch["row_pair_mask"] = row_pair_mask
    batch["complex_structure_mask"] = complex_mask
    batch["expanded_op_matrix"] = expanded_matrix
    batch["expanded_op_offset"] = expanded_offset
    batch["expanded_target_coords"] = expanded_target
    batch["expanded_atom_mask"] = expanded_mask
    batch["lattice_mean_values"] = lattice_mean.detach().clone().float()
    batch["lattice_std_values"] = lattice_std.detach().clone().float()
    return batch


def row_pair_geometry_loss(
    coord_mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    complex_only: bool,
    min_sep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_rep = batch["representative_coord_values"]
    free_mask = batch["coord_mask"]
    pred_rep = torch.remainder(coord_mu * free_mask + target_rep * (1.0 - free_mask), 1.0)
    row_mask = batch["row_pair_mask"] > 0.5
    complex_mask = batch["complex_structure_mask"] > 0.5
    pair_losses: list[torch.Tensor] = []
    sep_losses: list[torch.Tensor] = []
    for b in range(pred_rep.shape[0]):
        if complex_only and not bool(complex_mask[b].detach().cpu()):
            continue
        idx = row_mask[b]
        if int(idx.sum().detach().cpu()) < 2:
            continue
        pred = pred_rep[b, idx]
        target = target_rep[b, idx]
        pred_diff = torch.abs(pred[:, None, :] - pred[None, :, :])
        pred_diff = torch.minimum(pred_diff, 1.0 - pred_diff)
        target_diff = torch.abs(target[:, None, :] - target[None, :, :])
        target_diff = torch.minimum(target_diff, 1.0 - target_diff)
        pred_dist = torch.sqrt(pred_diff.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        target_dist = torch.sqrt(target_diff.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        tri = torch.triu(torch.ones_like(pred_dist, dtype=torch.bool), diagonal=1)
        if int(tri.sum().detach().cpu()) == 0:
            continue
        pair_losses.append(F.smooth_l1_loss(pred_dist[tri], target_dist[tri], reduction="mean"))
        sep_losses.append(F.relu(float(min_sep) - pred_dist[tri]).pow(2).mean())
    zero = coord_mu.sum() * 0.0
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    sep_loss = torch.stack(sep_losses).mean() if sep_losses else zero
    return pair_loss, sep_loss


def expanded_structure_geometry_loss(
    coord_mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    complex_only: bool,
    min_sep: float,
    max_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_rep = batch["representative_coord_values"]
    free_mask = batch["coord_mask"]
    pred_rep = torch.remainder(coord_mu * free_mask + target_rep * (1.0 - free_mask), 1.0)
    expanded = torch.einsum("bsaoj,bsj->bsao", batch["expanded_op_matrix"], pred_rep)
    expanded = torch.remainder(expanded + batch["expanded_op_offset"], 1.0)
    target = batch["expanded_target_coords"]
    atom_mask = batch["expanded_atom_mask"] > 0.5
    complex_mask = batch["complex_structure_mask"] > 0.5
    pair_losses: list[torch.Tensor] = []
    sep_losses: list[torch.Tensor] = []
    atom_cap = max(0, int(max_atoms))
    for b in range(expanded.shape[0]):
        if complex_only and not bool(complex_mask[b].detach().cpu()):
            continue
        coords = expanded[b][atom_mask[b]]
        target_coords = target[b][atom_mask[b]]
        if atom_cap > 0 and coords.shape[0] > atom_cap:
            coords = coords[:atom_cap]
            target_coords = target_coords[:atom_cap]
        if int(coords.shape[0]) < 2:
            continue
        pred_diff = torch.abs(coords[:, None, :] - coords[None, :, :])
        pred_diff = torch.minimum(pred_diff, 1.0 - pred_diff)
        target_diff = torch.abs(target_coords[:, None, :] - target_coords[None, :, :])
        target_diff = torch.minimum(target_diff, 1.0 - target_diff)
        pred_dist = torch.sqrt(pred_diff.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        target_dist = torch.sqrt(target_diff.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        tri = torch.triu(torch.ones_like(pred_dist, dtype=torch.bool), diagonal=1)
        if int(tri.sum().detach().cpu()) == 0:
            continue
        pair_losses.append(F.smooth_l1_loss(pred_dist[tri], target_dist[tri], reduction="mean"))
        sep_losses.append(F.relu(float(min_sep) - pred_dist[tri]).pow(2).mean())
    zero = coord_mu.sum() * 0.0
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    sep_loss = torch.stack(sep_losses).mean() if sep_losses else zero
    return pair_loss, sep_loss


def lattice_basis_from_normalized(raw_lattice: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = batch["lattice_masks"]
    raw = raw_lattice * mask + batch["lattice_values"] * (1.0 - mask)
    values = raw * batch["lattice_std_values"].view(1, 6) + batch["lattice_mean_values"].view(1, 6)
    lengths = torch.exp(values[:, :3]).clamp(0.5, 80.0)
    angles = torch.deg2rad((values[:, 3:6] * ANGLE_SCALE).clamp(30.0, 150.0))
    a = lengths[:, 0]
    b = lengths[:, 1]
    c = lengths[:, 2]
    alpha = angles[:, 0]
    beta = angles[:, 1]
    gamma = angles[:, 2]
    cos_a = torch.cos(alpha)
    cos_b = torch.cos(beta)
    cos_g = torch.cos(gamma)
    sin_g = torch.sin(gamma).clamp_min(1.0e-4)
    a_vec = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    b_vec = torch.stack([b * cos_g, b * sin_g, torch.zeros_like(a)], dim=-1)
    c_x = c * cos_b
    c_y = c * (cos_a - cos_b * cos_g) / sin_g
    c_z = torch.sqrt((c.pow(2) - c_x.pow(2) - c_y.pow(2)).clamp_min(1.0e-6))
    c_vec = torch.stack([c_x, c_y, c_z], dim=-1)
    return torch.stack([a_vec, b_vec, c_vec], dim=1)


def expanded_fractional_coords(coord_mu: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    target_rep = batch["representative_coord_values"]
    free_mask = batch["coord_mask"]
    pred_rep = torch.remainder(coord_mu * free_mask + target_rep * (1.0 - free_mask), 1.0)
    expanded = torch.einsum("bsaoj,bsj->bsao", batch["expanded_op_matrix"], pred_rep)
    return torch.remainder(expanded + batch["expanded_op_offset"], 1.0)


def cartesian_structure_geometry_loss(
    coord_mu: torch.Tensor,
    lattice_mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    complex_only: bool,
    min_sep: float,
    max_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_frac = expanded_fractional_coords(coord_mu, batch)
    target_frac = batch["expanded_target_coords"]
    atom_mask = batch["expanded_atom_mask"] > 0.5
    complex_mask = batch["complex_structure_mask"] > 0.5
    pred_basis = lattice_basis_from_normalized(lattice_mu, batch)
    target_basis = lattice_basis_from_normalized(batch["lattice_values"], batch)
    pair_losses: list[torch.Tensor] = []
    sep_losses: list[torch.Tensor] = []
    atom_cap = max(0, int(max_atoms))
    for b in range(pred_frac.shape[0]):
        if complex_only and not bool(complex_mask[b].detach().cpu()):
            continue
        pred = pred_frac[b][atom_mask[b]]
        target = target_frac[b][atom_mask[b]]
        if atom_cap > 0 and pred.shape[0] > atom_cap:
            pred = pred[:atom_cap]
            target = target[:atom_cap]
        if int(pred.shape[0]) < 2:
            continue
        pred_diff = pred[:, None, :] - pred[None, :, :]
        pred_diff = pred_diff - torch.round(pred_diff.detach())
        target_diff = target[:, None, :] - target[None, :, :]
        target_diff = target_diff - torch.round(target_diff.detach())
        pred_cart = torch.matmul(pred_diff, pred_basis[b])
        target_cart = torch.matmul(target_diff, target_basis[b])
        pred_dist = torch.sqrt(pred_cart.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        target_dist = torch.sqrt(target_cart.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        tri = torch.triu(torch.ones_like(pred_dist, dtype=torch.bool), diagonal=1)
        if int(tri.sum().detach().cpu()) == 0:
            continue
        scale = target_dist[tri].detach().mean().clamp_min(1.0)
        pair_losses.append(F.smooth_l1_loss(pred_dist[tri] / scale, target_dist[tri] / scale, reduction="mean"))
        sep_losses.append(F.relu(float(min_sep) - pred_dist[tri]).pow(2).mean())
    zero = coord_mu.sum() * 0.0
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    sep_loss = torch.stack(sep_losses).mean() if sep_losses else zero
    return pair_loss, sep_loss


def relative_cartesian_separation_loss(
    coord_mu: torch.Tensor,
    lattice_mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    complex_only: bool,
    target_fraction: float,
    max_atoms: int,
) -> torch.Tensor:
    pred_frac = expanded_fractional_coords(coord_mu, batch)
    target_frac = batch["expanded_target_coords"]
    atom_mask = batch["expanded_atom_mask"] > 0.5
    complex_mask = batch["complex_structure_mask"] > 0.5
    pred_basis = lattice_basis_from_normalized(lattice_mu, batch)
    target_basis = lattice_basis_from_normalized(batch["lattice_values"], batch)
    losses: list[torch.Tensor] = []
    atom_cap = max(0, int(max_atoms))
    frac = max(0.0, float(target_fraction))
    if frac <= 0.0:
        return coord_mu.sum() * 0.0
    for b in range(pred_frac.shape[0]):
        if complex_only and not bool(complex_mask[b].detach().cpu()):
            continue
        pred = pred_frac[b][atom_mask[b]]
        target = target_frac[b][atom_mask[b]]
        if atom_cap > 0 and pred.shape[0] > atom_cap:
            pred = pred[:atom_cap]
            target = target[:atom_cap]
        if int(pred.shape[0]) < 2:
            continue
        pred_diff = pred[:, None, :] - pred[None, :, :]
        pred_diff = pred_diff - torch.round(pred_diff.detach())
        target_diff = target[:, None, :] - target[None, :, :]
        target_diff = target_diff - torch.round(target_diff.detach())
        pred_cart = torch.matmul(pred_diff, pred_basis[b])
        target_cart = torch.matmul(target_diff, target_basis[b])
        pred_dist = torch.sqrt(pred_cart.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        target_dist = torch.sqrt(target_cart.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        tri = torch.triu(torch.ones_like(pred_dist, dtype=torch.bool), diagonal=1) & (target_dist > 1.0e-6)
        if int(tri.sum().detach().cpu()) == 0:
            continue
        threshold = target_dist[tri].detach() * frac
        scale = target_dist[tri].detach().mean().clamp_min(1.0)
        losses.append((F.relu(threshold - pred_dist[tri]) / scale).pow(2).mean())
    zero = coord_mu.sum() * 0.0
    return torch.stack(losses).mean() if losses else zero


def local_cartesian_geometry_loss(
    coord_mu: torch.Tensor,
    lattice_mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    complex_only: bool,
    min_sep: float,
    max_atoms: int,
    pair_cutoff: float,
    neighbors: int,
    separation_active_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_frac = expanded_fractional_coords(coord_mu, batch)
    target_frac = batch["expanded_target_coords"]
    atom_mask = batch["expanded_atom_mask"] > 0.5
    complex_mask = batch["complex_structure_mask"] > 0.5
    pred_basis = lattice_basis_from_normalized(lattice_mu, batch)
    target_basis = lattice_basis_from_normalized(batch["lattice_values"], batch)
    pair_losses: list[torch.Tensor] = []
    sep_losses: list[torch.Tensor] = []
    atom_cap = max(0, int(max_atoms))
    cutoff = float(pair_cutoff)
    neighbor_k = max(0, int(neighbors))
    for b in range(pred_frac.shape[0]):
        if complex_only and not bool(complex_mask[b].detach().cpu()):
            continue
        pred = pred_frac[b][atom_mask[b]]
        target = target_frac[b][atom_mask[b]]
        if atom_cap > 0 and pred.shape[0] > atom_cap:
            pred = pred[:atom_cap]
            target = target[:atom_cap]
        natoms = int(pred.shape[0])
        if natoms < 2:
            continue
        pred_diff = pred[:, None, :] - pred[None, :, :]
        pred_diff = pred_diff - torch.round(pred_diff.detach())
        target_diff = target[:, None, :] - target[None, :, :]
        target_diff = target_diff - torch.round(target_diff.detach())
        pred_cart = torch.matmul(pred_diff, pred_basis[b])
        target_cart = torch.matmul(target_diff, target_basis[b])
        pred_dist = torch.sqrt(pred_cart.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        target_dist = torch.sqrt(target_cart.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        tri = torch.triu(torch.ones_like(pred_dist, dtype=torch.bool), diagonal=1)
        if int(tri.sum().detach().cpu()) == 0:
            continue
        local_mask = target_dist <= cutoff if cutoff > 0.0 else torch.zeros_like(target_dist, dtype=torch.bool)
        if neighbor_k > 0:
            k = min(neighbor_k, natoms - 1)
            with torch.no_grad():
                masked_target = target_dist + torch.eye(natoms, dtype=target_dist.dtype, device=target_dist.device) * 1.0e6
                nn_idx = torch.topk(masked_target, k=k, largest=False, dim=-1).indices
                row_idx = torch.arange(natoms, device=target_dist.device).view(-1, 1).expand(-1, k)
                nn_mask = torch.zeros_like(local_mask)
                nn_mask[row_idx, nn_idx] = True
                local_mask = local_mask | nn_mask | nn_mask.t()
        pair_mask = local_mask & tri
        if int(pair_mask.sum().detach().cpu()) == 0:
            continue
        scale = target_dist[pair_mask].detach().mean().clamp_min(1.0)
        pair_losses.append(F.smooth_l1_loss(pred_dist[pair_mask] / scale, target_dist[pair_mask] / scale, reduction="mean"))
        violations = F.relu(float(min_sep) - pred_dist[tri])
        if bool(separation_active_only):
            active = violations > 0.0
            if int(active.sum().detach().cpu()) > 0:
                sep_losses.append(violations[active].pow(2).mean())
        else:
            sep_losses.append(violations.pow(2).mean())
    zero = coord_mu.sum() * 0.0
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    sep_loss = torch.stack(sep_losses).mean() if sep_losses else zero
    return pair_loss, sep_loss


def training_aux_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "decoder_conditioning": {
            "previous_representative_coord": True,
        },
        "symbolic_loss": {
            "orbit_loss_weight": float(args.orbit_loss_weight),
            "element_loss_weight": float(args.element_loss_weight),
            "complex_symbolic_loss_weight": float(args.complex_symbolic_loss_weight),
        },
        "geometry_loss": {
            "complex_geometry_loss_weight": float(args.complex_geometry_loss_weight),
        },
        "row_pair_aux": {
            "row_pair_loss_weight": float(args.row_pair_loss_weight),
            "row_separation_loss_weight": float(args.row_separation_loss_weight),
            "row_pair_complex_only": bool(args.row_pair_complex_only),
            "row_pair_min_sep": float(args.row_pair_min_sep),
        },
        "expanded_structure_aux": {
            "expanded_pair_loss_weight": float(args.expanded_pair_loss_weight),
            "expanded_separation_loss_weight": float(args.expanded_separation_loss_weight),
            "expanded_pair_complex_only": bool(args.expanded_pair_complex_only),
            "expanded_pair_min_sep": float(args.expanded_pair_min_sep),
            "expanded_pair_max_atoms": int(args.expanded_pair_max_atoms),
        },
        "cartesian_structure_aux": {
            "cartesian_pair_loss_weight": float(args.cartesian_pair_loss_weight),
            "cartesian_separation_loss_weight": float(args.cartesian_separation_loss_weight),
            "cartesian_pair_complex_only": bool(args.cartesian_pair_complex_only),
            "cartesian_pair_min_sep": float(args.cartesian_pair_min_sep),
            "cartesian_pair_max_atoms": int(args.cartesian_pair_max_atoms),
        },
        "relative_cartesian_separation_aux": {
            "relative_cartesian_separation_loss_weight": float(args.relative_cartesian_separation_loss_weight),
            "relative_cartesian_separation_complex_only": bool(args.relative_cartesian_separation_complex_only),
            "relative_cartesian_separation_frac": float(args.relative_cartesian_separation_frac),
            "relative_cartesian_separation_max_atoms": int(args.relative_cartesian_separation_max_atoms),
        },
        "local_cartesian_structure_aux": {
            "local_cartesian_pair_loss_weight": float(args.local_cartesian_pair_loss_weight),
            "local_cartesian_separation_loss_weight": float(args.local_cartesian_separation_loss_weight),
            "local_cartesian_complex_only": bool(args.local_cartesian_complex_only),
            "local_cartesian_pair_min_sep": float(args.local_cartesian_pair_min_sep),
            "local_cartesian_pair_cutoff": float(args.local_cartesian_pair_cutoff),
            "local_cartesian_pair_neighbors": int(args.local_cartesian_pair_neighbors),
            "local_cartesian_pair_max_atoms": int(args.local_cartesian_pair_max_atoms),
            "local_cartesian_separation_active_only": bool(args.local_cartesian_separation_active_only),
        },
    }


def weighted_masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal: torch.Tensor,
    *,
    ignore_index: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    masked = logits.masked_fill(~legal, -1.0e9)
    losses = F.cross_entropy(
        masked.reshape(-1, masked.shape[-1]),
        targets.reshape(-1),
        ignore_index=int(ignore_index),
        reduction="none",
    ).view_as(targets)
    valid = targets != int(ignore_index)
    if weights is None:
        denom = valid.float().sum().clamp_min(1.0)
        return (losses * valid.float()).sum() / denom
    step_weights = weights.to(device=losses.device, dtype=losses.dtype) * valid.float()
    return (losses * step_weights).sum() / step_weights.sum().clamp_min(1.0)


def loss_geom_sampler(
    orbit_logits: torch.Tensor,
    element_logits: torch.Tensor,
    coord_mu: torch.Tensor,
    coord_scale_raw: torch.Tensor,
    lattice_mu: torch.Tensor,
    lattice_scale_raw: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    coord_nll_weight: float,
    lattice_nll_weight: float,
    coord_scale_floor: float,
    coord_scale_ceiling: float,
    lattice_scale_floor: float,
    lattice_scale_ceiling: float,
    complex_geometry_loss_weight: float,
    orbit_loss_weight: float,
    element_loss_weight: float,
    complex_symbolic_loss_weight: float,
    row_pair_loss_weight: float,
    row_separation_loss_weight: float,
    row_pair_complex_only: bool,
    row_pair_min_sep: float,
    expanded_pair_loss_weight: float,
    expanded_separation_loss_weight: float,
    expanded_pair_complex_only: bool,
    expanded_pair_min_sep: float,
    expanded_pair_max_atoms: int,
    cartesian_pair_loss_weight: float,
    cartesian_separation_loss_weight: float,
    cartesian_pair_complex_only: bool,
    cartesian_pair_min_sep: float,
    cartesian_pair_max_atoms: int,
    relative_cartesian_separation_loss_weight: float,
    relative_cartesian_separation_complex_only: bool,
    relative_cartesian_separation_frac: float,
    relative_cartesian_separation_max_atoms: int,
    local_cartesian_pair_loss_weight: float,
    local_cartesian_separation_loss_weight: float,
    local_cartesian_complex_only: bool,
    local_cartesian_pair_min_sep: float,
    local_cartesian_pair_cutoff: float,
    local_cartesian_pair_neighbors: int,
    local_cartesian_pair_max_atoms: int,
    local_cartesian_separation_active_only: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    symbolic_step_weights = 1.0 + (float(complex_symbolic_loss_weight) - 1.0) * batch["complex_structure_mask"].view(-1, 1)
    orbit_loss = weighted_masked_cross_entropy(
        orbit_logits,
        batch["target_orbit_ids"],
        batch["orbit_legal"],
        ignore_index=-100,
        weights=symbolic_step_weights,
    )
    element_valid = batch["target_element_ids"] >= 0
    element_legal = batch["element_legal"].clone()
    element_legal[..., 0] = element_legal[..., 0] | ~element_valid
    element_targets = batch["target_element_ids"].masked_fill(~element_valid, 0)
    element_loss = weighted_masked_cross_entropy(
        element_logits,
        element_targets,
        element_legal,
        ignore_index=0,
        weights=symbolic_step_weights,
    )

    coord_scale = positive_scale(coord_scale_raw, coord_scale_floor, coord_scale_ceiling)
    coord_diff = torch.abs(coord_mu - batch["coord_values"])
    coord_wrapped = torch.minimum(coord_diff, 1.0 - coord_diff)
    coord_nll = 0.5 * (coord_wrapped / coord_scale).pow(2) + torch.log(coord_scale)
    geometry_step_weights = 1.0 + (float(complex_geometry_loss_weight) - 1.0) * batch["complex_structure_mask"]
    coord_weights = geometry_step_weights.view(-1, 1, 1) * batch["coord_mask"]
    coord_nll_loss = (coord_nll * coord_weights).sum() / coord_weights.sum().clamp_min(1.0)
    coord_mae = (coord_wrapped * batch["coord_mask"]).sum() / batch["coord_mask"].sum().clamp_min(1.0)
    coord_scale_mean = (coord_scale * batch["coord_mask"]).sum() / batch["coord_mask"].sum().clamp_min(1.0)

    lattice_scale = positive_scale(lattice_scale_raw, lattice_scale_floor, lattice_scale_ceiling)
    lattice_diff = torch.abs(lattice_mu - batch["lattice_values"])
    lattice_nll = 0.5 * (lattice_diff / lattice_scale).pow(2) + torch.log(lattice_scale)
    lattice_weights = geometry_step_weights.view(-1, 1) * batch["lattice_masks"]
    lattice_nll_loss = (lattice_nll * lattice_weights).sum() / lattice_weights.sum().clamp_min(1.0)
    lattice_mae = (lattice_diff * batch["lattice_masks"]).sum() / batch["lattice_masks"].sum().clamp_min(1.0)
    lattice_scale_mean = (lattice_scale * batch["lattice_masks"]).sum() / batch["lattice_masks"].sum().clamp_min(1.0)

    pair_loss, sep_loss = row_pair_geometry_loss(
        coord_mu,
        batch,
        complex_only=bool(row_pair_complex_only),
        min_sep=float(row_pair_min_sep),
    )
    expanded_weight_active = (float(expanded_pair_loss_weight) != 0.0) or (float(expanded_separation_loss_weight) != 0.0)
    if expanded_weight_active:
        expanded_pair_loss, expanded_sep_loss = expanded_structure_geometry_loss(
            coord_mu,
            batch,
            complex_only=bool(expanded_pair_complex_only),
            min_sep=float(expanded_pair_min_sep),
            max_atoms=int(expanded_pair_max_atoms),
        )
    else:
        expanded_pair_loss = coord_mu.sum() * 0.0
        expanded_sep_loss = coord_mu.sum() * 0.0
    cartesian_weight_active = (float(cartesian_pair_loss_weight) != 0.0) or (float(cartesian_separation_loss_weight) != 0.0)
    if cartesian_weight_active:
        cartesian_pair_loss, cartesian_sep_loss = cartesian_structure_geometry_loss(
            coord_mu,
            lattice_mu,
            batch,
            complex_only=bool(cartesian_pair_complex_only),
            min_sep=float(cartesian_pair_min_sep),
            max_atoms=int(cartesian_pair_max_atoms),
        )
    else:
        cartesian_pair_loss = coord_mu.sum() * 0.0
        cartesian_sep_loss = coord_mu.sum() * 0.0
    if float(relative_cartesian_separation_loss_weight) != 0.0:
        relative_cartesian_sep_loss = relative_cartesian_separation_loss(
            coord_mu,
            lattice_mu,
            batch,
            complex_only=bool(relative_cartesian_separation_complex_only),
            target_fraction=float(relative_cartesian_separation_frac),
            max_atoms=int(relative_cartesian_separation_max_atoms),
        )
    else:
        relative_cartesian_sep_loss = coord_mu.sum() * 0.0
    local_cartesian_weight_active = (float(local_cartesian_pair_loss_weight) != 0.0) or (float(local_cartesian_separation_loss_weight) != 0.0)
    if local_cartesian_weight_active:
        local_cartesian_pair_loss, local_cartesian_sep_loss = local_cartesian_geometry_loss(
            coord_mu,
            lattice_mu,
            batch,
            complex_only=bool(local_cartesian_complex_only),
            min_sep=float(local_cartesian_pair_min_sep),
            max_atoms=int(local_cartesian_pair_max_atoms),
            pair_cutoff=float(local_cartesian_pair_cutoff),
            neighbors=int(local_cartesian_pair_neighbors),
            separation_active_only=bool(local_cartesian_separation_active_only),
        )
    else:
        local_cartesian_pair_loss = coord_mu.sum() * 0.0
        local_cartesian_sep_loss = coord_mu.sum() * 0.0

    loss = (
        float(orbit_loss_weight) * orbit_loss
        + float(element_loss_weight) * element_loss
        + float(coord_nll_weight) * coord_nll_loss
        + float(lattice_nll_weight) * lattice_nll_loss
        + float(row_pair_loss_weight) * pair_loss
        + float(row_separation_loss_weight) * sep_loss
        + float(expanded_pair_loss_weight) * expanded_pair_loss
        + float(expanded_separation_loss_weight) * expanded_sep_loss
        + float(cartesian_pair_loss_weight) * cartesian_pair_loss
        + float(cartesian_separation_loss_weight) * cartesian_sep_loss
        + float(relative_cartesian_separation_loss_weight) * relative_cartesian_sep_loss
        + float(local_cartesian_pair_loss_weight) * local_cartesian_pair_loss
        + float(local_cartesian_separation_loss_weight) * local_cartesian_sep_loss
    )
    return loss, {
        "orbit_loss": float(orbit_loss.detach().cpu()),
        "element_loss": float(element_loss.detach().cpu()),
        "coord_nll_loss": float(coord_nll_loss.detach().cpu()),
        "coord_wrapped_mae": float(coord_mae.detach().cpu()),
        "coord_scale_mean": float(coord_scale_mean.detach().cpu()),
        "lattice_nll_loss": float(lattice_nll_loss.detach().cpu()),
        "lattice_normalized_mae": float(lattice_mae.detach().cpu()),
        "lattice_scale_mean": float(lattice_scale_mean.detach().cpu()),
        "row_pair_loss": float(pair_loss.detach().cpu()),
        "row_separation_loss": float(sep_loss.detach().cpu()),
        "expanded_pair_loss": float(expanded_pair_loss.detach().cpu()),
        "expanded_separation_loss": float(expanded_sep_loss.detach().cpu()),
        "cartesian_pair_loss": float(cartesian_pair_loss.detach().cpu()),
        "cartesian_separation_loss": float(cartesian_sep_loss.detach().cpu()),
        "relative_cartesian_separation_loss": float(relative_cartesian_sep_loss.detach().cpu()),
        "local_cartesian_pair_loss": float(local_cartesian_pair_loss.detach().cpu()),
        "local_cartesian_separation_loss": float(local_cartesian_sep_loss.detach().cpu()),
    }


def train_model(
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    engine: OrbitEngine,
    args: argparse.Namespace,
    ckpt_dir: Path,
) -> tuple[MiniCFJointV2GeomSamplerNet, Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
    vocab = build_v2_vocab(train_records + val_records, engine)
    lattice_mean, lattice_std = lattice_stats(train_records)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MiniCFJointV2GeomSamplerNet(vocab, emb_dim=int(args.emb_dim), hidden_dim=int(args.hidden_dim)).to(device)
    torch.set_float32_matmul_precision("high")
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    include_expanded = (
        (float(args.expanded_pair_loss_weight) != 0.0)
        or (float(args.expanded_separation_loss_weight) != 0.0)
        or (float(args.cartesian_pair_loss_weight) != 0.0)
        or (float(args.cartesian_separation_loss_weight) != 0.0)
        or (float(args.relative_cartesian_separation_loss_weight) != 0.0)
        or (float(args.local_cartesian_pair_loss_weight) != 0.0)
        or (float(args.local_cartesian_separation_loss_weight) != 0.0)
    )
    loader = DataLoader(
        JointV2Dataset(train_records),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda xs: collate_geom_sampler(
            xs,
            vocab=vocab,
            engine=engine,
            lattice_mean=mean_t.cpu(),
            lattice_std=std_t.cpu(),
            include_expanded=include_expanded,
        ),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        JointV2Dataset(val_records),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=lambda xs: collate_geom_sampler(
            xs,
            vocab=vocab,
            engine=engine,
            lattice_mean=mean_t.cpu(),
            lattice_std=std_t.cpu(),
            include_expanded=include_expanded,
        ),
        num_workers=0,
    )
    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        sums = Counter()
        steps = 0
        for raw in loader:
            batch = move_batch(raw, device)
            opt.zero_grad(set_to_none=True)
            outputs = model.forward_distribution(batch)
            loss, parts = loss_geom_sampler(
                *outputs,
                batch,
                coord_nll_weight=float(args.coord_nll_weight),
                lattice_nll_weight=float(args.lattice_nll_weight),
                coord_scale_floor=float(args.coord_scale_floor),
                coord_scale_ceiling=float(args.coord_scale_ceiling),
                lattice_scale_floor=float(args.lattice_scale_floor),
                lattice_scale_ceiling=float(args.lattice_scale_ceiling),
                complex_geometry_loss_weight=float(args.complex_geometry_loss_weight),
                orbit_loss_weight=float(args.orbit_loss_weight),
                element_loss_weight=float(args.element_loss_weight),
                complex_symbolic_loss_weight=float(args.complex_symbolic_loss_weight),
                row_pair_loss_weight=float(args.row_pair_loss_weight),
                row_separation_loss_weight=float(args.row_separation_loss_weight),
                row_pair_complex_only=bool(args.row_pair_complex_only),
                row_pair_min_sep=float(args.row_pair_min_sep),
                expanded_pair_loss_weight=float(args.expanded_pair_loss_weight),
                expanded_separation_loss_weight=float(args.expanded_separation_loss_weight),
                expanded_pair_complex_only=bool(args.expanded_pair_complex_only),
                expanded_pair_min_sep=float(args.expanded_pair_min_sep),
                expanded_pair_max_atoms=int(args.expanded_pair_max_atoms),
                cartesian_pair_loss_weight=float(args.cartesian_pair_loss_weight),
                cartesian_separation_loss_weight=float(args.cartesian_separation_loss_weight),
                cartesian_pair_complex_only=bool(args.cartesian_pair_complex_only),
                cartesian_pair_min_sep=float(args.cartesian_pair_min_sep),
                cartesian_pair_max_atoms=int(args.cartesian_pair_max_atoms),
                relative_cartesian_separation_loss_weight=float(args.relative_cartesian_separation_loss_weight),
                relative_cartesian_separation_complex_only=bool(args.relative_cartesian_separation_complex_only),
                relative_cartesian_separation_frac=float(args.relative_cartesian_separation_frac),
                relative_cartesian_separation_max_atoms=int(args.relative_cartesian_separation_max_atoms),
                local_cartesian_pair_loss_weight=float(args.local_cartesian_pair_loss_weight),
                local_cartesian_separation_loss_weight=float(args.local_cartesian_separation_loss_weight),
                local_cartesian_complex_only=bool(args.local_cartesian_complex_only),
                local_cartesian_pair_min_sep=float(args.local_cartesian_pair_min_sep),
                local_cartesian_pair_cutoff=float(args.local_cartesian_pair_cutoff),
                local_cartesian_pair_neighbors=int(args.local_cartesian_pair_neighbors),
                local_cartesian_pair_max_atoms=int(args.local_cartesian_pair_max_atoms),
                local_cartesian_separation_active_only=bool(args.local_cartesian_separation_active_only),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sums["loss"] += float(loss.detach().cpu())
            for key, value in parts.items():
                sums[key] += float(value)
            steps += 1
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / max(1, steps),
            "seconds": time.time() - started,
        }
        for key in (
            "orbit_loss",
            "element_loss",
            "coord_nll_loss",
            "coord_wrapped_mae",
            "coord_scale_mean",
            "lattice_nll_loss",
            "lattice_normalized_mae",
            "lattice_scale_mean",
            "row_pair_loss",
            "row_separation_loss",
            "expanded_pair_loss",
            "expanded_separation_loss",
            "cartesian_pair_loss",
            "cartesian_separation_loss",
            "relative_cartesian_separation_loss",
            "local_cartesian_pair_loss",
            "local_cartesian_separation_loss",
        ):
            row[key] = sums[key] / max(1, steps)
        if row["train_loss"] < best_train_loss:
            best_train_loss = float(row["train_loss"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs):
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for raw in val_loader:
                    batch = move_batch(raw, device)
                    outputs = model.forward_distribution(batch)
                    loss, _parts = loss_geom_sampler(
                        *outputs,
                        batch,
                        coord_nll_weight=float(args.coord_nll_weight),
                        lattice_nll_weight=float(args.lattice_nll_weight),
                        coord_scale_floor=float(args.coord_scale_floor),
                        coord_scale_ceiling=float(args.coord_scale_ceiling),
                        lattice_scale_floor=float(args.lattice_scale_floor),
                        lattice_scale_ceiling=float(args.lattice_scale_ceiling),
                        complex_geometry_loss_weight=float(args.complex_geometry_loss_weight),
                        orbit_loss_weight=float(args.orbit_loss_weight),
                        element_loss_weight=float(args.element_loss_weight),
                        complex_symbolic_loss_weight=float(args.complex_symbolic_loss_weight),
                        row_pair_loss_weight=float(args.row_pair_loss_weight),
                        row_separation_loss_weight=float(args.row_separation_loss_weight),
                        row_pair_complex_only=bool(args.row_pair_complex_only),
                        row_pair_min_sep=float(args.row_pair_min_sep),
                        expanded_pair_loss_weight=float(args.expanded_pair_loss_weight),
                        expanded_separation_loss_weight=float(args.expanded_separation_loss_weight),
                        expanded_pair_complex_only=bool(args.expanded_pair_complex_only),
                        expanded_pair_min_sep=float(args.expanded_pair_min_sep),
                        expanded_pair_max_atoms=int(args.expanded_pair_max_atoms),
                        cartesian_pair_loss_weight=float(args.cartesian_pair_loss_weight),
                        cartesian_separation_loss_weight=float(args.cartesian_separation_loss_weight),
                        cartesian_pair_complex_only=bool(args.cartesian_pair_complex_only),
                        cartesian_pair_min_sep=float(args.cartesian_pair_min_sep),
                        cartesian_pair_max_atoms=int(args.cartesian_pair_max_atoms),
                        relative_cartesian_separation_loss_weight=float(args.relative_cartesian_separation_loss_weight),
                        relative_cartesian_separation_complex_only=bool(args.relative_cartesian_separation_complex_only),
                        relative_cartesian_separation_frac=float(args.relative_cartesian_separation_frac),
                        relative_cartesian_separation_max_atoms=int(args.relative_cartesian_separation_max_atoms),
                        local_cartesian_pair_loss_weight=float(args.local_cartesian_pair_loss_weight),
                        local_cartesian_separation_loss_weight=float(args.local_cartesian_separation_loss_weight),
                        local_cartesian_complex_only=bool(args.local_cartesian_complex_only),
                        local_cartesian_pair_min_sep=float(args.local_cartesian_pair_min_sep),
                        local_cartesian_pair_cutoff=float(args.local_cartesian_pair_cutoff),
                        local_cartesian_pair_neighbors=int(args.local_cartesian_pair_neighbors),
                        local_cartesian_pair_max_atoms=int(args.local_cartesian_pair_max_atoms),
                        local_cartesian_separation_active_only=bool(args.local_cartesian_separation_active_only),
                    )
                    val_loss += float(loss.detach().cpu())
                    val_steps += 1
            row["val_loss_report_only"] = val_loss / max(1, val_steps)
            print(json.dumps({"stage": "fixed_order_geom_sampler_epoch", **row}, sort_keys=True), flush=True)
        history.append(row)

    ckpt_dir = under_root(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    aux_config = training_aux_config(args)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "vocab": vocab.to_json(),
            "lattice_mean": lattice_mean,
            "lattice_std": lattice_std,
            "config": config,
            "history": history,
            "best_train_loss": best_train_loss,
            "model_kind": "MiniCFJointV2GeomSamplerNet",
            "selection_rule": "lowest_train_loss_not_eval_label_selection",
            "geometry_objective": "heteroscedastic_free_param_lattice_nll",
            **aux_config,
            "rng_state": {
                "python_random": random.getstate()[1][0],
                "numpy_random": int(np.random.get_state()[1][0]),
                "torch_initial_seed": int(torch.initial_seed()),
            },
        },
        ckpt_dir / "last.pt",
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab.to_json(),
            "lattice_mean": lattice_mean,
            "lattice_std": lattice_std,
            "config": config,
            "history": history,
            "best_train_loss": best_train_loss,
            "model_kind": "MiniCFJointV2GeomSamplerNet",
            "selection_rule": "lowest_train_loss_not_eval_label_selection",
            "geometry_objective": "heteroscedastic_free_param_lattice_nll",
            **aux_config,
        },
        ckpt_dir / "best_train.pt",
    )
    info = {
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.time() - started,
        "best_train_loss": best_train_loss,
        "history": history,
        "checkpoint_best": safe_rel(ckpt_dir / "best_train.pt"),
        "checkpoint_last": safe_rel(ckpt_dir / "last.pt"),
        "geometry_objective": "heteroscedastic_free_param_lattice_nll",
        **aux_config,
    }
    return model, vocab, mean_t, std_t, info


def load_eval_only_model(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    ckpt_dir: Path,
) -> tuple[MiniCFJointV2GeomSamplerNet, Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
    ckpt_path = checkpoint_path.expanduser().resolve()
    root = ROOT.resolve()
    if ckpt_path != root and root not in ckpt_path.parents:
        raise RuntimeError(f"refusing to load eval-only checkpoint outside opentry_5: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config") or {}
    vocab = Vocab.from_json(checkpoint["vocab"])
    emb_dim = int(config.get("emb_dim", args.emb_dim))
    hidden_dim = int(config.get("hidden_dim", args.hidden_dim))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MiniCFJointV2GeomSamplerNet(vocab, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean_t = torch.tensor(checkpoint["lattice_mean"], dtype=torch.float32)
    std_t = torch.tensor(checkpoint["lattice_std"], dtype=torch.float32)
    ckpt_dir = under_root(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        ckpt_dir / "eval_only_source.json",
        {
            "eval_only": True,
            "source_checkpoint": safe_rel(ckpt_path),
            "model_kind": checkpoint.get("model_kind"),
            "selection_rule": checkpoint.get("selection_rule"),
            "best_train_loss": checkpoint.get("best_train_loss"),
            "loaded_config": config,
        },
    )
    aux_config = {
        k: checkpoint.get(k)
        for k in (
            "row_pair_aux",
            "expanded_structure_aux",
            "cartesian_structure_aux",
            "relative_cartesian_separation_aux",
            "local_cartesian_structure_aux",
            "decoder_conditioning",
            "symbolic_loss",
            "geometry_loss",
        )
        if k in checkpoint
    }
    info = {
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": 0.0,
        "eval_only": True,
        "source_checkpoint": safe_rel(ckpt_path),
        "best_train_loss": checkpoint.get("best_train_loss"),
        "history": checkpoint.get("history") or [],
        "checkpoint_best": safe_rel(ckpt_path),
        "checkpoint_last": None,
        "geometry_objective": checkpoint.get("geometry_objective", "heteroscedastic_free_param_lattice_nll"),
        **aux_config,
    }
    return model, vocab, mean_t, std_t, info


@torch.no_grad()
def decode_geom_sampler_candidate(
    model: MiniCFJointV2GeomSamplerNet,
    record: dict[str, Any],
    *,
    engine: OrbitEngine,
    vocab: Any,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
    gen_index: int,
    seed: int,
    temperature: float,
    args: argparse.Namespace,
    max_steps: int = 64,
) -> tuple[dict[str, Any] | None, str | None]:
    model.eval()
    try:
        formula_element_ids, formula_counts_t, formula_elements = formula_tensors(record, vocab, device)
        sg_key = str(int(record["sg"]))
        if sg_key not in vocab.sg_to_id:
            return None, "sg_not_in_vocab"
    except Exception as exc:  # noqa: BLE001
        return None, f"input_vocab_error:{type(exc).__name__}:{exc}"

    deterministic = int(gen_index) == 0
    generator = None
    if not deterministic:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    sg_id = torch.tensor(vocab.sg_to_id[sg_key], dtype=torch.long, device=device)
    ctx = model.context({"formula_element_ids": formula_element_ids.view(1, -1), "formula_counts": formula_counts_t.view(1, -1), "sg_ids": sg_id.view(1)})
    formula_embs = model.element_emb(formula_element_ids.view(1, -1)).squeeze(0)
    lattice_mu_raw, lattice_scale_raw = model.lattice_params(ctx)
    lattice_mu_raw = lattice_mu_raw.squeeze(0)
    lattice_scale = positive_scale(
        lattice_scale_raw.squeeze(0),
        float(args.lattice_scale_floor),
        float(args.lattice_scale_ceiling),
    )
    if deterministic:
        lattice_raw = lattice_mu_raw
    else:
        lattice_raw = lattice_mu_raw + torch.randn(lattice_mu_raw.shape, generator=generator, device=device) * lattice_scale * float(args.lattice_sample_scale)
    lattice = decode_lattice(lattice_raw, lattice_mean, lattice_std, int(record["sg"]))
    total_atoms = max(1, sum(int(v) for v in record["formula_counts"].values()))
    sg_multiplicities = sorted({int(o.multiplicity) for o in engine.get_orbits(int(record["sg"]))})
    if sg_multiplicities:
        max_rows_from_formula = int(sum(int(v) for v in record["formula_counts"].values())) // max(1, min(sg_multiplicities))
        max_steps = min(int(max_steps), max(4, max_rows_from_formula + 2))

    close_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[Any, ...] | None], bool] = {}
    legal_elements_cache: dict[tuple[str, tuple[tuple[str, int], ...], tuple[Any, ...] | None], list[str]] = {}
    legal_orbits_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[Any, ...] | None], list[str]] = {}

    def remaining_key(remaining: dict[str, int]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((str(k), int(v)) for k, v in remaining.items() if int(v) > 0))

    def can_close_cached(remaining: dict[str, int], last_key: tuple[Any, ...] | None) -> bool:
        key = (remaining_key(remaining), last_key)
        if key not in close_cache:
            close_cache[key] = can_close_remaining_ordered(engine=engine, sg=int(record["sg"]), remaining=remaining, last_key=last_key)
        return close_cache[key]

    def legal_elements_cached(orbit_id: str, remaining: dict[str, int], last_key: tuple[Any, ...] | None) -> list[str]:
        key = (str(orbit_id), remaining_key(remaining), last_key)
        if key not in legal_elements_cache:
            orbit = engine.get_orbit_by_id(str(orbit_id))
            out: list[str] = []
            for element in sorted(record["formula_counts"]):
                if element not in vocab.element_to_id:
                    continue
                if int(remaining.get(element, 0)) < int(orbit.multiplicity):
                    continue
                new_key = candidate_key_for(engine, orbit_id, element)
                if last_key is not None and new_key < last_key:
                    continue
                next_remaining = dict(remaining)
                next_remaining[element] = int(next_remaining[element]) - int(orbit.multiplicity)
                if can_close_cached(next_remaining, new_key):
                    out.append(element)
            legal_elements_cache[key] = out
        return legal_elements_cache[key]

    def legal_orbits_cached(remaining: dict[str, int], last_key: tuple[Any, ...] | None) -> list[str]:
        key = (remaining_key(remaining), last_key)
        if key not in legal_orbits_cache:
            if sum(int(v) for v in remaining.values()) == 0:
                legal_orbits_cache[key] = [STOP_ORBIT]
            else:
                out: list[str] = []
                for orbit in engine.get_orbits(int(record["sg"])):
                    oid = str(orbit.canonical_orbit_id)
                    if oid in vocab.orbit_to_id and legal_elements_cached(oid, remaining, last_key):
                        out.append(oid)
                legal_orbits_cache[key] = out
        return legal_orbits_cache[key]

    def step_fast(
        *,
        remaining_counts_t: torch.Tensor,
        prev_orbit_id: int,
        prev_element_id: int,
        prev_coord_values: torch.Tensor,
        step_features: torch.Tensor,
        hidden: torch.Tensor | None,
        chosen_summary: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = remaining_counts_t / remaining_counts_t.sum().clamp_min(1.0)
        rem_vec = (formula_embs * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        prev_orbit = model.orbit_emb(torch.tensor([int(prev_orbit_id)], dtype=torch.long, device=device))
        prev_element = model.element_emb(torch.tensor([int(prev_element_id)], dtype=torch.long, device=device))
        prev_coord = model.prev_coord_proj(prev_coord_values.view(1, 3))
        step_emb = model.step_emb(torch.tensor([min(int(step_index), 64)], dtype=torch.long, device=device))
        prev_row = prev_orbit + prev_element + prev_coord
        new_summary = (chosen_summary * float(max(0, step_index)) + prev_row) / float(max(1, step_index + 1))
        inp = model.step_proj(torch.cat([ctx, rem_vec, prev_orbit, prev_element, prev_coord, step_emb, new_summary, step_features.view(1, -1)], dim=-1)).unsqueeze(1)
        out, hidden_out = model.gru(inp, hidden)
        h = out[:, -1, :]
        return model.orbit_head(h).squeeze(0), h.squeeze(0), hidden_out, new_summary.detach()

    rows: list[dict[str, Any]] = []
    params: dict[int, dict[str, float]] = {}
    remaining = {str(k): int(v) for k, v in record["formula_counts"].items()}
    last_key: tuple[Any, ...] | None = None
    prev_orbit = vocab.start_orbit_id
    prev_element = 0
    prev_coord_values = torch.zeros(3, dtype=torch.float32, device=device)
    hidden: torch.Tensor | None = None
    summary = torch.zeros((1, model.element_emb.embedding_dim), dtype=torch.float32, device=device)

    for step in range(max_steps):
        rem_vec = torch.tensor([float(remaining.get(e, 0)) for e in formula_elements], dtype=torch.float32, device=device)
        chosen_atoms = total_atoms - sum(int(v) for v in remaining.values())
        step_denom = max(1.0, float(max_steps - 1))
        feats = torch.tensor(
            [float(step) / step_denom, float(sum(int(v) for v in remaining.values())) / total_atoms, float(chosen_atoms) / total_atoms, float(len(rows)) / 64.0],
            dtype=torch.float32,
            device=device,
        )
        orbit_logits, h, hidden_out, summary_out = step_fast(
            remaining_counts_t=rem_vec,
            prev_orbit_id=int(prev_orbit),
            prev_element_id=int(prev_element),
            prev_coord_values=prev_coord_values,
            step_features=feats,
            hidden=hidden,
            chosen_summary=summary,
            step_index=step,
        )
        legal_orbits = legal_orbits_cached(remaining, last_key)
        legal_orbit_ids = [vocab.orbit_to_id[o] for o in legal_orbits if o in vocab.orbit_to_id]
        if not legal_orbit_ids:
            return None, "no_legal_orbit"
        orbit_idx = choose_from_logits(
            orbit_logits,
            legal_orbit_ids,
            deterministic=deterministic,
            temperature=temperature,
            generator=generator,
        )
        orbit_id = vocab.id_to_orbit[int(orbit_idx)]
        if orbit_id == STOP_ORBIT:
            if sum(int(v) for v in remaining.values()) == 0:
                skel, wa = canonical_keys_from_rows(rows)
                return {"rows": rows, "params": params, "lattice": lattice, "canonical_skeleton_key": skel, "canonical_wa_key": wa}, None
            return None, "premature_stop"

        legal_elements = legal_elements_cached(orbit_id, remaining, last_key)
        elem_ids = [vocab.element_to_id[e] for e in legal_elements if e in vocab.element_to_id]
        if not elem_ids:
            return None, "no_legal_element"
        orbit_cond = model.orbit_emb(torch.tensor([int(orbit_idx)], dtype=torch.long, device=device))
        elem_logits = model.element_head(torch.cat([h.view(1, -1), orbit_cond], dim=-1)).squeeze(0)
        elem_idx = choose_from_logits(
            elem_logits,
            elem_ids,
            deterministic=deterministic,
            temperature=temperature,
            generator=generator,
        )
        element = vocab.id_to_element[int(elem_idx)]
        if element == NONE_ELEMENT:
            return None, "none_element_selected"
        orbit = engine.get_orbit_by_id(orbit_id)
        next_remaining = dict(remaining)
        next_remaining[element] = int(next_remaining[element]) - int(orbit.multiplicity)
        new_key = candidate_key_for(engine, orbit_id, element)
        if next_remaining[element] < 0 or not can_close_cached(next_remaining, new_key):
            return None, "sampled_infeasible_element"

        elem_cond = model.element_emb(torch.tensor([int(elem_idx)], dtype=torch.long, device=device))
        prev_coord_emb = model.prev_coord_proj(prev_coord_values.view(1, 3))
        step_emb = model.step_emb(torch.tensor([min(int(step), 64)], dtype=torch.long, device=device))
        rem_weights = rem_vec / rem_vec.sum().clamp_min(1.0)
        rem_emb_for_coord = (formula_embs * rem_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        coord_input = torch.cat([h.view(1, -1), ctx, rem_emb_for_coord, orbit_cond, elem_cond, prev_coord_emb, step_emb, feats.view(1, -1)], dim=-1)
        coord_mu, coord_scale_raw = model.coord_params(coord_input)
        coord_mu = coord_mu.squeeze(0)
        coord_scale = positive_scale(
            coord_scale_raw.squeeze(0),
            float(args.coord_scale_floor),
            float(args.coord_scale_ceiling),
        )
        if deterministic:
            coord_pred = coord_mu
        else:
            coord_pred = torch.remainder(
                coord_mu + torch.randn(coord_mu.shape, generator=generator, device=device) * coord_scale * float(args.coord_sample_scale),
                1.0,
            )
        row = {
            "element": element,
            "orbit_id": orbit_id,
            "multiplicity": int(orbit.multiplicity),
            "letter": orbit.letter,
            "enumeration": orbit.enumeration,
            "site_symmetry": orbit.site_symmetry,
            "free_symbols": list(orbit.free_symbols),
        }
        params[len(rows)] = {sym: float(coord_pred[j].detach().cpu()) % 1.0 for j, sym in enumerate(COORD_ORDER) if sym in set(orbit.free_symbols)}
        free_symbols = {str(s) for s in orbit.free_symbols}
        rep_next: list[float] = []
        for axis, symbol in enumerate(COORD_ORDER):
            if symbol in free_symbols:
                rep_next.append(float(coord_pred[axis].detach().cpu()) % 1.0)
            else:
                fixed = orbit.fixed_values[axis] if axis < len(orbit.fixed_values) else None
                rep_next.append((0.0 if fixed is None else float(fixed)) % 1.0)
        prev_coord_values = torch.tensor(rep_next, dtype=torch.float32, device=device)
        rows.append(row)
        remaining = next_remaining
        last_key = new_key
        prev_orbit = int(orbit_idx)
        prev_element = int(elem_idx)
        hidden = hidden_out
        summary = summary_out
    return None, "max_steps_exceeded"


@torch.no_grad()
def generate_fixed_order_rows(
    *,
    records: list[dict[str, Any]],
    model: MiniCFJointV2GeomSamplerNet,
    vocab: Any,
    engine: OrbitEngine,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    args: argparse.Namespace,
    partial_path: Path,
    progress_path: Path,
) -> list[dict[str, Any]]:
    eval_device = torch.device("cpu")
    model = model.to(eval_device)
    rows: list[dict[str, Any]] = []
    started = time.time()
    seeds = SEEDS[: int(args.k)]
    partial_path = under_root(partial_path)
    progress_path = under_root(progress_path)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "fixed_order_geom_sampler_generation_start", "samples": len(records), "k": len(seeds)}, sort_keys=True), flush=True)
    with partial_path.open("w", encoding="utf-8") as partial_f, progress_path.open("w", encoding="utf-8") as progress_f:
        for sample_index, record in enumerate(records):
            if sample_index and sample_index % 16 == 0:
                print(json.dumps({"stage": "fixed_order_geom_sampler_generation_progress", "done": sample_index, "seconds": time.time() - started}, sort_keys=True), flush=True)
            target_skel, target_wa = canonical_keys_from_rows(canonical_rows(record))
            for gen_index, seed in enumerate(seeds):
                gen_started = time.time()
                progress_f.write(
                    json.dumps(
                        {
                            "event": "candidate_start",
                            "sample_index": sample_index,
                            "sample_id": record["sample_id"],
                            "gen_index": gen_index,
                            "seed": int(seed),
                            "seconds": time.time() - started,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                progress_f.flush()
                temp = 1.0e-6 if gen_index == 0 else (float(args.temperature_medium) if gen_index <= 4 else float(args.temperature_diverse))
                candidate, error = decode_geom_sampler_candidate(
                    model,
                    record,
                    engine=engine,
                    vocab=vocab,
                    lattice_mean=mean_t.cpu(),
                    lattice_std=std_t.cpu(),
                    device=eval_device,
                    gen_index=gen_index,
                    seed=seed,
                    temperature=temp,
                    args=args,
                    max_steps=int(args.max_steps),
                )
                base = {
                    "mode": MODE,
                    "sample_index": sample_index,
                    "sample_id": record["sample_id"],
                    "gen_index": gen_index,
                    "seed": int(seed),
                    "temperature": temp,
                    "candidate_order": "generation_index_then_fixed_seed",
                    "geometry_sampler": "heteroscedastic_nll_fixed_seed",
                    "generation_time_seconds": time.time() - gen_started,
                }
                if candidate is None:
                    row = {
                        **base,
                        "raw_generation_success": False,
                        "generated_text": "",
                        "error": error or "decode_failed",
                        "formula_closure_success": False,
                        "atom_count_ok": False,
                        "skeleton_hit": False,
                        "wa_hit": False,
                    }
                else:
                    rendered = render_candidate(engine, record, candidate, gen_index, "fixed_order_geom_sampler")
                    row = {
                        **base,
                        "raw_generation_success": bool(rendered["ok"]),
                        "generated_text": rendered["cif"],
                        "error": rendered.get("error"),
                        "formula_closure_success": True,
                        "atom_count_ok": bool(rendered.get("atom_count_ok")),
                        "canonical_skeleton_key": candidate["canonical_skeleton_key"],
                        "canonical_wa_key": candidate["canonical_wa_key"],
                        "target_canonical_skeleton_key": record.get("canonical_skeleton_key_v2") or target_skel,
                        "target_canonical_wa_key": record.get("canonical_wa_key_v2") or target_wa,
                        "skeleton_hit": candidate["canonical_skeleton_key"] == (record.get("canonical_skeleton_key_v2") or target_skel),
                        "wa_hit": candidate["canonical_wa_key"] == (record.get("canonical_wa_key_v2") or target_wa),
                    }
                rows.append(row)
                partial_f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                partial_f.flush()
                progress_f.write(
                    json.dumps(
                        {
                            "event": "candidate_done",
                            "sample_index": sample_index,
                            "sample_id": record["sample_id"],
                            "gen_index": gen_index,
                            "success": bool(row.get("raw_generation_success")),
                            "error": row.get("error"),
                            "elapsed": row.get("generation_time_seconds"),
                            "seconds": time.time() - started,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                progress_f.flush()
    print(json.dumps({"stage": "fixed_order_geom_sampler_generation_done", "samples": len(records), "seconds": time.time() - started}, sort_keys=True), flush=True)
    return rows


def write_reports(
    *,
    args: argparse.Namespace,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    source_clean_dir: Path,
    train_info: dict[str, Any],
    curriculum_info: dict[str, Any],
    summary: dict[str, Any],
    out_dir: Path,
    ckpt_dir: Path,
    experiment_id: str,
    elapsed_seconds: float,
) -> None:
    report = {
        "experiment_id": experiment_id,
        "fold": args.fold,
        "scope": {
            "train_split": "clean_train_smoke_from_grouped_train_core",
            "eval_split": f"{args.fold}_clean_val_smoke",
            "train_samples": len(train_records),
            "eval_samples": len(val_records),
            "test_used": False,
            "val512_used": False,
        },
        "no_ranking": True,
        "candidate_order": "generation_index_then_fixed_seed",
        "geometry_sampler": "heteroscedastic_nll_fixed_seed",
        "seeds": SEEDS[: int(args.k)],
        "selection_rule": "checkpoint_by_train_loss_only; eval labels not used for candidate order",
        "train": train_info,
        "curriculum": curriculum_info,
        "sampling": {
            "coord_sample_scale": float(args.coord_sample_scale),
            "lattice_sample_scale": float(args.lattice_sample_scale),
            "coord_scale_floor": float(args.coord_scale_floor),
            "coord_scale_ceiling": float(args.coord_scale_ceiling),
            "lattice_scale_floor": float(args.lattice_scale_floor),
            "lattice_scale_ceiling": float(args.lattice_scale_ceiling),
        },
        "training_auxiliaries": training_aux_config(args),
        "summary": summary,
        "paths": {
            "clean_data": safe_rel(source_clean_dir),
            "checkpoint_dir": safe_rel(ckpt_dir),
            "generations": safe_rel(out_dir / "generations.jsonl"),
            "metrics": safe_rel(out_dir / "metrics.jsonl"),
        },
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(out_dir / "report.json", report)
    write_md(
        out_dir / "report.md",
        "\n".join(
            [
                f"# {experiment_id} Fixed-Order Geometry Sampler Smoke ({args.fold})",
                "",
                f"- train samples: {len(train_records)}",
                f"- rows>=7 train records: {curriculum_info['effective_rows_ge7_records']}",
                f"- eval samples: {len(val_records)}",
                f"- checkpoint: `{report['paths']['checkpoint_dir']}`",
                "- geometry objective: heteroscedastic free-param/lattice NLL plus optional training-only geometry auxiliaries",
                "- decoder conditioning: previous representative coordinate",
                "- candidate order: generation_index then fixed seed",
                "- no ranking/selector/scorer: true",
                "- test used: false",
                "- val512 used: false",
                f"- match@1/5/20: {pct(summary['top1']['match@1'])} / {pct(summary['top5']['match@5'])} / {pct(summary['top20']['match@20'])}",
                f"- rows>=7 match@1/5/20: {pct(summary['top1']['rows_ge_7_match@1'])} / {pct(summary['top5']['rows_ge_7_match@5'])} / {pct(summary['top20']['rows_ge_7_match@20'])}",
                f"- readable@20: {pct(summary['top20']['readable@20'])}",
                f"- composition exact@20: {pct(summary['top20']['composition_exact@20'])}",
                f"- SG/Wyckoff legal@20: {pct(summary['top20']['SG_Wyckoff_legal@20'])}",
                f"- WA-hit but match-fail@20: {pct(summary['top20']['WA_hit_but_match_fail_rate@20'])}",
                f"- collision-like candidate rate@20: {pct(summary['top20']['collision_like_candidate_rate@20'])}",
                "",
                "This is a non-oracle smoke. It does not satisfy the terminal gate unless later grouped folds, dev_gate, and val512 pass under the frozen protocol.",
            ]
        ),
    )
    append_text(
        ROOT / "reports/opentry_5_experiment_log.md",
        f"""
## {experiment_id}: Fixed-order geometry-sampler MiniCFJoint-v2 smoke ({args.fold})
- Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
- Core hypothesis: rows>=7 failures require distributional free-param/lattice sampling instead of a single coordinate point estimate.
- Difference vs historical failed routes: no selector, no reranker, no source-prior insertion, no candidate score ordering; each candidate is one fixed generation_index/seed path.
- Model/data side: model-side joint generator smoke on opentry_5 grouped clean data.
- Contains sorting/filtering: no.
- candidate order: generation_index_then_fixed_seed; invalid candidates remain in place.
- Geometry objective: heteroscedastic free-param/lattice NLL; previous representative coordinate conditioning; fixed-seed sampling only.
- Read files: {safe_rel(source_clean_dir / 'clean_train.jsonl')}, {safe_rel(source_clean_dir / 'clean_val.jsonl')}, model/New_model/symcif_experiment/artifacts/wyckoff_lookup_full.json.
- Written files: {safe_rel(out_dir / 'report.json')}, {safe_rel(out_dir / 'generations.jsonl')}, {safe_rel(out_dir / 'metrics.jsonl')}, {safe_rel(ckpt_dir / 'best_train.pt')}, {safe_rel(ckpt_dir / 'last.pt')}.
- Data split: {args.fold} clean smoke; train={len(train_records)}, eval={len(val_records)}.
- Train curriculum: {json.dumps(curriculum_info, sort_keys=True)}.
- Test read: no.
- val512 read: no; val512 cumulative count unchanged.
- Model: MiniCFJointV2GeomSamplerNet fixed-order decoder with previous representative coordinate conditioning.
- Parameters: {train_info['parameters']}.
- GPU/CPU: train {train_info['device']}; generation/eval CPU.
- Training time: {train_info['train_seconds']:.2f}s.
- Inference/eval wall time: {elapsed_seconds:.2f}s total.
- readable@20: {pct(summary['top20']['readable@20'])}.
- composition exact@20: {pct(summary['top20']['composition_exact@20'])}.
- SG/Wyckoff legal@20: {pct(summary['top20']['SG_Wyckoff_legal@20'])}.
- match@1: {pct(summary['top1']['match@1'])}.
- match@5: {pct(summary['top5']['match@5'])}.
- match@20: {pct(summary['top20']['match@20'])}.
- rows>=7 match@1/5/20: {pct(summary['top1']['rows_ge_7_match@1'])}, {pct(summary['top5']['rows_ge_7_match@5'])}, {pct(summary['top20']['rows_ge_7_match@20'])}.
- rows>=7 positive-any@20: {pct(summary['top20']['rows_ge_7_positive_any@20'])}.
- rows>=7 new positives@20: {summary['top20']['rows_ge_7_new_positive_samples@20']}.
- W/A-hit match-fail@20: {pct(summary['top20']['WA_hit_but_match_fail_rate@20'])}.
- skeleton-hit match-fail@20: {pct(summary['top20']['skeleton_hit_but_match_fail_rate@20'])}.
- collision-like candidate rate@20: {pct(summary['top20']['collision_like_candidate_rate@20'])}.
- Gate pass: false.
- Stop model family: false.
- Next: run the paired grouped fold before deciding whether to stop or scale this geometry objective.
""",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["fold_a", "fold_b"], required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--temperature-medium", type=float, default=0.8)
    parser.add_argument("--temperature-diverse", type=float, default=1.05)
    parser.add_argument("--coord-sample-scale", type=float, default=0.8)
    parser.add_argument("--lattice-sample-scale", type=float, default=0.5)
    parser.add_argument("--orbit-loss-weight", type=float, default=1.0)
    parser.add_argument("--element-loss-weight", type=float, default=1.0)
    parser.add_argument("--complex-symbolic-loss-weight", type=float, default=1.0)
    parser.add_argument("--complex-geometry-loss-weight", type=float, default=1.0)
    parser.add_argument("--coord-nll-weight", type=float, default=3.0)
    parser.add_argument("--lattice-nll-weight", type=float, default=0.5)
    parser.add_argument("--row-pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--row-separation-loss-weight", type=float, default=0.0)
    parser.add_argument("--row-pair-min-sep", type=float, default=0.12)
    parser.add_argument("--row-pair-complex-only", action="store_true")
    parser.add_argument("--expanded-pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--expanded-separation-loss-weight", type=float, default=0.0)
    parser.add_argument("--expanded-pair-min-sep", type=float, default=0.10)
    parser.add_argument("--expanded-pair-max-atoms", type=int, default=96)
    parser.add_argument("--expanded-pair-complex-only", action="store_true")
    parser.add_argument("--cartesian-pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--cartesian-separation-loss-weight", type=float, default=0.0)
    parser.add_argument("--cartesian-pair-min-sep", type=float, default=1.0)
    parser.add_argument("--cartesian-pair-max-atoms", type=int, default=96)
    parser.add_argument("--cartesian-pair-complex-only", action="store_true")
    parser.add_argument("--relative-cartesian-separation-loss-weight", type=float, default=0.0)
    parser.add_argument("--relative-cartesian-separation-frac", type=float, default=0.5)
    parser.add_argument("--relative-cartesian-separation-max-atoms", type=int, default=96)
    parser.add_argument("--relative-cartesian-separation-complex-only", action="store_true")
    parser.add_argument("--local-cartesian-pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--local-cartesian-separation-loss-weight", type=float, default=0.0)
    parser.add_argument("--local-cartesian-pair-min-sep", type=float, default=1.0)
    parser.add_argument("--local-cartesian-pair-cutoff", type=float, default=4.0)
    parser.add_argument("--local-cartesian-pair-neighbors", type=int, default=4)
    parser.add_argument("--local-cartesian-pair-max-atoms", type=int, default=96)
    parser.add_argument("--local-cartesian-complex-only", action="store_true")
    parser.add_argument("--local-cartesian-separation-active-only", action="store_true")
    parser.add_argument("--coord-scale-floor", type=float, default=0.015)
    parser.add_argument("--coord-scale-ceiling", type=float, default=0.18)
    parser.add_argument("--lattice-scale-floor", type=float, default=0.02)
    parser.add_argument("--lattice-scale-ceiling", type=float, default=0.75)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_PROJECT / "artifacts/wyckoff_lookup_full.json")
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-sites", type=int, default=300)
    parser.add_argument("--seed", type=int, default=8017)
    parser.add_argument("--rows-ge7-quota", type=float, default=0.35)
    parser.add_argument("--rows-ge7-repeat", type=int, default=3)
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    experiment_id = str(args.experiment_id)
    out_dir = ROOT / "eval" / f"fixed_order_geom_sampler_smoke_{experiment_id}_{args.fold}"
    ckpt_dir = ROOT / "checkpoints" / f"fixed_order_geom_sampler_smoke_{experiment_id}_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    raw_train_limit = None if float(args.rows_ge7_quota) > 0.0 or int(args.rows_ge7_repeat) > 1 else args.train_limit
    train_records_raw, val_records, source_clean_dir = load_clean_records(args.fold, raw_train_limit, args.eval_limit)
    train_records, curriculum_info = select_train_curriculum(train_records_raw, args)
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in train_records + val_records}
    engine = OrbitEngine(args.lookup_json, sg_symbols)
    if args.eval_only_checkpoint is not None:
        model, vocab, mean_t, std_t, train_info = load_eval_only_model(
            args=args,
            checkpoint_path=args.eval_only_checkpoint,
            ckpt_dir=ckpt_dir,
        )
    else:
        model, vocab, mean_t, std_t, train_info = train_model(
            train_records=train_records,
            val_records=val_records,
            engine=engine,
            args=args,
            ckpt_dir=ckpt_dir,
        )
    generation_rows = generate_fixed_order_rows(
        records=val_records,
        model=model,
        vocab=vocab,
        engine=engine,
        mean_t=mean_t,
        std_t=std_t,
        args=args,
        partial_path=out_dir / "generations.partial.jsonl",
        progress_path=out_dir / "generation_progress.jsonl",
    )
    write_jsonl(out_dir / "generations.jsonl", generation_rows)
    eval_args = argparse.Namespace(
        eval_workers=int(args.eval_workers),
        bond_timeout_seconds=float(args.bond_timeout_seconds),
        valid_timeout_seconds=float(args.valid_timeout_seconds),
        match_timeout_seconds=float(args.rmsd_timeout_seconds),
        max_match_sites=int(args.max_sites),
        max_eval_sites=int(args.max_sites),
        parse_timeout_seconds=float(args.parse_timeout_seconds),
        sg_timeout_seconds=float(args.sg_timeout_seconds),
        sample_timeout_seconds=float(args.sample_timeout_seconds),
    )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        grouped[int(row["sample_index"])].append(row)
    metrics = evaluate_mode_with_hard_timeouts(
        mode=MODE,
        case_payload=case_payload_from_clean_records(val_records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    write_jsonl(out_dir / "metrics.jsonl", metrics)
    summary = summarize_fixed_order(val_records, generation_rows, metrics, top_ks=[1, 5, 20])
    write_reports(
        args=args,
        train_records=train_records,
        val_records=val_records,
        source_clean_dir=source_clean_dir,
        train_info=train_info,
        curriculum_info=curriculum_info,
        summary=summary,
        out_dir=out_dir,
        ckpt_dir=ckpt_dir,
        experiment_id=experiment_id,
        elapsed_seconds=time.time() - started,
    )
    print(json.dumps({"status": "ok", "experiment_id": experiment_id, "fold": args.fold, "summary": summary}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
