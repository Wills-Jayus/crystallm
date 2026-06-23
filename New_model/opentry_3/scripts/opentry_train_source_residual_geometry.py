#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_render_wyckoff_cifs_e07e08 as rb  # noqa: E402


ANGLE_SCALE = 180.0
ROW_NUMERIC_DIM = 13
GLOBAL_NUMERIC_DIM = 13


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def lattice_vector(lattice: dict[str, Any]) -> list[float]:
    return [
        math.log(max(float(lattice["a"]), 0.5)),
        math.log(max(float(lattice["b"]), 0.5)),
        math.log(max(float(lattice["c"]), 0.5)),
        float(lattice["alpha"]) / ANGLE_SCALE,
        float(lattice["beta"]) / ANGLE_SCALE,
        float(lattice["gamma"]) / ANGLE_SCALE,
    ]


def lattice_stats(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    values = torch.tensor([lattice_vector(dict(record["lattice"])) for record in records], dtype=torch.float32)
    mean = values.mean(dim=0)
    std = values.std(dim=0).clamp_min(1e-4)
    return mean.tolist(), std.tolist()


def normalize_lattice(lattice: dict[str, Any], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (torch.tensor(lattice_vector(lattice), dtype=torch.float32) - mean) / std


def build_vocabs(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    elements = sorted({str(e) for rec in records for e in rec["formula_counts"]} | {str(row["element"]) for rec in records for row in rec["wa_table"]})
    sgs = sorted({str(int(rec["sg"])) for rec in records}, key=lambda x: int(x))
    orbits = sorted({str(row["orbit_id"]) for rec in records for row in rec["wa_table"]})
    site_syms = sorted({str(row.get("site_symmetry") or "UNKNOWN") for rec in records for row in rec["wa_table"]})
    letters = sorted({f"{int(rec['sg'])}|{row.get('letter')}" for rec in records for row in rec["wa_table"]})
    return {
        "element": {value: idx + 1 for idx, value in enumerate(elements)},
        "sg": {value: idx + 1 for idx, value in enumerate(sgs)},
        "orbit": {value: idx + 1 for idx, value in enumerate(orbits)},
        "site_sym": {value: idx + 1 for idx, value in enumerate(site_syms)},
        "letter": {value: idx + 1 for idx, value in enumerate(letters)},
    }


def row_param_target(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    params = {str(k): float(v) % 1.0 for k, v in dict(row.get("free_params") or {}).items()}
    free_symbols = {str(sym) for sym in row.get("free_symbols") or params.keys()}
    values: list[float] = []
    mask: list[float] = []
    for symbol in ("x", "y", "z"):
        values.append(float(params.get(symbol, 0.0)))
        mask.append(1.0 if symbol in free_symbols and symbol in params else 0.0)
    return values, mask


def base_param_values(params: dict[int, dict[str, float]], row_idx: int, row: dict[str, Any]) -> tuple[list[float], list[float]]:
    raw_params = params.get(int(row_idx))
    if raw_params is None:
        raw_params = params.get(str(row_idx))  # JSONL caches stringify integer keys.
    row_params = {str(k): float(v) % 1.0 for k, v in dict(raw_params or {}).items()}
    free_symbols = {str(sym) for sym in row.get("free_symbols") or []}
    values: list[float] = []
    mask: list[float] = []
    for symbol in ("x", "y", "z"):
        values.append(float(row_params.get(symbol, 0.0)))
        mask.append(1.0 if symbol in free_symbols and symbol in row_params else 0.0)
    return values, mask


def complex_sample_weight(record: dict[str, Any], boost: float) -> float:
    if float(boost) <= 1.0:
        return 1.0
    hits = 0
    if int(record.get("n_sites", len(record.get("wa_table") or []))) >= 7:
        hits += 1
    if rb.gb.atom_count(record) >= 12:
        hits += 1
    if int(record.get("num_elements", len(record.get("formula_counts") or {}))) >= 4:
        hits += 1
    return 1.0 + (float(boost) - 1.0) * float(hits)


def build_selector(train_records: list[dict[str, Any]]) -> rb.OpentryGeometrySelector:
    return rb.OpentryGeometrySelector(train_records, rb.gb.GeometryIndex(train_records, []))


def fast_chem_quality_candidates(selector: rb.OpentryGeometrySelector, target: dict[str, Any], k: int) -> list[dict[str, Any]]:
    sg = int(target["sg"])
    atom_bucket = rb.gb.atom_bucket(rb.gb.atom_count(target))
    pool_limit = max(int(k) * 8, 64)
    pools: list[list[dict[str, Any]]] = [
        selector.by_sg_row.get((sg, rb.gb.row_count(target)), []),
        selector.by_sg_atom_bucket.get((sg, atom_bucket), []),
        selector.by_sg.get(sg, []),
        selector.by_crystal_bucket.get((rb.gb.crystal_system(sg), atom_bucket), []),
        selector.base_index.candidates(target, "row_conditioned_knn", max(int(k) * 4, 32)),
    ]
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_pool in pools:
        if not source_pool:
            continue
        reduced = source_pool
        if len(reduced) > pool_limit:
            reduced = sorted(reduced, key=lambda rec: selector.cheap_chem_source_distance(target, rec))[:pool_limit]
        for rec in reduced:
            sid = str(rec.get("sample_id"))
            if sid in seen:
                continue
            seen.add(sid)
            pool.append(rec)
        if len(pool) >= max(int(k) * 10, 80):
            break
    scored: list[tuple[float, float, float, dict[str, Any]]] = []
    for rec in pool:
        chem_row_dist = selector.row_condition_chem_distance(target, rec)
        align_cost = selector.source_alignment_cost(target, rec, chemical=True)
        quality_penalty = selector.source_quality_penalty(rec)
        score = chem_row_dist + 0.35 * align_cost + quality_penalty
        scored.append((score, chem_row_dist, quality_penalty, rec))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [rec for _, _, _, rec in scored[: max(int(k), 1)]]


def select_source(
    selector: rb.OpentryGeometrySelector,
    record: dict[str, Any],
    *,
    source_pool_k: int,
    exclude_sample_id: str | None,
) -> dict[str, Any] | None:
    sources = fast_chem_quality_candidates(selector, record, int(source_pool_k))
    for source in sources:
        if exclude_sample_id is not None and str(source.get("sample_id")) == str(exclude_sample_id):
            continue
        return source
    return None


def make_examples(
    records: list[dict[str, Any]],
    selector: rb.OpentryGeometrySelector,
    *,
    source_pool_k: int,
    exclude_self: bool,
    vpa_strength: float,
    complex_weight: float,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        exclude = str(record.get("sample_id")) if exclude_self else None
        source = select_source(selector, record, source_pool_k=int(source_pool_k), exclude_sample_id=exclude)
        if source is None:
            continue
        params, align_cost = selector.source_aligned_params(record, source, 0, chemical=True)
        lattice = selector.vpa_calibrated_lattice(source, record, strength=float(vpa_strength))
        examples.append(
            {
                "record": record,
                "base_params": params,
                "base_lattice": lattice,
                "source_sample_id": str(source.get("sample_id")),
                "source_distance": float(rb.gb.row_condition_distance(record, source)),
                "align_cost": float(align_cost),
                "sample_weight": complex_sample_weight(record, float(complex_weight)),
            }
        )
    return examples


class ResidualGeometryDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def load_examples_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = ensure_under_opentry(path)
    examples: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def collate_residual(
    examples: list[dict[str, Any]],
    *,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = len(examples)
    max_formula = max(1, max(len(ex["record"]["formula_counts"]) for ex in examples))
    max_rows = max(1, max(len(ex["record"]["wa_table"]) for ex in examples))

    formula_element_ids = torch.zeros((batch_size, max_formula), dtype=torch.long)
    formula_weights = torch.zeros((batch_size, max_formula), dtype=torch.float32)
    sg_ids = torch.zeros((batch_size,), dtype=torch.long)
    global_numeric = torch.zeros((batch_size, GLOBAL_NUMERIC_DIM), dtype=torch.float32)
    row_element_ids = torch.zeros((batch_size, max_rows), dtype=torch.long)
    row_orbit_ids = torch.zeros((batch_size, max_rows), dtype=torch.long)
    row_site_sym_ids = torch.zeros((batch_size, max_rows), dtype=torch.long)
    row_letter_ids = torch.zeros((batch_size, max_rows), dtype=torch.long)
    row_numeric = torch.zeros((batch_size, max_rows, ROW_NUMERIC_DIM), dtype=torch.float32)
    row_mask = torch.zeros((batch_size, max_rows), dtype=torch.float32)
    base_coord_values = torch.zeros((batch_size, max_rows, 3), dtype=torch.float32)
    coord_values = torch.zeros((batch_size, max_rows, 3), dtype=torch.float32)
    coord_mask = torch.zeros((batch_size, max_rows, 3), dtype=torch.float32)
    base_lattice_values = torch.zeros((batch_size, 6), dtype=torch.float32)
    lattice_values = torch.zeros((batch_size, 6), dtype=torch.float32)
    sample_weight = torch.ones((batch_size,), dtype=torch.float32)

    for i, example in enumerate(examples):
        record = example["record"]
        rows = rb.v2.canonical_rows(record)
        counts = {str(k): int(v) for k, v in dict(record["formula_counts"]).items()}
        total_atoms = max(1, sum(counts.values()))
        sg = int(record["sg"])
        sg_ids[i] = vocabs["sg"].get(str(sg), 0)
        for j, (element, count) in enumerate(sorted(counts.items())):
            formula_element_ids[i, j] = vocabs["element"].get(str(element), 0)
            formula_weights[i, j] = float(count) / float(total_atoms)
        base_lattice = normalize_lattice(dict(example["base_lattice"]), lattice_mean, lattice_std)
        target_lattice = normalize_lattice(dict(record["lattice"]), lattice_mean, lattice_std)
        base_lattice_values[i] = base_lattice
        lattice_values[i] = target_lattice
        sample_weight[i] = float(example.get("sample_weight", 1.0))
        global_values = [
            float(sg) / 230.0,
            float(total_atoms) / 300.0,
            float(len(rows)) / 64.0,
            float(len(counts)) / 12.0,
            1.0 if len(rows) >= 7 else 0.0,
            min(4.0, float(example.get("align_cost", 0.0))) / 4.0,
            min(8.0, float(example.get("source_distance", 0.0))) / 8.0,
        ] + [float(x) for x in base_lattice.tolist()]
        global_numeric[i] = torch.tensor(global_values, dtype=torch.float32)

        for j, row in enumerate(rows):
            row_mask[i, j] = 1.0
            row_element_ids[i, j] = vocabs["element"].get(str(row["element"]), 0)
            row_orbit_ids[i, j] = vocabs["orbit"].get(str(row["orbit_id"]), 0)
            row_site_sym_ids[i, j] = vocabs["site_sym"].get(str(row.get("site_symmetry") or "UNKNOWN"), 0)
            row_letter_ids[i, j] = vocabs["letter"].get(f"{sg}|{row.get('letter')}", 0)
            free_symbols = {str(sym) for sym in row.get("free_symbols") or []}
            base_vals, base_mask = base_param_values(example["base_params"], j, row)
            target_vals, target_mask = row_param_target(row)
            base_coord_values[i, j] = torch.tensor(base_vals, dtype=torch.float32)
            coord_values[i, j] = torch.tensor(target_vals, dtype=torch.float32)
            coord_mask[i, j] = torch.tensor(target_mask, dtype=torch.float32)
            row_numeric[i, j] = torch.tensor(
                [
                    float(int(row.get("multiplicity", 1))) / 64.0,
                    float(len(free_symbols)) / 3.0,
                    1.0 if "x" in free_symbols else 0.0,
                    1.0 if "y" in free_symbols else 0.0,
                    1.0 if "z" in free_symbols else 0.0,
                    float(j) / 64.0,
                    *base_vals,
                    *base_mask,
                    1.0 if example.get("source_sample_id") else 0.0,
                ],
                dtype=torch.float32,
            )

    return {
        "formula_element_ids": formula_element_ids,
        "formula_weights": formula_weights,
        "sg_ids": sg_ids,
        "global_numeric": global_numeric,
        "row_element_ids": row_element_ids,
        "row_orbit_ids": row_orbit_ids,
        "row_site_sym_ids": row_site_sym_ids,
        "row_letter_ids": row_letter_ids,
        "row_numeric": row_numeric,
        "row_mask": row_mask,
        "base_coord_values": base_coord_values,
        "coord_values": coord_values,
        "coord_mask": coord_mask,
        "base_lattice_values": base_lattice_values,
        "lattice_values": lattice_values,
        "sample_weight": sample_weight,
    }


class SourceResidualGeometryNet(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        hidden_dim: int = 256,
        emb_dim: int = 64,
        lattice_delta_scale: float = 0.75,
        coord_delta_scale: float = 0.35,
        enable_delta_gate: bool = False,
        gate_bias_init: float = -1.0,
    ) -> None:
        super().__init__()
        self.lattice_delta_scale = float(lattice_delta_scale)
        self.coord_delta_scale = float(coord_delta_scale)
        self.enable_delta_gate = bool(enable_delta_gate)
        self.element_emb = nn.Embedding(vocab_sizes["element"] + 1, emb_dim)
        self.sg_emb = nn.Embedding(vocab_sizes["sg"] + 1, emb_dim)
        self.orbit_emb = nn.Embedding(vocab_sizes["orbit"] + 1, emb_dim)
        self.site_sym_emb = nn.Embedding(vocab_sizes["site_sym"] + 1, emb_dim // 2)
        self.letter_emb = nn.Embedding(vocab_sizes["letter"] + 1, emb_dim // 2)
        row_in = emb_dim * 2 + emb_dim // 2 + emb_dim // 2 + ROW_NUMERIC_DIM
        self.row_encoder = nn.Sequential(
            nn.LayerNorm(row_in),
            nn.Linear(row_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        ctx_in = hidden_dim + emb_dim * 2 + GLOBAL_NUMERIC_DIM
        self.context = nn.Sequential(
            nn.LayerNorm(ctx_in),
            nn.Linear(ctx_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.lattice_delta_head = nn.Linear(hidden_dim, 6)
        self.coord_delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        if self.enable_delta_gate:
            self.lattice_gate_head = nn.Linear(hidden_dim, 6)
            self.coord_gate_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
            nn.init.constant_(self.lattice_gate_head.bias, float(gate_bias_init))
            last = self.coord_gate_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.constant_(last.bias, float(gate_bias_init))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        formula_emb = self.element_emb(batch["formula_element_ids"])
        weights = batch["formula_weights"]
        formula_vec = (formula_emb * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        sg_vec = self.sg_emb(batch["sg_ids"])
        row_vec = torch.cat(
            [
                self.element_emb(batch["row_element_ids"]),
                self.orbit_emb(batch["row_orbit_ids"]),
                self.site_sym_emb(batch["row_site_sym_ids"]),
                self.letter_emb(batch["row_letter_ids"]),
                batch["row_numeric"],
            ],
            dim=-1,
        )
        row_h = self.row_encoder(row_vec)
        row_mask = batch["row_mask"].unsqueeze(-1)
        row_pool = (row_h * row_mask).sum(dim=1) / row_mask.sum(dim=1).clamp_min(1.0)
        ctx = self.context(torch.cat([row_pool, formula_vec, sg_vec, batch["global_numeric"]], dim=-1))
        lattice_delta = self.lattice_delta_scale * torch.tanh(self.lattice_delta_head(ctx))
        aux: dict[str, torch.Tensor] = {}
        if self.enable_delta_gate:
            lattice_gate = torch.sigmoid(self.lattice_gate_head(ctx))
            lattice_delta = lattice_delta * lattice_gate
            aux["lattice_gate"] = lattice_gate
        lattice_pred = batch["base_lattice_values"] + lattice_delta
        ctx_rows = ctx.unsqueeze(1).expand(-1, row_h.shape[1], -1)
        coord_features = torch.cat([row_h, ctx_rows], dim=-1)
        coord_delta = self.coord_delta_scale * torch.tanh(self.coord_delta_head(coord_features))
        if self.enable_delta_gate:
            coord_gate = torch.sigmoid(self.coord_gate_head(coord_features))
            coord_delta = coord_delta * coord_gate
            aux["coord_gate"] = coord_gate
        coord_pred = batch["base_coord_values"] + coord_delta
        return lattice_pred, coord_pred, aux


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loss_fn(
    lattice_pred: torch.Tensor,
    coord_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    coord_weight: float,
    *,
    aux: dict[str, torch.Tensor] | None = None,
    gate_l1_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = batch["sample_weight"].clamp_min(1e-6)
    lattice_per = (lattice_pred - batch["lattice_values"]).square().mean(dim=1)
    lattice_loss = (lattice_per * weights).sum() / weights.sum()
    diff = torch.abs(coord_pred - batch["coord_values"])
    wrapped = torch.minimum(diff, 1.0 - diff)
    coord_num = (wrapped.square() * batch["coord_mask"]).sum(dim=(1, 2))
    coord_den = batch["coord_mask"].sum(dim=(1, 2)).clamp_min(1.0)
    coord_per = coord_num / coord_den
    coord_loss = (coord_per * weights).sum() / weights.sum()
    loss = lattice_loss + float(coord_weight) * coord_loss
    parts = {"lattice_loss": float(lattice_loss.detach().cpu()), "coord_loss": float(coord_loss.detach().cpu())}
    if aux and float(gate_l1_weight) > 0.0:
        gate_terms: list[torch.Tensor] = []
        if "lattice_gate" in aux:
            gate_terms.append(aux["lattice_gate"].mean())
        if "coord_gate" in aux:
            coord_gate = aux["coord_gate"]
            gate_num = (coord_gate * batch["coord_mask"]).sum()
            gate_den = batch["coord_mask"].sum().clamp_min(1.0)
            gate_terms.append(gate_num / gate_den)
        if gate_terms:
            gate_loss = torch.stack(gate_terms).mean()
            loss = loss + float(gate_l1_weight) * gate_loss
            parts["gate_l1_loss"] = float(gate_loss.detach().cpu())
            if "lattice_gate" in aux:
                parts["lattice_gate_mean"] = float(aux["lattice_gate"].detach().mean().cpu())
            if "coord_gate" in aux:
                mask = batch["coord_mask"]
                parts["coord_gate_mean"] = float(((aux["coord_gate"].detach() * mask).sum() / mask.sum().clamp_min(1.0)).cpu())
    return loss, parts


@torch.no_grad()
def evaluate(
    model: SourceResidualGeometryNet,
    examples: list[dict[str, Any]],
    *,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    coord_weight: float,
    gate_l1_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    lattice_losses: list[float] = []
    coord_losses: list[float] = []
    loader = DataLoader(
        ResidualGeometryDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda xs: collate_residual(xs, vocabs=vocabs, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu()),
        num_workers=0,
    )
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        lattice_pred, coord_pred, aux = model(batch)
        loss, parts = loss_fn(lattice_pred, coord_pred, batch, coord_weight=float(coord_weight), aux=aux, gate_l1_weight=float(gate_l1_weight))
        losses.append(float(loss.detach().cpu()))
        lattice_losses.append(parts["lattice_loss"])
        coord_losses.append(parts["coord_loss"])
    out = {
        "loss": float(sum(losses) / max(1, len(losses))),
        "lattice_loss": float(sum(lattice_losses) / max(1, len(lattice_losses))),
        "coord_loss": float(sum(coord_losses) / max(1, len(coord_losses))),
    }
    if hasattr(model, "enable_delta_gate") and bool(model.enable_delta_gate):
        gate_l1_values: list[float] = []
        lattice_gate_values: list[float] = []
        coord_gate_values: list[float] = []
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            _, _, aux = model(batch)
            if "lattice_gate" in aux:
                lattice_gate_values.append(float(aux["lattice_gate"].detach().mean().cpu()))
            if "coord_gate" in aux:
                mask = batch["coord_mask"]
                coord_gate_values.append(float(((aux["coord_gate"].detach() * mask).sum() / mask.sum().clamp_min(1.0)).cpu()))
            if lattice_gate_values or coord_gate_values:
                vals = lattice_gate_values[-1:] + coord_gate_values[-1:]
                gate_l1_values.append(float(sum(vals) / len(vals)))
        out["gate_l1_loss"] = float(sum(gate_l1_values) / max(1, len(gate_l1_values)))
        out["lattice_gate_mean"] = float(sum(lattice_gate_values) / max(1, len(lattice_gate_values))) if lattice_gate_values else 0.0
        out["coord_gate_mean"] = float(sum(coord_gate_values) / max(1, len(coord_gate_values))) if coord_gate_values else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a train-only source-conditioned residual geometry model for opentry_3.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coord-weight", type=float, default=4.0)
    parser.add_argument("--complex-weight", type=float, default=3.0)
    parser.add_argument("--source-pool-k", type=int, default=24)
    parser.add_argument("--vpa-strength", type=float, default=0.5)
    parser.add_argument("--lattice-delta-scale", type=float, default=0.75)
    parser.add_argument("--coord-delta-scale", type=float, default=0.35)
    parser.add_argument("--enable-delta-gate", action="store_true")
    parser.add_argument("--gate-bias-init", type=float, default=-1.0)
    parser.add_argument("--gate-l1-weight", type=float, default=0.0)
    parser.add_argument("--cached-examples-dir", type=Path, default=None)
    parser.add_argument("--train-examples", type=Path, default=None)
    parser.add_argument("--val-examples", type=Path, default=None)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.set_float32_matmul_precision("high")

    cached_dir = ensure_under_opentry(args.cached_examples_dir) if args.cached_examples_dir is not None else None
    train_examples_path = args.train_examples
    val_examples_path = args.val_examples
    if cached_dir is not None:
        train_examples_path = train_examples_path or cached_dir / "train_examples.jsonl"
        val_examples_path = val_examples_path or cached_dir / "val_examples.jsonl"

    if train_examples_path is not None or val_examples_path is not None:
        if train_examples_path is None or val_examples_path is None:
            raise SystemExit("--train-examples and --val-examples must be provided together")
        train_examples = load_examples_jsonl(train_examples_path)
        val_examples = load_examples_jsonl(val_examples_path)
        train_records = [dict(ex["record"]) for ex in train_examples]
        val_records = [dict(ex["record"]) for ex in val_examples]
        example_source = {
            "mode": "cached",
            "train_examples": str(ensure_under_opentry(train_examples_path)),
            "val_examples": str(ensure_under_opentry(val_examples_path)),
        }
    else:
        train_records = rb.load_records(args.data_root / "train.jsonl")
        val_records = rb.load_records(args.data_root / "val.jsonl")
        if int(args.max_train_records) > 0:
            train_records = train_records[: int(args.max_train_records)]
        if int(args.max_val_records) > 0:
            val_records = val_records[: int(args.max_val_records)]

        selector = build_selector(train_records)
        train_examples = make_examples(
            train_records,
            selector,
            source_pool_k=int(args.source_pool_k),
            exclude_self=True,
            vpa_strength=float(args.vpa_strength),
            complex_weight=float(args.complex_weight),
        )
        val_examples = make_examples(
            val_records,
            selector,
            source_pool_k=int(args.source_pool_k),
            exclude_self=False,
            vpa_strength=float(args.vpa_strength),
            complex_weight=1.0,
        )
        example_source = {"mode": "online"}
    if not train_examples or not val_examples:
        raise SystemExit("No train/val examples built")

    vocabs = build_vocabs(train_records)
    lattice_mean, lattice_std = lattice_stats(train_records)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = SourceResidualGeometryNet(
        {name: len(vocab) for name, vocab in vocabs.items()},
        hidden_dim=int(args.hidden_dim),
        emb_dim=int(args.emb_dim),
        lattice_delta_scale=float(args.lattice_delta_scale),
        coord_delta_scale=float(args.coord_delta_scale),
        enable_delta_gate=bool(args.enable_delta_gate),
        gate_bias_init=float(args.gate_bias_init),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    weights = [float(ex.get("sample_weight", 1.0)) for ex in train_examples]
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(train_examples), replacement=True)
    loader = DataLoader(
        ResidualGeometryDataset(train_examples),
        batch_size=int(args.batch_size),
        sampler=sampler,
        collate_fn=lambda xs: collate_residual(xs, vocabs=vocabs, lattice_mean=mean_t, lattice_std=std_t),
        num_workers=0,
    )

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    best_path = out_dir / "ckpt_best.pt"
    last_path = out_dir / "ckpt_last.pt"
    print(
        json.dumps(
            {
                "train_records": len(train_records),
                "val_records": len(val_records),
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "device": str(device),
                "epochs": int(args.epochs),
                "source_pool_k": int(args.source_pool_k),
                "vpa_strength": float(args.vpa_strength),
                "lattice_delta_scale": float(args.lattice_delta_scale),
                "coord_delta_scale": float(args.coord_delta_scale),
                "enable_delta_gate": bool(args.enable_delta_gate),
                "gate_bias_init": float(args.gate_bias_init),
                "gate_l1_weight": float(args.gate_l1_weight),
                "example_source": example_source,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        train_lattice = 0.0
        train_coord = 0.0
        train_gate = 0.0
        train_lattice_gate = 0.0
        train_coord_gate = 0.0
        steps = 0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            lattice_pred, coord_pred, aux = model(batch)
            loss, parts = loss_fn(
                lattice_pred,
                coord_pred,
                batch,
                coord_weight=float(args.coord_weight),
                aux=aux,
                gate_l1_weight=float(args.gate_l1_weight),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_lattice += parts["lattice_loss"]
            train_coord += parts["coord_loss"]
            train_gate += float(parts.get("gate_l1_loss", 0.0))
            train_lattice_gate += float(parts.get("lattice_gate_mean", 0.0))
            train_coord_gate += float(parts.get("coord_gate_mean", 0.0))
            steps += 1
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, steps),
            "train_lattice_loss": train_lattice / max(1, steps),
            "train_coord_loss": train_coord / max(1, steps),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        if bool(args.enable_delta_gate):
            row.update(
                {
                    "train_gate_l1_loss": train_gate / max(1, steps),
                    "train_lattice_gate_mean": train_lattice_gate / max(1, steps),
                    "train_coord_gate_mean": train_coord_gate / max(1, steps),
                }
            )
        if epoch == 1 or epoch % max(1, int(args.eval_every)) == 0 or epoch == int(args.epochs):
            val = evaluate(
                model,
                val_examples,
                vocabs=vocabs,
                lattice_mean=mean_t,
                lattice_std=std_t,
                device=device,
                batch_size=int(args.batch_size),
                coord_weight=float(args.coord_weight),
                gate_l1_weight=float(args.gate_l1_weight),
            )
            row.update({f"val_{key}": value for key, value in val.items()})
            if val["loss"] < best_val:
                best_val = float(val["loss"])
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocabs": vocabs,
                        "lattice_mean": lattice_mean,
                        "lattice_std": lattice_std,
                        "config": jsonable_args(args),
                        "best_epoch": epoch,
                        "best_val": best_val,
                    },
                    best_path,
                )
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "lattice_mean": lattice_mean,
            "lattice_std": lattice_std,
            "config": jsonable_args(args),
            "best_epoch": None,
            "best_val": best_val,
        },
        last_path,
    )
    write_json(
        out_dir / "training_summary.json",
        {
            "config": jsonable_args(args),
            "best_val_loss": best_val,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "history": history,
            "example_source": example_source,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "vocab_sizes": {name: len(vocab) for name, vocab in vocabs.items()},
            "mean_sample_weight": float(sum(weights) / max(1, len(weights))),
            "max_sample_weight": float(max(weights) if weights else 1.0),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
