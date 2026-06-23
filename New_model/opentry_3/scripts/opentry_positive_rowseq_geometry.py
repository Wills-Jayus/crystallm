#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import OrderedDict
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

import opentry_selfscore_rendered_cifs as selfscore  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402


ANGLE_SCALE = 180.0
CELL_RE = re.compile(r"^(_cell_(?:length_[abc]|angle_(?:alpha|beta|gamma)|volume))\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")
ATOM_LABEL_RE = re.compile(r"^[A-Z][A-Za-z]?(\d+)_0$")


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_wa_key(wa_key: str) -> list[tuple[str, str]]:
    chunks = str(wa_key or "").split("|setting=")
    out: list[tuple[str, str]] = []
    for idx, chunk in enumerate(chunks):
        text = chunk if idx == 0 else "setting=" + chunk
        if ":" not in text:
            continue
        orbit_id, element = text.rsplit(":", 1)
        out.append((orbit_id, element))
    return out


def parse_rendered_cif_targets(cif: str) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    cell: dict[str, float] = {}
    coords: dict[int, dict[str, float]] = {}
    in_atom_loop = False
    for line in str(cif or "").splitlines():
        text = line.strip()
        match = CELL_RE.match(text)
        if match:
            cell[match.group(1).replace("_cell_", "")] = float(match.group(2))
            continue
        if text.startswith("loop_"):
            in_atom_loop = False
            continue
        if text.startswith("_atom_site_type_symbol"):
            in_atom_loop = True
            continue
        if not in_atom_loop or text.startswith("_") or not text:
            continue
        parts = text.split()
        if len(parts) < 7:
            continue
        label_match = ATOM_LABEL_RE.match(str(parts[1]))
        if label_match is None:
            continue
        row_idx = int(label_match.group(1))
        if row_idx in coords:
            continue
        try:
            coords[row_idx] = {
                "x": float(parts[3]) % 1.0,
                "y": float(parts[4]) % 1.0,
                "z": float(parts[5]) % 1.0,
            }
        except Exception:
            continue
    required = {"length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"}
    if not required.issubset(cell):
        raise ValueError("missing_cell")
    return cell, coords


def lattice_target_from_cell(cell: dict[str, float]) -> list[float]:
    return [
        math.log(max(0.1, float(cell["length_a"]))),
        math.log(max(0.1, float(cell["length_b"]))),
        math.log(max(0.1, float(cell["length_c"]))),
        float(cell["angle_alpha"]) / ANGLE_SCALE,
        float(cell["angle_beta"]) / ANGLE_SCALE,
        float(cell["angle_gamma"]) / ANGLE_SCALE,
    ]


def load_structured_by_id(data_root: Path, split: str) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in read_jsonl(data_root / f"{split}.jsonl")}


def rows_from_wa_key(engine: OrbitEngine, wa_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit_id, element in split_wa_key(wa_key):
        orbit = engine.get_orbit_by_id(str(orbit_id))
        rows.append(
            {
                "element": str(element),
                "orbit_id": orbit.canonical_orbit_id,
                "multiplicity": int(orbit.multiplicity),
                "letter": orbit.letter,
                "enumeration": orbit.enumeration,
                "site_symmetry": orbit.site_symmetry,
                "free_symbols": list(orbit.free_symbols),
            }
        )
    return v2.canonical_rows({"wa_table": rows})


def make_pseudo_record(base: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    formula_counts = {str(k): int(v) for k, v in dict(base["formula_counts"]).items()}
    record = {
        "sample_id": str(base["sample_id"]),
        "formula_counts": formula_counts,
        "sg": int(base["sg"]),
        "sg_symbol": str(base.get("sg_symbol") or ""),
        "atom_count": int(base.get("atom_count") or sum(formula_counts.values())),
        "n_sites": len(rows),
        "num_elements": len(formula_counts),
        "wa_table": rows,
        "lattice": dict(base.get("lattice") or {}),
    }
    skel, wa = v2.canonical_keys_from_rows(rows)
    record["canonical_skeleton_key"] = skel
    record["canonical_wa_key"] = wa
    return record


def build_examples(
    *,
    features_path: Path,
    structured_by_id: dict[str, dict[str, Any]],
    engine: OrbitEngine,
    max_examples: int,
    min_positive_row_count: int,
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    stats = {
        "feature_rows": 0,
        "positive_rows": 0,
        "used_examples": 0,
        "skipped_not_positive": 0,
        "skipped_missing_record": 0,
        "skipped_parse": 0,
        "skipped_too_many_rows": 0,
        "rows_ge_7_examples": 0,
    }
    for feature in read_jsonl(features_path):
        stats["feature_rows"] += 1
        if not bool(feature.get("label_match")):
            stats["skipped_not_positive"] += 1
            continue
        stats["positive_rows"] += 1
        if int(feature.get("target_row_count", 0) or 0) < int(min_positive_row_count):
            continue
        base = structured_by_id.get(str(feature.get("sample_id") or ""))
        if base is None:
            stats["skipped_missing_record"] += 1
            continue
        try:
            rows = rows_from_wa_key(engine, str(feature.get("canonical_wa_key") or ""))
            if len(rows) > int(max_rows):
                stats["skipped_too_many_rows"] += 1
                continue
            cell, coords = parse_rendered_cif_targets(str(feature.get("cif") or ""))
        except Exception:
            stats["skipped_parse"] += 1
            continue
        record = make_pseudo_record(base, rows)
        coord_values: list[list[float]] = []
        coord_mask: list[list[float]] = []
        for row_idx, row in enumerate(rows):
            row_coord = coords.get(row_idx, {})
            symbols = {str(sym) for sym in row.get("free_symbols") or []}
            coord_values.append([float(row_coord.get(sym, 0.0)) % 1.0 for sym in ("x", "y", "z")])
            coord_mask.append([1.0 if sym in symbols and sym in row_coord else 0.0 for sym in ("x", "y", "z")])
        row_count = len(rows)
        atom_count = int(record["atom_count"])
        examples.append(
            {
                "sample_id": str(record["sample_id"]),
                "record": record,
                "lattice_target": lattice_target_from_cell(cell),
                "coord_values": coord_values,
                "coord_mask": coord_mask,
                "label_rmsd": feature.get("label_rmsd"),
                "source_feature_rank": feature.get("rank"),
                "source_sample_id": feature.get("source_sample_id"),
                "sample_weight": (8.0 if row_count >= 7 else 1.0) * (1.5 if atom_count >= 12 else 1.0),
            }
        )
        stats["used_examples"] += 1
        stats["rows_ge_7_examples"] += int(row_count >= 7)
        if int(max_examples) > 0 and len(examples) >= int(max_examples):
            break
    return examples, stats


def build_vocabs(examples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    records = [ex["record"] for ex in examples]
    elements = sorted({str(el) for rec in records for el in rec["formula_counts"]} | {str(row["element"]) for rec in records for row in rec["wa_table"]})
    sgs = sorted({str(int(rec["sg"])) for rec in records}, key=lambda value: int(value))
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


def lattice_stats(examples: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    values = torch.tensor([ex["lattice_target"] for ex in examples], dtype=torch.float32)
    return values.mean(dim=0).tolist(), values.std(dim=0).clamp_min(1.0e-4).tolist()


class PositiveGeometryDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_positive(
    examples: list[dict[str, Any]],
    *,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    max_rows_limit: int,
) -> dict[str, torch.Tensor]:
    batch = len(examples)
    max_formula = max(1, max(len(ex["record"]["formula_counts"]) for ex in examples))
    max_rows = min(int(max_rows_limit), max(1, max(len(ex["record"]["wa_table"]) for ex in examples)))
    formula_element_ids = torch.zeros((batch, max_formula), dtype=torch.long)
    formula_weights = torch.zeros((batch, max_formula), dtype=torch.float32)
    sg_ids = torch.zeros((batch,), dtype=torch.long)
    numeric = torch.zeros((batch, 5), dtype=torch.float32)
    row_element_ids = torch.zeros((batch, max_rows), dtype=torch.long)
    row_orbit_ids = torch.zeros((batch, max_rows), dtype=torch.long)
    row_site_sym_ids = torch.zeros((batch, max_rows), dtype=torch.long)
    row_letter_ids = torch.zeros((batch, max_rows), dtype=torch.long)
    row_numeric = torch.zeros((batch, max_rows, 6), dtype=torch.float32)
    row_mask = torch.zeros((batch, max_rows), dtype=torch.float32)
    coord_values = torch.zeros((batch, max_rows, 3), dtype=torch.float32)
    coord_mask = torch.zeros((batch, max_rows, 3), dtype=torch.float32)
    lattice_values = torch.zeros((batch, 6), dtype=torch.float32)
    sample_weight = torch.ones((batch,), dtype=torch.float32)
    for i, ex in enumerate(examples):
        rec = ex["record"]
        counts = {str(k): int(v) for k, v in rec["formula_counts"].items()}
        total_atoms = max(1, sum(counts.values()))
        sg = int(rec["sg"])
        sg_ids[i] = vocabs["sg"].get(str(sg), 0)
        for j, (element, count) in enumerate(sorted(counts.items())):
            formula_element_ids[i, j] = vocabs["element"].get(str(element), 0)
            formula_weights[i, j] = float(count) / float(total_atoms)
        rows = list(rec["wa_table"])[:max_rows]
        numeric[i] = torch.tensor(
            [
                float(sg) / 230.0,
                float(total_atoms) / 300.0,
                float(len(rec["wa_table"])) / 96.0,
                float(len(counts)) / 12.0,
                1.0 if len(rec["wa_table"]) >= 7 else 0.0,
            ],
            dtype=torch.float32,
        )
        lattice_values[i] = (torch.tensor(ex["lattice_target"], dtype=torch.float32) - lattice_mean) / lattice_std
        for j, row in enumerate(rows):
            row_mask[i, j] = 1.0
            row_element_ids[i, j] = vocabs["element"].get(str(row["element"]), 0)
            row_orbit_ids[i, j] = vocabs["orbit"].get(str(row["orbit_id"]), 0)
            row_site_sym_ids[i, j] = vocabs["site_sym"].get(str(row.get("site_symmetry") or "UNKNOWN"), 0)
            row_letter_ids[i, j] = vocabs["letter"].get(f"{sg}|{row.get('letter')}", 0)
            free_symbols = {str(sym) for sym in row.get("free_symbols") or []}
            row_numeric[i, j] = torch.tensor(
                [
                    float(int(row.get("multiplicity", 1))) / 64.0,
                    float(len(free_symbols)) / 3.0,
                    1.0 if "x" in free_symbols else 0.0,
                    1.0 if "y" in free_symbols else 0.0,
                    1.0 if "z" in free_symbols else 0.0,
                    float(j) / 96.0,
                ],
                dtype=torch.float32,
            )
            coord_values[i, j] = torch.tensor(ex["coord_values"][j], dtype=torch.float32)
            coord_mask[i, j] = torch.tensor(ex["coord_mask"][j], dtype=torch.float32)
        sample_weight[i] = float(ex.get("sample_weight", 1.0))
    return {
        "formula_element_ids": formula_element_ids,
        "formula_weights": formula_weights,
        "sg_ids": sg_ids,
        "numeric": numeric,
        "row_element_ids": row_element_ids,
        "row_orbit_ids": row_orbit_ids,
        "row_site_sym_ids": row_site_sym_ids,
        "row_letter_ids": row_letter_ids,
        "row_numeric": row_numeric,
        "row_mask": row_mask,
        "coord_values": coord_values,
        "coord_mask": coord_mask,
        "lattice_values": lattice_values,
        "sample_weight": sample_weight,
    }


class RowSeqPositiveGeometryNet(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], hidden_dim: int = 192, emb_dim: int = 48, layers: int = 3, heads: int = 4, max_rows: int = 96) -> None:
        super().__init__()
        self.max_rows = int(max_rows)
        self.element_emb = nn.Embedding(vocab_sizes["element"] + 1, emb_dim)
        self.sg_emb = nn.Embedding(vocab_sizes["sg"] + 1, emb_dim)
        self.orbit_emb = nn.Embedding(vocab_sizes["orbit"] + 1, emb_dim)
        self.site_sym_emb = nn.Embedding(vocab_sizes["site_sym"] + 1, emb_dim // 2)
        self.letter_emb = nn.Embedding(vocab_sizes["letter"] + 1, emb_dim // 2)
        row_in = emb_dim * 2 + emb_dim // 2 + emb_dim // 2 + 6
        ctx_in = emb_dim * 2 + 5
        self.row_proj = nn.Sequential(nn.LayerNorm(row_in), nn.Linear(row_in, hidden_dim), nn.GELU())
        self.ctx_proj = nn.Sequential(nn.LayerNorm(ctx_in), nn.Linear(ctx_in, hidden_dim), nn.GELU())
        self.pos_emb = nn.Embedding(max_rows + 1, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4, dropout=0.08, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.lattice_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 12))
        self.coord_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        formula_emb = self.element_emb(batch["formula_element_ids"])
        formula_ctx = (formula_emb * batch["formula_weights"].unsqueeze(-1)).sum(dim=1)
        sg_ctx = self.sg_emb(batch["sg_ids"])
        ctx = self.ctx_proj(torch.cat([formula_ctx, sg_ctx, batch["numeric"]], dim=-1))
        row_input = torch.cat(
            [
                self.element_emb(batch["row_element_ids"]),
                self.orbit_emb(batch["row_orbit_ids"]),
                self.site_sym_emb(batch["row_site_sym_ids"]),
                self.letter_emb(batch["row_letter_ids"]),
                batch["row_numeric"],
            ],
            dim=-1,
        )
        row_h = self.row_proj(row_input) + ctx.unsqueeze(1)
        positions = torch.arange(row_h.shape[1], device=row_h.device).clamp_max(self.max_rows)
        row_h = row_h + self.pos_emb(positions).unsqueeze(0)
        cls = self.cls.expand(row_h.shape[0], -1, -1) + ctx.unsqueeze(1)
        tokens = torch.cat([cls, row_h], dim=1)
        padding = torch.cat([torch.zeros((row_h.shape[0], 1), dtype=torch.bool, device=row_h.device), batch["row_mask"] <= 0.0], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        lattice = self.lattice_head(encoded[:, 0])
        coord = self.coord_head(encoded[:, 1:])
        lattice_mu, lattice_logstd = lattice[:, :6], lattice[:, 6:].clamp(-4.5, 1.0)
        coord_mu = torch.sigmoid(coord[:, :, :3])
        coord_logstd = coord[:, :, 3:].clamp(-5.0, -0.2)
        return lattice_mu, lattice_logstd, coord_mu, coord_logstd


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def loss_fn(
    lattice_mu: torch.Tensor,
    lattice_logstd: torch.Tensor,
    coord_mu: torch.Tensor,
    coord_logstd: torch.Tensor,
    batch: dict[str, torch.Tensor],
    coord_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = batch["sample_weight"].clamp_min(0.05)
    lat_sigma = lattice_logstd.exp().clamp_min(1.0e-4)
    lat_nll = 0.5 * ((batch["lattice_values"] - lattice_mu) / lat_sigma) ** 2 + lattice_logstd
    lat_per = lat_nll.mean(dim=1)
    diff = (batch["coord_values"] - coord_mu + 0.5) % 1.0 - 0.5
    coord_sigma = coord_logstd.exp().clamp_min(1.0e-4)
    coord_nll = 0.5 * (diff / coord_sigma) ** 2 + coord_logstd
    mask = batch["coord_mask"]
    coord_per = (coord_nll * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
    loss_per = lat_per + float(coord_weight) * coord_per
    loss = (loss_per * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "lattice_nll": float(((lat_per * weights).sum() / weights.sum().clamp_min(1.0)).detach().cpu()),
        "coord_nll": float(((coord_per * weights).sum() / weights.sum().clamp_min(1.0)).detach().cpu()),
    }


@torch.no_grad()
def evaluate_model(model: RowSeqPositiveGeometryNet, examples: list[dict[str, Any]], vocabs: dict[str, dict[str, int]], lattice_mean: torch.Tensor, lattice_std: torch.Tensor, device: torch.device, batch_size: int, max_rows: int, coord_weight: float) -> dict[str, float]:
    loader = DataLoader(
        PositiveGeometryDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda xs: collate_positive(xs, vocabs=vocabs, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu(), max_rows_limit=max_rows),
    )
    total = 0.0
    total_w = 0.0
    lat = 0.0
    coord = 0.0
    model.eval()
    for raw in loader:
        batch = move_batch(raw, device)
        outputs = model(batch)
        loss, parts = loss_fn(*outputs, batch, coord_weight=coord_weight)
        weight = float(batch["sample_weight"].sum().detach().cpu())
        total += float(loss.detach().cpu()) * weight
        lat += parts["lattice_nll"] * weight
        coord += parts["coord_nll"] * weight
        total_w += weight
    return {"loss": total / max(1.0, total_w), "lattice_nll": lat / max(1.0, total_w), "coord_nll": coord / max(1.0, total_w)}


def train(args: argparse.Namespace) -> int:
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_by_id = load_structured_by_id(args.data_root, "train")
    val_by_id = load_structured_by_id(args.data_root, "val")
    engine = OrbitEngine(args.lookup_json)
    train_examples, train_stats = build_examples(
        features_path=args.train_features,
        structured_by_id=train_by_id,
        engine=engine,
        max_examples=int(args.max_train_examples),
        min_positive_row_count=int(args.min_train_row_count),
        max_rows=int(args.max_rows),
    )
    val_examples, val_stats = build_examples(
        features_path=args.val_features,
        structured_by_id=val_by_id,
        engine=engine,
        max_examples=int(args.max_val_examples),
        min_positive_row_count=int(args.min_val_row_count),
        max_rows=int(args.max_rows),
    )
    if not train_examples or not val_examples:
        raise SystemExit("No train/val positive examples available.")
    vocabs = build_vocabs(train_examples)
    lattice_mean, lattice_std = lattice_stats(train_examples)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RowSeqPositiveGeometryNet(
        {key: len(value) for key, value in vocabs.items()},
        hidden_dim=int(args.hidden_dim),
        emb_dim=int(args.emb_dim),
        layers=int(args.layers),
        heads=int(args.heads),
        max_rows=int(args.max_rows),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    weights = torch.tensor([float(ex.get("sample_weight", 1.0)) for ex in train_examples], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_examples), replacement=True)
    loader = DataLoader(
        PositiveGeometryDataset(train_examples),
        batch_size=int(args.batch_size),
        sampler=sampler,
        collate_fn=lambda xs: collate_positive(xs, vocabs=vocabs, lattice_mean=mean_t, lattice_std=std_t, max_rows_limit=int(args.max_rows)),
        num_workers=0,
    )
    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    print(json.dumps({"train_examples": len(train_examples), "val_examples": len(val_examples), "device": str(device), "pid": __import__("os").getpid()}, sort_keys=True), flush=True)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        sums = {"loss": 0.0, "lattice_nll": 0.0, "coord_nll": 0.0}
        steps = 0
        for raw in loader:
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = loss_fn(*model(batch), batch, coord_weight=float(args.coord_weight))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for key in sums:
                sums[key] += float(parts[key])
            steps += 1
        row: dict[str, Any] = {f"train_{key}": value / max(1, steps) for key, value in sums.items()}
        row["epoch"] = epoch
        if epoch == 1 or epoch % max(1, int(args.eval_every)) == 0 or epoch == int(args.epochs):
            val = evaluate_model(model, val_examples, vocabs, mean_t, std_t, device, int(args.batch_size), int(args.max_rows), float(args.coord_weight))
            row.update({f"val_{key}": value for key, value in val.items()})
            if val["loss"] < best_val:
                best_val = float(val["loss"])
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocabs": vocabs,
                        "lattice_mean": lattice_mean,
                        "lattice_std": lattice_std,
                        "config": vars(args),
                        "best_epoch": best_epoch,
                        "best_val": best_val,
                    },
                    out_dir / "ckpt_best.pt",
                )
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "lattice_mean": lattice_mean,
            "lattice_std": lattice_std,
            "config": vars(args),
            "best_epoch": best_epoch,
            "best_val": best_val,
        },
        out_dir / "ckpt_last.pt",
    )
    write_json(
        out_dir / "training_summary.json",
        {
            "train_stats": train_stats,
            "val_stats": val_stats,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "history": history,
            "vocab_sizes": {key: len(value) for key, value in vocabs.items()},
            "note": "Train examples are train rendered StructureMatcher positives only; val positives are used for model selection/loss only; no test labels are read.",
        },
    )
    return 0


def load_ckpt(path: Path, device: torch.device) -> tuple[RowSeqPositiveGeometryNet, dict[str, dict[str, int]], torch.Tensor, torch.Tensor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    vocabs = ckpt["vocabs"]
    config = ckpt.get("config") or {}
    model = RowSeqPositiveGeometryNet(
        {key: len(value) for key, value in vocabs.items()},
        hidden_dim=int(config.get("hidden_dim", 192)),
        emb_dim=int(config.get("emb_dim", 48)),
        layers=int(config.get("layers", 3)),
        heads=int(config.get("heads", 4)),
        max_rows=int(config.get("max_rows", 96)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocabs, torch.tensor(ckpt["lattice_mean"], dtype=torch.float32, device=device), torch.tensor(ckpt["lattice_std"], dtype=torch.float32, device=device), config


def grouped_rendered(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    out: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        out.setdefault(str(row.get("sample_id") or ""), []).append(row)
    for values in out.values():
        values.sort(key=lambda row: int(row.get("rank", 10**9) or 10**9))
    return out


@torch.no_grad()
def predict_one(
    model: RowSeqPositiveGeometryNet,
    vocabs: dict[str, dict[str, int]],
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    record: dict[str, Any],
    *,
    device: torch.device,
    max_rows: int,
    variant: int,
    temperature: float,
    seed_text: str,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    example = {
        "record": record,
        "lattice_target": [0.0] * 6,
        "coord_values": [[0.0, 0.0, 0.0] for _ in record["wa_table"]],
        "coord_mask": [[1.0 if sym in set(row.get("free_symbols") or []) else 0.0 for sym in ("x", "y", "z")] for row in record["wa_table"]],
        "sample_weight": 1.0,
    }
    batch = collate_positive([example], vocabs=vocabs, lattice_mean=mean_t.detach().cpu(), lattice_std=std_t.detach().cpu(), max_rows_limit=max_rows)
    batch = move_batch(batch, device)
    lattice_mu, lattice_logstd, coord_mu, coord_logstd = model(batch)
    if int(variant) > 0 and float(temperature) > 0.0:
        digest = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        gen = torch.Generator(device=device)
        gen.manual_seed(digest)
        lattice_raw = lattice_mu[0] + float(temperature) * lattice_logstd[0].exp() * torch.randn(lattice_mu[0].shape, generator=gen, device=device)
        coord_raw = (coord_mu[0] + float(temperature) * coord_logstd[0].exp() * torch.randn(coord_mu[0].shape, generator=gen, device=device)) % 1.0
    else:
        lattice_raw = lattice_mu[0]
        coord_raw = coord_mu[0]
    lattice_values = (lattice_raw * std_t + mean_t).detach().cpu().tolist()
    lattice = v2.lattice_from_target([float(x) for x in lattice_values], int(record["sg"]))
    params: dict[int, dict[str, float]] = {}
    for row_idx, row in enumerate(record["wa_table"][:max_rows]):
        row_params: dict[str, float] = {}
        for sym_idx, sym in enumerate(("x", "y", "z")):
            if sym in set(row.get("free_symbols") or []):
                row_params[sym] = float(coord_raw[row_idx, sym_idx].detach().cpu()) % 1.0
        params[row_idx] = row_params
    return params, lattice


def render(args: argparse.Namespace) -> int:
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, vocabs, mean_t, std_t, config = load_ckpt(args.ckpt, device)
    structured_by_id = load_structured_by_id(args.data_root, args.split)
    engine = OrbitEngine(args.lookup_json)
    groups = grouped_rendered(read_jsonl(args.input_rendered_jsonl))
    sample_ids = list(groups)
    if int(args.max_records) > 0:
        sample_ids = sample_ids[: int(args.max_records)]
    out_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "ckpt": str(args.ckpt),
        "input_rendered_jsonl": str(args.input_rendered_jsonl),
        "selected_samples": len(sample_ids),
        "min_target_row_count": int(args.min_target_row_count),
        "max_input_candidates_per_sample": int(args.max_input_candidates_per_sample),
        "variants_per_wa": int(args.variants_per_wa),
        "temperature": float(args.temperature),
        "samples_with_output": 0,
        "render_attempts": 0,
        "render_ok": 0,
        "skipped_target_rows_lt_min": 0,
        "skipped_missing_record": 0,
        "skipped_too_many_rows": 0,
        "deduped_wa": 0,
        "note": "Inference uses predicted W/A, formula, GT-SG, and train-positive row-seq geometry model only. No val/test StructureMatcher labels are used for generation.",
    }
    max_rows = int(config.get("max_rows", args.max_rows))
    for sample_id in sample_ids:
        base = structured_by_id.get(sample_id)
        if base is None:
            summary["skipped_missing_record"] += 1
            continue
        target_row_count = int(base.get("n_sites", len(base.get("wa_table") or [])))
        if int(args.min_target_row_count) > 0 and target_row_count < int(args.min_target_row_count):
            summary["skipped_target_rows_lt_min"] += 1
            continue
        sample_rows: list[dict[str, Any]] = []
        seen_wa: set[str] = set()
        input_candidates = groups.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]
        for cand in input_candidates:
            wa_key = str(cand.get("canonical_wa_key") or "")
            if not wa_key or wa_key in seen_wa:
                summary["deduped_wa"] += int(bool(wa_key))
                continue
            seen_wa.add(wa_key)
            try:
                rows = rows_from_wa_key(engine, wa_key)
            except Exception:
                continue
            if len(rows) > max_rows:
                summary["skipped_too_many_rows"] += 1
                continue
            record = make_pseudo_record(base, rows)
            for variant in range(int(args.variants_per_wa)):
                if len(sample_rows) >= int(args.top_k):
                    break
                summary["render_attempts"] += 1
                try:
                    params, lattice = predict_one(
                        model,
                        vocabs,
                        mean_t,
                        std_t,
                        record,
                        device=device,
                        max_rows=max_rows,
                        variant=variant,
                        temperature=float(args.temperature),
                        seed_text=f"{sample_id}|{wa_key}|{variant}|{args.seed}",
                    )
                    rendered = v2.render_candidate(engine, record, {"rows": rows, "params": params, "lattice": lattice}, len(sample_rows) + 1, f"rowseq_v{variant}")
                    cif = str(rendered.get("cif") or "")
                    metric = validate_cif(cif, record["formula_counts"], int(record["sg"])) if cif else {
                        "readable": False,
                        "formula_ok": False,
                        "sg_ok": False,
                        "atom_count_ok": False,
                        "composition_exact": False,
                        "atom_count_after_expansion": None,
                        "detected_sg": None,
                        "error": rendered.get("error") or "empty_cif",
                    }
                    metric.update(selfscore.cif_self_features(cif))
                    skel, wa = v2.canonical_keys_from_rows(rows)
                    row = {
                        "sample_id": sample_id,
                        "rank": len(sample_rows) + 1,
                        "original_rank": cand.get("rank"),
                        "geometry_mode": "positive_rowseq_geometry",
                        "geometry_variant": int(variant),
                        "geometry_source": "train_positive_rowseq",
                        "geometry_lattice_mode": "rowseq_positive_nll",
                        "candidate_score": cand.get("candidate_score"),
                        "canonical_skeleton_key": skel,
                        "canonical_wa_key": wa,
                        "cif": cif,
                        **metric,
                    }
                    sample_rows.append(row)
                    summary["render_ok"] += int(bool(cif))
                except Exception as exc:  # noqa: BLE001
                    sample_rows.append(
                        {
                            "sample_id": sample_id,
                            "rank": len(sample_rows) + 1,
                            "original_rank": cand.get("rank"),
                            "geometry_mode": "positive_rowseq_geometry",
                            "geometry_variant": int(variant),
                            "geometry_source": "train_positive_rowseq",
                            "canonical_skeleton_key": cand.get("canonical_skeleton_key"),
                            "canonical_wa_key": wa_key,
                            "cif": "",
                            "readable": False,
                            "formula_ok": False,
                            "sg_ok": False,
                            "atom_count_ok": False,
                            "composition_exact": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        if sample_rows:
            summary["samples_with_output"] += 1
        out_rows.extend(sample_rows[: int(args.top_k)])
    summary["rendered_rows"] = len(out_rows)
    write_jsonl(out_dir / "rendered_topk.jsonl", out_rows)
    write_json(out_dir / "positive_rowseq_render_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/render a train-positive row-sequence probabilistic geometry decoder.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--train-features", type=Path, required=True)
    p_train.add_argument("--val-features", type=Path, required=True)
    p_train.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    p_train.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    p_train.add_argument("--out-dir", type=Path, required=True)
    p_train.add_argument("--max-train-examples", type=int, default=0)
    p_train.add_argument("--max-val-examples", type=int, default=0)
    p_train.add_argument("--min-train-row-count", type=int, default=0)
    p_train.add_argument("--min-val-row-count", type=int, default=0)
    p_train.add_argument("--max-rows", type=int, default=96)
    p_train.add_argument("--epochs", type=int, default=8)
    p_train.add_argument("--batch-size", type=int, default=96)
    p_train.add_argument("--hidden-dim", type=int, default=192)
    p_train.add_argument("--emb-dim", type=int, default=48)
    p_train.add_argument("--layers", type=int, default=3)
    p_train.add_argument("--heads", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1.5e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--coord-weight", type=float, default=2.0)
    p_train.add_argument("--eval-every", type=int, default=1)
    p_train.add_argument("--seed", type=int, default=20260615)
    p_train.add_argument("--device", default="cpu")
    p_render = sub.add_parser("render")
    p_render.add_argument("--ckpt", type=Path, required=True)
    p_render.add_argument("--input-rendered-jsonl", type=Path, required=True)
    p_render.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    p_render.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    p_render.add_argument("--split", default="val")
    p_render.add_argument("--out-dir", type=Path, required=True)
    p_render.add_argument("--max-records", type=int, default=64)
    p_render.add_argument("--min-target-row-count", type=int, default=7)
    p_render.add_argument("--max-input-candidates-per-sample", type=int, default=12)
    p_render.add_argument("--variants-per-wa", type=int, default=3)
    p_render.add_argument("--top-k", type=int, default=50)
    p_render.add_argument("--temperature", type=float, default=0.35)
    p_render.add_argument("--max-rows", type=int, default=96)
    p_render.add_argument("--seed", type=int, default=20260615)
    p_render.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        return train(args)
    if args.command == "render":
        return render(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
