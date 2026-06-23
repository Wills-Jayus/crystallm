#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TARGET_SGS = {2, 65, 71, 127}
ANGLE_SCALE = 180.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


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


def build_vocabs(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    elements = sorted({str(e) for r in records for e in r["formula_counts"]} | {str(w["element"]) for r in records for w in r["wa_table"]})
    sgs = sorted({str(int(r["sg"])) for r in records}, key=lambda x: int(x))
    orbits = sorted({str(w["orbit_id"]) for r in records for w in r["wa_table"]})
    site_syms = sorted({str(w.get("site_symmetry") or "UNKNOWN") for r in records for w in r["wa_table"]})
    letters = sorted({f"{int(r['sg'])}|{w.get('letter')}" for r in records for w in r["wa_table"]})
    return {
        "element": {v: i + 1 for i, v in enumerate(elements)},
        "sg": {v: i + 1 for i, v in enumerate(sgs)},
        "orbit": {v: i + 1 for i, v in enumerate(orbits)},
        "site_sym": {v: i + 1 for i, v in enumerate(site_syms)},
        "letter": {v: i + 1 for i, v in enumerate(letters)},
    }


def lattice_stats(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    targets = torch.tensor([lattice_target(r) for r in records], dtype=torch.float32)
    mean = targets.mean(dim=0)
    std = targets.std(dim=0).clamp_min(1e-4)
    return mean.tolist(), std.tolist()


def coord_target(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    params = {str(k): float(v) % 1.0 for k, v in dict(row.get("free_params") or {}).items()}
    free_symbols = {str(s) for s in row.get("free_symbols") or params.keys()}
    values: list[float] = []
    mask: list[float] = []
    for symbol in ("x", "y", "z"):
        values.append(float(params.get(symbol, 0.0)))
        mask.append(1.0 if symbol in free_symbols and symbol in params else 0.0)
    return values, mask


class GeometryDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], weights: list[float] | None = None):
        self.records = records
        self.weights = weights or [1.0 for _ in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def rare_sgs(records: list[dict[str, Any]], max_count: int) -> set[int]:
    counts = Counter(int(r.get("sg", 0)) for r in records)
    return {sg for sg, count in counts.items() if count <= int(max_count)}


def complex_weight(record: dict[str, Any], boost: float, rare_sg_set: set[int] | None = None, high_orbit_threshold: int = 12) -> float:
    if boost <= 1.0:
        return 1.0
    rare_sg_set = rare_sg_set or set()
    hits = 0
    if int(record.get("n_sites", 0)) >= 6:
        hits += 1
    if int(record.get("num_elements", 0)) >= 4:
        hits += 1
    if int(record.get("sg", 0)) in TARGET_SGS or int(record.get("sg", 0)) in rare_sg_set:
        hits += 1
    max_mult = max((int(row.get("multiplicity", 1)) for row in record.get("wa_table", [])), default=1)
    if max_mult >= int(high_orbit_threshold):
        hits += 1
    return 1.0 + (float(boost) - 1.0) * float(hits)


def collate_geometry(
    records: list[dict[str, Any]],
    *,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = len(records)
    max_formula = max(1, max(len(r["formula_counts"]) for r in records))
    max_rows = max(1, max(len(r["wa_table"]) for r in records))
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

    for i, record in enumerate(records):
        counts = {str(k): int(v) for k, v in record["formula_counts"].items()}
        total_atoms = max(1, sum(counts.values()))
        sg = int(record["sg"])
        sg_ids[i] = vocabs["sg"].get(str(sg), 0)
        for j, (element, count) in enumerate(sorted(counts.items())):
            formula_element_ids[i, j] = vocabs["element"].get(str(element), 0)
            formula_weights[i, j] = float(count) / float(total_atoms)
        numeric[i] = torch.tensor(
            [
                float(sg) / 230.0,
                float(total_atoms) / 300.0,
                float(int(record.get("n_sites", len(record["wa_table"])))) / 64.0,
                float(int(record.get("num_elements", len(counts)))) / 12.0,
                1.0 if int(record.get("n_sites", 0)) >= 6 else 0.0,
            ],
            dtype=torch.float32,
        )
        lattice_values[i] = (torch.tensor(lattice_target(record), dtype=torch.float32) - lattice_mean) / lattice_std
        for j, row in enumerate(record["wa_table"]):
            row_mask[i, j] = 1.0
            row_element_ids[i, j] = vocabs["element"].get(str(row["element"]), 0)
            row_orbit_ids[i, j] = vocabs["orbit"].get(str(row["orbit_id"]), 0)
            row_site_sym_ids[i, j] = vocabs["site_sym"].get(str(row.get("site_symmetry") or "UNKNOWN"), 0)
            row_letter_ids[i, j] = vocabs["letter"].get(f"{sg}|{row.get('letter')}", 0)
            free_symbols = {str(s) for s in row.get("free_symbols") or []}
            row_numeric[i, j] = torch.tensor(
                [
                    float(int(row.get("multiplicity", 1))) / 64.0,
                    float(len(free_symbols)) / 3.0,
                    1.0 if "x" in free_symbols else 0.0,
                    1.0 if "y" in free_symbols else 0.0,
                    1.0 if "z" in free_symbols else 0.0,
                    float(j) / 64.0,
                ],
                dtype=torch.float32,
            )
            vals, mask = coord_target(row)
            coord_values[i, j] = torch.tensor(vals, dtype=torch.float32)
            coord_mask[i, j] = torch.tensor(mask, dtype=torch.float32)

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
    }


class GeometryNet(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], hidden_dim: int = 256, emb_dim: int = 64) -> None:
        super().__init__()
        self.element_emb = nn.Embedding(vocab_sizes["element"] + 1, emb_dim)
        self.sg_emb = nn.Embedding(vocab_sizes["sg"] + 1, emb_dim)
        self.orbit_emb = nn.Embedding(vocab_sizes["orbit"] + 1, emb_dim)
        self.site_sym_emb = nn.Embedding(vocab_sizes["site_sym"] + 1, emb_dim // 2)
        self.letter_emb = nn.Embedding(vocab_sizes["letter"] + 1, emb_dim // 2)
        row_in = emb_dim * 2 + emb_dim // 2 + emb_dim // 2 + 6
        self.row_encoder = nn.Sequential(
            nn.LayerNorm(row_in),
            nn.Linear(row_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        ctx_in = hidden_dim + emb_dim * 2 + 5
        self.context = nn.Sequential(
            nn.LayerNorm(ctx_in),
            nn.Linear(ctx_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.lattice_head = nn.Linear(hidden_dim, 6)
        self.coord_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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
        ctx = self.context(torch.cat([row_pool, formula_vec, sg_vec, batch["numeric"]], dim=-1))
        lattice = self.lattice_head(ctx)
        ctx_rows = ctx.unsqueeze(1).expand(-1, row_h.shape[1], -1)
        coords = torch.sigmoid(self.coord_head(torch.cat([row_h, ctx_rows], dim=-1)))
        return lattice, coords


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def loss_fn(
    lattice_pred: torch.Tensor,
    coord_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    coord_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    lattice_loss = F.mse_loss(lattice_pred, batch["lattice_values"])
    diff = torch.abs(coord_pred - batch["coord_values"])
    wrapped = torch.minimum(diff, 1.0 - diff)
    mask = batch["coord_mask"]
    coord_loss = (wrapped.square() * mask).sum() / mask.sum().clamp_min(1.0)
    loss = lattice_loss + float(coord_weight) * coord_loss
    return loss, {"lattice_loss": float(lattice_loss.detach().cpu()), "coord_loss": float(coord_loss.detach().cpu())}


@torch.no_grad()
def evaluate(
    model: GeometryNet,
    records: list[dict[str, Any]],
    *,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    coord_weight: float,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    lattice_losses: list[float] = []
    coord_losses: list[float] = []
    loader = DataLoader(
        GeometryDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda xs: collate_geometry(xs, vocabs=vocabs, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu()),
        num_workers=0,
    )
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        lattice_pred, coord_pred = model(batch)
        loss, parts = loss_fn(lattice_pred, coord_pred, batch, coord_weight=coord_weight)
        losses.append(float(loss.detach().cpu()))
        lattice_losses.append(parts["lattice_loss"])
        coord_losses.append(parts["coord_loss"])
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "lattice_loss": float(sum(lattice_losses) / max(1, len(lattice_losses))),
        "coord_loss": float(sum(coord_losses) / max(1, len(coord_losses))),
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SymCIF-v4 orbit-level geometry model.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coord-weight", type=float, default=2.0)
    parser.add_argument("--complex-weight", type=float, default=1.0)
    parser.add_argument("--rare-sg-max-count", type=int, default=25)
    parser.add_argument("--high-orbit-threshold", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(args.data_root / "train.jsonl")
    val_records = read_jsonl(args.data_root / "val.jsonl")
    vocabs = build_vocabs(train_records + val_records)
    lattice_mean, lattice_std = lattice_stats(train_records)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    rare_sg_set = rare_sgs(train_records, args.rare_sg_max_count)
    weights = [
        complex_weight(
            r,
            args.complex_weight,
            rare_sg_set=rare_sg_set,
            high_orbit_threshold=args.high_orbit_threshold,
        )
        for r in train_records
    ]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = GeometryNet({k: len(v) for k, v in vocabs.items()}, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(train_records), replacement=True)
    loader = DataLoader(
        GeometryDataset(train_records, weights=weights),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda xs: collate_geometry(xs, vocabs=vocabs, lattice_mean=mean_t, lattice_std=std_t),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    best_path = args.out_dir / "ckpt_best.pt"
    last_path = args.out_dir / "ckpt_last.pt"
    print(
        json.dumps(
            {
                "train_records": len(train_records),
                "val_records": len(val_records),
                "device": str(device),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "complex_weight": args.complex_weight,
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        parts_sum = {"lattice_loss": 0.0, "coord_loss": 0.0}
        steps = 0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            lattice_pred, coord_pred = model(batch)
            loss, parts = loss_fn(lattice_pred, coord_pred, batch, coord_weight=args.coord_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            parts_sum["lattice_loss"] += parts["lattice_loss"]
            parts_sum["coord_loss"] += parts["coord_loss"]
            steps += 1
        scheduler.step()
        train_row = {
            "epoch": epoch,
            "train_loss": running / max(1, steps),
            "train_lattice_loss": parts_sum["lattice_loss"] / max(1, steps),
            "train_coord_loss": parts_sum["coord_loss"] / max(1, steps),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        if epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            val = evaluate(
                model,
                val_records,
                vocabs=vocabs,
                lattice_mean=mean_t,
                lattice_std=std_t,
                device=device,
                batch_size=args.batch_size,
                coord_weight=args.coord_weight,
            )
            train_row.update({f"val_{k}": v for k, v in val.items()})
            if val["loss"] < best_val:
                best_val = val["loss"]
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
        history.append(train_row)
        if epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            print(json.dumps(train_row, sort_keys=True), flush=True)

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
        args.out_dir / "training_summary.json",
        {
            "config": jsonable_args(args),
            "best_val_loss": best_val,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "history": history,
            "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
            "oversampling": {
                "complex_weight": args.complex_weight,
                "rare_sg_max_count": args.rare_sg_max_count,
                "rare_sg_count": len(rare_sg_set),
                "rare_sgs": sorted(rare_sg_set),
                "high_orbit_threshold": args.high_orbit_threshold,
                "mean_weight": float(sum(weights) / max(1, len(weights))),
                "max_weight": float(max(weights) if weights else 1.0),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
