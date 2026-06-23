#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_symcif_v4_geometry_model as geom  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train opentry_3 train-only geometry net for W/A -> lattice/free params.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coord-weight", type=float, default=2.0)
    parser.add_argument("--complex-weight", type=float, default=3.0)
    parser.add_argument("--rare-sg-max-count", type=int, default=25)
    parser.add_argument("--high-orbit-threshold", type=int, default=12)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    torch.set_float32_matmul_precision("high")

    train_records = geom.read_jsonl(args.data_root / "train.jsonl")
    val_records = geom.read_jsonl(args.data_root / "val.jsonl")
    if int(args.max_train_records) > 0:
        train_records = train_records[: int(args.max_train_records)]
    if int(args.max_val_records) > 0:
        val_records = val_records[: int(args.max_val_records)]

    vocabs = geom.build_vocabs(train_records)
    lattice_mean, lattice_std = geom.lattice_stats(train_records)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)

    rare_sg_set = geom.rare_sgs(train_records, int(args.rare_sg_max_count))
    weights = [
        geom.complex_weight(
            record,
            float(args.complex_weight),
            rare_sg_set=rare_sg_set,
            high_orbit_threshold=int(args.high_orbit_threshold),
        )
        for record in train_records
    ]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = geom.GeometryNet(
        {name: len(vocab) for name, vocab in vocabs.items()},
        hidden_dim=int(args.hidden_dim),
        emb_dim=int(args.emb_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(train_records), replacement=True)
    loader = DataLoader(
        geom.GeometryDataset(train_records, weights=weights),
        batch_size=int(args.batch_size),
        sampler=sampler,
        collate_fn=lambda xs: geom.collate_geometry(xs, vocabs=vocabs, lattice_mean=mean_t, lattice_std=std_t),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
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
                "device": str(device),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "complex_weight": float(args.complex_weight),
                "vocab_source": "train_only",
                "pid": os.getpid(),
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
        steps = 0
        for raw_batch in loader:
            batch = geom.move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            lattice_pred, coord_pred = model(batch)
            loss, parts = geom.loss_fn(lattice_pred, coord_pred, batch, coord_weight=float(args.coord_weight))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_lattice += parts["lattice_loss"]
            train_coord += parts["coord_loss"]
            steps += 1
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, steps),
            "train_lattice_loss": train_lattice / max(1, steps),
            "train_coord_loss": train_coord / max(1, steps),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        if epoch == 1 or epoch % max(1, int(args.eval_every)) == 0 or epoch == int(args.epochs):
            val = geom.evaluate(
                model,
                val_records,
                vocabs=vocabs,
                lattice_mean=mean_t,
                lattice_std=std_t,
                device=device,
                batch_size=int(args.batch_size),
                coord_weight=float(args.coord_weight),
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
                        "vocab_source": "train_only",
                    },
                    best_path,
                )
        history.append(row)
        if epoch == 1 or epoch % max(1, int(args.eval_every)) == 0 or epoch == int(args.epochs):
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
            "vocab_source": "train_only",
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
            "vocab_sizes": {name: len(vocab) for name, vocab in vocabs.items()},
            "vocab_source": "train_only",
            "oversampling": {
                "complex_weight": float(args.complex_weight),
                "rare_sg_max_count": int(args.rare_sg_max_count),
                "rare_sg_count": len(rare_sg_set),
                "rare_sgs": sorted(rare_sg_set),
                "high_orbit_threshold": int(args.high_orbit_threshold),
                "mean_weight": float(sum(weights) / max(1, len(weights))),
                "max_weight": float(max(weights) if weights else 1.0),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
