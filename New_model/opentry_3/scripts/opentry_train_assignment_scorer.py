#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SCRIPT_DIR = OPENTRY_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_wyckoff_sequence_models import (  # noqa: E402
    AssignmentNet,
    build_decoder_priors,
    build_vocab,
    ensure_under_opentry,
    make_assignment_states,
    read_jsonl,
    tensor_batch,
    vocab_to_jsonable,
    weighted_ce,
    write_json,
)


def limit_arg(value: int) -> int | None:
    return None if int(value) <= 0 else int(value)


def batch_indexes(n: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i : i + int(batch_size)] for i in range(0, n, int(batch_size))]


def train_epoch(
    model: AssignmentNet,
    optimizer: torch.optim.Optimizer,
    states: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.train()
    rng = random.Random(seed)
    total_loss = 0.0
    batches = 0
    for indexes in batch_indexes(len(states), batch_size, rng):
        optimizer.zero_grad(set_to_none=True)
        sg = tensor_batch(states, indexes, "sg_id", device)
        formula = tensor_batch(states, indexes, "formula_vec", device)
        remaining = tensor_batch(states, indexes, "remaining_vec", device)
        orbit = tensor_batch(states, indexes, "orbit", device)
        numeric = torch.stack(
            [
                tensor_batch(states, indexes, "remaining_total", device),
                tensor_batch(states, indexes, "step", device),
                tensor_batch(states, indexes, "multiplicity", device),
            ],
            dim=-1,
        )
        target = tensor_batch(states, indexes, "target", device)
        weight = tensor_batch(states, indexes, "weight", device)
        loss = weighted_ce(model(sg, formula, remaining, orbit, numeric), target, weight)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return {"loss": total_loss / max(1, batches), "batches": float(batches)}


@torch.no_grad()
def evaluate_assignment(
    model: AssignmentNet,
    states: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    mrr = 0.0
    loss_total = 0.0
    batches = 0
    weight_total = 0.0
    for start in range(0, len(states), int(batch_size)):
        indexes = list(range(start, min(len(states), start + int(batch_size))))
        sg = tensor_batch(states, indexes, "sg_id", device)
        formula = tensor_batch(states, indexes, "formula_vec", device)
        remaining = tensor_batch(states, indexes, "remaining_vec", device)
        orbit = tensor_batch(states, indexes, "orbit", device)
        numeric = torch.stack(
            [
                tensor_batch(states, indexes, "remaining_total", device),
                tensor_batch(states, indexes, "step", device),
                tensor_batch(states, indexes, "multiplicity", device),
            ],
            dim=-1,
        )
        target = tensor_batch(states, indexes, "target", device)
        weight = tensor_batch(states, indexes, "weight", device)
        logits = model(sg, formula, remaining, orbit, numeric)
        loss_vec = F.cross_entropy(logits, target, reduction="none")
        loss_total += float((loss_vec * weight).sum().detach().cpu())
        weight_total += float(weight.sum().detach().cpu())
        target_score = logits.gather(1, target.view(-1, 1)).squeeze(1)
        ranks = (logits > target_score.view(-1, 1)).sum(dim=1) + 1
        ranks_cpu = ranks.detach().cpu()
        total += int(ranks_cpu.numel())
        top1 += int((ranks_cpu <= 1).sum())
        top3 += int((ranks_cpu <= 3).sum())
        top5 += int((ranks_cpu <= 5).sum())
        mrr += float((1.0 / ranks_cpu.float()).sum())
        batches += 1
    denom = max(1, total)
    return {
        "states": float(total),
        "loss": loss_total / max(1.0, weight_total),
        "top1": top1 / denom,
        "top3": top3 / denom,
        "top5": top5 / denom,
        "mrr": mrr / denom,
        "batches": float(batches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a train-only formula+GT-SG+skeleton assignment scorer for exact W/A DP.")
    parser.add_argument("--data-dir", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52")
    parser.add_argument("--run-dir", type=Path, default=OPENTRY_ROOT / "runs" / "e63_assignment_scorer_fulltrain")
    parser.add_argument("--out-dir", type=Path, default=OPENTRY_ROOT / "reports" / "e63_assignment_scorer_fulltrain")
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--complex-weight", type=float, default=3.0)
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

    start = time.time()
    train_records = read_jsonl(args.data_dir / "train.jsonl", limit_arg(args.max_train_records))
    val_records = read_jsonl(args.data_dir / "val.jsonl", limit_arg(args.max_val_records))
    vocab = build_vocab(train_records)
    train_states = make_assignment_states(train_records, vocab, args.complex_weight)
    val_states = make_assignment_states(val_records, vocab, args.complex_weight)
    model = AssignmentNet(
        num_sgs=len(vocab.sgs),
        formula_dim=len(vocab.elements),
        num_orbits=len(vocab.orbits),
        num_elements=len(vocab.elements),
        hidden=int(args.hidden),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    priors = build_decoder_priors(train_records, vocab)
    write_json(run_dir / "vocab.json", vocab_to_jsonable(vocab))
    write_json(run_dir / "decoder_priors_train_only.json", priors)

    summary: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "states": {"train_assignment": len(train_states), "val_assignment": len(val_states)},
        "vocab": {"elements": len(vocab.elements), "sgs": len(vocab.sgs), "orbits": len(vocab.orbits)},
        "epochs": [],
    }
    write_json(out_dir / "training_summary_running.json", summary)

    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_epoch(
            model,
            optimizer,
            train_states,
            batch_size=int(args.batch_size),
            device=device,
            seed=int(args.seed) + epoch,
        )
        val_metrics = evaluate_assignment(model, val_states, batch_size=int(args.batch_size), device=device)
        score = float(val_metrics["mrr"])
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_s": time.time() - start,
        }
        summary["epochs"].append(row)
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
                    "val": val_metrics,
                },
                run_dir / "best.pt",
            )

    summary["best_epoch"] = best_epoch
    summary["best_score_val_mrr"] = best_score
    summary["elapsed_s_total"] = time.time() - start
    write_json(out_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
