#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine
from symcif_v4.step_policy import NUMERIC_DIM, StepPolicyNet, StepPolicyVocab, build_frequency_priors, vocab_sizes
from train_orbit_aware_listwise_reranker import (
    OrbitAwareListwiseRanker,
    RerankVocab,
    build_samples,
    evaluate_samples,
    make_batch,
    topk_metrics,
    rank_split,
)
from train_step_policy_wa import make_training_states, state_loss


LOSS_WEIGHTS = {
    "step_policy": 1.0,
    "wa_listwise_rank": 0.5,
    "skeleton_set": 0.2,
    "value_reachable": 0.1,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


class SkeletonSetNet(nn.Module):
    def __init__(self, *, num_elements: int, num_sgs: int, num_orbits: int, hidden_dim: int = 192, emb_dim: int = 64) -> None:
        super().__init__()
        self.element_emb = nn.Embedding(num_elements + 1, emb_dim)
        self.sg_emb = nn.Embedding(num_sgs + 1, emb_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim * 2 + 6),
            nn.Linear(emb_dim * 2 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.presence_head = nn.Linear(hidden_dim, num_orbits)
        self.count_head = nn.Linear(hidden_dim, num_orbits)

    def forward(self, sg_id: torch.Tensor, element_ids: torch.Tensor, element_weights: torch.Tensor, numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        elem = self.element_emb(element_ids)
        denom = element_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        elem_vec = (elem * element_weights.unsqueeze(-1)).sum(dim=1) / denom
        sg_vec = self.sg_emb(sg_id)
        h = self.net(torch.cat([sg_vec, elem_vec, numeric], dim=-1))
        return self.presence_head(h), self.count_head(h)


class ValueReachableNet(nn.Module):
    def __init__(self, *, num_elements: int, num_sgs: int, hidden_dim: int = 128, emb_dim: int = 48) -> None:
        super().__init__()
        self.element_emb = nn.Embedding(num_elements + 1, emb_dim)
        self.sg_emb = nn.Embedding(num_sgs + 1, emb_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim * 2 + 10),
            nn.Linear(emb_dim * 2 + 10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sg_id: torch.Tensor, element_ids: torch.Tensor, element_weights: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        elem = self.element_emb(element_ids)
        denom = element_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        elem_vec = (elem * element_weights.unsqueeze(-1)).sum(dim=1) / denom
        sg_vec = self.sg_emb(sg_id)
        return self.net(torch.cat([sg_vec, elem_vec, numeric], dim=-1)).squeeze(-1)


class StructuredWAMultitaskModel(nn.Module):
    def __init__(
        self,
        *,
        step_vocab_sizes: dict[str, int],
        rerank_vocab_sizes: dict[str, int],
        element_vocab_size: int,
        sg_vocab_size: int,
        orbit_count: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.step_policy = StepPolicyNet(step_vocab_sizes, numeric_dim=NUMERIC_DIM, hidden_dim=hidden_dim)
        self.reranker = OrbitAwareListwiseRanker(rerank_vocab_sizes, hidden_dim=hidden_dim)
        self.skeleton_set = SkeletonSetNet(
            num_elements=element_vocab_size,
            num_sgs=sg_vocab_size,
            num_orbits=orbit_count,
            hidden_dim=hidden_dim,
        )
        self.value_reachable = ValueReachableNet(
            num_elements=element_vocab_size,
            num_sgs=sg_vocab_size,
            hidden_dim=max(96, hidden_dim // 2),
        )


def element_sg_maps(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    elements = sorted({str(e) for row in records for e in row["formula_counts"]})
    sgs = sorted({str(int(row["sg"])) for row in records})
    orbits = sorted({str(wa["orbit_id"]) for row in records for wa in row["wa_table"]})
    return (
        {element: i + 1 for i, element in enumerate(elements)},
        {sg: i + 1 for i, sg in enumerate(sgs)},
        {orbit: i for i, orbit in enumerate(orbits)},
    )


def record_feature_batch(
    records: list[dict[str, Any]],
    element_to_id: dict[str, int],
    sg_to_id: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_elements = max(1, max(len(r["formula_counts"]) for r in records))
    element_ids = torch.zeros((len(records), max_elements), dtype=torch.long)
    element_weights = torch.zeros((len(records), max_elements), dtype=torch.float32)
    sg_ids = torch.zeros((len(records),), dtype=torch.long)
    numeric = torch.zeros((len(records), 6), dtype=torch.float32)
    for i, record in enumerate(records):
        total = max(1, int(record["atom_count"]))
        sg_ids[i] = sg_to_id.get(str(int(record["sg"])), 0)
        for j, (element, count) in enumerate(sorted(record["formula_counts"].items())):
            element_ids[i, j] = element_to_id.get(str(element), 0)
            element_weights[i, j] = float(int(count)) / float(total)
        counts = [int(v) for v in record["formula_counts"].values()]
        numeric[i] = torch.tensor(
            [
                float(int(record["sg"])) / 230.0,
                float(total) / 256.0,
                float(int(record["n_sites"])) / 64.0,
                float(len(record["formula_counts"])) / 10.0,
                float(max(counts) if counts else 0) / float(total),
                float(min(counts) if counts else 0) / float(total),
            ],
            dtype=torch.float32,
        )
    return sg_ids.to(device), element_ids.to(device), element_weights.to(device), numeric.to(device)


def skeleton_targets(records: list[dict[str, Any]], orbit_to_index: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    presence = torch.zeros((len(records), len(orbit_to_index)), dtype=torch.float32)
    counts = torch.zeros((len(records), len(orbit_to_index)), dtype=torch.float32)
    for i, record in enumerate(records):
        for row in record["wa_table"]:
            idx = orbit_to_index[str(row["orbit_id"])]
            presence[i, idx] = 1.0
            counts[i, idx] += 1.0
    return presence.to(device), counts.to(device)


def value_batch(
    states: list[dict[str, Any]],
    element_to_id: dict[str, int],
    sg_to_id: dict[str, int],
    device: torch.device,
    *,
    make_negative: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_elements = max(1, max(len(s["formula_counts"]) for s in states))
    element_ids = torch.zeros((len(states), max_elements), dtype=torch.long)
    weights = torch.zeros((len(states), max_elements), dtype=torch.float32)
    sg_ids = torch.zeros((len(states),), dtype=torch.long)
    numeric = torch.zeros((len(states), 10), dtype=torch.float32)
    labels = torch.ones((len(states),), dtype=torch.float32)
    for i, state in enumerate(states):
        formula_counts = {str(k): int(v) for k, v in state["formula_counts"].items()}
        remaining = {str(k): int(v) for k, v in state["remaining_counts"].items()}
        total = max(1, sum(formula_counts.values()))
        if make_negative:
            labels[i] = 0.0
            largest = max(remaining, key=lambda key: remaining[key])
            if remaining[largest] > 0:
                remaining[largest] = max(0, remaining[largest] - 1)
        sg_ids[i] = sg_to_id.get(str(int(state["sg"])), 0)
        for j, (element, count) in enumerate(sorted(remaining.items())):
            element_ids[i, j] = element_to_id.get(str(element), 0)
            weights[i, j] = float(count) / float(total)
        rem_total = sum(remaining.values())
        numeric[i] = torch.tensor(
            [
                float(int(state["sg"])) / 230.0,
                float(total) / 256.0,
                float(rem_total) / float(total),
                float(int(state["step_index"])) / 64.0,
                float(int(state["chosen_count"])) / 64.0,
                float(max(remaining.values()) if remaining else 0) / float(total),
                float(min(remaining.values()) if remaining else 0) / float(total),
                float(len(remaining)) / 10.0,
                1.0 if rem_total == 0 else 0.0,
                1.0 if make_negative else 0.0,
            ],
            dtype=torch.float32,
        )
    return (
        sg_ids.to(device),
        element_ids.to(device),
        weights.to(device),
        numeric.to(device),
        labels.to(device),
    )


@torch.no_grad()
def eval_step_policy(
    model: StepPolicyNet,
    states: list[dict[str, Any]],
    vocab: StepPolicyVocab,
    priors: dict[str, Any],
    device: torch.device,
    max_states: int,
) -> dict[str, Any]:
    model.eval()
    rows = states[: int(max_states)]
    hits = {1: 0, 5: 0, 20: 0}
    loss_sum = 0.0
    for state in rows:
        loss, logits = state_loss(model, state, vocab, priors, device)
        loss_sum += float(loss.detach().cpu())
        order = torch.argsort(logits, descending=True).detach().cpu().tolist()
        rank = order.index(int(state["target_index"])) + 1
        for k in hits:
            hits[k] += int(rank <= k)
    denom = max(1, len(rows))
    return {"states": len(rows), "loss": loss_sum / denom, **{f"top{k}": v / denom for k, v in hits.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train WA-only structured multitask SymCIF-v4 model.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--candidate-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_hybrid_search")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "structured_multitask_wa_v1")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_hybrid_search")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-step-states-per-epoch", type=int, default=30000)
    parser.add_argument("--max-rank-samples-per-epoch", type=int, default=3500)
    parser.add_argument("--skeleton-batch-size", type=int, default=256)
    parser.add_argument("--value-batch-size", type=int, default=256)
    parser.add_argument("--max-actions-per-state", type=int, default=256)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    train_records = read_jsonl(args.data_root / "train.jsonl")
    val_records = read_jsonl(args.data_root / "val.jsonl")
    if args.max_train_records is not None:
        train_records = train_records[: int(args.max_train_records)]
    if args.max_val_records is not None:
        val_records = val_records[: int(args.max_val_records)]
    engine = OrbitEngine.from_structured_root(args.lookup_json, args.data_root)
    step_vocab = StepPolicyVocab.from_records(train_records)
    step_priors = build_frequency_priors(train_records)
    train_states, train_state_skipped = make_training_states(train_records, engine, args.max_actions_per_state, args.seed)
    val_states, val_state_skipped = make_training_states(val_records, engine, None, args.seed)
    rank_samples, rank_skipped, train_candidate_rows = build_samples(
        args.data_root,
        args.candidate_dir,
        "train",
        max_candidates=320,
        seed=args.seed,
        max_rows=args.max_train_records,
    )
    val_rank_samples, val_rank_skipped, val_candidate_rows = build_samples(
        args.data_root,
        args.candidate_dir,
        "val",
        max_candidates=500,
        seed=args.seed,
        max_rows=args.max_val_records,
    )
    rerank_vocab = RerankVocab.from_rows(train_records, train_candidate_rows + val_candidate_rows)
    element_to_id, sg_to_id, orbit_to_index = element_sg_maps(train_records)
    model = StructuredWAMultitaskModel(
        step_vocab_sizes=vocab_sizes(step_vocab),
        rerank_vocab_sizes=rerank_vocab.sizes(),
        element_vocab_size=max(element_to_id.values(), default=0),
        sg_vocab_size=max(sg_to_id.values(), default=0),
        orbit_count=len(orbit_to_index),
        hidden_dim=args.hidden_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    rng = random.Random(args.seed)
    history: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "device": str(device),
                "torch_cuda_available": torch.cuda.is_available(),
                "train_records": len(train_records),
                "val_records": len(val_records),
                "train_states": len(train_states),
                "rank_samples": len(rank_samples),
                "rank_skipped": len(rank_skipped),
                "loss_weights": LOSS_WEIGHTS,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        rng.shuffle(train_states)
        rng.shuffle(rank_samples)
        loss_totals = {key: 0.0 for key in LOSS_WEIGHTS}
        counts = {key: 0 for key in LOSS_WEIGHTS}

        for state in train_states[: int(args.max_step_states_per_epoch)]:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss, _ = state_loss(model.step_policy, state, step_vocab, step_priors, device)
                weighted = LOSS_WEIGHTS["step_policy"] * loss
            scaler.scale(weighted).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_totals["step_policy"] += float(loss.detach().cpu())
            counts["step_policy"] += 1

        for sample in rank_samples[: int(args.max_rank_samples_per_epoch)]:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model.reranker(make_batch(sample["record"], sample["candidates"], rerank_vocab, device))
                target = torch.tensor([int(sample["target_index"])], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits.unsqueeze(0), target)
                weighted = LOSS_WEIGHTS["wa_listwise_rank"] * loss
            scaler.scale(weighted).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_totals["wa_listwise_rank"] += float(loss.detach().cpu())
            counts["wa_listwise_rank"] += 1

        rng.shuffle(train_records)
        for start in range(0, len(train_records), int(args.skeleton_batch_size)):
            batch_records = train_records[start : start + int(args.skeleton_batch_size)]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                sg_ids, element_ids, element_weights, numeric = record_feature_batch(batch_records, element_to_id, sg_to_id, device)
                target_presence, target_counts = skeleton_targets(batch_records, orbit_to_index, device)
                presence_logits, count_logits = model.skeleton_set(sg_ids, element_ids, element_weights, numeric)
                loss = F.binary_cross_entropy_with_logits(presence_logits, target_presence)
                loss = loss + 0.25 * F.smooth_l1_loss(F.relu(count_logits), target_counts)
                weighted = LOSS_WEIGHTS["skeleton_set"] * loss
            scaler.scale(weighted).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_totals["skeleton_set"] += float(loss.detach().cpu())
            counts["skeleton_set"] += 1

        value_states = train_states[:]
        rng.shuffle(value_states)
        for start in range(0, min(len(value_states), int(args.max_step_states_per_epoch)), int(args.value_batch_size)):
            states = value_states[start : start + int(args.value_batch_size)]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pos = value_batch(states, element_to_id, sg_to_id, device, make_negative=False)
                neg = value_batch(states, element_to_id, sg_to_id, device, make_negative=True)
                pos_logits = model.value_reachable(*pos[:4])
                neg_logits = model.value_reachable(*neg[:4])
                logits = torch.cat([pos_logits, neg_logits], dim=0)
                labels = torch.cat([pos[4], neg[4]], dim=0)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                weighted = LOSS_WEIGHTS["value_reachable"] * loss
            scaler.scale(weighted).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_totals["value_reachable"] += float(loss.detach().cpu())
            counts["value_reachable"] += 1

        step_eval = eval_step_policy(model.step_policy, val_states, step_vocab, step_priors, device, max_states=5000)
        rank_eval = evaluate_samples(model.reranker, val_rank_samples, rerank_vocab, device, max_eval_samples=250)
        row = {
            "epoch": epoch,
            "loss": {key: loss_totals[key] / max(1, counts[key]) for key in LOSS_WEIGHTS},
            "counts": counts,
            "val_step_policy": step_eval,
            "val_listwise_rank": rank_eval,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step_vocab": step_vocab.to_jsonable(),
            "rerank_vocab": rerank_vocab.to_jsonable(),
            "element_to_id": element_to_id,
            "sg_to_id": sg_to_id,
            "orbit_to_index": orbit_to_index,
            "config": vars(args),
            "history": history,
            "loss_weights": LOSS_WEIGHTS,
        },
        args.run_dir / "ckpt.pt",
    )
    predictions = rank_split(model.reranker, args.data_root, args.candidate_dir, "test", rerank_vocab, device)
    write_jsonl(args.out_dir / "test_multitask_reranked_predictions.jsonl", predictions)
    summary = {
        "config": {**jsonable_config(args), "device": str(device)},
        "loss_weights": LOSS_WEIGHTS,
        "train_state_skipped": len(train_state_skipped),
        "val_state_skipped": len(val_state_skipped),
        "rank_samples": len(rank_samples),
        "rank_skipped": len(rank_skipped),
        "val_rank_samples": len(val_rank_samples),
        "val_rank_skipped": len(val_rank_skipped),
        "test": topk_metrics(predictions),
        "history": history,
    }
    (args.out_dir / "structured_multitask_wa_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
