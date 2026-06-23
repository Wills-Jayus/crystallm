#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine
from symcif_v4.orbit_token import OrbitToken
from symcif_v4.step_policy import (
    NUMERIC_DIM,
    StepPolicyNet,
    StepPolicyVocab,
    build_frequency_priors,
    canonical_sequence,
    encode_action_batch,
    vocab_sizes,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def legal_actions(
    *,
    sg: int,
    formula_counts: dict[str, int],
    remaining_counts: dict[str, int],
    orbits: list[OrbitToken],
    used_fixed_orbits: set[str],
) -> list[tuple[str, OrbitToken]]:
    actions: list[tuple[str, OrbitToken]] = []
    for element in sorted(formula_counts):
        remaining = int(remaining_counts.get(element, 0))
        if remaining <= 0:
            continue
        for orbit in orbits:
            if int(orbit.sg) != int(sg):
                continue
            if remaining < int(orbit.multiplicity):
                continue
            if orbit.is_fully_fixed and orbit.canonical_orbit_id in used_fixed_orbits:
                continue
            actions.append((str(element), orbit))
    return actions


def orbit_from_row(engine: OrbitEngine, row: dict[str, Any]) -> OrbitToken:
    return engine.get_orbit_by_id(str(row["orbit_id"]))


TARGET_SGS = {2, 65, 71, 127}


def rare_sgs(records: list[dict[str, Any]], max_count: int) -> set[int]:
    counts = Counter(int(r.get("sg", 0)) for r in records)
    return {sg for sg, count in counts.items() if count <= int(max_count)}


def max_orbit_multiplicity(record: dict[str, Any]) -> int:
    return max((int(row.get("multiplicity", 1)) for row in record.get("wa_table", [])), default=1)


def complex_record_weight(
    record: dict[str, Any],
    boost: float,
    *,
    rare_sg_set: set[int],
    high_orbit_threshold: int,
) -> float:
    if float(boost) <= 1.0:
        return 1.0
    hits = 0
    if int(record.get("n_sites", 0)) >= 6:
        hits += 1
    if int(record.get("num_elements", 0)) >= 4:
        hits += 1
    if int(record.get("sg", 0)) in TARGET_SGS or int(record.get("sg", 0)) in rare_sg_set:
        hits += 1
    if max_orbit_multiplicity(record) >= int(high_orbit_threshold):
        hits += 1
    return 1.0 + (float(boost) - 1.0) * float(hits)


def make_training_states(
    records: list[dict[str, Any]],
    engine: OrbitEngine,
    max_actions_per_state: int | None,
    seed: int,
    record_weights: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    record_weights = record_weights or {}
    states: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in records:
        sg = int(record["sg"])
        formula_counts = {str(k): int(v) for k, v in record["formula_counts"].items()}
        remaining = dict(formula_counts)
        used_fixed: set[str] = set()
        orbits = engine.get_orbits(sg)
        sequence = canonical_sequence(record)
        for step_index, row in enumerate(sequence):
            target_orbit = orbit_from_row(engine, row)
            target = (str(row["element"]), target_orbit.canonical_orbit_id)
            actions = legal_actions(
                sg=sg,
                formula_counts=formula_counts,
                remaining_counts=remaining,
                orbits=orbits,
                used_fixed_orbits=used_fixed,
            )
            target_indexes = [i for i, (element, orbit) in enumerate(actions) if (element, orbit.canonical_orbit_id) == target]
            if not target_indexes:
                skipped.append(
                    {
                        "sample_id": record.get("sample_id"),
                        "step_index": step_index,
                        "reason": "target_not_legal",
                        "target": f"{target[0]}@{target[1]}",
                    }
                )
                break
            target_index = target_indexes[0]
            if max_actions_per_state is not None and len(actions) > int(max_actions_per_state):
                keep = {target_index}
                others = [i for i in range(len(actions)) if i != target_index]
                rng.shuffle(others)
                keep.update(others[: max(0, int(max_actions_per_state) - 1)])
                sorted_keep = sorted(keep)
                remap = {old: new for new, old in enumerate(sorted_keep)}
                actions = [actions[i] for i in sorted_keep]
                target_index = remap[target_index]
            states.append(
                {
                    "sg": sg,
                    "formula_counts": dict(formula_counts),
                    "remaining_counts": dict(remaining),
                    "actions": actions,
                    "target_index": int(target_index),
                    "step_index": int(step_index),
                    "chosen_count": int(step_index),
                    "sample_id": record.get("sample_id"),
                    "sample_weight": float(record_weights.get(str(record.get("sample_id")), 1.0)),
                }
            )
            element = str(row["element"])
            remaining[element] = int(remaining[element]) - int(target_orbit.multiplicity)
            if target_orbit.is_fully_fixed:
                used_fixed.add(target_orbit.canonical_orbit_id)
    return states, skipped


def move_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(t.to(device, non_blocking=True) for t in batch)  # type: ignore[return-value]


def state_loss(
    model: StepPolicyNet,
    state: dict[str, Any],
    vocab: StepPolicyVocab,
    priors: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = encode_action_batch(
        sg=int(state["sg"]),
        formula_counts=state["formula_counts"],
        remaining_counts=state["remaining_counts"],
        element_orbits=state["actions"],
        vocab=vocab,
        priors=priors,
        step_index=int(state["step_index"]),
        chosen_count=int(state["chosen_count"]),
    )
    logits = model(*move_batch(batch, device))
    target = torch.tensor([int(state["target_index"])], dtype=torch.long, device=device)
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    return loss, logits


@torch.no_grad()
def evaluate_states(
    model: StepPolicyNet,
    states: list[dict[str, Any]],
    vocab: StepPolicyVocab,
    priors: dict[str, Any],
    device: torch.device,
    max_states: int | None = None,
) -> dict[str, Any]:
    model.eval()
    rows = states if max_states is None else states[: int(max_states)]
    total_loss = 0.0
    hits = {1: 0, 5: 0, 20: 0}
    mrr = 0.0
    for state in rows:
        loss, logits = state_loss(model, state, vocab, priors, device)
        total_loss += float(loss.detach().cpu())
        target = int(state["target_index"])
        order = torch.argsort(logits, descending=True).detach().cpu().tolist()
        rank = order.index(target) + 1
        mrr += 1.0 / float(rank)
        for k in hits:
            if rank <= k:
                hits[k] += 1
    denom = max(1, len(rows))
    return {
        "states": len(rows),
        "loss": total_loss / denom,
        "top1": hits[1] / denom,
        "top5": hits[5] / denom,
        "top20": hits[20] / denom,
        "mrr": mrr / denom,
    }


def jsonable_priors(priors: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {name: {str(k): int(v) for k, v in counter.items()} for name, counter in priors.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a step-level element@OrbitToken policy for SymCIF-v4 WA search.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "step_policy_wa_v1")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_policy_search")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-actions-per-state", type=int, default=256)
    parser.add_argument("--eval-max-states", type=int, default=None)
    parser.add_argument("--complex-weight", type=float, default=1.0)
    parser.add_argument("--rare-sg-max-count", type=int, default=25)
    parser.add_argument("--high-orbit-threshold", type=int, default=12)
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
    vocab = StepPolicyVocab.from_records(train_records)
    priors = build_frequency_priors(train_records)
    rare_sg_set = rare_sgs(train_records, args.rare_sg_max_count)
    train_weights = {
        str(r.get("sample_id")): complex_record_weight(
            r,
            args.complex_weight,
            rare_sg_set=rare_sg_set,
            high_orbit_threshold=args.high_orbit_threshold,
        )
        for r in train_records
    }
    train_states, train_skipped = make_training_states(
        train_records,
        engine,
        args.max_actions_per_state,
        args.seed,
        record_weights=train_weights,
    )
    val_states, val_skipped = make_training_states(val_records, engine, None, args.seed)

    model = StepPolicyNet(vocab_sizes(vocab), numeric_dim=NUMERIC_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, Any]] = []
    best_val = -1.0
    best_path = args.run_dir / "ckpt.pt"
    print(
        json.dumps(
            {
                "device": str(device),
                "torch_cuda_available": torch.cuda.is_available(),
                "train_records": len(train_records),
                "val_records": len(val_records),
                "train_states": len(train_states),
                "val_states": len(val_states),
                "train_skipped": len(train_skipped),
                "val_skipped": len(val_skipped),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        random.shuffle(train_states)
        total_loss = 0.0
        for i, state in enumerate(train_states, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = state_loss(model, state, vocab, priors, device)
            loss = loss * float(state.get("sample_weight", 1.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            if i % 5000 == 0:
                print(f"[train] epoch={epoch} states={i}/{len(train_states)} loss={total_loss / i:.4f}", flush=True)
        train_eval = evaluate_states(model, train_states, vocab, priors, device, max_states=args.eval_max_states)
        val_eval = evaluate_states(model, val_states, vocab, priors, device, max_states=args.eval_max_states)
        row = {
            "epoch": epoch,
            "train_loss_online": total_loss / max(1, len(train_states)),
            "train_eval": train_eval,
            "val_eval": val_eval,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if float(val_eval["top5"]) > best_val:
            best_val = float(val_eval["top5"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab.to_jsonable(),
                    "vocab_sizes": vocab_sizes(vocab),
                    "numeric_dim": NUMERIC_DIM,
                    "history": history,
                "priors": jsonable_priors(priors),
                "config": vars(args),
                "oversampling": {
                    "complex_weight": float(args.complex_weight),
                    "rare_sg_max_count": int(args.rare_sg_max_count),
                    "rare_sg_count": len(rare_sg_set),
                    "rare_sgs": sorted(rare_sg_set),
                    "high_orbit_threshold": int(args.high_orbit_threshold),
                    "mean_record_weight": sum(train_weights.values()) / max(1, len(train_weights)),
                    "max_record_weight": max(train_weights.values()) if train_weights else 1.0,
                },
                "device": str(device),
            },
                best_path,
            )

    summary = {
        "device": str(device),
        "torch_cuda_available": torch.cuda.is_available(),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_states": len(train_states),
        "val_states": len(val_states),
        "train_skipped": train_skipped[:200],
        "val_skipped": val_skipped[:200],
        "history": history,
        "best_checkpoint": str(best_path),
        "best_val_top5": best_val,
        "oversampling": {
            "complex_weight": float(args.complex_weight),
            "rare_sg_max_count": int(args.rare_sg_max_count),
            "rare_sg_count": len(rare_sg_set),
            "rare_sgs": sorted(rare_sg_set),
            "high_orbit_threshold": int(args.high_orbit_threshold),
            "mean_record_weight": sum(train_weights.values()) / max(1, len(train_weights)),
            "max_record_weight": max(train_weights.values()) if train_weights else 1.0,
        },
    }
    (args.out_dir / "step_policy_training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
