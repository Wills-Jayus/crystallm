#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SCRIPT_DIR = OPENTRY_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_wyckoff_sequence_models import (  # noqa: E402
    Vocab,
    build_vocab,
    ensure_under_opentry,
    hit_at,
    read_jsonl,
    seq_key_from_orbits,
    vocab_from_jsonable,
    vocab_to_jsonable,
    write_json,
    write_jsonl,
)


def formula_count_features(record: dict[str, Any], vocab: Vocab, count_scale: float) -> torch.Tensor:
    frac = torch.zeros((len(vocab.elements),), dtype=torch.float32)
    counts = torch.zeros((len(vocab.elements),), dtype=torch.float32)
    total = max(1, int(record["atom_count"]))
    scale = max(1.0, float(count_scale))
    for element, value in record["formula_counts"].items():
        idx = vocab.element_to_id.get(str(element))
        if idx is None:
            continue
        count = int(value)
        frac[idx] = float(count) / float(total)
        counts[idx] = min(float(count) / scale, 4.0)
    return torch.cat([frac, counts], dim=0)


def record_weight(record: dict[str, Any], complex_weight: float) -> float:
    weight = 1.0
    if bool(record.get("complex_flag")):
        weight *= float(complex_weight)
    if int(record.get("row_count", 0)) >= 7:
        weight *= 1.5
    if int(record.get("atom_count", 0)) >= 12:
        weight *= 1.15
    return float(weight)


def make_states(records: list[dict[str, Any]], vocab: Vocab, *, complex_weight: float, count_scale: float) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for rec in records:
        seq = [str(row["orbit_id"]) for row in rec["skeleton_sequence"]]
        if any(oid not in vocab.orbit_to_id for oid in seq):
            continue
        total = int(rec["atom_count"])
        remaining = total
        last = len(vocab.orbits) + 1
        formula_feat = formula_count_features(rec, vocab, count_scale)
        weight = record_weight(rec, complex_weight)
        for step, orbit_id in enumerate(seq):
            states.append(
                {
                    "sg_id": vocab.sg_to_id.get(int(rec["sg"]), 0),
                    "sg": int(rec["sg"]),
                    "formula_feat": formula_feat,
                    "last": last,
                    "numeric": torch.tensor(
                        [
                            float(remaining) / max(1.0, float(total)),
                            min(float(remaining) / max(1.0, float(count_scale)), 4.0),
                            min(float(total) / max(1.0, float(count_scale)), 4.0),
                            float(len(rec["formula_counts"])) / 16.0,
                            float(step) / 64.0,
                        ],
                        dtype=torch.float32,
                    ),
                    "target": vocab.orbit_to_id[orbit_id],
                    "weight": weight,
                }
            )
            remaining -= int(vocab.orbit_info[orbit_id].multiplicity)
            last = vocab.orbit_to_id[orbit_id]
        states.append(
            {
                "sg_id": vocab.sg_to_id.get(int(rec["sg"]), 0),
                "sg": int(rec["sg"]),
                "formula_feat": formula_feat,
                "last": last,
                "numeric": torch.tensor(
                    [
                        0.0,
                        0.0,
                        min(float(total) / max(1.0, float(count_scale)), 4.0),
                        float(len(rec["formula_counts"])) / 16.0,
                        float(len(seq)) / 64.0,
                    ],
                    dtype=torch.float32,
                ),
                "target": len(vocab.orbits),
                "weight": weight,
            }
        )
    return states


class CountAwareSkeletonNet(nn.Module):
    def __init__(self, num_sgs: int, formula_dim: int, num_orbit_tokens: int, hidden: int) -> None:
        super().__init__()
        emb = max(32, hidden // 4)
        self.sg_emb = nn.Embedding(num_sgs + 1, emb)
        self.last_emb = nn.Embedding(num_orbit_tokens, emb)
        self.net = nn.Sequential(
            nn.LayerNorm(formula_dim + emb * 2 + 5),
            nn.Linear(formula_dim + emb * 2 + 5, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_orbit_tokens - 1),
        )

    def forward(self, sg_id: torch.Tensor, formula_feat: torch.Tensor, last: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.sg_emb(sg_id), self.last_emb(last), formula_feat, numeric], dim=-1)
        return self.net(x)


def batch_indexes(n: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    indexes = list(range(n))
    rng.shuffle(indexes)
    return [indexes[i : i + batch_size] for i in range(0, n, batch_size)]


def tensor_batch(states: list[dict[str, Any]], indexes: list[int], key: str, device: torch.device) -> torch.Tensor:
    values = [states[i][key] for i in indexes]
    if values and isinstance(values[0], torch.Tensor):
        return torch.stack(values).to(device)
    dtype = torch.long if key in {"sg_id", "last", "target"} else torch.float32
    return torch.tensor(values, dtype=dtype, device=device)


def train_epoch(
    model: CountAwareSkeletonNet,
    optimizer: torch.optim.Optimizer,
    states: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> float:
    model.train()
    rng = random.Random(seed)
    total_loss = 0.0
    total_batches = 0
    for indexes in batch_indexes(len(states), batch_size, rng):
        optimizer.zero_grad(set_to_none=True)
        sg = tensor_batch(states, indexes, "sg_id", device)
        formula = tensor_batch(states, indexes, "formula_feat", device)
        last = tensor_batch(states, indexes, "last", device)
        numeric = tensor_batch(states, indexes, "numeric", device)
        target = tensor_batch(states, indexes, "target", device)
        weight = tensor_batch(states, indexes, "weight", device)
        logits = model(sg, formula, last, numeric)
        loss_vec = F.cross_entropy(logits, target, reduction="none")
        loss = (loss_vec * weight).sum() / weight.sum().clamp_min(1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(1, total_batches)


def legal_actions(vocab: Vocab, sg: int, remaining: int) -> list[int]:
    eos = len(vocab.orbits)
    if int(remaining) == 0:
        return [eos]
    out: list[int] = []
    for orbit_id in vocab.sg_orbits.get(int(sg), []):
        if int(vocab.orbit_info[orbit_id].multiplicity) <= int(remaining):
            out.append(vocab.orbit_to_id[orbit_id])
    return out


@torch.no_grad()
def beam_skeleton_count_aware(
    model: CountAwareSkeletonNet,
    record: dict[str, Any],
    vocab: Vocab,
    device: torch.device,
    *,
    beam_size: int,
    branch: int,
    max_steps: int,
    count_scale: float,
) -> list[dict[str, Any]]:
    sg = int(record["sg"])
    total = int(record["atom_count"])
    sg_id = torch.tensor([vocab.sg_to_id.get(sg, 0)], dtype=torch.long, device=device)
    formula_feat = formula_count_features(record, vocab, count_scale).unsqueeze(0).to(device)
    start_id = len(vocab.orbits) + 1
    eos_id = len(vocab.orbits)
    beams: list[dict[str, Any]] = [{"orbits": [], "last": start_id, "remaining": total, "score": 0.0}]
    complete: dict[str, dict[str, Any]] = {}
    model.eval()
    for step in range(int(max_steps)):
        next_beams: list[dict[str, Any]] = []
        legal_by_beam = [legal_actions(vocab, sg, int(beam["remaining"])) for beam in beams]
        active = [(beam, legal) for beam, legal in zip(beams, legal_by_beam) if legal]
        if not active:
            break
        last_batch = torch.tensor([int(beam["last"]) for beam, _ in active], dtype=torch.long, device=device)
        numeric_batch = torch.tensor(
            [
                [
                    float(beam["remaining"]) / max(1.0, float(total)),
                    min(float(beam["remaining"]) / max(1.0, float(count_scale)), 4.0),
                    min(float(total) / max(1.0, float(count_scale)), 4.0),
                    float(len(record["formula_counts"])) / 16.0,
                    float(step) / 64.0,
                ]
                for beam, _ in active
            ],
            dtype=torch.float32,
            device=device,
        )
        sg_batch = sg_id.expand(len(active))
        formula_batch = formula_feat.expand(len(active), -1)
        logits_batch = model(sg_batch, formula_batch, last_batch, numeric_batch)
        for beam_idx, (beam, legal) in enumerate(active):
            if not legal:
                continue
            legal_tensor = torch.tensor(legal, dtype=torch.long, device=device)
            scores = F.log_softmax(logits_batch[beam_idx, legal_tensor], dim=-1)
            top_n = min(len(legal), int(branch))
            values, indexes = torch.topk(scores, k=top_n)
            for value, local_idx in zip(values.detach().cpu().tolist(), indexes.detach().cpu().tolist()):
                action = legal[int(local_idx)]
                score = float(beam["score"]) + float(value)
                if action == eos_id:
                    key = seq_key_from_orbits(list(beam["orbits"]))
                    if key and key not in complete:
                        complete[key] = {
                            "skeleton_key": key,
                            "orbits": list(beam["orbits"]),
                            "score": score,
                            "source": "count_aware_skeleton_model",
                            "composition_exact": int(beam["remaining"]) == 0,
                        }
                    continue
                orbit_id = vocab.orbits[action]
                mult = int(vocab.orbit_info[orbit_id].multiplicity)
                if mult > int(beam["remaining"]):
                    continue
                next_beams.append(
                    {
                        "orbits": [*beam["orbits"], orbit_id],
                        "last": int(action),
                        "remaining": int(beam["remaining"]) - mult,
                        "score": score,
                    }
                )
        next_beams.sort(key=lambda item: item["score"] / max(1, len(item["orbits"])), reverse=True)
        beams = next_beams[: int(beam_size)]
        if not beams:
            break
    ranked = sorted(complete.values(), key=lambda item: item["score"] / max(1, len(item["orbits"])), reverse=True)
    return ranked[: int(beam_size)]


@torch.no_grad()
def evaluate(
    model: CountAwareSkeletonNet,
    records: list[dict[str, Any]],
    vocab: Vocab,
    device: torch.device,
    *,
    beam_size: int,
    branch: int,
    max_steps: int,
    count_scale: float,
    out_candidates: Path | None = None,
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    out_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        candidates = beam_skeleton_count_aware(
            model,
            record,
            vocab,
            device,
            beam_size=beam_size,
            branch=branch,
            max_steps=max_steps,
            count_scale=count_scale,
        )
        keys = [str(x["skeleton_key"]) for x in candidates]
        row = {
            "index": idx,
            "sample_id": record["keys"].get("sample_id"),
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "row_count": int(record["row_count"]),
            "atom_count": int(record["atom_count"]),
            "complex_flag": bool(record.get("complex_flag")),
            "target_skeleton_key": str(record["canonical_skeleton_key"]),
            "candidate_count": len(candidates),
            "unique_skeleton": len(set(keys)),
            "composition_exact_any": any(bool(x.get("composition_exact")) for x in candidates),
        }
        for k in (1, 5, 20, 50):
            row[f"skeleton_hit@{k}"] = hit_at(keys, str(record["canonical_skeleton_key"]), k)
        per_sample.append(row)
        out_rows.append({"sample_id": row["sample_id"], "skeleton_candidates": candidates[:50]})
    if out_candidates is not None:
        write_jsonl(out_candidates, out_rows)
    return {
        "full": summarize(per_sample),
        "rows_ge_7": summarize([x for x in per_sample if int(x["row_count"]) >= 7]),
        "atoms_ge_12": summarize([x for x in per_sample if int(x["atom_count"]) >= 12]),
        "per_sample": per_sample,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denom = max(1, len(rows))
    out: dict[str, Any] = {
        "samples": len(rows),
        "candidate_nonempty_rate": sum(int(x["candidate_count"] > 0) for x in rows) / denom,
        "composition_exact_rate": sum(int(x["composition_exact_any"]) for x in rows) / denom,
        "unique_skeleton_mean": sum(int(x["unique_skeleton"]) for x in rows) / denom,
    }
    for k in (1, 5, 20, 50):
        out[f"skeleton@{k}"] = sum(int(x[f"skeleton_hit@{k}"]) for x in rows) / denom
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a count-aware A-stage Wyckoff skeleton generator.")
    parser.add_argument("--data-dir", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=128)
    parser.add_argument("--eval-split", choices=["train", "val"], default="val")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--complex-weight", type=float, default=2.5)
    parser.add_argument("--count-scale", type=float, default=64.0)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--branch", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--eval-only-ckpt", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    start = time.time()
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

    if args.eval_only_ckpt is not None:
        ckpt = torch.load(args.eval_only_ckpt, map_location=device)
        vocab = vocab_from_jsonable(ckpt["vocab"])
        ckpt_config = ckpt.get("config", {})
        hidden = int(ckpt_config.get("hidden", args.hidden))
        count_scale = float(ckpt_config.get("count_scale", args.count_scale))
        val_records = read_jsonl(args.data_dir / f"{args.eval_split}.jsonl", int(args.max_val_records))
        model = CountAwareSkeletonNet(
            num_sgs=len(vocab.sgs),
            formula_dim=len(vocab.elements) * 2,
            num_orbit_tokens=len(vocab.orbits) + 2,
            hidden=hidden,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        final_eval = evaluate(
            model,
            val_records,
            vocab,
            device,
            beam_size=int(args.beam_size),
            branch=int(args.branch),
            max_steps=int(args.max_steps),
            count_scale=count_scale,
            out_candidates=out_dir / "skeleton_candidates.jsonl",
        )
        write_jsonl(out_dir / "skeleton_per_sample.jsonl", final_eval["per_sample"])
        summary = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "device": str(device),
            "ckpt_epoch": int(ckpt.get("epoch", -1)),
            "ckpt_score": float(ckpt.get("score", 0.0)),
            "val_records": len(val_records),
            "eval_split": str(args.eval_split),
            "vocab": {"elements": len(vocab.elements), "sgs": len(vocab.sgs), "orbits": len(vocab.orbits)},
            "final_eval": {k: v for k, v in final_eval.items() if k != "per_sample"},
            "elapsed_s_total": time.time() - start,
            "notes": [
                "Eval-only run; no training performed.",
                "Input uses formula fractions plus absolute element counts; row_count is not an input feature.",
            ],
        }
        write_json(out_dir / "training_summary.json", summary)
        write_json(out_dir / "skeleton_summary.json", {k: v for k, v in final_eval.items() if k != "per_sample"})
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return 0

    max_train = None if int(args.max_train_records) <= 0 else int(args.max_train_records)
    train_records = read_jsonl(args.data_dir / "train.jsonl", max_train)
    val_records = read_jsonl(args.data_dir / "val.jsonl", int(args.max_val_records))
    vocab = build_vocab(train_records)
    write_json(run_dir / "vocab.json", vocab_to_jsonable(vocab))
    states = make_states(train_records, vocab, complex_weight=args.complex_weight, count_scale=args.count_scale)
    model = CountAwareSkeletonNet(
        num_sgs=len(vocab.sgs),
        formula_dim=len(vocab.elements) * 2,
        num_orbit_tokens=len(vocab.orbits) + 2,
        hidden=int(args.hidden),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    summary: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "states": len(states),
        "vocab": {"elements": len(vocab.elements), "sgs": len(vocab.sgs), "orbits": len(vocab.orbits)},
        "notes": [
            "Input uses formula fractions plus absolute element counts; row_count is not an input feature.",
            "row_count/complex flags are used only for training weights and subset reporting.",
        ],
        "epochs": [],
    }
    write_json(out_dir / "training_summary_running.json", summary)
    best_score = -math.inf
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        loss = train_epoch(model, optimizer, states, batch_size=int(args.batch_size), device=device, seed=int(args.seed) + epoch)
        eval_result: dict[str, Any] | None = None
        if args.eval_every_epoch:
            eval_result = evaluate(
                model,
                val_records,
                vocab,
                device,
                beam_size=int(args.beam_size),
                branch=int(args.branch),
                max_steps=int(args.max_steps),
                count_scale=float(args.count_scale),
            )
            score = float(eval_result["full"]["skeleton@50"]) + 0.5 * float(eval_result["rows_ge_7"]["skeleton@50"])
        else:
            score = -float(loss)
        row = {
            "epoch": epoch,
            "loss": loss,
            "eval": None if eval_result is None else {k: v for k, v in eval_result.items() if k != "per_sample"},
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
                },
                run_dir / "best.pt",
            )

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    vocab = vocab_from_jsonable(ckpt["vocab"])
    model = CountAwareSkeletonNet(
        num_sgs=len(vocab.sgs),
        formula_dim=len(vocab.elements) * 2,
        num_orbit_tokens=len(vocab.orbits) + 2,
        hidden=int(args.hidden),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    final_eval = evaluate(
        model,
        val_records,
        vocab,
        device,
        beam_size=int(args.beam_size),
        branch=int(args.branch),
        max_steps=int(args.max_steps),
        count_scale=float(args.count_scale),
        out_candidates=out_dir / "skeleton_candidates.jsonl",
    )
    write_jsonl(out_dir / "skeleton_per_sample.jsonl", final_eval["per_sample"])
    summary["best_epoch"] = best_epoch
    summary["best_score"] = best_score
    summary["final_eval"] = {k: v for k, v in final_eval.items() if k != "per_sample"}
    summary["elapsed_s_total"] = time.time() - start
    write_json(out_dir / "training_summary.json", summary)
    write_json(out_dir / "skeleton_summary.json", {k: v for k, v in final_eval.items() if k != "per_sample"})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
