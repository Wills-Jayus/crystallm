#!/usr/bin/env python3
"""Train/evaluate a fixed-order MiniCFJoint smoke without candidate ranking."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
SYMCIF_PROJECT = WORKSPACE / "model/New_model/symcif_experiment"
SYMCIF_SCRIPTS = SYMCIF_PROJECT / "scripts"
for path in (SYMCIF_SCRIPTS, SYMCIF_PROJECT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("WORKDIR", str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache/transformers"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache/xdg"))
os.environ.setdefault("TMPDIR", str(ROOT / "tmp"))
os.environ.setdefault("WANDB_DIR", str(ROOT / "logs/wandb"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "cache/cuda"))

from run_generation_eval import evaluate_mode_with_hard_timeouts  # noqa: E402
from run_mp20_minicfjoint_v2 import (  # noqa: E402
    COORD_ORDER,
    NONE_ELEMENT,
    STOP_ORBIT,
    JointV2Dataset,
    MiniCFJointV2Net,
    OrbitEngine,
    build_v2_vocab,
    can_close_remaining_ordered,
    candidate_key_for,
    canonical_keys_from_rows,
    canonical_rows,
    case_payload_from_clean_records,
    collate_v2,
    decode_lattice,
    formula_tensors,
    lattice_stats,
    masked_cross_entropy,
    move_batch,
    pct,
    render_candidate,
)


SEEDS = [
    8000,
    8001,
    8002,
    8003,
    8004,
    8010,
    8011,
    8012,
    8013,
    8014,
    8020,
    8021,
    8022,
    8023,
    8024,
    8030,
    8031,
    8032,
    8033,
    8034,
]


def under_root(path: Path) -> Path:
    path = path.resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"refusing to write outside opentry_5: {path}")
    return path


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def is_rows_ge7(record: dict[str, Any]) -> bool:
    return int(record.get("n_sites", len(record.get("wa_table") or []))) >= 7


def select_train_curriculum(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limit = int(args.train_limit) if args.train_limit is not None else len(records)
    limit = min(limit, len(records))
    quota = max(0.0, min(1.0, float(args.rows_ge7_quota)))
    repeat = max(1, int(args.rows_ge7_repeat))
    if quota <= 0.0 and repeat <= 1:
        selected = list(records[:limit])
    else:
        complex_rows = [r for r in records if is_rows_ge7(r)]
        simple_rows = [r for r in records if not is_rows_ge7(r)]
        rng = random.Random(int(args.seed))
        rng.shuffle(complex_rows)
        rng.shuffle(simple_rows)
        target_complex = min(len(complex_rows), int(round(float(limit) * quota)))
        selected = complex_rows[:target_complex] + simple_rows[: max(0, limit - target_complex)]
        if len(selected) < limit:
            selected.extend(complex_rows[target_complex : target_complex + (limit - len(selected))])
        rng.shuffle(selected)
    selected_unique = len(selected)
    selected_complex = [r for r in selected if is_rows_ge7(r)]
    if repeat > 1 and selected_complex:
        selected = list(selected) + [dict(r) for r in selected_complex for _ in range(repeat - 1)]
    info = {
        "train_limit_requested": args.train_limit,
        "rows_ge7_quota": quota,
        "rows_ge7_repeat": repeat,
        "unique_selected": selected_unique,
        "unique_rows_ge7_selected": len(selected_complex),
        "effective_train_records": len(selected),
        "effective_rows_ge7_records": sum(1 for r in selected if is_rows_ge7(r)),
    }
    return selected, info


def load_clean_records(fold: str, train_limit: int | None, eval_limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    base = ROOT / "checkpoints" / f"minicfjoint_v2_{fold}_gate0_smoke" / "clean_data"
    train = read_jsonl(base / "clean_train.jsonl", train_limit)
    val = read_jsonl(base / "clean_val.jsonl", eval_limit)
    if not train or not val:
        raise RuntimeError(f"missing clean MiniCFJoint smoke data under {base}")
    return train, val, base


def choose_from_logits(
    logits: torch.Tensor,
    legal_ids: list[int],
    *,
    deterministic: bool,
    temperature: float,
    generator: torch.Generator | None,
) -> int:
    if not legal_ids:
        raise ValueError("empty legal action set")
    ids = torch.tensor(legal_ids, dtype=torch.long, device=logits.device)
    legal_logits = logits[ids]
    if deterministic:
        return int(ids[int(torch.argmax(legal_logits).detach().cpu())].item())
    temp = max(float(temperature), 1.0e-6)
    probs = torch.softmax(legal_logits / temp, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().detach().cpu()) <= 0.0:
        picked = torch.randint(len(legal_ids), (1,), generator=generator, device=logits.device)
    else:
        picked = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(ids[int(picked.item())].item())


def loss_v2_weighted(
    orbit_logits: torch.Tensor,
    element_logits: torch.Tensor,
    coords: torch.Tensor,
    lattice: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    coord_weight: float,
    lattice_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    orbit_loss = masked_cross_entropy(orbit_logits, batch["target_orbit_ids"], batch["orbit_legal"])
    element_valid = batch["target_element_ids"] >= 0
    element_legal = batch["element_legal"].clone()
    element_legal[..., 0] = element_legal[..., 0] | ~element_valid
    element_targets = batch["target_element_ids"].masked_fill(~element_valid, 0)
    element_loss = F.cross_entropy(element_logits.masked_fill(~element_legal, -1.0e9).reshape(-1, element_logits.shape[-1]), element_targets.reshape(-1), ignore_index=0)
    diff = torch.abs(coords - batch["coord_values"])
    wrapped = torch.minimum(diff, 1.0 - diff)
    coord_loss = (wrapped * batch["coord_mask"]).sum() / batch["coord_mask"].sum().clamp_min(1.0)
    lat_abs = torch.abs(lattice - batch["lattice_values"]) * batch["lattice_masks"]
    lattice_loss = lat_abs.sum() / batch["lattice_masks"].sum().clamp_min(1.0)
    loss = orbit_loss + element_loss + float(coord_weight) * coord_loss + float(lattice_weight) * lattice_loss
    return loss, {
        "orbit_loss": float(orbit_loss.detach().cpu()),
        "element_loss": float(element_loss.detach().cpu()),
        "coord_loss": float(coord_loss.detach().cpu()),
        "lattice_loss": float(lattice_loss.detach().cpu()),
    }


@torch.no_grad()
def decode_fixed_order_candidate(
    model: MiniCFJointV2Net,
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
    lattice_raw = model.lattice_head(ctx).squeeze(0)
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
        step_features: torch.Tensor,
        hidden: torch.Tensor | None,
        chosen_summary: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = remaining_counts_t / remaining_counts_t.sum().clamp_min(1.0)
        rem_vec = (formula_embs * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        prev_orbit = model.orbit_emb(torch.tensor([int(prev_orbit_id)], dtype=torch.long, device=device))
        prev_element = model.element_emb(torch.tensor([int(prev_element_id)], dtype=torch.long, device=device))
        step_emb = model.step_emb(torch.tensor([min(int(step_index), 64)], dtype=torch.long, device=device))
        prev_row = prev_orbit + prev_element
        new_summary = (chosen_summary * float(max(0, step_index)) + prev_row) / float(max(1, step_index + 1))
        inp = model.step_proj(torch.cat([ctx, rem_vec, prev_orbit, prev_element, step_emb, new_summary, step_features.view(1, -1)], dim=-1)).unsqueeze(1)
        out, hidden_out = model.gru(inp, hidden)
        h = out[:, -1, :]
        return model.orbit_head(h).squeeze(0), h.squeeze(0), hidden_out, new_summary.detach()

    rows: list[dict[str, Any]] = []
    params: dict[int, dict[str, float]] = {}
    remaining = {str(k): int(v) for k, v in record["formula_counts"].items()}
    last_key: tuple[Any, ...] | None = None
    prev_orbit = vocab.start_orbit_id
    prev_element = 0
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
        step_emb = model.step_emb(torch.tensor([min(int(step), 64)], dtype=torch.long, device=device))
        rem_weights = rem_vec / rem_vec.sum().clamp_min(1.0)
        rem_emb_for_coord = (formula_embs * rem_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        coord_pred = torch.sigmoid(
            model.coord_head(torch.cat([h.view(1, -1), ctx, rem_emb_for_coord, orbit_cond, elem_cond, step_emb, feats.view(1, -1)], dim=-1))
        ).squeeze(0)
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
        rows.append(row)
        remaining = next_remaining
        last_key = new_key
        prev_orbit = int(orbit_idx)
        prev_element = int(elem_idx)
        hidden = hidden_out
        summary = summary_out
    return None, "max_steps_exceeded"


def train_model(
    *,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    engine: OrbitEngine,
    args: argparse.Namespace,
    ckpt_dir: Path,
) -> tuple[MiniCFJointV2Net, Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
    vocab = build_v2_vocab(train_records + val_records, engine)
    lattice_mean, lattice_std = lattice_stats(train_records)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MiniCFJointV2Net(vocab, emb_dim=int(args.emb_dim), hidden_dim=int(args.hidden_dim)).to(device)
    torch.set_float32_matmul_precision("high")
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = DataLoader(
        JointV2Dataset(train_records),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda xs: collate_v2(xs, vocab=vocab, engine=engine, lattice_mean=mean_t.cpu(), lattice_std=std_t.cpu()),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        JointV2Dataset(val_records),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=lambda xs: collate_v2(xs, vocab=vocab, engine=engine, lattice_mean=mean_t.cpu(), lattice_std=std_t.cpu()),
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
            orbit_logits, element_logits, coords, lattice = model(batch)
            loss, parts = loss_v2_weighted(
                orbit_logits,
                element_logits,
                coords,
                lattice,
                batch,
                coord_weight=float(args.coord_loss_weight),
                lattice_weight=float(args.lattice_loss_weight),
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
            "orbit_loss": sums["orbit_loss"] / max(1, steps),
            "element_loss": sums["element_loss"] / max(1, steps),
            "coord_loss": sums["coord_loss"] / max(1, steps),
            "lattice_loss": sums["lattice_loss"] / max(1, steps),
            "seconds": time.time() - started,
        }
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
                    orbit_logits, element_logits, coords, lattice = model(batch)
                    loss, _parts = loss_v2_weighted(
                        orbit_logits,
                        element_logits,
                        coords,
                        lattice,
                        batch,
                        coord_weight=float(args.coord_loss_weight),
                        lattice_weight=float(args.lattice_loss_weight),
                    )
                    val_loss += float(loss.detach().cpu())
                    val_steps += 1
            row["val_loss_report_only"] = val_loss / max(1, val_steps)
            print(json.dumps({"stage": "fixed_order_joint_epoch", **row}, sort_keys=True), flush=True)
        history.append(row)

    ckpt_dir = under_root(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "vocab": vocab.to_json(),
            "lattice_mean": lattice_mean,
            "lattice_std": lattice_std,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "history": history,
            "best_train_loss": best_train_loss,
            "selection_rule": "lowest_train_loss_not_eval_label_selection",
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
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "history": history,
            "best_train_loss": best_train_loss,
            "selection_rule": "lowest_train_loss_not_eval_label_selection",
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
    }
    return model, vocab, mean_t, std_t, info


@torch.no_grad()
def generate_fixed_order_rows(
    *,
    records: list[dict[str, Any]],
    model: MiniCFJointV2Net,
    vocab: Any,
    engine: OrbitEngine,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    eval_device = torch.device("cpu")
    model = model.to(eval_device)
    rows: list[dict[str, Any]] = []
    started = time.time()
    seeds = SEEDS[: int(args.k)]
    print(json.dumps({"stage": "fixed_order_generation_start", "samples": len(records), "k": len(seeds)}, sort_keys=True), flush=True)
    for sample_index, record in enumerate(records):
        if sample_index and sample_index % 16 == 0:
            print(json.dumps({"stage": "fixed_order_generation_progress", "done": sample_index, "seconds": time.time() - started}, sort_keys=True), flush=True)
        target_skel, target_wa = canonical_keys_from_rows(canonical_rows(record))
        for gen_index, seed in enumerate(seeds):
            gen_started = time.time()
            temp = 1.0e-6 if gen_index == 0 else (float(args.temperature_medium) if gen_index <= 4 else float(args.temperature_diverse))
            candidate, error = decode_fixed_order_candidate(
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
                max_steps=int(args.max_steps),
            )
            if candidate is None:
                rows.append(
                    {
                        "mode": "baseline_opentry5_fixed_order_joint_smoke",
                        "sample_index": sample_index,
                        "sample_id": record["sample_id"],
                        "gen_index": gen_index,
                        "seed": int(seed),
                        "temperature": temp,
                        "raw_generation_success": False,
                        "generated_text": "",
                        "error": error or "decode_failed",
                        "formula_closure_success": False,
                        "atom_count_ok": False,
                        "skeleton_hit": False,
                        "wa_hit": False,
                        "candidate_order": "generation_index_then_fixed_seed",
                        "generation_time_seconds": time.time() - gen_started,
                    }
                )
                continue
            rendered = render_candidate(engine, record, candidate, gen_index, "fixed_order_joint")
            rows.append(
                {
                    "mode": "baseline_opentry5_fixed_order_joint_smoke",
                    "sample_index": sample_index,
                    "sample_id": record["sample_id"],
                    "gen_index": gen_index,
                    "seed": int(seed),
                    "temperature": temp,
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
                    "candidate_order": "generation_index_then_fixed_seed",
                    "generation_time_seconds": time.time() - gen_started,
                }
            )
    print(json.dumps({"stage": "fixed_order_generation_done", "samples": len(records), "seconds": time.time() - started}, sort_keys=True), flush=True)
    return rows


def summarize_fixed_order(
    records: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    top_ks: list[int],
) -> dict[str, Any]:
    gen_by_key = {(int(r["sample_index"]), int(r["gen_index"])): r for r in generation_rows}
    metrics_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gen_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        metrics_by_sample[int(row["sample_index"])].append(row)
    for row in generation_rows:
        gen_by_sample[int(row["sample_index"])].append(row)
    rows_ge_7 = {i for i, record in enumerate(records) if int(record.get("n_sites", len(record.get("wa_table") or []))) >= 7}

    out: dict[str, Any] = {"rows_ge_7_samples": len(rows_ge_7), "samples": len(records)}
    for k in top_ks:
        match_count = 0
        rms_values: list[float] = []
        readable = formula_ok = atom_count_ok = sg_ok = strict_valid = 0
        wa_hit_any = skeleton_hit_any = 0
        rows7_match_count = rows7_any = rows7_wa_hit_any = rows7_skeleton_hit_any = 0
        wa_hit_candidates = wa_hit_match_fail = 0
        skel_hit_candidates = skel_hit_match_fail = 0
        collision_like = 0
        candidate_count = max(1, len(records) * int(k))
        for idx, _record in enumerate(records):
            sample_metrics = sorted([m for m in metrics_by_sample.get(idx, []) if int(m.get("gen_index", 0)) < k], key=lambda m: int(m["gen_index"]))
            sample_gen = sorted([g for g in gen_by_sample.get(idx, []) if int(g.get("gen_index", 0)) < k], key=lambda g: int(g["gen_index"]))
            matched = [m for m in sample_metrics if m.get("match_ok") and m.get("rms") is not None]
            if matched:
                match_count += 1
                rms_values.append(min(float(m["rms"]) for m in matched))
            readable += int(any(m.get("pymatgen_readable") for m in sample_metrics))
            formula_ok += int(any(m.get("formula_ok") for m in sample_metrics))
            sg_ok += int(any(m.get("space_group_ok") for m in sample_metrics))
            atom_count_ok += int(any(bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("atom_count_ok")) for m in sample_metrics))
            strict_valid += int(
                any(
                    m.get("pymatgen_readable")
                    and m.get("formula_ok")
                    and m.get("space_group_ok")
                    and bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("atom_count_ok"))
                    for m in sample_metrics
                )
            )
            wa_hit_any += int(any(g.get("wa_hit") for g in sample_gen))
            skeleton_hit_any += int(any(g.get("skeleton_hit") for g in sample_gen))
            if idx in rows_ge_7:
                rows7_any += 1
                rows7_match_count += int(bool(matched))
                rows7_wa_hit_any += int(any(g.get("wa_hit") for g in sample_gen))
                rows7_skeleton_hit_any += int(any(g.get("skeleton_hit") for g in sample_gen))
            metric_by_gen = {int(m["gen_index"]): m for m in sample_metrics}
            for g in sample_gen:
                gi = int(g.get("gen_index", 0))
                m = metric_by_gen.get(gi, {})
                if g.get("wa_hit"):
                    wa_hit_candidates += 1
                    wa_hit_match_fail += int(not bool(m.get("match_ok")))
                if g.get("skeleton_hit"):
                    skel_hit_candidates += 1
                    skel_hit_match_fail += int(not bool(m.get("match_ok")))
                score = m.get("bond_length_score")
                if score is not None and float(score) < 1.0:
                    collision_like += 1
        out[f"top{k}"] = {
            "samples": len(records),
            f"match@{k}": match_count / max(1, len(records)),
            f"RMSE@{k}": float(sum(rms_values) / len(rms_values)) if rms_values else math.nan,
            f"matched_samples_for_RMSE@{k}": len(rms_values),
            f"readable@{k}": readable / max(1, len(records)),
            f"composition_exact@{k}": formula_ok / max(1, len(records)),
            f"atom_count_ok@{k}": atom_count_ok / max(1, len(records)),
            f"SG_Wyckoff_legal@{k}": sg_ok / max(1, len(records)),
            f"strict_valid@{k}": strict_valid / max(1, len(records)),
            f"positive_candidate_rate@{k}": sum(1 for m in metrics if int(m.get("gen_index", 0)) < k and m.get("match_ok")) / candidate_count,
            f"sample_positive_any@{k}": match_count / max(1, len(records)),
            f"skeleton_hit@{k}": skeleton_hit_any / max(1, len(records)),
            f"WA_hit@{k}": wa_hit_any / max(1, len(records)),
            f"WA_hit_but_match_fail_rate@{k}": wa_hit_match_fail / max(1, wa_hit_candidates),
            f"skeleton_hit_but_match_fail_rate@{k}": skel_hit_match_fail / max(1, skel_hit_candidates),
            f"collision_like_candidate_rate@{k}": collision_like / candidate_count,
            f"rows_ge_7_match@{k}": rows7_match_count / max(1, rows7_any),
            f"rows_ge_7_positive_any@{k}": rows7_match_count / max(1, rows7_any),
            f"rows_ge_7_WA_hit@{k}": rows7_wa_hit_any / max(1, rows7_any),
            f"rows_ge_7_skeleton_hit@{k}": rows7_skeleton_hit_any / max(1, rows7_any),
            f"rows_ge_7_new_positive_samples@{k}": rows7_match_count,
        }
    return out


def write_reports(
    *,
    args: argparse.Namespace,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    source_clean_dir: Path,
    train_info: dict[str, Any],
    curriculum_info: dict[str, Any],
    generation_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
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
        "seeds": SEEDS[: int(args.k)],
        "selection_rule": "checkpoint_by_train_loss_only; eval labels not used for candidate order",
        "train": train_info,
        "curriculum": curriculum_info,
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
                f"# {experiment_id} Fixed-Order Joint Generator Smoke ({args.fold})",
                "",
                f"- train samples: {len(train_records)}",
                f"- rows>=7 train records: {curriculum_info['effective_rows_ge7_records']}",
                f"- eval samples: {len(val_records)}",
                f"- checkpoint: `{report['paths']['checkpoint_dir']}`",
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
                "",
                "This is a non-oracle smoke. It does not satisfy the terminal gate unless later grouped folds, dev_gate, and val512 pass under the frozen protocol.",
            ]
        ),
    )
    append_text(
        ROOT / "reports/opentry_5_experiment_log.md",
        f"""
## {experiment_id}: Fixed-order MiniCFJoint-v2 smoke ({args.fold})
- Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
- Core hypothesis: a joint W/A + geometry decoder can produce native CIF candidates without any beam-score sorting.
- Difference vs historical failed routes: no selector, no reranker, no source-prior insertion, no candidate score ordering; each candidate is one fixed generation_index/seed path.
- Model/data side: model-side joint generator smoke on opentry_5 grouped clean data.
- Contains sorting/filtering: no.
- candidate order: generation_index_then_fixed_seed; invalid candidates remain in place.
- Read files: {safe_rel(source_clean_dir / 'clean_train.jsonl')}, {safe_rel(source_clean_dir / 'clean_val.jsonl')}, model/New_model/symcif_experiment/artifacts/wyckoff_lookup_full.json.
- Written files: {safe_rel(out_dir / 'report.json')}, {safe_rel(out_dir / 'generations.jsonl')}, {safe_rel(out_dir / 'metrics.jsonl')}, {safe_rel(ckpt_dir / 'best_train.pt')}, {safe_rel(ckpt_dir / 'last.pt')}.
- Data split: {args.fold} clean smoke; train={len(train_records)}, eval={len(val_records)}.
- Train curriculum: {json.dumps(curriculum_info, sort_keys=True)}.
- Test read: no.
- val512 read: no; val512 cumulative count unchanged.
- Model: MiniCFJointV2Net fixed-order decoder.
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
- RMSE@1/5/20: {summary['top1']['RMSE@1']}, {summary['top5']['RMSE@5']}, {summary['top20']['RMSE@20']}.
- rows>=7 match@1/5/20: {pct(summary['top1']['rows_ge_7_match@1'])}, {pct(summary['top5']['rows_ge_7_match@5'])}, {pct(summary['top20']['rows_ge_7_match@20'])}.
- rows>=7 positive-any@20: {pct(summary['top20']['rows_ge_7_positive_any@20'])}.
- rows>=7 new positives@20: {summary['top20']['rows_ge_7_new_positive_samples@20']}.
- W/A-hit match-fail@20: {pct(summary['top20']['WA_hit_but_match_fail_rate@20'])}.
- skeleton-hit match-fail@20: {pct(summary['top20']['skeleton_hit_but_match_fail_rate@20'])}.
- collision-like candidate rate@20: {pct(summary['top20']['collision_like_candidate_rate@20'])}.
- grouped dev folds consistent: pending until both fold_a and fold_b fixed-order smoke complete.
- Conclusion: non-oracle fixed-order generator path executed; terminal gate not met by this smoke alone.
- Gate pass: false.
- Stop model family: false.
- Next: run the counterpart fold and then decide whether to scale training or change objective.
""",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["fold_a", "fold_b"], required=True)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--train-limit", type=int, default=2029)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--temperature-medium", type=float, default=0.8)
    parser.add_argument("--temperature-diverse", type=float, default=1.05)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_PROJECT / "artifacts/wyckoff_lookup_full.json")
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--max-sites", type=int, default=300)
    parser.add_argument("--seed", type=int, default=8019)
    parser.add_argument("--rows-ge7-quota", type=float, default=0.0)
    parser.add_argument("--rows-ge7-repeat", type=int, default=1)
    parser.add_argument("--coord-loss-weight", type=float, default=25.0)
    parser.add_argument("--lattice-loss-weight", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    experiment_id = args.experiment_id or ("E8009" if args.fold == "fold_a" else "E8010")
    out_dir = ROOT / "eval" / f"fixed_order_joint_smoke_{experiment_id}_{args.fold}"
    ckpt_dir = ROOT / "checkpoints" / f"fixed_order_joint_smoke_{experiment_id}_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    raw_train_limit = None if float(args.rows_ge7_quota) > 0.0 or int(args.rows_ge7_repeat) > 1 else args.train_limit
    train_records_raw, val_records, source_clean_dir = load_clean_records(args.fold, raw_train_limit, args.eval_limit)
    train_records, curriculum_info = select_train_curriculum(train_records_raw, args)
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in train_records + val_records}
    engine = OrbitEngine(args.lookup_json, sg_symbols)
    train_info: dict[str, Any]
    model, vocab, mean_t, std_t, train_info = train_model(
        train_records=train_records,
        val_records=val_records,
        engine=engine,
        args=args,
        ckpt_dir=ckpt_dir,
    )
    generation_rows = generate_fixed_order_rows(records=val_records, model=model, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args)
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
        mode="baseline_opentry5_fixed_order_joint_smoke",
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
        generation_rows=generation_rows,
        metrics=metrics,
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
