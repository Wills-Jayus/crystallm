#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch import nn


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = ensure_under_opentry(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}


class SourceModeScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def feature_names_from_groups(groups: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for group in groups:
        for cand in group.get("candidates") or []:
            names.update(str(key) for key in dict(cand.get("features") or {}))
    return sorted(names)


def vector_for_candidate(candidate: dict[str, Any], feature_names: list[str]) -> list[float]:
    features = dict(candidate.get("features") or {})
    values: list[float] = []
    for name in feature_names:
        raw = float(features.get(name, 0.0))
        if not math.isfinite(raw):
            raw = 0.0
        values.append(raw)
    return values


def feature_stats(groups: list[dict[str, Any]], feature_names: list[str]) -> tuple[list[float], list[float]]:
    rows: list[list[float]] = []
    for group in groups:
        for cand in group.get("candidates") or []:
            rows.append(vector_for_candidate(cand, feature_names))
    if not rows:
        raise SystemExit("No candidate feature rows")
    tensor = torch.tensor(rows, dtype=torch.float32)
    mean_t = tensor.mean(dim=0)
    std_t = tensor.std(dim=0).clamp_min(1e-4)
    return mean_t.tolist(), std_t.tolist()


def group_tensor(group: dict[str, Any], feature_names: list[str], mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    rows = [vector_for_candidate(cand, feature_names) for cand in group.get("candidates") or []]
    if not rows:
        return torch.zeros((0, len(feature_names)), dtype=torch.float32)
    x = torch.tensor(rows, dtype=torch.float32)
    return (x - mean_t) / std_t


def group_weight(group: dict[str, Any], complex_weight: float) -> float:
    if float(complex_weight) <= 1.0:
        return 1.0
    flags = dict(group.get("complex_flags") or {})
    hits = int(bool(flags.get("rows_ge_7"))) + int(bool(flags.get("atoms_ge_12"))) + int(bool(flags.get("num_elements_ge_4")))
    return 1.0 + (float(complex_weight) - 1.0) * float(hits)


@torch.no_grad()
def evaluate(
    model: SourceModeScorer,
    groups: list[dict[str, Any]],
    *,
    feature_names: list[str],
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()

    def subset(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"groups": 0}
        selected_errors: list[float] = []
        heuristic_errors: list[float] = []
        best_errors: list[float] = []
        hit_best = 0
        improved = 0
        selected_ranks: list[float] = []
        best_ranks: list[float] = []
        for group in items:
            x = group_tensor(group, feature_names, mean_t, std_t).to(device)
            if x.shape[0] == 0:
                continue
            scores = model(x)
            idx = int(torch.argmax(scores).detach().cpu().item())
            candidates = list(group.get("candidates") or [])
            selected = candidates[idx]
            selected_error = float(selected["combined_error"])
            heuristic_error = float(group["heuristic_error"])
            best_error = float(group["best_error"])
            selected_errors.append(selected_error)
            heuristic_errors.append(heuristic_error)
            best_errors.append(best_error)
            selected_ranks.append(float(selected["source_rank"]))
            best_ranks.append(float(group["best_source_rank"]))
            if idx == int(group["best_source_index"]):
                hit_best += 1
            if selected_error < heuristic_error - 1e-8:
                improved += 1
        denom = max(1, len(selected_errors))
        return {
            "groups": len(items),
            "selected_error_mean": float(mean(selected_errors)) if selected_errors else None,
            "heuristic_error_mean": float(mean(heuristic_errors)) if heuristic_errors else None,
            "best_error_mean": float(mean(best_errors)) if best_errors else None,
            "selected_gain_vs_heuristic_mean": float(mean([h - s for h, s in zip(heuristic_errors, selected_errors)])) if selected_errors else None,
            "oracle_gain_vs_heuristic_mean": float(mean([h - b for h, b in zip(heuristic_errors, best_errors)])) if selected_errors else None,
            "hit_best_rate": float(hit_best / denom),
            "improved_over_heuristic_rate": float(improved / denom),
            "selected_source_rank_mean": float(mean(selected_ranks)) if selected_ranks else None,
            "best_source_rank_mean": float(mean(best_ranks)) if best_ranks else None,
        }

    rows_ge7 = [g for g in groups if bool(dict(g.get("complex_flags") or {}).get("rows_ge_7"))]
    atoms_ge12 = [g for g in groups if bool(dict(g.get("complex_flags") or {}).get("atoms_ge_12"))]
    return {"full": subset(groups), "rows_ge_7": subset(rows_ge7), "atoms_ge_12": subset(atoms_ge12)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a small listwise source-mode selector from train-only geometry transfer labels.")
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--complex-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    examples_dir = ensure_under_opentry(args.examples_dir)
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_groups = read_jsonl(examples_dir / "train_source_mode_examples.jsonl")
    val_groups = read_jsonl(examples_dir / "val_source_mode_examples.jsonl")
    if not train_groups or not val_groups:
        raise SystemExit("No train/val source-mode groups")

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    feature_names = feature_names_from_groups(train_groups)
    mean_values, std_values = feature_stats(train_groups, feature_names)
    mean_t = torch.tensor(mean_values, dtype=torch.float32)
    std_t = torch.tensor(std_values, dtype=torch.float32)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = SourceModeScorer(len(feature_names), hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = out_dir / "ckpt_best.pt"
    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(train_groups)
        model.train()
        losses: list[float] = []
        for group in train_groups:
            x = group_tensor(group, feature_names, mean_t, std_t).to(device)
            if x.shape[0] <= 1:
                continue
            target = torch.tensor([int(group["best_source_index"])], dtype=torch.long, device=device)
            scores = model(x).unsqueeze(0)
            loss = nn.functional.cross_entropy(scores, target)
            loss = loss * float(group_weight(group, float(args.complex_weight)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_eval = evaluate(model, train_groups, feature_names=feature_names, mean_t=mean_t, std_t=std_t, device=device)
        val_eval = evaluate(model, val_groups, feature_names=feature_names, mean_t=mean_t, std_t=std_t, device=device)
        val_selected = val_eval["full"].get("selected_error_mean")
        row = {
            "epoch": epoch,
            "train_loss": float(mean(losses)) if losses else None,
            "train_full": train_eval["full"],
            "val_full": val_eval["full"],
            "val_rows_ge_7": val_eval["rows_ge_7"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if val_selected is not None and float(val_selected) < best_val:
            best_val = float(val_selected)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": feature_names,
                    "feature_mean": mean_values,
                    "feature_std": std_values,
                    "config": jsonable_args(args),
                    "best_epoch": epoch,
                    "best_val_selected_error": best_val,
                },
                best_path,
            )

    final_train = evaluate(model, train_groups, feature_names=feature_names, mean_t=mean_t, std_t=std_t, device=device)
    final_val = evaluate(model, val_groups, feature_names=feature_names, mean_t=mean_t, std_t=std_t, device=device)
    last_path = out_dir / "ckpt_last.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": feature_names,
            "feature_mean": mean_values,
            "feature_std": std_values,
            "config": jsonable_args(args),
            "best_epoch": None,
            "best_val_selected_error": best_val,
        },
        last_path,
    )
    write_json(
        out_dir / "source_mode_training_summary.json",
        {
            "config": jsonable_args(args),
            "feature_names": feature_names,
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "best_val_selected_error": best_val,
            "history": history,
            "final_train": final_train,
            "final_val": final_val,
            "leakage_guard": {
                "training_labels": "train lattice+free-param transfer error",
                "validation_labels": "val lattice+free-param transfer error for model selection only",
                "test_records_used": 0,
                "structurematcher_labels_used": False,
                "gt_lattice_or_free_params_in_features": False,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
