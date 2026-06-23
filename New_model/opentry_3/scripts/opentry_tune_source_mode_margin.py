#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
if str(OPENTRY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(OPENTRY_ROOT / "scripts"))

import opentry_train_source_mode_selector as train_source_mode  # noqa: E402


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


def parse_thresholds(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        if text in {"model", "-inf"}:
            value = -1.0e9
        elif text in {"heuristic", "inf"}:
            value = 1.0e9
        else:
            value = float(text)
        if value not in out:
            out.append(value)
    return out


def threshold_label(value: float) -> str:
    if value <= -1.0e8:
        return "pure_model"
    if value >= 1.0e8:
        return "heuristic_rank0"
    return f"margin>={value:g}"


def load_model(ckpt_path: Path, device: torch.device) -> tuple[train_source_mode.SourceModeScorer, dict[str, Any]]:
    ckpt = torch.load(ensure_under_opentry(ckpt_path), map_location="cpu")
    feature_names = [str(x) for x in ckpt["feature_names"]]
    config = dict(ckpt.get("config") or {})
    model = train_source_mode.SourceModeScorer(
        len(feature_names),
        hidden_dim=int(config.get("hidden_dim", 96)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    runtime = {
        "feature_names": feature_names,
        "feature_mean": torch.tensor(ckpt["feature_mean"], dtype=torch.float32, device=device),
        "feature_std": torch.tensor(ckpt["feature_std"], dtype=torch.float32, device=device),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_selected_error": ckpt.get("best_val_selected_error"),
    }
    return model, runtime


@torch.no_grad()
def score_group(
    model: train_source_mode.SourceModeScorer,
    runtime: dict[str, Any],
    group: dict[str, Any],
    device: torch.device,
) -> tuple[list[float], int]:
    feature_names = list(runtime["feature_names"])
    rows = [
        train_source_mode.vector_for_candidate(candidate, feature_names)
        for candidate in list(group.get("candidates") or [])
    ]
    if not rows:
        return [], 0
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    x = (x - runtime["feature_mean"]) / runtime["feature_std"].clamp_min(1e-4)
    scores = model(x).detach().cpu().tolist()
    raw_best_idx = int(max(range(len(scores)), key=lambda idx: float(scores[idx])))
    return [float(v) for v in scores], raw_best_idx


def choose_index(scores: list[float], raw_best_idx: int, threshold: float) -> tuple[int, float, bool]:
    if not scores:
        return 0, 0.0, False
    margin = float(scores[raw_best_idx] - scores[0])
    if raw_best_idx != 0 and margin < float(threshold):
        return 0, margin, True
    return int(raw_best_idx), margin, False


def summarize_groups(
    groups: list[dict[str, Any]],
    scored: dict[str, tuple[list[float], int]],
    threshold: float,
) -> dict[str, Any]:
    selected_errors: list[float] = []
    heuristic_errors: list[float] = []
    best_errors: list[float] = []
    margins: list[float] = []
    selected_ranks: list[float] = []
    hit_best = 0
    improved = 0
    overridden = 0
    model_nonzero = 0
    for group in groups:
        candidates = list(group.get("candidates") or [])
        if not candidates:
            continue
        scores, raw_best_idx = scored[str(group["sample_id"])]
        selected_idx, margin, overrode = choose_index(scores, raw_best_idx, threshold)
        selected = candidates[selected_idx]
        selected_error = float(selected["combined_error"])
        heuristic_error = float(group["heuristic_error"])
        best_error = float(group["best_error"])
        selected_errors.append(selected_error)
        heuristic_errors.append(heuristic_error)
        best_errors.append(best_error)
        margins.append(float(margin))
        selected_ranks.append(float(selected["source_rank"]))
        if selected_idx == int(group["best_source_index"]):
            hit_best += 1
        if selected_error < heuristic_error - 1e-8:
            improved += 1
        if overrode:
            overridden += 1
        if raw_best_idx != 0:
            model_nonzero += 1
    denom = max(1, len(selected_errors))
    return {
        "groups": len(selected_errors),
        "selected_error_mean": float(mean(selected_errors)) if selected_errors else None,
        "heuristic_error_mean": float(mean(heuristic_errors)) if heuristic_errors else None,
        "best_error_mean": float(mean(best_errors)) if best_errors else None,
        "selected_gain_vs_heuristic_mean": float(mean([h - s for h, s in zip(heuristic_errors, selected_errors)])) if selected_errors else None,
        "oracle_gain_vs_heuristic_mean": float(mean([h - b for h, b in zip(heuristic_errors, best_errors)])) if selected_errors else None,
        "hit_best_rate": float(hit_best / denom),
        "improved_over_heuristic_rate": float(improved / denom),
        "model_nonzero_choice_rate": float(model_nonzero / denom),
        "overrode_to_rank0_rate": float(overridden / denom),
        "selected_source_rank_mean": float(mean(selected_ranks)) if selected_ranks else None,
        "margin_mean": float(mean(margins)) if margins else None,
    }


def evaluate_split(
    groups: list[dict[str, Any]],
    model: train_source_mode.SourceModeScorer,
    runtime: dict[str, Any],
    thresholds: list[float],
    device: torch.device,
) -> dict[str, Any]:
    scored = {
        str(group["sample_id"]): score_group(model, runtime, group, device)
        for group in groups
    }
    rows_ge7 = [g for g in groups if bool(dict(g.get("complex_flags") or {}).get("rows_ge_7"))]
    atoms_ge12 = [g for g in groups if bool(dict(g.get("complex_flags") or {}).get("atoms_ge_12"))]
    out: dict[str, Any] = {}
    for threshold in thresholds:
        out[threshold_label(threshold)] = {
            "threshold": float(threshold),
            "full": summarize_groups(groups, scored, threshold),
            "rows_ge_7": summarize_groups(rows_ge7, scored, threshold),
            "atoms_ge_12": summarize_groups(atoms_ge12, scored, threshold),
        }
    return out


def write_tsv(path: Path, results: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "split",
        "label",
        "threshold",
        "subset",
        "selected_error_mean",
        "heuristic_error_mean",
        "best_error_mean",
        "selected_gain_vs_heuristic_mean",
        "hit_best_rate",
        "improved_over_heuristic_rate",
        "model_nonzero_choice_rate",
        "overrode_to_rank0_rate",
        "selected_source_rank_mean",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for split, split_result in results.items():
            for label, entry in split_result.items():
                for subset in ("full", "rows_ge_7", "atoms_ge_12"):
                    metrics = dict(entry[subset])
                    row = {
                        "split": split,
                        "label": label,
                        "threshold": entry["threshold"],
                        "subset": subset,
                        **metrics,
                    }
                    f.write("\t".join(str(row.get(col, "")) for col in cols) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune a no-move margin for the source-mode selector using train/val transfer labels.")
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--source-mode-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default="model,0,0.05,0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0,3.0,heuristic",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    examples_dir = ensure_under_opentry(args.examples_dir)
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, runtime = load_model(args.source_mode_ckpt, device)
    thresholds = parse_thresholds(str(args.thresholds))

    train_groups = read_jsonl(examples_dir / "train_source_mode_examples.jsonl")
    val_groups = read_jsonl(examples_dir / "val_source_mode_examples.jsonl")
    results = {
        "train": evaluate_split(train_groups, model, runtime, thresholds, device),
        "val": evaluate_split(val_groups, model, runtime, thresholds, device),
    }
    best_label = min(
        results["val"],
        key=lambda label: float(results["val"][label]["full"].get("selected_error_mean") or math.inf),
    )
    summary = {
        "examples_dir": str(examples_dir),
        "source_mode_ckpt": str(ensure_under_opentry(args.source_mode_ckpt)),
        "best_val_label": best_label,
        "best_val_threshold": float(results["val"][best_label]["threshold"]),
        "best_val_full": results["val"][best_label]["full"],
        "best_val_rows_ge_7": results["val"][best_label]["rows_ge_7"],
        "results": results,
    }
    write_json(out_dir / "source_mode_margin_tuning_summary.json", summary)
    write_tsv(out_dir / "source_mode_margin_tuning.tsv", results)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
