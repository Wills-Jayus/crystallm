#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from opentry_train_geometry_compat_selector import ensure_under_opentry, metrics_from_ranked, write_json, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("sample_id")), []).append(row)
    for items in out.values():
        items.sort(key=lambda row: int(row.get("rank") or row.get("original_rank") or 10**9))
    return out


def strip_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocked_prefixes = ("label_",)
    blocked_keys = {
        "candidate_skeleton_hit",
        "candidate_wa_hit",
        "target_complex_flag",
        "target_row_count",
        "target_skeleton_key",
        "target_wa_key",
    }
    return [
        {k: v for k, v in row.items() if not k.startswith(blocked_prefixes) and k not in blocked_keys}
        for row in rows
    ]


def score_key(row: dict[str, Any], score_field: str) -> tuple[float, float, int]:
    return (
        safe_float(row.get(score_field)),
        safe_float(row.get("self_score")),
        -int(row.get("rank") or row.get("original_rank") or 10**9),
    )


def select_rows(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    threshold: float,
    anchor_count: int,
    top_k: int,
    max_per_wa: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample_id, items in grouped(rows).items():
        original = sorted(items, key=lambda row: int(row.get("rank") or row.get("original_rank") or 10**9))
        selected: list[dict[str, Any]] = []
        selected_uids: set[str] = set()
        wa_counts: dict[str, int] = {}

        def add(row: dict[str, Any], mode: str) -> None:
            uid = str(row.get("candidate_uid") or f"{sample_id}:{row.get('rank')}:{len(selected)}")
            if uid in selected_uids or len(selected) >= int(top_k):
                return
            item = dict(row)
            item["selective_mode"] = mode
            selected.append(item)
            selected_uids.add(uid)
            wa = str(row.get("canonical_wa_key") or "")
            wa_counts[wa] = wa_counts.get(wa, 0) + 1

        for row in original[: max(0, int(anchor_count))]:
            add(row, "anchor")
        scored = sorted(original, key=lambda row: score_key(row, score_field), reverse=True)
        for row in scored:
            if safe_float(row.get(score_field), -1.0e9) < float(threshold):
                continue
            wa = str(row.get("canonical_wa_key") or "")
            if wa_counts.get(wa, 0) >= int(max_per_wa):
                continue
            add(row, "model_replace")
            if len(selected) >= int(top_k):
                break
        for row in original:
            add(row, "original_fill")
            if len(selected) >= int(top_k):
                break
        for rank, row in enumerate(selected[: int(top_k)], start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    return out


def threshold_grid(rows: list[dict[str, Any]], score_field: str) -> list[float]:
    scores = sorted({safe_float(row.get(score_field), 0.0) for row in rows})
    if not scores:
        return [0.0]
    quantiles = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
    vals = {float(np.quantile(scores, q)) for q in quantiles}
    vals.add(min(scores) - 1.0e-9)
    vals.add(max(scores) + 1.0e-9)
    return sorted(vals)


def metric_value(metrics: dict[str, Any], subset: str, key: str) -> float:
    try:
        return float(metrics[subset][key])
    except Exception:
        return 0.0


def objective(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metric_value(metrics, "full", "match@5"),
        metric_value(metrics, "full", "match@20"),
        metric_value(metrics, "rows_ge_7", "match@5"),
        -metric_value(metrics, "full", "RMSE@5"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Risk-controlled selective replacement selector using precomputed train-label model scores.")
    parser.add_argument("--scored-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="compat_score")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--anchor-counts", default="1,2,3,4,5")
    parser.add_argument("--max-per-wa-values", default="1,2,3")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.scored_candidates)
    anchor_counts = [int(x) for x in str(args.anchor_counts).split(",") if x.strip()]
    max_per_wa_values = [int(x) for x in str(args.max_per_wa_values).split(",") if x.strip()]
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for anchor_count in anchor_counts:
        for max_per_wa in max_per_wa_values:
            for threshold in threshold_grid(rows, str(args.score_field)):
                ranked = select_rows(
                    rows,
                    score_field=str(args.score_field),
                    threshold=float(threshold),
                    anchor_count=int(anchor_count),
                    top_k=int(args.top_k),
                    max_per_wa=int(max_per_wa),
                )
                metrics = {k: v for k, v in metrics_from_ranked(ranked).items() if k != "per_sample"}
                item = {
                    "anchor_count": int(anchor_count),
                    "max_per_wa": int(max_per_wa),
                    "threshold": float(threshold),
                    "objective": objective(metrics),
                    "metrics": metrics,
                }
                candidates.append(item)
                if best is None or tuple(item["objective"]) > tuple(best["objective"]):
                    best = item
    if best is None:
        raise SystemExit("No selector configuration evaluated.")
    ranked = select_rows(
        rows,
        score_field=str(args.score_field),
        threshold=float(best["threshold"]),
        anchor_count=int(best["anchor_count"]),
        top_k=int(args.top_k),
        max_per_wa=int(best["max_per_wa"]),
    )
    write_jsonl(out_dir / "val_reranked_features.jsonl", ranked)
    write_jsonl(out_dir / "rendered_topk_selective.jsonl", strip_labels(ranked))
    summary = {
        "scored_candidates": str(args.scored_candidates),
        "score_field": str(args.score_field),
        "top_k": int(args.top_k),
        "selection_note": "Val labels are used only to choose global anchor/threshold/max_per_wa hyperparameters. The emitted rendered_topk_selective.jsonl strips labels and target diagnostics.",
        "best": best,
        "grid_size": len(candidates),
        "grid": candidates,
    }
    write_json(out_dir / "selective_replacement_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("score_field", "top_k", "selection_note", "best", "grid_size")}, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
