#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (1, 5, 20)


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_score_matrix(path: Path, *, max_rank: int | None = None) -> dict[str, Any]:
    df = pd.read_json(path, lines=True)
    required = {"sample_id", "material_id", "rank", "match", "rmsd", "target_rows_ge7", "score"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    if max_rank is not None:
        df = df[df["rank"] <= int(max_rank)].copy()
    df["match"] = df["match"].fillna(False).astype(bool)
    df = df.sort_values(["sample_id", "rank"], kind="mergesort").reset_index(drop=True)
    counts = df.groupby("sample_id", sort=False)["rank"].size().to_numpy()
    if len(counts) == 0 or counts.min() != counts.max():
        raise RuntimeError(f"{path} has incomplete sample rows")
    n_samples = int(len(counts))
    k = int(counts[0])
    ranks = df["rank"].to_numpy(dtype=np.int16).reshape(n_samples, k)
    expected = np.arange(1, k + 1, dtype=np.int16)
    if not np.all(ranks == expected[None, :]):
        raise RuntimeError(f"{path} ranks are not complete 1..{k} per sample")
    sample_ids = df["sample_id"].to_numpy(dtype=object).reshape(n_samples, k)[:, 0].astype(str)
    material_ids = df["material_id"].to_numpy(dtype=object).reshape(n_samples, k)[:, 0].astype(str)
    rows_ge7 = df["target_rows_ge7"].to_numpy(dtype=bool).reshape(n_samples, k)[:, 0]
    matches = df["match"].to_numpy(dtype=bool).reshape(n_samples, k)
    scores = df["score"].to_numpy(dtype=np.float64).reshape(n_samples, k)
    rmsd_series = pd.to_numeric(df["rmsd"], errors="coerce")
    rmsd = rmsd_series.to_numpy(dtype=np.float64).reshape(n_samples, k)
    model = str(df["model"].dropna().iloc[0]) if "model" in df.columns and df["model"].notna().any() else path.stem
    seed = int(df["seed"].dropna().iloc[0]) if "seed" in df.columns and df["seed"].notna().any() else -1
    return {
        "path": path,
        "model": model,
        "seed": seed,
        "sample_ids": sample_ids,
        "material_ids": material_ids,
        "rows_ge7": rows_ge7,
        "matches": matches,
        "rmsd": rmsd,
        "scores": scores,
        "k": k,
    }


def order_by_score(scores: np.ndarray) -> np.ndarray:
    n, k = scores.shape
    rank_tiebreak = np.tile(np.arange(k, dtype=np.int16), (n, 1))
    return np.lexsort((rank_tiebreak, -scores), axis=1)


def selected_indices(score_order: np.ndarray, *, strategy: str, anchor_keep: int | None) -> np.ndarray:
    n, k = score_order.shape
    if strategy == "baseline":
        return np.tile(np.arange(20, dtype=np.int16), (n, 1))
    if strategy == "unconstrained":
        return score_order[:, :20].astype(np.int16)
    if strategy == "conservative":
        if anchor_keep is None:
            raise ValueError("anchor_keep required")
        if anchor_keep > 20:
            raise ValueError("anchor_keep must be <= 20")
        out = np.empty((n, 20), dtype=np.int16)
        out[:, :anchor_keep] = np.arange(anchor_keep, dtype=np.int16)[None, :]
        fill = 20 - anchor_keep
        keep_mask = score_order >= anchor_keep
        for i in range(n):
            out[i, anchor_keep:] = score_order[i, keep_mask[i]][:fill]
        return out
    raise ValueError(strategy)


def per_sample_stats(matches: np.ndarray, rmsd: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    picked_match = np.take_along_axis(matches, indices, axis=1)
    picked_rmsd = np.take_along_axis(rmsd, indices, axis=1)
    stats: dict[str, np.ndarray] = {}
    for budget in BUDGETS:
        m = picked_match[:, :budget]
        stats[f"hit@{budget}"] = m.any(axis=1)
        masked_rmsd = np.where(m, picked_rmsd[:, :budget], np.nan)
        all_nan = np.isnan(masked_rmsd).all(axis=1)
        min_rmsd = np.nanmin(masked_rmsd, axis=1)
        min_rmsd[all_nan] = np.nan
        stats[f"rmsd@{budget}"] = min_rmsd
    return stats


def summarize(stats: dict[str, np.ndarray], rows_ge7: np.ndarray) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "samples": int(rows_ge7.shape[0]),
        "rows_ge7_samples": int(rows_ge7.sum()),
    }
    for budget in BUDGETS:
        hit = stats[f"hit@{budget}"]
        rmsd = stats[f"rmsd@{budget}"]
        hit7 = hit[rows_ge7]
        rmsd7 = rmsd[rows_ge7]
        out[f"match@{budget}"] = float(hit.mean())
        out[f"rows>=7_match@{budget}"] = None if len(hit7) == 0 else float(hit7.mean())
        out[f"rmsd@{budget}"] = None if np.isnan(rmsd).all() else float(np.nanmean(rmsd))
        out[f"rows>=7_rmsd@{budget}"] = None if np.isnan(rmsd7).all() else float(np.nanmean(rmsd7))
    return out


def mixed_summary(
    base: dict[str, np.ndarray],
    alt: dict[str, np.ndarray],
    route_mask: np.ndarray,
    rows_ge7: np.ndarray,
) -> dict[str, float | int | None]:
    stats: dict[str, np.ndarray] = {}
    for budget in BUDGETS:
        stats[f"hit@{budget}"] = np.where(route_mask, alt[f"hit@{budget}"], base[f"hit@{budget}"])
        stats[f"rmsd@{budget}"] = np.where(route_mask, alt[f"rmsd@{budget}"], base[f"rmsd@{budget}"])
    return summarize(stats, rows_ge7)


def deltas(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = [
        "match@1",
        "match@5",
        "match@20",
        "rows>=7_match@1",
        "rows>=7_match@5",
        "rows>=7_match@20",
    ]
    return {key: float(metrics[key] - baseline[key]) for key in keys}


def route_masks(scores: np.ndarray, score_order: np.ndarray, quantiles: list[float]) -> list[dict[str, Any]]:
    best_idx = score_order[:, 0]
    best_score = scores[np.arange(scores.shape[0]), best_idx]
    rank1_score = scores[:, 0]
    best_minus_rank1 = best_score - rank1_score
    best_rank = best_idx + 1
    specs: list[dict[str, Any]] = []
    for name, values in [("best_score", best_score), ("best_minus_rank1", best_minus_rank1)]:
        for q in quantiles:
            threshold = float(np.quantile(values, q))
            specs.append(
                {
                    "rule": f"{name}_ge_q{q:g}",
                    "threshold": threshold,
                    "mask": values >= threshold,
                }
            )
            threshold_le = float(np.quantile(values, 1.0 - q))
            specs.append(
                {
                    "rule": f"{name}_le_q{1.0 - q:g}",
                    "threshold": threshold_le,
                    "mask": values <= threshold_le,
                }
            )
    for max_rank in [1, 2, 3, 5, 10, 20, 50]:
        specs.append({"rule": f"best_rank_le_{max_rank}", "threshold": float(max_rank), "mask": best_rank <= max_rank})
    for min_rank in [2, 3, 5, 10, 20, 21]:
        specs.append({"rule": f"best_rank_ge_{min_rank}", "threshold": float(min_rank), "mask": best_rank >= min_rank})
    return [spec for spec in specs if bool(np.any(spec["mask"]))]


def passes_gate(row: dict[str, Any]) -> bool:
    d = row["deltas"]
    if d["match@20"] < -0.002 or d["rows>=7_match@20"] < -0.002:
        return False
    improved_budgets = [b for b in BUDGETS if d[f"match@{b}"] >= 0.01]
    if not improved_budgets:
        return False
    return all(d[f"rows>=7_match@{b}"] >= -0.002 for b in improved_budgets)


def make_report(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['dataset_label']} Fast K50 Score Route Sweep",
        "",
        f"Created: {result['created_at']}",
        "",
        "Validation-only sweep from saved OOF scores. Samples not routed keep baseline K20 order.",
        "",
        "## Baseline",
        "",
        "| metric | value | rows>=7 |",
        "| --- | ---: | ---: |",
    ]
    base = result["baseline_metrics"]
    for budget in BUDGETS:
        lines.append(
            f"| match@{budget} | {100.0 * base[f'match@{budget}']:.3f}% | "
            f"{100.0 * base[f'rows>=7_match@{budget}']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Top Routes",
            "",
            "| score file | strategy | anchor_keep | rule | routed | d@1 | rows7 d@1 | d@5 | rows7 d@5 | d@20 | rows7 d@20 |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["route_results"][:40]:
        d = row["deltas"]
        lines.append(
            f"| {Path(row['scores']).name} | {row['strategy']} | {row.get('anchor_keep')} | "
            f"{row['route_rule']} | {row['routed_samples']} | "
            f"{100*d['match@1']:.3f} pp | {100*d['rows>=7_match@1']:.3f} pp | "
            f"{100*d['match@5']:.3f} pp | {100*d['rows>=7_match@5']:.3f} pp | "
            f"{100*d['match@20']:.3f} pp | {100*d['rows>=7_match@20']:.3f} pp |"
        )
    best = result.get("best_gate_candidate")
    lines.extend(["", "## Best Gate Candidate", ""])
    if best is None:
        lines.append("No route satisfied the validation gate.")
    else:
        d = best["deltas"]
        lines.extend(
            [
                f"- Scores: {best['scores']}",
                f"- Model: {best['model']} seed={best['seed']}",
                f"- Strategy: {best['strategy']} anchor_keep={best.get('anchor_keep')}",
                f"- Route: {best['route_rule']} threshold={best['threshold']}",
                f"- Routed samples: {best['routed_samples']}",
                f"- match@1 delta: {100*d['match@1']:.3f} pp; rows>=7: {100*d['rows>=7_match@1']:.3f} pp",
                f"- match@5 delta: {100*d['match@5']:.3f} pp; rows>=7: {100*d['rows>=7_match@5']:.3f} pp",
                f"- match@20 delta: {100*d['match@20']:.3f} pp; rows>=7: {100*d['rows>=7_match@20']:.3f} pp",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized validation-only sweep for K50 OOF score routes.")
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--anchor-keeps", default="6,8,10,12,14,16,18")
    parser.add_argument("--quantiles", default="0.0,0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--dataset-label", default="MPTS-52")
    parser.add_argument("--max-rank", type=int, default=None, help="Optional rank cutoff for K30/K40 validation-only sweeps.")
    parser.add_argument("--out", default=str(ROOT / "metrics/mpts52_k50_score_route_sweep_fast.json"))
    parser.add_argument("--report", default=str(ROOT / "reports/mpts52_k50_score_route_sweep_fast.md"))
    args = parser.parse_args()

    anchor_keeps = [int(x) for x in args.anchor_keeps.split(",") if x.strip()]
    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    all_results: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None
    input_files: list[str] = []

    for score_arg in args.scores:
        data = load_score_matrix(Path(score_arg), max_rank=args.max_rank)
        input_files.append(str(data["path"].resolve()))
        order = order_by_score(data["scores"])
        baseline = per_sample_stats(data["matches"], data["rmsd"], selected_indices(order, strategy="baseline", anchor_keep=None))
        if baseline_metrics is None:
            baseline_metrics = summarize(baseline, data["rows_ge7"])
        alt_specs: list[tuple[str, int | None, dict[str, np.ndarray]]] = [
            ("unconstrained", None, per_sample_stats(data["matches"], data["rmsd"], selected_indices(order, strategy="unconstrained", anchor_keep=None)))
        ]
        for keep in anchor_keeps:
            alt_specs.append(
                (
                    "conservative",
                    keep,
                    per_sample_stats(data["matches"], data["rmsd"], selected_indices(order, strategy="conservative", anchor_keep=keep)),
                )
            )
        for route in route_masks(data["scores"], order, quantiles):
            mask = route["mask"].astype(bool)
            for strategy, anchor_keep, alt in alt_specs:
                metrics = mixed_summary(baseline, alt, mask, data["rows_ge7"])
                row = {
                    "scores": str(data["path"].resolve()),
                    "model": data["model"],
                    "seed": data["seed"],
                    "strategy": strategy,
                    "anchor_keep": anchor_keep,
                    "route_rule": route["rule"],
                    "threshold": route["threshold"],
                    "routed_samples": int(mask.sum()),
                    "metrics": metrics,
                    "deltas": deltas(metrics, baseline_metrics),
                }
                row["passes_gate"] = passes_gate(row)
                all_results.append(row)

    if baseline_metrics is None:
        raise RuntimeError("no score files provided")
    all_results.sort(
        key=lambda row: (
            int(row["passes_gate"]),
            max(row["deltas"]["match@1"], row["deltas"]["match@5"], row["deltas"]["match@20"]),
            row["deltas"]["match@20"],
            row["deltas"]["rows>=7_match@20"],
            row["deltas"]["rows>=7_match@1"],
            row["deltas"]["rows>=7_match@5"],
        ),
        reverse=True,
    )
    best = next((row for row in all_results if row["passes_gate"]), None)
    result = {
        "created_at": now_iso(),
        "dataset_label": str(args.dataset_label),
        "input_score_files": input_files,
        "baseline_metrics": baseline_metrics,
        "route_results": all_results,
        "best_gate_candidate": best,
        "gate": {
            "match@20_delta_min": -0.002,
            "rows>=7_match@20_delta_min": -0.002,
            "required_any_match_delta_min": 0.01,
            "rows>=7_corresponding_delta_min": -0.002,
        },
        "note": "Validation-only vectorized route sweep from OOF scores; no official test labels used.",
    }
    write_json(Path(args.out), result)
    write_text(Path(args.report), make_report(result))


if __name__ == "__main__":
    main()
