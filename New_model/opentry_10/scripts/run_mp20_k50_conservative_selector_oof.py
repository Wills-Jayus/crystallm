#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_mp20_k20_rerank_oof import (
    DEFAULT_FEATURES,
    DEFAULT_LABELS,
    DEFAULT_TRUE_TAR,
    ROOT,
    bootstrap_delta,
    load_table,
    model_specs,
    now_iso,
    run_oof,
    summarize_metrics,
    under_root,
    write_json,
    write_text,
)


def baseline_k20_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        ordered = group[group["rank"] <= 20].sort_values("rank")
        rows.append(sample_record(str(sample_id), ordered))
    return rows


def sample_record(sample_id: str, ordered: pd.DataFrame) -> dict[str, Any]:
    material_id = str(ordered["material_id"].iloc[0])
    rows_ge7 = bool(ordered["target_rows_ge7"].iloc[0])
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "material_id": material_id,
        "rows_ge7": rows_ge7,
        "selected_ranks": [int(x) for x in ordered["rank"].tolist()],
    }
    for budget in [1, 5, 20]:
        top = ordered.head(budget)
        matched = top[top["match"]]
        record[f"hit@{budget}"] = bool(len(matched) > 0)
        record[f"rmsd@{budget}"] = None if len(matched) == 0 else float(matched["rmsd"].min())
    return record


def select_conservative(group: pd.DataFrame, anchor_keep: int, budget: int = 20) -> pd.DataFrame:
    anchor = group[group["rank"] <= int(anchor_keep)].sort_values("rank")
    pool = group[group["rank"] > int(anchor_keep)].sort_values(["score", "rank"], ascending=[False, True])
    selected = pd.concat([anchor, pool], axis=0).drop_duplicates(subset=["rank"], keep="first").head(int(budget))
    return selected


def select_unconstrained(group: pd.DataFrame, budget: int = 20) -> pd.DataFrame:
    return group.sort_values(["score", "rank"], ascending=[False, True]).head(int(budget))


def evaluate_strategy(pred: pd.DataFrame, strategy: str, anchor_keep: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, group in pred.groupby("sample_id", sort=False):
        if strategy == "conservative":
            assert anchor_keep is not None
            selected = select_conservative(group, int(anchor_keep), budget=20)
        elif strategy == "unconstrained":
            selected = select_unconstrained(group, budget=20)
        else:
            raise ValueError(strategy)
        rows.append(sample_record(str(sample_id), selected))
    return rows


def selected_rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    used_supplemental = 0
    supplemental_slots = 0
    anchor_slots = 0
    for row in rows:
        ranks = row["selected_ranks"]
        supp = [r for r in ranks if r > 20]
        used_supplemental += int(bool(supp))
        supplemental_slots += len(supp)
        anchor_slots += len([r for r in ranks if r <= 20])
    return {
        "samples_with_supplemental": used_supplemental,
        "supplemental_slots": supplemental_slots,
        "anchor_slots": anchor_slots,
        "mean_supplemental_slots_per_sample": float(supplemental_slots / max(1, len(rows))),
    }


def make_report(result: dict[str, Any]) -> str:
    dataset_label = result.get("dataset_label", "MP-20")
    lines = [
        f"# {dataset_label} K50 Conservative Selector OOF Search",
        "",
        f"Created: {result['created_at']}",
        "",
        "The selector is trained with K50 validation labels in 5-fold GroupKFold by sample_id. Evaluation is out-of-fold only.",
        "",
        "## Baseline K20",
        "",
        "| metric | value | rows>=7 |",
        "| --- | ---: | ---: |",
    ]
    baseline = result["baseline_metrics"]
    for budget in [1, 5, 20]:
        lines.append(
            f"| match@{budget} | {100.0 * baseline[f'match@{budget}']:.3f}% | "
            f"{100.0 * baseline[f'rows>=7_match@{budget}']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Strategy Results",
            "",
            "| model | seed | strategy | anchor_keep | match@1 delta | match@5 delta | match@20 delta | rows>=7 @20 delta | supp slots/sample | CI95 delta@20 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["strategy_results"]:
        d = row["deltas"]
        m = row["metrics"]
        ci = row.get("bootstrap", {}).get("match@20_delta_ci95")
        ci_text = "NA" if ci is None else f"[{100.0 * ci[0]:.3f}, {100.0 * ci[1]:.3f}] pp"
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['strategy']} | {row.get('anchor_keep')} | "
            f"{100.0 * d['match@1']:.3f} pp | {100.0 * d['match@5']:.3f} pp | "
            f"{100.0 * d['match@20']:.3f} pp | {100.0 * d['rows>=7_match@20']:.3f} pp | "
            f"{row['selection_summary']['mean_supplemental_slots_per_sample']:.3f} | {ci_text} |"
        )
    best = result.get("best_gate_candidate")
    lines.extend(["", "## Best Gate Candidate", ""])
    if best is None:
        lines.append("No strategy satisfied the conservative validation gate in this run.")
    else:
        lines.extend(
            [
                f"- Model: {best['model']} seed={best['seed']}",
                f"- Strategy: {best['strategy']} anchor_keep={best.get('anchor_keep')}",
                f"- match@1 delta: {100.0 * best['deltas']['match@1']:.3f} pp",
                f"- match@5 delta: {100.0 * best['deltas']['match@5']:.3f} pp",
                f"- match@20 delta: {100.0 * best['deltas']['match@20']:.3f} pp",
                f"- rows>=7 match@20 delta: {100.0 * best['deltas']['rows>=7_match@20']:.3f} pp",
            ]
        )
    lines.append("")
    lines.append("This report is validation-only and does not freeze an official-test strategy by itself.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K50 conservative selector OOF search.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--true-tar", default=str(DEFAULT_TRUE_TAR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--models", default="logistic,hgb")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--anchor-keeps", default="14,16,18")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "metrics/mp20_k50_conservative_selector_oof.json"))
    parser.add_argument("--report", default=str(ROOT / "reports/mp20_k50_conservative_selector_oof.md"))
    parser.add_argument("--score-prefix", default="mp20_val_k50")
    parser.add_argument("--score-dir", default=str(ROOT / "features/k50_selector_oof_scores"))
    parser.add_argument("--dataset-label", default="MP-20")
    parser.add_argument("--task-name", default="mp20_validation_k50_conservative_selector_oof")
    args = parser.parse_args()

    names = [name.strip() for name in args.models.split(",") if name.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    anchor_keeps = [int(x.strip()) for x in args.anchor_keeps.split(",") if x.strip()]
    df, input_summary = load_table(
        Path(args.features),
        Path(args.labels),
        Path(args.true_tar),
        max_rank=50,
        allow_partial=bool(args.allow_partial),
    )
    baseline_rows = baseline_k20_rows(df)
    baseline_metrics = summarize_metrics(baseline_rows)

    strategy_results: list[dict[str, Any]] = []
    score_dir = under_root(Path(args.score_dir))
    score_dir.mkdir(parents=True, exist_ok=True)
    for spec in model_specs(names, seeds):
        pred, folds, feature_info = run_oof(df, spec, int(args.folds))
        score_path = score_dir / f"{args.score_prefix}_{spec.name}_seed{spec.seed}_scores.jsonl"
        pred_out = pred.copy()
        pred_out["model"] = spec.name
        pred_out["seed"] = int(spec.seed)
        pred_out.to_json(score_path, orient="records", lines=True)

        for strategy, anchor_keep in [("unconstrained", None)] + [("conservative", k) for k in anchor_keeps]:
            rows = evaluate_strategy(pred, strategy=strategy, anchor_keep=anchor_keep)
            metrics = summarize_metrics(rows)
            deltas = {
                key: (metrics[key] - baseline_metrics[key])
                for key in [
                    "match@1",
                    "match@5",
                    "match@20",
                    "rows>=7_match@1",
                    "rows>=7_match@5",
                    "rows>=7_match@20",
                ]
            }
            strategy_results.append(
                {
                    "model": spec.name,
                    "seed": int(spec.seed),
                    "strategy": strategy,
                    "anchor_keep": anchor_keep,
                    "metrics": metrics,
                    "deltas": deltas,
                    "bootstrap": bootstrap_delta(baseline_rows, rows, n=int(args.bootstrap), seed=3000 + spec.seed),
                    "selection_summary": selected_rank_summary(rows),
                    "folds": folds,
                    "feature_info": feature_info,
                    "scores": str(score_path.resolve()),
                }
            )

    gate_candidates = [
        row
        for row in strategy_results
        if row["deltas"]["match@20"] >= -0.002
        and row["deltas"]["rows>=7_match@20"] >= -0.002
        and max(row["deltas"]["match@1"], row["deltas"]["match@5"], row["deltas"]["match@20"]) >= 0.01
    ]
    best = None
    if gate_candidates:
        best = max(
            gate_candidates,
            key=lambda row: (
                max(row["deltas"]["match@1"], row["deltas"]["match@5"], row["deltas"]["match@20"]),
                row["deltas"]["match@20"],
                row["deltas"]["rows>=7_match@20"],
            ),
        )

    result = {
        "created_at": now_iso(),
        "task": str(args.task_name),
        "dataset_label": str(args.dataset_label),
        "input_summary": input_summary,
        "baseline_metrics": baseline_metrics,
        "strategy_results": strategy_results,
        "best_gate_candidate": best,
        "note": "Validation-only OOF selector search. Official test remains untouched.",
    }
    write_json(Path(args.out), result)
    write_text(Path(args.report), make_report(result))


if __name__ == "__main__":
    main()
