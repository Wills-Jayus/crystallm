#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows7 = [r for r in rows if r["rows_ge7"]]
    out: dict[str, Any] = {"samples": len(rows), "rows_ge7_samples": len(rows7)}
    for budget in [1, 5, 20]:
        hits = [r for r in rows if r[f"hit@{budget}"]]
        hits7 = [r for r in rows7 if r[f"hit@{budget}"]]
        rms = [float(r[f"rmsd@{budget}"]) for r in hits if r[f"rmsd@{budget}"] is not None]
        rms7 = [float(r[f"rmsd@{budget}"]) for r in hits7 if r[f"rmsd@{budget}"] is not None]
        out[f"match@{budget}"] = None if not rows else float(len(hits) / len(rows))
        out[f"rmsd@{budget}"] = None if not rms else float(sum(rms) / len(rms))
        out[f"rows>=7_match@{budget}"] = None if not rows7 else float(len(hits7) / len(rows7))
        out[f"rows>=7_rmsd@{budget}"] = None if not rms7 else float(sum(rms7) / len(rms7))
    return out


def baseline_k20_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        rows.append(sample_record(str(sample_id), group[group["rank"] <= 20].sort_values("rank")))
    return rows


def select_conservative(group: pd.DataFrame, anchor_keep: int, budget: int = 20) -> pd.DataFrame:
    anchor = group[group["rank"] <= int(anchor_keep)].sort_values("rank")
    pool = group[group["rank"] > int(anchor_keep)].sort_values(["score", "rank"], ascending=[False, True])
    return pd.concat([anchor, pool], axis=0).drop_duplicates(subset=["rank"], keep="first").head(int(budget))


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
    supplemental_slots = 0
    samples_with_supplemental = 0
    for row in rows:
        supp = [r for r in row["selected_ranks"] if r > 20]
        supplemental_slots += len(supp)
        samples_with_supplemental += int(bool(supp))
    return {
        "samples_with_supplemental": int(samples_with_supplemental),
        "supplemental_slots": int(supplemental_slots),
        "mean_supplemental_slots_per_sample": float(supplemental_slots / max(1, len(rows))),
    }


def infer_model_seed(df: pd.DataFrame, path: Path) -> tuple[str, int]:
    if "model" in df.columns and df["model"].notna().any():
        model = str(df["model"].dropna().iloc[0])
    else:
        model = path.stem
    if "seed" in df.columns and df["seed"].notna().any():
        seed = int(df["seed"].dropna().iloc[0])
    else:
        seed = -1
    return model, seed


def make_report(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['dataset_label']} K50 Selector Score Evaluation",
        "",
        f"Created: {result['created_at']}",
        "",
        "This report evaluates already generated OOF score files. It does not train a model.",
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
            "## Strategies",
            "",
            "| model | seed | strategy | anchor_keep | delta@1 | delta@5 | delta@20 | rows>=7 delta@20 | supp slots/sample |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["strategy_results"]:
        d = row["deltas"]
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['strategy']} | {row.get('anchor_keep')} | "
            f"{100.0 * d['match@1']:.3f} pp | {100.0 * d['match@5']:.3f} pp | "
            f"{100.0 * d['match@20']:.3f} pp | {100.0 * d['rows>=7_match@20']:.3f} pp | "
            f"{row['selection_summary']['mean_supplemental_slots_per_sample']:.3f} |"
        )
    best = result.get("best_gate_candidate")
    lines.extend(["", "## Best Gate Candidate", ""])
    if best is None:
        lines.append("No evaluated score file satisfied the validation gate.")
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
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate K50 selector strategies from saved OOF score JSONL files.")
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--anchor-keeps", default="10,12,14,16,18")
    parser.add_argument("--dataset-label", default="MPTS-52")
    parser.add_argument("--out", default=str(ROOT / "metrics/mpts52_k50_selector_scores_eval.json"))
    parser.add_argument("--report", default=str(ROOT / "reports/mpts52_k50_selector_scores_eval.md"))
    args = parser.parse_args()

    anchor_keeps = [int(x.strip()) for x in args.anchor_keeps.split(",") if x.strip()]
    strategy_results: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None
    input_files: list[str] = []
    for score_arg in args.scores:
        path = Path(score_arg)
        pred = pd.read_json(path, lines=True)
        input_files.append(str(path.resolve()))
        pred["match"] = pred["match"].fillna(False).astype(bool)
        pred["rank"] = pd.to_numeric(pred["rank"], errors="coerce").astype(int)
        model, seed = infer_model_seed(pred, path)
        baseline_rows = baseline_k20_rows(pred)
        if baseline_metrics is None:
            baseline_metrics = summarize_metrics(baseline_rows)
        for strategy, anchor_keep in [("unconstrained", None)] + [("conservative", k) for k in anchor_keeps]:
            rows = evaluate_strategy(pred, strategy=strategy, anchor_keep=anchor_keep)
            metrics = summarize_metrics(rows)
            deltas = {
                key: metrics[key] - baseline_metrics[key]
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
                    "model": model,
                    "seed": int(seed),
                    "strategy": strategy,
                    "anchor_keep": anchor_keep,
                    "metrics": metrics,
                    "deltas": deltas,
                    "selection_summary": selected_rank_summary(rows),
                    "scores": str(path.resolve()),
                }
            )
    assert baseline_metrics is not None
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
        "dataset_label": str(args.dataset_label),
        "input_score_files": input_files,
        "baseline_metrics": baseline_metrics,
        "strategy_results": strategy_results,
        "best_gate_candidate": best,
        "note": "Validation-only evaluation of saved OOF scores.",
    }
    write_json(Path(args.out), result)
    write_text(Path(args.report), make_report(result))


if __name__ == "__main__":
    main()
