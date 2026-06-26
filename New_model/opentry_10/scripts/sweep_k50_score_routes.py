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
    row0 = ordered.iloc[0]
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "rows_ge7": bool(row0["target_rows_ge7"]),
    }
    for budget in [1, 5, 20]:
        top = ordered.head(budget)
        matched = top[top["match"]]
        record[f"hit@{budget}"] = bool(len(matched) > 0)
        record[f"rmsd@{budget}"] = None if len(matched) == 0 else float(matched["rmsd"].min())
    return record


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def baseline_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [sample_record(str(sid), g[g["rank"] <= 20].sort_values("rank")) for sid, g in df.groupby("sample_id", sort=False)]


def sample_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("sample_id", sort=False):
        rank_order = g.sort_values("rank")
        score_order = g.sort_values(["score", "rank"], ascending=[False, True])
        best = score_order.iloc[0]
        rank1 = rank_order.iloc[0]
        rows.append(
            {
                "sample_id": sid,
                "best_score": float(best["score"]),
                "best_rank": int(best["rank"]),
                "rank1_score": float(rank1["score"]),
                "best_minus_rank1": float(best["score"] - rank1["score"]),
                "best_is_anchor": bool(int(best["rank"]) <= 20),
            }
        )
    return pd.DataFrame(rows)


def selected_rows(
    df: pd.DataFrame,
    route_ids: set[str],
    *,
    routed_strategy: str,
    anchor_keep: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for sid, g in df.groupby("sample_id", sort=False):
        sid_s = str(sid)
        if sid_s not in route_ids:
            selected = g[g["rank"] <= 20].sort_values("rank")
        elif routed_strategy == "unconstrained":
            selected = g.sort_values(["score", "rank"], ascending=[False, True]).head(20)
        elif routed_strategy == "conservative":
            assert anchor_keep is not None
            anchor = g[g["rank"] <= int(anchor_keep)].sort_values("rank")
            pool = g[g["rank"] > int(anchor_keep)].sort_values(["score", "rank"], ascending=[False, True])
            selected = pd.concat([anchor, pool], axis=0).drop_duplicates("rank").head(20)
        else:
            raise ValueError(routed_strategy)
        rows.append(sample_record(sid_s, selected))
    return rows


def make_report(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['dataset_label']} K50 Score Route Sweep",
        "",
        f"Created: {result['created_at']}",
        "",
        "Routing uses saved OOF scores only. Samples not routed keep baseline K20 order.",
        "",
        "## Best Routes",
        "",
        "| score file | strategy | anchor_keep | rule | threshold | routed | d@1 | d@5 | d@20 | rows>=7 d@20 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["route_results"][:30]:
        d = row["deltas"]
        lines.append(
            f"| {Path(row['scores']).name} | {row['routed_strategy']} | {row.get('anchor_keep')} | "
            f"{row['route_rule']} | {row['threshold']:.6f} | {row['routed_samples']} | "
            f"{100*d['match@1']:.3f} pp | {100*d['match@5']:.3f} pp | "
            f"{100*d['match@20']:.3f} pp | {100*d['rows>=7_match@20']:.3f} pp |"
        )
    best = result.get("best_gate_candidate")
    lines.extend(["", "## Best Gate Candidate", ""])
    if best is None:
        lines.append("No route satisfied the validation gate.")
    else:
        lines.extend(
            [
                f"- Scores: {best['scores']}",
                f"- Strategy: {best['routed_strategy']} anchor_keep={best.get('anchor_keep')}",
                f"- Rule: {best['route_rule']} threshold={best['threshold']}",
                f"- Routed samples: {best['routed_samples']}",
                f"- match@1 delta: {100*best['deltas']['match@1']:.3f} pp",
                f"- match@5 delta: {100*best['deltas']['match@5']:.3f} pp",
                f"- match@20 delta: {100*best['deltas']['match@20']:.3f} pp",
                f"- rows>=7 match@20 delta: {100*best['deltas']['rows>=7_match@20']:.3f} pp",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep validation-only routes for saved K50 OOF score files.")
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--anchor-keeps", default="10,12,14,16,18")
    parser.add_argument("--quantiles", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--dataset-label", default="MPTS-52")
    parser.add_argument("--out", default=str(ROOT / "metrics/mpts52_k50_score_route_sweep.json"))
    parser.add_argument("--report", default=str(ROOT / "reports/mpts52_k50_score_route_sweep.md"))
    args = parser.parse_args()

    anchor_keeps = [int(x) for x in args.anchor_keeps.split(",") if x.strip()]
    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    all_results: list[dict[str, Any]] = []
    baseline_metrics: dict[str, Any] | None = None

    for score_path_arg in args.scores:
        score_path = Path(score_path_arg)
        df = pd.read_json(score_path, lines=True)
        df["match"] = df["match"].fillna(False).astype(bool)
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype(int)
        feats = sample_features(df)
        base_rows = baseline_rows(df)
        if baseline_metrics is None:
            baseline_metrics = summarize(base_rows)
        assert baseline_metrics is not None

        route_specs: list[tuple[str, pd.Series]] = []
        for col in ["best_score", "best_minus_rank1"]:
            values = feats[col].to_numpy(dtype=float)
            for q in quantiles:
                threshold = float(np.quantile(values, q))
                route_specs.append((f"{col}_ge", pd.Series(feats[col].to_numpy(dtype=float) >= threshold, index=feats["sample_id"])))
        for max_rank in [1, 2, 3, 5, 10, 20, 50]:
            route_specs.append((f"best_rank_le_{max_rank}", pd.Series(feats["best_rank"].to_numpy(dtype=int) <= max_rank, index=feats["sample_id"])))

        strategy_specs: list[tuple[str, int | None]] = [("unconstrained", None)] + [("conservative", k) for k in anchor_keeps]
        for rule_name, route_mask in route_specs:
            threshold = 0.0
            if rule_name.endswith("_ge"):
                col = rule_name[:-3]
                threshold = float(feats.loc[route_mask.to_numpy(), col].min()) if route_mask.any() else float("inf")
            route_ids = set(str(x) for x in route_mask[route_mask].index)
            if not route_ids:
                continue
            for strategy, anchor_keep in strategy_specs:
                rows = selected_rows(df, route_ids, routed_strategy=strategy, anchor_keep=anchor_keep)
                metrics = summarize(rows)
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
                all_results.append(
                    {
                        "scores": str(score_path.resolve()),
                        "routed_strategy": strategy,
                        "anchor_keep": anchor_keep,
                        "route_rule": rule_name,
                        "threshold": threshold,
                        "routed_samples": int(len(route_ids)),
                        "metrics": metrics,
                        "deltas": deltas,
                    }
                )
    all_results.sort(
        key=lambda row: (
            max(row["deltas"]["match@1"], row["deltas"]["match@5"], row["deltas"]["match@20"]),
            row["deltas"]["match@20"],
            row["deltas"]["rows>=7_match@20"],
        ),
        reverse=True,
    )
    gate_candidates = [
        row
        for row in all_results
        if row["deltas"]["match@20"] >= -0.002
        and row["deltas"]["rows>=7_match@20"] >= -0.002
        and max(row["deltas"]["match@1"], row["deltas"]["match@5"], row["deltas"]["match@20"]) >= 0.01
    ]
    best = gate_candidates[0] if gate_candidates else None
    result = {
        "created_at": now_iso(),
        "dataset_label": str(args.dataset_label),
        "baseline_metrics": baseline_metrics,
        "route_results": all_results,
        "best_gate_candidate": best,
        "note": "Validation-only route sweep from OOF scores; no test labels used.",
    }
    write_json(Path(args.out), result)
    write_text(Path(args.report), make_report(result))


if __name__ == "__main__":
    main()
