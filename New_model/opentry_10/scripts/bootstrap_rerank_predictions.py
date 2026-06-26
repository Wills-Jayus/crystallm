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


def sample_rows(df: pd.DataFrame, *, score_col: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid, g in df.groupby("sample_id", sort=False):
        ordered = g.sort_values("rank") if score_col is None else g.sort_values([score_col, "rank"], ascending=[False, True])
        row: dict[str, Any] = {"sample_id": str(sid), "rows_ge7": bool(ordered["target_rows_ge7"].iloc[0])}
        for budget in [1, 5, 20]:
            top = ordered.head(budget)
            hit = bool(top["match"].any())
            row[f"hit@{budget}"] = hit
            row[f"rmsd@{budget}"] = None if not hit else float(top.loc[top["match"], "rmsd"].min())
        rows.append(row)
    return rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows7 = [r for r in rows if r["rows_ge7"]]
    out: dict[str, Any] = {"samples": len(rows), "rows_ge7_samples": len(rows7)}
    for budget in [1, 5, 20]:
        hits = [r for r in rows if r[f"hit@{budget}"]]
        hits7 = [r for r in rows7 if r[f"hit@{budget}"]]
        out[f"match@{budget}"] = float(len(hits) / len(rows))
        out[f"rows>=7_match@{budget}"] = float(len(hits7) / len(rows7)) if rows7 else None
    return out


def bootstrap(base: list[dict[str, Any]], pred: list[dict[str, Any]], *, n: int, seed: int) -> dict[str, Any]:
    base_by_id = {r["sample_id"]: r for r in base}
    pred_by_id = {r["sample_id"]: r for r in pred}
    ids = np.array(sorted(set(base_by_id) & set(pred_by_id)))
    rows7_ids = np.array([i for i in ids if base_by_id[str(i)]["rows_ge7"]])
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for budget in [1, 5, 20]:
        base_hits = np.array([base_by_id[str(i)][f"hit@{budget}"] for i in ids], dtype=float)
        pred_hits = np.array([pred_by_id[str(i)][f"hit@{budget}"] for i in ids], dtype=float)
        sample_idx = rng.integers(0, len(ids), size=(int(n), len(ids)))
        deltas = pred_hits[sample_idx].mean(axis=1) - base_hits[sample_idx].mean(axis=1)
        lo, hi = np.quantile(deltas, [0.025, 0.975])
        out[f"match@{budget}_delta_ci95"] = [float(lo), float(hi)]
        if len(rows7_ids) > 0:
            base_hits7 = np.array([base_by_id[str(i)][f"hit@{budget}"] for i in rows7_ids], dtype=float)
            pred_hits7 = np.array([pred_by_id[str(i)][f"hit@{budget}"] for i in rows7_ids], dtype=float)
            sample_idx7 = rng.integers(0, len(rows7_ids), size=(int(n), len(rows7_ids)))
            deltas7 = pred_hits7[sample_idx7].mean(axis=1) - base_hits7[sample_idx7].mean(axis=1)
            lo7, hi7 = np.quantile(deltas7, [0.025, 0.975])
            out[f"rows>=7_match@{budget}_delta_ci95"] = [float(lo7), float(hi7)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap saved rerank OOF predictions without retraining.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--out", default=str(ROOT / "metrics/mpts52_rerank_hgb_seed2_bootstrap.json"))
    args = parser.parse_args()

    df = pd.read_json(args.predictions, lines=True)
    df["match"] = df["match"].fillna(False).astype(bool)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype(int)
    base = sample_rows(df, score_col=None)
    pred = sample_rows(df, score_col="score")
    result = {
        "created_at": now_iso(),
        "predictions": str(Path(args.predictions).resolve()),
        "baseline_metrics": metrics(base),
        "rerank_metrics": metrics(pred),
        "deltas": {
            key: metrics(pred)[key] - metrics(base)[key]
            for key in [
                "match@1",
                "match@5",
                "match@20",
                "rows>=7_match@1",
                "rows>=7_match@5",
                "rows>=7_match@20",
            ]
        },
        "bootstrap": bootstrap(base, pred, n=int(args.bootstrap), seed=int(args.seed)),
        "note": "Validation OOF bootstrap only; no test labels used.",
    }
    write_json(Path(args.out), result)


if __name__ == "__main__":
    main()
