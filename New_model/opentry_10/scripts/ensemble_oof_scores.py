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


def main() -> None:
    parser = argparse.ArgumentParser(description="Average aligned OOF candidate scores into an ensemble score file.")
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-name", default="score_ensemble")
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    score_paths = [Path(p).resolve() for p in args.scores]
    if len(score_paths) < 2:
        raise RuntimeError("at least two score files are required")

    base: pd.DataFrame | None = None
    score_cols: list[str] = []
    input_rows: list[dict[str, Any]] = []
    key_cols = ["sample_id", "rank"]
    passthrough_cols = [
        "sample_id",
        "material_id",
        "rank",
        "match",
        "rmsd",
        "target_rows_ge7",
    ]

    for idx, path in enumerate(score_paths):
        df = pd.read_json(path, lines=True)
        missing = set(key_cols + ["score"]) - set(df.columns)
        if missing:
            raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
        df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
        df = df.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
        score_col = f"score_{idx}"
        score_cols.append(score_col)
        input_rows.append({"path": str(path), "rows": int(len(df)), "score_col": score_col})
        if base is None:
            keep_cols = [c for c in passthrough_cols if c in df.columns]
            base = df[keep_cols + ["score"]].rename(columns={"score": score_col})
        else:
            merged = df[key_cols + ["score"]].rename(columns={"score": score_col})
            base = base.merge(merged, on=key_cols, how="inner", validate="one_to_one")

    assert base is not None
    expected_rows = input_rows[0]["rows"]
    if len(base) != expected_rows:
        raise RuntimeError(f"score files are not fully aligned: merged rows={len(base)} expected={expected_rows}")

    scores = base[score_cols].to_numpy(dtype=np.float64)
    base["score_mean"] = scores.mean(axis=1)
    base["score_median"] = np.median(scores, axis=1)
    base["score_min"] = scores.min(axis=1)
    base["score_max"] = scores.max(axis=1)
    base["score_std"] = scores.std(axis=1)
    base["score"] = base["score_mean"]
    base["model"] = str(args.model_name)
    base["seed"] = int(args.seed)

    out = under_root(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [c for c in passthrough_cols if c in base.columns] + [
        "score",
        "score_mean",
        "score_median",
        "score_min",
        "score_max",
        "score_std",
        "model",
        "seed",
    ]
    base[out_cols].to_json(out, orient="records", lines=True)

    sample_counts = base.groupby("sample_id", sort=False)["rank"].size()
    manifest = {
        "created_at": now_iso(),
        "inputs": input_rows,
        "output": str(out.resolve()),
        "rows": int(len(base)),
        "samples": int(sample_counts.shape[0]),
        "min_rows_per_sample": int(sample_counts.min()),
        "max_rows_per_sample": int(sample_counts.max()),
        "score_columns": score_cols,
        "ensemble_score": "mean",
        "model_name": str(args.model_name),
        "seed": int(args.seed),
        "test_feedback_used": False,
        "note": "Validation OOF score ensembling only; StructureMatcher labels remain evaluation columns, not inference features.",
    }
    write_json(Path(args.manifest), manifest)


if __name__ == "__main__":
    main()
