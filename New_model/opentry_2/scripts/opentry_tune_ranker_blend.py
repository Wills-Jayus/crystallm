#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2").resolve()
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import opentry_validation_ranker_eval_gt_sg as ranker  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise ValueError(f"refusing to write outside opentry_2: {resolved}")
    return resolved


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Any) -> None:
    path = ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def feature_names(records: Sequence[Dict[str, Any]]) -> List[str]:
    names = list(ranker._feature_names())
    present = {name for rec in records for name in (rec.get("features") or {})}
    return [name for name in names if name in present]


def matrix(records: Sequence[Dict[str, Any]], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[float((rec.get("features") or {}).get(name, 0.0)) for name in names] for rec in records], dtype=float)
    y = np.array([int(rec.get("label", 0)) for rec in records], dtype=int)
    return x, y


def stable_split_ids(records: Sequence[Dict[str, Any]], holdout_frac: float) -> Tuple[set[str], set[str]]:
    ids = sorted({str(rec["id"]) for rec in records})
    scored = []
    for mid in ids:
        h = hashlib.sha1(mid.encode("utf-8")).hexdigest()
        scored.append((int(h[:12], 16) / float(16**12), mid))
    holdout = {mid for score, mid in scored if score < float(holdout_frac)}
    train = set(ids) - holdout
    if not holdout or not train:
        cut = max(1, int(round(len(ids) * float(holdout_frac))))
        holdout = {mid for _score, mid in sorted(scored)[:cut]}
        train = set(ids) - holdout
    return train, holdout


def make_model(config: Dict[str, Any], seed: int) -> Any:
    kind = str(config["kind"])
    common = {
        "n_estimators": int(config["n_estimators"]),
        "max_depth": None if config["max_depth"] is None else int(config["max_depth"]),
        "min_samples_leaf": int(config["min_samples_leaf"]),
        "class_weight": "balanced_subsample" if kind == "rf" else "balanced",
        "random_state": int(seed),
        "n_jobs": int(config["jobs"]),
    }
    if kind == "rf":
        return RandomForestClassifier(**common)
    if kind == "extra":
        common.pop("class_weight", None)
        return ExtraTreesClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=None if config["max_depth"] is None else int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=int(config["jobs"]),
        )
    raise ValueError(f"unknown kind: {kind}")


def score_model(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] >= 2:
        return np.asarray(proba[:, 1], dtype=float)
    return np.asarray(proba[:, 0], dtype=float)


def blended_scores(
    records: Sequence[Dict[str, Any]],
    model_scores: np.ndarray,
    *,
    ranker_weight: float,
    rank_inv_weight: float,
    rank_log_inv_weight: float,
) -> np.ndarray:
    out = np.asarray(model_scores, dtype=float) * float(ranker_weight)
    if rank_inv_weight or rank_log_inv_weight:
        inv = np.array([float((rec.get("features") or {}).get("sample_rank_inv", 0.0)) for rec in records], dtype=float)
        loginv = np.array([float((rec.get("features") or {}).get("sample_rank_log_inv", 0.0)) for rec in records], dtype=float)
        out = out + float(rank_inv_weight) * inv + float(rank_log_inv_weight) * loginv
    return out


def hit_metrics(records: Sequence[Dict[str, Any]], scores: Sequence[float], budgets: Iterable[int]) -> Dict[str, Any]:
    by_id: Dict[str, List[Tuple[float, int, int]]] = {}
    for rec, score in zip(records, scores):
        by_id.setdefault(str(rec["id"]), []).append((float(score), int(rec.get("idx", 0)), int(rec.get("label", 0))))
    out: Dict[str, Any] = {"n_ids": len(by_id)}
    for budget in budgets:
        hits = 0
        for items in by_id.values():
            ranked = sorted(items, key=lambda x: (-x[0], x[1]))
            if any(label > 0 for _score, _idx, label in ranked[: int(budget)]):
                hits += 1
        out[f"hit_at_{int(budget)}"] = float(hits / max(1, len(by_id)))
    return out


def configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for kind in ["rf", "extra"]:
        for depth in [10, 14, 18, None]:
            for leaf in [1, 2, 4, 8, 16]:
                out.append(
                    {
                        "kind": kind,
                        "max_depth": depth,
                        "min_samples_leaf": leaf,
                        "n_estimators": int(args.estimators),
                        "jobs": int(args.jobs),
                    }
                )
    return out


def blend_grid() -> List[Dict[str, float]]:
    out = []
    for inv in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
        for loginv in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]:
            out.append({"ranker_weight": 1.0, "rank_inv_weight": inv, "rank_log_inv_weight": loginv})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune a non-oracle candidate ranker/blend on validation-heldout labels.")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.records_jsonl))
    names = feature_names(records)
    train_ids, holdout_ids = stable_split_ids(records, float(args.holdout_frac))
    train_records = [rec for rec in records if str(rec["id"]) in train_ids]
    holdout_records = [rec for rec in records if str(rec["id"]) in holdout_ids]
    x_train, y_train = matrix(train_records, names)
    x_holdout, y_holdout = matrix(holdout_records, names)
    if not train_records or not holdout_records:
        raise SystemExit("empty train or holdout split")

    all_results: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for i, cfg in enumerate(configs(args), start=1):
        model = make_model(cfg, seed=int(args.seed) + i)
        model.fit(x_train, y_train)
        base = score_model(model, x_holdout)
        for blend in blend_grid():
            scores = blended_scores(holdout_records, base, **blend)
            metrics = hit_metrics(holdout_records, scores, budgets=[1, 5, 20, 50])
            result = {
                "config": cfg,
                "blend": blend,
                "metrics": metrics,
                "holdout_positive_rate": float(y_holdout.sum() / max(1, len(y_holdout))),
            }
            all_results.append(result)
            key = (
                metrics["hit_at_5"],
                metrics["hit_at_1"],
                metrics["hit_at_20"],
                -float(blend["rank_inv_weight"] + blend["rank_log_inv_weight"]),
            )
            if best is None or key > best["key"]:
                best = {"key": key, **result}
        print(f"[tune] {i}/{len(configs(args))} best_hit5={best['metrics']['hit_at_5']:.4f}", flush=True)

    assert best is not None
    final_model = make_model(best["config"], seed=int(args.seed) + 9999)
    x_all, y_all = matrix(records, names)
    final_model.fit(x_all, y_all)
    (out_dir / "ranker_model.pkl").write_bytes(pickle.dumps(final_model))
    write_json(out_dir / "score_config.json", best["blend"])
    summary = {
        "records_jsonl": str(Path(args.records_jsonl).expanduser().resolve()),
        "n_records": len(records),
        "n_ids": len({str(rec["id"]) for rec in records}),
        "train_ids": len(train_ids),
        "holdout_ids": len(holdout_ids),
        "train_records": len(train_records),
        "holdout_records": len(holdout_records),
        "train_positive_rate": float(y_train.sum() / max(1, len(y_train))),
        "holdout_positive_rate": float(y_holdout.sum() / max(1, len(y_holdout))),
        "all_positive_rate": float(y_all.sum() / max(1, len(y_all))),
        "feature_names": names,
        "best": {k: v for k, v in best.items() if k != "key"},
        "ranker_model": str((out_dir / "ranker_model.pkl").resolve()),
        "score_config": str((out_dir / "score_config.json").resolve()),
    }
    write_json(out_dir / "tune_summary.json", summary)
    write_json(out_dir / "all_tune_results.json", all_results)
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
