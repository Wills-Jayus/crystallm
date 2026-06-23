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


def matrix(records: Sequence[Dict[str, Any]]) -> np.ndarray:
    names = list(ranker._feature_names())
    return np.array([[float((rec.get("features") or {}).get(name, 0.0)) for name in names] for rec in records], dtype=float)


def score_model(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba[:, 0], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def stable_holdout_ids(records: Sequence[Dict[str, Any]], holdout_frac: float) -> set[str]:
    ids = sorted({str(rec["id"]) for rec in records})
    holdout = set()
    for mid in ids:
        h = hashlib.sha1(mid.encode("utf-8")).hexdigest()
        score = int(h[:12], 16) / float(16**12)
        if score < float(holdout_frac):
            holdout.add(mid)
    if not holdout:
        holdout = set(ids[: max(1, int(len(ids) * holdout_frac))])
    return holdout


def blend_scores(records: Sequence[Dict[str, Any]], scores: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    out = np.asarray(scores, dtype=float) * float(weights.get("ranker_weight", 1.0))
    inv = np.array([float((rec.get("features") or {}).get("sample_rank_inv", 0.0)) for rec in records], dtype=float)
    loginv = np.array([float((rec.get("features") or {}).get("sample_rank_log_inv", 0.0)) for rec in records], dtype=float)
    out = out + float(weights.get("rank_inv_weight", 0.0)) * inv + float(weights.get("rank_log_inv_weight", 0.0)) * loginv
    return out


def policies() -> List[str]:
    base = [
        "",
        "r1,o1,o2,o3,o4",
        "r1,r2,o1,o2,o3",
        "r1,o1,r2,o2,r3",
        "r1,r2,r3,o1,o2",
        "r1,o1,o2,r2,r3",
        "r1,r2,o1,r3,o2",
        "r1,o1,r2,r3,r4",
        "o1,r1,r2,r3,r4",
        "o1,o2,r1,r2,r3",
        "r1,r2,r3,r4,o1",
        "r1,r2,r3,o1,r4",
    ]
    # Add wider top-10 policies that can affect K20 after ranker fill.
    for n_orig in [1, 2, 3, 4, 5]:
        toks = ["r1"] + [f"o{i}" for i in range(1, n_orig + 1)] + [f"r{i}" for i in range(2, 12 - n_orig)]
        base.append(",".join(toks))
    return list(dict.fromkeys(base))


def order_for_items(items: Sequence[Tuple[Dict[str, Any], float]], policy: str) -> List[Tuple[Dict[str, Any], float]]:
    ranker_order = sorted(items, key=lambda x: (-x[1], int(x[0].get("idx", 0))))
    original_order = sorted(items, key=lambda x: int(x[0].get("idx", 0)))
    chosen: List[Tuple[Dict[str, Any], float]] = []
    seen: set[int] = set()
    if policy:
        for raw_tok in policy.replace("+", ",").split(","):
            tok = raw_tok.strip().lower()
            if len(tok) < 2:
                continue
            source = tok[0]
            try:
                pos = int(tok[1:]) - 1
            except ValueError:
                continue
            pool = ranker_order if source == "r" else original_order if source == "o" else []
            if 0 <= pos < len(pool):
                idx = int(pool[pos][0].get("idx", 0))
                if idx not in seen:
                    chosen.append(pool[pos])
                    seen.add(idx)
    for pool in (ranker_order, original_order):
        for item in pool:
            idx = int(item[0].get("idx", 0))
            if idx not in seen:
                chosen.append(item)
                seen.add(idx)
    return chosen


def hit_metrics(records: Sequence[Dict[str, Any]], scores: Sequence[float], policy: str, budgets: Iterable[int]) -> Dict[str, Any]:
    by_id: Dict[str, List[Tuple[Dict[str, Any], float]]] = {}
    for rec, score in zip(records, scores):
        by_id.setdefault(str(rec["id"]), []).append((rec, float(score)))
    out: Dict[str, Any] = {"n_ids": len(by_id)}
    for budget in budgets:
        hits = 0
        for items in by_id.values():
            ordered = order_for_items(items, policy)
            if any(int(rec.get("label", 0)) > 0 for rec, _score in ordered[: int(budget)]):
                hits += 1
        out[f"hit_at_{int(budget)}"] = float(hits / max(1, len(by_id)))
    return out


def parse_model_specs(text: str) -> List[Dict[str, str]]:
    specs = []
    for part in str(text).split(","):
        if not part.strip():
            continue
        bits = part.split(":", 2)
        if len(bits) < 2:
            raise ValueError(f"model spec must be name:path[:score_config], got {part}")
        specs.append({"name": bits[0], "path": bits[1], "score_config": bits[2] if len(bits) > 2 else ""})
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune fixed ranker/original interleave policies on validation-heldout labels.")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--model-specs", required=True, help="Comma specs: name:pkl[:score_config]")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    args = parser.parse_args()

    records_all = load_jsonl(Path(args.records_jsonl))
    holdout_ids = stable_holdout_ids(records_all, float(args.holdout_frac))
    records = [rec for rec in records_all if str(rec["id"]) in holdout_ids]
    x = matrix(records)
    results: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for spec in parse_model_specs(args.model_specs):
        model = pickle.loads(Path(spec["path"]).expanduser().read_bytes())
        weights = {"ranker_weight": 1.0, "rank_inv_weight": 0.0, "rank_log_inv_weight": 0.0}
        if spec.get("score_config"):
            loaded = json.loads(Path(spec["score_config"]).expanduser().read_text(encoding="utf-8"))
            weights.update({k: float(v) for k, v in loaded.items() if k in weights})
        scores = blend_scores(records, score_model(model, x), weights)
        for policy in policies():
            metrics = hit_metrics(records, scores, policy, budgets=[1, 5, 20, 50])
            result = {"model": spec["name"], "model_path": spec["path"], "score_config": spec.get("score_config", ""), "weights": weights, "policy": policy, "metrics": metrics}
            results.append(result)
            key = (metrics["hit_at_5"], metrics["hit_at_1"], metrics["hit_at_20"], -len(policy))
            if best is None or key > best["key"]:
                best = {"key": key, **result}
    assert best is not None
    out_dir = ensure_under_opentry(Path(args.out_dir))
    write_json(out_dir / "policy_tune_results.json", results)
    write_json(
        out_dir / "best_policy.json",
        {
            "records_jsonl": str(Path(args.records_jsonl).expanduser().resolve()),
            "holdout_ids": len(holdout_ids),
            "holdout_records": len(records),
            "best": {k: v for k, v in best.items() if k != "key"},
        },
    )
    print(json.dumps({k: v for k, v in best.items() if k != "key"}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
