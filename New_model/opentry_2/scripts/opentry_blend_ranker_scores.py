#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2").resolve()
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import opentry_apply_ranker_to_existing_cifs as apply_ranker  # noqa: E402
import opentry_validation_ranker_eval_gt_sg as ranker  # noqa: E402


def _ensure_under_opentry(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise ValueError(f"refusing to write outside opentry_2: {resolved}")
    return resolved


def _write_json(path: Path, obj: Any) -> None:
    path = _ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _score_model(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] >= 2:
            return np.asarray(p[:, 1], dtype=float)
        return np.asarray(p[:, 0], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _hit_metrics(records: Sequence[Dict[str, Any]], scores: Sequence[float], budgets: Sequence[int]) -> Dict[str, Any]:
    by_id: Dict[str, List[Tuple[float, int, int]]] = {}
    for rec, score in zip(records, scores):
        by_id.setdefault(str(rec["id"]), []).append((float(score), int(rec.get("idx", 0)), int(rec.get("label", 0))))
    out: Dict[str, Any] = {"n_ids": len(by_id)}
    for k in budgets:
        hits = 0
        for items in by_id.values():
            ranked = sorted(items, key=lambda x: (-x[0], x[1]))
            if any(label > 0 for _score, _idx, label in ranked[: int(k)]):
                hits += 1
        out[f"hit@{int(k)}"] = float(hits / max(1, len(by_id)))
    return out


def cmd_tune(args: argparse.Namespace) -> int:
    records = _load_jsonl(Path(args.records_jsonl).expanduser().resolve())
    names = ranker._feature_names()
    X = np.array([[float((rec.get("features") or {}).get(name, 0.0)) for name in names] for rec in records], dtype=float)
    e21 = pickle.loads(Path(args.ranker_a).expanduser().resolve().read_bytes())
    e26 = pickle.loads(Path(args.ranker_b).expanduser().resolve().read_bytes())
    s_a = _score_model(e21, X)
    s_b = _score_model(e26, X)
    rank_prior = np.array([1.0 / max(1.0, float(rec.get("idx", 1))) for rec in records], dtype=float)
    rank_log_prior = np.array([1.0 / np.log(max(2.0, float(rec.get("idx", 1) + 1))) for rec in records], dtype=float)

    weights = [float(x) for x in str(args.weights).split(",") if x.strip()]
    rank_weights = [float(x) for x in str(args.rank_weights).split(",") if x.strip()]
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    results: List[Dict[str, Any]] = []
    for wa in weights:
        for wb in weights:
            for wr in rank_weights:
                for wlr in rank_weights:
                    score = wa * s_a + wb * s_b + wr * rank_prior + wlr * rank_log_prior
                    metrics = _hit_metrics(records, score, budgets)
                    rec = {
                        "w_a": float(wa),
                        "w_b": float(wb),
                        "w_rank": float(wr),
                        "w_rank_log": float(wlr),
                        **metrics,
                    }
                    results.append(rec)
    results.sort(key=lambda r: (r.get("hit@5", 0.0), r.get("hit@1", 0.0), r.get("hit@20", 0.0)), reverse=True)
    out = {
        "records": len(records),
        "ids": int(_hit_metrics(records, s_a, budgets)["n_ids"]),
        "baseline_a": _hit_metrics(records, s_a, budgets),
        "baseline_b": _hit_metrics(records, s_b, budgets),
        "baseline_rank_prior": _hit_metrics(records, rank_prior, budgets),
        "top": results[: int(args.topn)],
    }
    _write_json(Path(args.out), out)
    print(json.dumps(out["top"][:5], ensure_ascii=False, indent=2), flush=True)
    return 0


def _load_ranking_scores(path: Path) -> Dict[str, Dict[int, float]]:
    scores: Dict[str, Dict[int, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mid = str(rec.get("material_id"))
            for item in rec.get("ranked") or []:
                scores.setdefault(mid, {})[int(item["old_idx"])] = float(item.get("score", 0.0))
    return scores


def cmd_apply(args: argparse.Namespace) -> int:
    out_dir = _ensure_under_opentry(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    tars_dir = out_dir / "tars"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tars_dir.mkdir(parents=True, exist_ok=True)

    limit = None if int(args.limit) <= 0 else int(args.limit)
    material_ids = apply_ranker._read_material_ids(Path(args.test_csv).expanduser().resolve(), start=int(args.start_index), limit=limit)
    rows = apply_ranker._load_rows_from_gt(material_ids, Path(args.test_gt_dir).expanduser().resolve())
    rows_by_id = {row.material_id: row for row in rows}

    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    grouped_paths: Dict[str, List[Path]] = {mid: [] for mid in rows_by_id}
    for path in sorted(candidate_dir.glob("*.cif")):
        parsed = apply_ranker._candidate_index(path)
        if parsed is None:
            continue
        mid, _idx = parsed
        if mid in grouped_paths:
            grouped_paths[mid].append(path)

    a_scores = _load_ranking_scores(Path(args.ranking_a).expanduser().resolve())
    b_scores = _load_ranking_scores(Path(args.ranking_b).expanduser().resolve())
    scores: Dict[str, Dict[int, float]] = {}
    for mid, paths in grouped_paths.items():
        for path in paths:
            parsed = apply_ranker._candidate_index(path)
            if parsed is None:
                continue
            _mid, idx = parsed
            rank_prior = 1.0 / max(1.0, float(idx))
            rank_log_prior = 1.0 / np.log(max(2.0, float(idx + 1)))
            score = (
                float(args.w_a) * float(a_scores.get(mid, {}).get(idx, 0.0))
                + float(args.w_b) * float(b_scores.get(mid, {}).get(idx, 0.0))
                + float(args.w_rank) * rank_prior
                + float(args.w_rank_log) * rank_log_prior
            )
            scores.setdefault(mid, {})[idx] = float(score)

    ranked_tar = tars_dir / "ranked_blend.tar.gz"
    ranking_stats = apply_ranker._make_ranked_tar(
        out_tar=ranked_tar,
        rows=rows,
        grouped_paths=grouped_paths,
        scores=scores,
        ranking_jsonl=out_dir / "ranking.jsonl",
        expected_k=int(args.expected_k),
    )
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    results: Dict[str, Any] = {}
    for budget in budgets:
        print(f"[blend] benchmark k={budget}", flush=True)
        results[f"ranked_k{budget}"] = apply_ranker._benchmark(
            py=str(args.py),
            bench_script=Path(args.bench_script).expanduser().resolve(),
            gen_tar=ranked_tar,
            true_tar=Path(args.true_tar).expanduser().resolve(),
            num_gens=int(budget),
            max_sites=int(args.max_sites),
            rmsd_timeout_seconds=float(args.rmsd_timeout_seconds),
            hard_timeout_seconds=float(args.hard_timeout_seconds),
            workers=int(args.workers),
            out_json=metrics_dir / f"ranked_k{budget}.json",
            out_txt=metrics_dir / f"ranked_k{budget}.txt",
        )
    summary = {
        "weights": {
            "w_a": float(args.w_a),
            "w_b": float(args.w_b),
            "w_rank": float(args.w_rank),
            "w_rank_log": float(args.w_rank_log),
        },
        "n_rows": int(len(rows)),
        "ranking_stats": ranking_stats,
        "results": results,
    }
    _write_json(out_dir / "run_summary.json", summary)
    with (out_dir / "summary_metrics.tsv").open("w", encoding="utf-8") as f:
        f.write("group\tK\tmatch_rate\trmse\n")
        for key, metrics in results.items():
            k = key.rsplit("k", 1)[-1]
            f.write(f"{key}\t{k}\t{metrics.get('match_rate')}\t{metrics.get('rms_dist')}\n")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune/apply blended ranker scores.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--records-jsonl", required=True)
    p_tune.add_argument("--ranker-a", required=True)
    p_tune.add_argument("--ranker-b", required=True)
    p_tune.add_argument("--out", required=True)
    p_tune.add_argument("--weights", default="0,0.25,0.5,0.75,1,1.5,2")
    p_tune.add_argument("--rank-weights", default="0,0.05,0.1,0.2,0.4,0.8")
    p_tune.add_argument("--budgets", default="1,5,20")
    p_tune.add_argument("--topn", type=int, default=20)
    p_tune.set_defaults(func=cmd_tune)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--test-csv", required=True)
    p_apply.add_argument("--test-gt-dir", required=True)
    p_apply.add_argument("--candidate-dir", required=True)
    p_apply.add_argument("--ranking-a", required=True)
    p_apply.add_argument("--ranking-b", required=True)
    p_apply.add_argument("--true-tar", required=True)
    p_apply.add_argument("--out-dir", required=True)
    p_apply.add_argument("--start-index", type=int, default=0)
    p_apply.add_argument("--limit", type=int, default=0)
    p_apply.add_argument("--expected-k", type=int, default=20)
    p_apply.add_argument("--budgets", default="1,5,20")
    p_apply.add_argument("--w-a", type=float, required=True)
    p_apply.add_argument("--w-b", type=float, required=True)
    p_apply.add_argument("--w-rank", type=float, required=True)
    p_apply.add_argument("--w-rank-log", type=float, required=True)
    p_apply.add_argument("--py", default=sys.executable)
    p_apply.add_argument("--bench-script", default="/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/bin/benchmark_metrics.py")
    p_apply.add_argument("--workers", type=int, default=96)
    p_apply.add_argument("--max-sites", type=int, default=512)
    p_apply.add_argument("--rmsd-timeout-seconds", type=float, default=20.0)
    p_apply.add_argument("--hard-timeout-seconds", type=float, default=60.0)
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
