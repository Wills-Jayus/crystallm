#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2").resolve()
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import opentry_validation_ranker_eval_gt_sg as ranker  # noqa: E402

_GEN_RE = ranker._GEN_RE


def _ensure_under_opentry(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise ValueError(f"refusing to write outside opentry_2: {resolved}")
    return resolved


def _read_material_ids(csv_path: Path, *, start: int, limit: Optional[int]) -> List[str]:
    ids: List[str] = []
    start_i = max(0, int(start))
    stop_i = None if limit is None else start_i + max(0, int(limit))
    seen = 0
    with csv_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            mid = str(row.get("material_id") or "").strip()
            if not mid:
                continue
            if seen < start_i:
                seen += 1
                continue
            if stop_i is not None and seen >= stop_i:
                break
            seen += 1
            ids.append(mid)
    return ids


def _load_rows_from_gt(material_ids: Sequence[str], gt_dir: Path) -> List[ranker.RowData]:
    rows: List[ranker.RowData] = []
    missing: List[str] = []
    for mid in material_ids:
        path = gt_dir / f"{mid}.cif"
        if not path.is_file():
            missing.append(mid)
            continue
        row = ranker._row_from_prepared_cif(mid, ranker._read_text(path))
        if row is None:
            missing.append(mid)
            continue
        rows.append(row)
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"missing/unusable GT CIFs for {len(missing)} rows: {preview}")
    return rows


def _candidate_index(path: Path) -> Optional[Tuple[str, int]]:
    m = _GEN_RE.match(path.name)
    if m is None:
        return None
    return m.group("mid"), int(m.group("idx"))


def _score_batch(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba[:, 0], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _blend_score(base: float, cfv: Any, weights: Dict[str, float]) -> float:
    names = ranker._feature_names()
    vals = {name: float(val) for name, val in zip(names, cfv.values)}
    return float(
        float(weights.get("ranker_weight", 1.0)) * float(base)
        + float(weights.get("rank_inv_weight", 0.0)) * vals.get("sample_rank_inv", 0.0)
        + float(weights.get("rank_log_inv_weight", 0.0)) * vals.get("sample_rank_log_inv", 0.0)
    )


def _score_rows_chunk(task: Tuple[str, List[Tuple[str, str]], Dict[str, List[str]], Dict[str, float]]) -> List[Tuple[str, int, float]]:
    ranker_pkl, row_payload, path_payload, weights = task
    model = pickle.loads(Path(ranker_pkl).expanduser().resolve().read_bytes())
    out: List[Tuple[str, int, float]] = []
    batch_x: List[np.ndarray] = []
    batch_keys: List[Tuple[str, int]] = []
    batch_feats: List[Any] = []
    for mid, cif in row_payload:
        row = ranker._row_from_prepared_cif(mid, cif)
        if row is None:
            continue
        paths = [Path(p) for p in path_payload.get(mid, [])]
        for path in sorted(paths, key=lambda p: _candidate_index(p)[1]):
            cfv = ranker._candidate_features(path, row)
            batch_x.append(cfv.values)
            batch_keys.append((row.material_id, int(cfv.idx)))
            batch_feats.append(cfv)
            if len(batch_x) >= 2048:
                vals = _score_batch(model, np.vstack(batch_x))
                out.extend((m, i, _blend_score(float(s), cfv, weights)) for (m, i), s, cfv in zip(batch_keys, vals.tolist(), batch_feats))
                batch_x.clear()
                batch_keys.clear()
                batch_feats.clear()
    if batch_x:
        vals = _score_batch(model, np.vstack(batch_x))
        out.extend((m, i, _blend_score(float(s), cfv, weights)) for (m, i), s, cfv in zip(batch_keys, vals.tolist(), batch_feats))
    return out


def _run_capture(cmd: Sequence[str]) -> str:
    proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    out = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _parse_metric_dict(stdout: str) -> Dict[str, Any]:
    import ast

    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, dict) else {}
    return {}


def _make_ranked_tar(
    *,
    out_tar: Path,
    rows: Sequence[ranker.RowData],
    grouped_paths: Dict[str, List[Path]],
    scores: Dict[str, Dict[int, float]],
    ranking_jsonl: Path,
    expected_k: int,
    interleave_policy: str = "",
) -> Dict[str, Any]:
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    ranking_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_added = 0
    n_missing = 0
    with tarfile.open(out_tar, "w:gz") as tar, ranking_jsonl.open("w", encoding="utf-8") as jf:
        for row in rows:
            paths = grouped_paths.get(row.material_id, [])
            ranked_paths = sorted(
                paths,
                key=lambda p: (-float(scores.get(row.material_id, {}).get(_candidate_index(p)[1], 0.0)), _candidate_index(p)[1]),
            )
            original_paths = sorted(paths, key=lambda p: _candidate_index(p)[1])
            if interleave_policy:
                chosen: List[Path] = []
                seen: set[Path] = set()
                for raw_tok in str(interleave_policy).replace("+", ",").split(","):
                    tok = raw_tok.strip().lower()
                    if len(tok) < 2:
                        continue
                    source = tok[0]
                    try:
                        pos = int(tok[1:]) - 1
                    except ValueError:
                        continue
                    pool = ranked_paths if source == "r" else original_paths if source == "o" else []
                    if 0 <= pos < len(pool) and pool[pos] not in seen:
                        chosen.append(pool[pos])
                        seen.add(pool[pos])
                for pool in (ranked_paths, original_paths):
                    for path in pool:
                        if len(chosen) >= int(expected_k):
                            break
                        if path not in seen:
                            chosen.append(path)
                            seen.add(path)
                    if len(chosen) >= int(expected_k):
                        break
                ranked_paths = chosen
            if len(ranked_paths) < int(expected_k):
                n_missing += int(expected_k) - len(ranked_paths)
            rec = {
                "material_id": row.material_id,
                "ranked": [
                    {
                        "new_idx": int(i),
                        "old_idx": int(_candidate_index(path)[1]),
                        "score": float(scores.get(row.material_id, {}).get(_candidate_index(path)[1], 0.0)),
                        "path": str(path),
                    }
                    for i, path in enumerate(ranked_paths[: int(expected_k)], start=1)
                ],
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for i, path in enumerate(ranked_paths[: int(expected_k)], start=1):
                tar.add(str(path), arcname=f"{row.material_id}__{i}.cif")
                n_added += 1
    return {"n_added": int(n_added), "n_missing": int(n_missing), "n_ids": int(len(rows))}


def _benchmark(
    *,
    py: str,
    bench_script: Path,
    gen_tar: Path,
    true_tar: Path,
    num_gens: int,
    max_sites: int,
    rmsd_timeout_seconds: float,
    hard_timeout_seconds: float,
    workers: int,
    out_json: Path,
    out_txt: Path,
) -> Dict[str, Any]:
    cmd = [
        str(py),
        str(bench_script),
        str(gen_tar),
        str(true_tar),
        "--num-gens",
        str(int(num_gens)),
        "--max-sites",
        str(int(max_sites)),
        "--rmsd-timeout-seconds",
        str(float(rmsd_timeout_seconds)),
        "--hard-timeout-seconds",
        str(float(hard_timeout_seconds)),
        "--workers",
        str(int(workers)),
    ]
    out = _run_capture(cmd)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(out, encoding="utf-8")
    metrics = _parse_metric_dict(out)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply an opentry validation ranker to existing postprocessed CIF candidates.")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--test-gt-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--ranker-pkl", required=True)
    parser.add_argument("--true-tar", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows after start-index.")
    parser.add_argument("--expected-k", type=int, default=20)
    parser.add_argument("--budgets", default="1,5,20")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--bench-script", default="/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/bin/benchmark_metrics.py")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--feature-chunk-ids", type=int, default=64)
    parser.add_argument("--score-config", default="", help="Optional JSON with ranker_weight/rank_inv_weight/rank_log_inv_weight.")
    parser.add_argument("--ranker-weight", type=float, default=1.0)
    parser.add_argument("--rank-inv-weight", type=float, default=0.0)
    parser.add_argument("--rank-log-inv-weight", type=float, default=0.0)
    parser.add_argument("--interleave-policy", default="", help="Optional fixed policy like r1,o1,o2,r2,r3 before ranker fill.")
    parser.add_argument("--max-sites", type=int, default=512)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--hard-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    out_dir = _ensure_under_opentry(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    tars_dir = out_dir / "tars"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tars_dir.mkdir(parents=True, exist_ok=True)

    limit = None if int(args.limit) <= 0 else int(args.limit)
    material_ids = _read_material_ids(Path(args.test_csv).expanduser().resolve(), start=int(args.start_index), limit=limit)
    rows = _load_rows_from_gt(material_ids, Path(args.test_gt_dir).expanduser().resolve())
    rows_by_id = {row.material_id: row for row in rows}

    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    score_weights = {
        "ranker_weight": float(args.ranker_weight),
        "rank_inv_weight": float(args.rank_inv_weight),
        "rank_log_inv_weight": float(args.rank_log_inv_weight),
    }
    if str(args.score_config or ""):
        loaded = json.loads(Path(args.score_config).expanduser().read_text(encoding="utf-8"))
        for key in ["ranker_weight", "rank_inv_weight", "rank_log_inv_weight"]:
            if key in loaded:
                score_weights[key] = float(loaded[key])
    grouped_paths: Dict[str, List[Path]] = {mid: [] for mid in rows_by_id}
    for path in sorted(candidate_dir.glob("*.cif")):
        parsed = _candidate_index(path)
        if parsed is None:
            continue
        mid, _ = parsed
        if mid in grouped_paths:
            grouped_paths[mid].append(path)

    scores: Dict[str, Dict[int, float]] = {}
    n_candidates = sum(len(v) for v in grouped_paths.values())
    n_scored = 0
    if int(args.feature_workers) > 1:
        chunk_size = max(1, int(args.feature_chunk_ids))
        chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
        tasks = []
        for chunk in chunks:
            payload_rows = [(row.material_id, row.cif) for row in chunk]
            payload_paths = {row.material_id: [str(p) for p in grouped_paths.get(row.material_id, [])] for row in chunk}
            tasks.append((str(Path(args.ranker_pkl).expanduser().resolve()), payload_rows, payload_paths, score_weights))
        done_chunks = 0
        with cf.ProcessPoolExecutor(max_workers=int(args.feature_workers)) as ex:
            for chunk_scores in ex.map(_score_rows_chunk, tasks):
                done_chunks += 1
                for mid, idx, score in chunk_scores:
                    scores.setdefault(mid, {})[int(idx)] = float(score)
                    n_scored += 1
                print(f"[rank-existing] scored chunks {done_chunks}/{len(tasks)} candidates={n_scored}", flush=True)
    else:
        model = pickle.loads(Path(args.ranker_pkl).expanduser().resolve().read_bytes())
        batch_x: List[np.ndarray] = []
        batch_keys: List[Tuple[str, int]] = []
        for row in rows:
            for path in sorted(grouped_paths.get(row.material_id, []), key=lambda p: _candidate_index(p)[1]):
                cfv = ranker._candidate_features(path, row)
                batch_x.append(cfv.values)
                batch_keys.append((row.material_id, int(cfv.idx)))
                batch_feats.append(cfv)
                n_scored += 1
                if len(batch_x) >= int(args.batch_size):
                    X = np.vstack(batch_x)
                    vals = _score_batch(model, X)
                    for (mid, idx), score, cfv in zip(batch_keys, vals, batch_feats):
                        scores.setdefault(mid, {})[idx] = _blend_score(float(score), cfv, score_weights)
                    batch_x.clear()
                    batch_keys.clear()
                    batch_feats.clear()
                    print(f"[rank-existing] scored {n_scored} candidates", flush=True)
        if batch_x:
            X = np.vstack(batch_x)
            vals = _score_batch(model, X)
            for (mid, idx), score, cfv in zip(batch_keys, vals, batch_feats):
                scores.setdefault(mid, {})[idx] = _blend_score(float(score), cfv, score_weights)
            print(f"[rank-existing] scored {n_scored} candidates", flush=True)
    print(f"[rank-existing] scored total {n_scored}/{n_candidates} candidates", flush=True)

    ranked_tar = tars_dir / "ranked_existing.tar.gz"
    ranking_stats = _make_ranked_tar(
        out_tar=ranked_tar,
        rows=rows,
        grouped_paths=grouped_paths,
        scores=scores,
        ranking_jsonl=out_dir / "ranking.jsonl",
        expected_k=int(args.expected_k),
        interleave_policy=str(args.interleave_policy or ""),
    )

    budgets = [int(x) for x in str(args.budgets).replace(" ", "").split(",") if x]
    results: Dict[str, Any] = {}
    for budget in budgets:
        print(f"[rank-existing] benchmark k={budget}", flush=True)
        results[f"ranked_k{budget}"] = _benchmark(
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
        "n_rows": int(len(rows)),
        "n_candidates": int(n_candidates),
        "n_features": int(len(ranker._feature_names())),
        "candidate_dir": str(candidate_dir),
        "ranker_pkl": str(Path(args.ranker_pkl).expanduser().resolve()),
        "score_weights": score_weights,
        "interleave_policy": str(args.interleave_policy or ""),
        "ranking_stats": ranking_stats,
        "results": results,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "summary_metrics.tsv").open("w", encoding="utf-8") as f:
        f.write("group\tK\tmatch_rate\trmse\tparse_rate_candidate\tvalid_rate_candidate\n")
        for key, metrics in results.items():
            k = key.rsplit("k", 1)[-1]
            f.write(
                "\t".join(
                    [
                        "ranked",
                        str(k),
                        str(metrics.get("match_rate")),
                        str(metrics.get("rms_dist")),
                        str(metrics.get("parse_rate_candidate")),
                        str(metrics.get("valid_rate_candidate")),
                    ]
                )
                + "\n"
            )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[done] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
