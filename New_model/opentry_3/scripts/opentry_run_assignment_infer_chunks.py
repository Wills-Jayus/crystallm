#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SCRIPT_DIR = OPENTRY_ROOT / "scripts"
ASSIGN_SCRIPT = SCRIPT_DIR / "diagnose_wyckoff_assignment_dp_infer.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_wyckoff_sequence_models import ensure_under_opentry, read_jsonl, write_json  # noqa: E402


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def run_chunk(args: argparse.Namespace, *, start: int, count: int, chunk_index: int, total_chunks: int) -> dict[str, Any]:
    chunk_root = args.out_dir / "chunks" / f"chunk_{chunk_index:04d}_{start:06d}_{start + count:06d}"
    chunk_root.mkdir(parents=True, exist_ok=True)
    log_path = chunk_root / "run.log"
    cmd = [
        sys.executable,
        str(ASSIGN_SCRIPT),
        "--data-dir",
        str(args.data_dir),
        "--split",
        str(args.split),
        "--train-repr",
        str(args.train_repr),
        "--vocab-json",
        str(args.vocab_json),
        "--priors-json",
        str(args.priors_json),
        "--predicted-candidates",
        str(args.predicted_candidates),
        "--out-dir",
        str(chunk_root),
        "--start-index",
        str(start),
        "--max-records",
        str(count),
        "--top-k",
        str(args.top_k),
        "--state-beam",
        str(args.state_beam),
        "--max-active-paths",
        str(args.max_active_paths),
        "--max-skeletons",
        str(args.max_skeletons),
        "--per-skeleton",
        str(args.per_skeleton),
        "--prior-weight",
        str(args.prior_weight),
        "--assignment-model-weight",
        str(args.assignment_model_weight),
        "--progress-every",
        str(args.chunk_progress_every),
        "--flush-every",
        str(args.flush_every),
    ]
    if args.allow_duplicate_fixed_skeletons:
        cmd.append("--allow-duplicate-fixed-skeletons")
    if args.assignment_model_ckpt is not None:
        cmd.extend(["--assignment-model-ckpt", str(args.assignment_model_ckpt)])
    cmd.extend(["--assignment-device", str(args.assignment_device)])
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(OPENTRY_ROOT / "tmp")
    env["XDG_CACHE_HOME"] = str(OPENTRY_ROOT / "cache")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(cmd, cwd=str(OPENTRY_ROOT.parents[2]), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    summary_path = chunk_root / "assignment_dp_summary.json"
    if proc.returncode != 0:
        return {
            "chunk_index": chunk_index,
            "start": start,
            "count": count,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "log": str(log_path),
        }
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    print(
        json.dumps(
            {
                "chunk_done": chunk_index,
                "chunks_total": total_chunks,
                "start": start,
                "count": count,
                "elapsed_s": elapsed,
                "candidate_nonempty_rate": summary.get("candidate_nonempty_rate"),
                "unique_mean": summary.get("unique_mean"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "chunk_index": chunk_index,
        "start": start,
        "count": count,
        "returncode": 0,
        "elapsed_s": elapsed,
        "log": str(log_path),
        "summary": summary,
        "per_sample": str(chunk_root / "predicted_skeleton_assignment_per_sample.jsonl"),
        "candidates": str(chunk_root / "predicted_skeleton_assignment_candidates.jsonl"),
    }


def append_file(src: Path, dst_handle: Any) -> int:
    count = 0
    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            dst_handle.write(line)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel chunk runner for inference-only assignment DP.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--train-repr", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, required=True)
    parser.add_argument("--priors-json", type=Path, required=True)
    parser.add_argument("--predicted-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--state-beam", type=int, required=True)
    parser.add_argument("--max-active-paths", type=int, required=True)
    parser.add_argument("--max-skeletons", type=int, required=True)
    parser.add_argument("--per-skeleton", type=int, required=True)
    parser.add_argument("--assignment-model-ckpt", type=Path, default=None)
    parser.add_argument("--assignment-device", default="cpu")
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--assignment-model-weight", type=float, default=0.0)
    parser.add_argument("--allow-duplicate-fixed-skeletons", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=640)
    parser.add_argument("--chunk-progress-every", type=int, default=250)
    parser.add_argument("--flush-every", type=int, default=25)
    args = parser.parse_args()

    start = time.time()
    args.out_dir = ensure_under_opentry(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_records = len(read_jsonl(args.data_dir / f"{args.split}.jsonl"))
    chunk_size = max(1, int(args.chunk_size))
    starts = list(range(0, total_records, chunk_size))
    total_chunks = len(starts)
    chunks = [(start_idx, min(chunk_size, total_records - start_idx), idx) for idx, start_idx in enumerate(starts)]
    print(json.dumps({"chunks": total_chunks, "records": total_records, "workers": int(args.workers)}, sort_keys=True), flush=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(run_chunk, args, start=start_idx, count=count, chunk_index=chunk_idx, total_chunks=total_chunks)
            for start_idx, count, chunk_idx in chunks
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if int(result.get("returncode", 1)) != 0:
                failures.append(result)
    results.sort(key=lambda item: int(item["start"]))
    if failures:
        write_json(args.out_dir / "chunk_failures.json", failures)
        print(json.dumps({"failed_chunks": len(failures), "failures": failures}, sort_keys=True), flush=True)
        return 1

    per_sample_out = args.out_dir / "predicted_skeleton_assignment_per_sample.jsonl"
    candidates_out = args.out_dir / "predicted_skeleton_assignment_candidates.jsonl"
    per_count = 0
    cand_count = 0
    with per_sample_out.open("w", encoding="utf-8") as per_handle, candidates_out.open("w", encoding="utf-8") as cand_handle:
        for result in results:
            per_count += append_file(Path(result["per_sample"]), per_handle)
            cand_count += append_file(Path(result["candidates"]), cand_handle)

    nonempty_weighted = 0.0
    unique_weighted = 0.0
    elapsed_chunks = 0.0
    for result in results:
        summary = result["summary"]
        records = int(summary.get("records", 0))
        nonempty_weighted += float(summary.get("candidate_nonempty_rate", 0.0)) * records
        unique_weighted += float(summary.get("unique_mean", 0.0)) * records
        elapsed_chunks += float(result.get("elapsed_s", 0.0))
    summary = {
        "config": json_safe_args(args),
        "records": total_records,
        "chunks": total_chunks,
        "workers": int(args.workers),
        "elapsed_s": time.time() - start,
        "chunk_elapsed_sum_s": elapsed_chunks,
        "per_sample_rows": per_count,
        "candidate_rows": cand_count,
        "candidate_nonempty_rate": nonempty_weighted / max(1, total_records),
        "unique_mean": unique_weighted / max(1, total_records),
        "label_policy": "Inference-only chunk runner: no target W/A, target skeleton, row_count label, hit metrics, or StructureMatcher labels are emitted.",
        "chunks_detail": results,
    }
    write_json(args.out_dir / "assignment_dp_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
