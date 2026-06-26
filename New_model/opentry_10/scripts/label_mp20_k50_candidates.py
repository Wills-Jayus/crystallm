#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import time

import benchmark_metrics_opentry10 as bm


ROOT = Path(__file__).resolve().parents[1]
GEN_JSONL = ROOT / "candidates/crystallm_gt_sg_mp20_val_k100.jsonl"
TRUE_TAR = ROOT / "generations/crystallm_gt_sg_val_anchor_symprec0p1/mp20_val_data_atomtype_gt_sg_symprec0p1_k100/tars/true.tar.gz"

_MATCHER = None
_RMSD_TIMEOUT_S = None
_MAX_SITES = None


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


def _init_worker(ltol: float, stol: float, angle_tol: float, rmsd_timeout_s: float, max_sites: int):
    global _MATCHER, _RMSD_TIMEOUT_S, _MAX_SITES
    _MATCHER = bm.StructureMatcher(stol=float(stol), angle_tol=float(angle_tol), ltol=float(ltol))
    _RMSD_TIMEOUT_S = float(rmsd_timeout_s)
    _MAX_SITES = int(max_sites)


def _label_material(task: dict[str, Any]) -> list[dict[str, Any]]:
    material_id = task["material_id"]
    true_cif = task["true_cif"]
    candidates = task["candidates"]

    class RmsdTimeout(Exception):
        pass

    @contextlib.contextmanager
    def time_limit(seconds: float | None):
        if seconds is None or seconds <= 0 or bm.signal is None:
            yield
            return

        def handler(signum, frame):  # noqa: ARG001
            raise RmsdTimeout()

        old_handler = bm.signal.signal(bm.signal.SIGALRM, handler)
        try:
            bm.signal.setitimer(bm.signal.ITIMER_REAL, float(seconds))
            yield
        finally:
            try:
                bm.signal.setitimer(bm.signal.ITIMER_REAL, 0)
            except Exception:
                pass
            try:
                bm.signal.signal(bm.signal.SIGALRM, old_handler)
            except Exception:
                pass

    records: list[dict[str, Any]] = []
    try:
        gt = bm.Structure.from_str(bm._normalize_cif_symmops_to_declared_sg(true_cif), fmt="cif")
        gt_parse_ok = True
    except Exception as exc:  # noqa: BLE001
        gt = None
        gt_parse_ok = False
        gt_error = f"{type(exc).__name__}: {exc}"

    for cand in candidates:
        cif = cand["generated_text"]
        rec = {
            "dataset": cand.get("dataset"),
            "split": cand.get("split"),
            "sample_id": cand.get("sample_id"),
            "material_id": material_id,
            "rank": int(cand["rank"]),
            "gen_index": cand.get("gen_index"),
            "generation_config": cand.get("generation_config"),
            "gt_parse_ok": gt_parse_ok,
            "parse_ok": False,
            "sensible": False,
            "valid": False,
            "match": False,
            "rmsd": None,
            "label_status": "ok",
            "error": None,
        }
        if not gt_parse_ok:
            rec["label_status"] = "gt_parse_error"
            rec["error"] = gt_error
            records.append(rec)
            continue
        try:
            rec["sensible"] = bool(bm.is_sensible(cif, 0.5, 2.0, 50.0, 130.0))
        except Exception:
            rec["sensible"] = False
        try:
            pred = bm.Structure.from_str(bm._normalize_cif_symmops_to_declared_sg(cif), fmt="cif")
            rec["parse_ok"] = True
        except Exception as exc:  # noqa: BLE001
            rec["label_status"] = "parse_error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
            continue
        try:
            rec["valid"] = bool(bm.is_valid(pred))
        except Exception:
            rec["valid"] = False
        try:
            if _MAX_SITES is not None and (int(pred.num_sites) > int(_MAX_SITES) or int(gt.num_sites) > int(_MAX_SITES)):
                rec["label_status"] = "skipped_large"
                records.append(rec)
                continue
        except Exception:
            pass
        try:
            with time_limit(_RMSD_TIMEOUT_S):
                rms_dist = _MATCHER.get_rms_dist(pred, gt)
            rmsd = None if rms_dist is None else float(rms_dist[0])
            rec["rmsd"] = rmsd
            rec["match"] = rmsd is not None
        except RmsdTimeout:
            rec["label_status"] = "rmsd_timeout"
        except Exception as exc:  # noqa: BLE001
            rec["label_status"] = "rmsd_error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)
    return records


def iter_material_tasks(input_path: Path, true_cifs: dict[str, str], max_rank: int, sample_limit: int | None):
    current_mid = None
    current_rows: list[dict[str, Any]] = []
    yielded = 0

    def flush():
        nonlocal yielded
        if not current_rows:
            return None
        mid = str(current_rows[0]["material_id"])
        yielded += 1
        return {
            "material_id": mid,
            "true_cif": true_cifs[mid],
            "candidates": current_rows,
        }

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rank = int(row.get("rank", 0))
            mid = str(row.get("material_id"))
            if current_mid is None:
                current_mid = mid
            if mid != current_mid:
                task = flush()
                if task is not None:
                    yield task
                    if sample_limit is not None and yielded >= sample_limit:
                        return
                current_mid = mid
                current_rows = []
            if 1 <= rank <= int(max_rank):
                current_rows.append(row)
        task = flush()
        if task is not None and (sample_limit is None or yielded <= sample_limit):
            yield task


def run_labeling(tasks, out_path: Path, workers: int, window: int, hard_timeout_s: float) -> dict[str, Any]:
    out_path = under_root(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q = deque(tasks)
    total = len(q)
    completed = 0
    hard_timeout_materials = 0
    label_counts: dict[str, int] = {}
    started_at = time.monotonic()

    with out_path.open("w", encoding="utf-8") as fout:
        pool = mp.Pool(
            processes=int(workers),
            initializer=_init_worker,
            initargs=(0.3, 0.5, 10.0, 5.0, 512),
        )
        running: list[dict[str, Any]] = []

        def submit(task):
            return {
                "task": task,
                "async": pool.apply_async(_label_material, (task,)),
                "started_at": time.monotonic(),
            }

        def fill():
            while q and len(running) < int(window):
                running.append(submit(q.popleft()))

        fill()
        try:
            while running:
                progressed = False
                keep = []
                for item in running:
                    ar = item["async"]
                    task = item["task"]
                    if ar.ready():
                        rows = ar.get(timeout=0)
                        for row in rows:
                            status = str(row.get("label_status"))
                            label_counts[status] = label_counts.get(status, 0) + 1
                            fout.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                        completed += 1
                        progressed = True
                    else:
                        keep.append(item)
                running = keep
                if progressed:
                    fill()
                    continue
                now = time.monotonic()
                timed_out = [item for item in running if now - float(item["started_at"]) >= float(hard_timeout_s)]
                if timed_out:
                    timeout_ids = {id(item) for item in timed_out}
                    for item in timed_out:
                        task = item["task"]
                        hard_timeout_materials += 1
                        completed += 1
                        for cand in task["candidates"]:
                            row = {
                                "dataset": cand.get("dataset"),
                                "split": cand.get("split"),
                                "sample_id": cand.get("sample_id"),
                                "material_id": task["material_id"],
                                "rank": int(cand["rank"]),
                                "gen_index": cand.get("gen_index"),
                                "generation_config": cand.get("generation_config"),
                                "gt_parse_ok": None,
                                "parse_ok": None,
                                "sensible": None,
                                "valid": None,
                                "match": False,
                                "rmsd": None,
                                "label_status": "material_hard_timeout",
                                "error": None,
                            }
                            label_counts["material_hard_timeout"] = label_counts.get("material_hard_timeout", 0) + 1
                            fout.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                    try:
                        pool.terminate()
                    finally:
                        pool.join()
                    retry_tasks = [item["task"] for item in running if id(item) not in timeout_ids]
                    for task in reversed(retry_tasks):
                        q.appendleft(task)
                    if q:
                        pool = mp.Pool(
                            processes=int(workers),
                            initializer=_init_worker,
                            initargs=(0.3, 0.5, 10.0, 5.0, 512),
                        )
                        running = []
                        fill()
                    else:
                        running = []
                    continue
                time.sleep(0.2)
        finally:
            try:
                pool.close()
            except Exception:
                pass
            try:
                pool.join()
            except Exception:
                pass

    return {
        "created_at": now_iso(),
        "output": str(out_path.resolve()),
        "materials": total,
        "completed_materials": completed,
        "hard_timeout_materials": hard_timeout_materials,
        "label_counts": dict(sorted(label_counts.items())),
        "elapsed_seconds": time.monotonic() - started_at,
        "workers": int(workers),
        "window": int(window),
        "hard_timeout_seconds": float(hard_timeout_s),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label MP-20 K50 validation candidates with timeout-safe StructureMatcher calls.")
    parser.add_argument("--input", default=str(GEN_JSONL))
    parser.add_argument("--true-tar", default=str(TRUE_TAR))
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--hard-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--out", default=str(ROOT / "labels/mp20_val_k50_candidate_labels.jsonl"))
    parser.add_argument("--summary", default=str(ROOT / "metrics/mp20_val_k50_candidate_label_summary.json"))
    args = parser.parse_args()

    true_cifs = bm.read_true_cifs(args.true_tar)
    sample_limit = None if int(args.sample_limit) <= 0 else int(args.sample_limit)
    tasks = list(iter_material_tasks(Path(args.input), true_cifs, int(args.max_rank), sample_limit))
    summary = run_labeling(
        tasks,
        Path(args.out),
        workers=int(args.workers),
        window=int(args.window),
        hard_timeout_s=float(args.hard_timeout_seconds),
    )
    summary.update(
        {
            "input": str(Path(args.input).resolve()),
            "true_tar": str(Path(args.true_tar).resolve()),
            "max_rank": int(args.max_rank),
            "sample_limit": sample_limit,
            "note": "Sample-limited outputs are sanity artifacts and must not be used as validation conclusions.",
        }
    )
    write_json(Path(args.summary), summary)


if __name__ == "__main__":
    main()
