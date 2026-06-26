#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import Any

import run_opentry10 as ctl


DATASET = "mpts_52"
PLAN_LOCK = ctl.STATE_DIR / "mpts52_parallel_worker.plan.lock"


def worker_id() -> str:
    return f"{os.uname().nodename}:{os.getpid()}:cuda={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"


def acquire_plan_lock():
    ctl.ensure_dir(ctl.STATE_DIR)
    lock = ctl.under_root(PLAN_LOCK).open("a+", encoding="utf-8")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    return lock


def acquire_shard_lock(shard: dict[str, Any]):
    run_dir = Path(shard["run_dir"])
    ctl.ensure_dir(run_dir)
    lock_path = ctl.under_root(run_dir / ".shard.lock")
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    lock.seek(0)
    lock.truncate()
    lock.write(
        json.dumps(
            {
                "dataset": DATASET,
                "shard_index": int(shard["shard_index"]),
                "pid": os.getpid(),
                "worker": worker_id(),
                "locked_at": ctl.now_iso(),
                "run_dir": str(run_dir),
            },
            sort_keys=True,
        )
        + "\n"
    )
    lock.flush()
    return lock


def release_shard_lock(lock) -> None:
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        lock.close()


def write_plan_shard_update(shard_index: int, updates: dict[str, Any]) -> None:
    plan_lock = acquire_plan_lock()
    try:
        plan = ctl.load_shard_plan(DATASET)
        for shard in plan["shards"]:
            if int(shard["shard_index"]) == int(shard_index):
                shard.update(updates)
                break
        else:
            raise RuntimeError(f"shard {shard_index} not found")
        ctl.update_shard_plan(DATASET, plan)
    finally:
        fcntl.flock(plan_lock.fileno(), fcntl.LOCK_UN)
        plan_lock.close()


def claim_next_shard() -> tuple[dict[str, Any] | None, Any | None]:
    plan_lock = acquire_plan_lock()
    try:
        plan = ctl.load_shard_plan(DATASET)
        for shard in plan["shards"]:
            if shard.get("status") == "completed":
                continue
            lock = acquire_shard_lock(shard)
            if lock is None:
                continue
            shard["status"] = "running"
            shard["started_at"] = ctl.now_iso()
            shard["worker"] = worker_id()
            shard.pop("last_error", None)
            shard.pop("failed_at", None)
            ctl.update_shard_plan(DATASET, plan)
            return dict(shard), lock
        return None, None
    finally:
        fcntl.flock(plan_lock.fileno(), fcntl.LOCK_UN)
        plan_lock.close()


def run_shard(shard: dict[str, Any]) -> None:
    run_dir = Path(shard["run_dir"])
    ctl.ensure_dir(run_dir / "cifs_raw/data_atomtype_gt_sg")
    ctl.ensure_dir(run_dir / "cifs_post/data_atomtype_gt_sg")
    ctl.write_json(
        run_dir / "generation_provenance.json",
        {
            "dataset": DATASET,
            "source_full_run_dir": str(ctl.val_run_dir(DATASET)),
            "shard": {k: v for k, v in shard.items() if k not in {"last_error", "failed_at"}},
            "generation_configs": ctl.ANCHOR_K100_CONFIGS,
            "created_at": ctl.now_iso(),
            "parallel_worker": worker_id(),
        },
    )
    for cfg in ctl.ANCHOR_K100_CONFIGS:
        cmd = ctl.anchor_shard_generate_command(DATASET, shard, cfg)
        stage_id = f"generate_validation_anchor_{ctl.dataset_short(DATASET)}_shard{int(shard['shard_index']):04d}_{cfg['name']}"
        ctl.run_logged(stage_id, cmd, max_attempts=3, oom_worker_arg="--workers")
    raw_count, raw_missing = ctl.count_shard_cifs(DATASET, shard, postprocessed=False)
    if raw_missing:
        raise RuntimeError(f"raw shard coverage incomplete: {raw_count}/{shard['expected_cifs']} first_missing={raw_missing[:3]}")
    ctl.run_logged(
        f"postprocess_validation_anchor_{ctl.dataset_short(DATASET)}_shard{int(shard['shard_index']):04d}",
        ctl.anchor_shard_postprocess_command(shard),
        max_attempts=2,
        oom_worker_arg="--workers",
    )
    post_count, post_missing = ctl.count_shard_cifs(DATASET, shard, postprocessed=True)
    if post_missing:
        raise RuntimeError(f"post shard coverage incomplete: {post_count}/{shard['expected_cifs']} first_missing={post_missing[:3]}")
    ctl.write_shard_manifest(DATASET, shard)


def maybe_write_complete_marker() -> None:
    plan_lock = acquire_plan_lock()
    try:
        plan = ctl.load_shard_plan(DATASET)
        remaining = [s for s in plan["shards"] if s.get("status") != "completed"]
        if remaining:
            return
        ctl.write_json(
            ctl.shard_complete_path(DATASET),
            {
                "dataset": DATASET,
                "dataset_short": ctl.dataset_short(DATASET),
                "completed_at": ctl.now_iso(),
                "num_shards": len(plan["shards"]),
                "samples_per_prompt": 100,
                "expected_cifs": sum(int(s["expected_cifs"]) for s in plan["shards"]),
                "plan": str(ctl.shard_plan_path(DATASET)),
                "completed_by": "run_mpts52_parallel_shard_worker.py",
            },
        )
    finally:
        fcntl.flock(plan_lock.fileno(), fcntl.LOCK_UN)
        plan_lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel MPTS-52 shard worker with plan and per-shard locks.")
    parser.add_argument("--max-shards", type=int, default=0, help="0 means run until no claimable shards remain.")
    args = parser.parse_args()

    completed = 0
    while int(args.max_shards) <= 0 or completed < int(args.max_shards):
        shard, lock = claim_next_shard()
        if shard is None:
            print("[worker] no claimable shards remain")
            maybe_write_complete_marker()
            return
        shard_index = int(shard["shard_index"])
        print(f"[worker] claimed shard {shard_index} on {worker_id()}", flush=True)
        try:
            run_shard(shard)
            checksum = ctl.checksum_path(Path(shard["run_dir"]))
            write_plan_shard_update(
                shard_index,
                {
                    "status": "completed",
                    "completed_at": ctl.now_iso(),
                    "output_checksum": checksum,
                    "worker": worker_id(),
                    "last_error": None,
                    "failed_at": None,
                },
            )
            completed += 1
            print(f"[worker] completed shard {shard_index}", flush=True)
        except Exception as exc:  # noqa: BLE001
            write_plan_shard_update(
                shard_index,
                {
                    "status": "failed",
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "failed_at": ctl.now_iso(),
                    "worker": worker_id(),
                },
            )
            raise
        finally:
            release_shard_lock(lock)

    maybe_write_complete_marker()


if __name__ == "__main__":
    main()
