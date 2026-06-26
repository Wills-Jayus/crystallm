#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_10")
PY = Path("/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python")
CONTROLLER = ROOT / "scripts/run_opentry10.py"
PLAN = ROOT / "state/validation_anchor_symprec0p1_shards_mpts52.json"
COMPLETE = ROOT / "state/validation_anchor_symprec0p1_shards_mpts52.complete.json"
STATE = ROOT / "state/mpts52_fast_finish_status.json"
LOG = ROOT / "logs/mpts52_fast_finish.log"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_state(payload: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def append_log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {line}\n")


def load_plan() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def shard_counts(plan: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shard in plan.get("shards", []):
        status = str(shard.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def all_shards_completed() -> bool:
    if COMPLETE.exists():
        return True
    plan = load_plan()
    return all(shard.get("status") == "completed" for shard in plan.get("shards", []))


def env_for_fast_eval(workers: int, window: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TMPDIR": str(ROOT / "tmp"),
            "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
            "TORCH_HOME": str(ROOT / "cache/torch"),
            "HF_HOME": str(ROOT / "cache/hf"),
            "CUDA_CACHE_PATH": str(ROOT / "cache/cuda"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OPENTRY10_BENCH_WORKERS": str(workers),
            "OPENTRY10_BENCH_WINDOW": str(window),
        }
    )
    return env


def run_fast_eval(workers: int, window: int) -> int:
    cmd = [
        str(PY),
        str(CONTROLLER),
        "--resume",
        "--only",
        ",".join(
            [
                "generate_validation_anchor_symprec0p1_mpts52_shards",
                "assemble_validation_anchor_symprec0p1_mpts52",
                "export_validation_anchor_symprec0p1_mpts52_jsonl",
                "copy_validation_anchor_symprec0p1_mpts52_metrics",
                "validation_anchor_report",
            ]
        ),
    ]
    append_log("starting fast finish: " + " ".join(cmd))
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"[{now_iso()}] env OPENTRY10_BENCH_WORKERS={workers} OPENTRY10_BENCH_WINDOW={window}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env_for_fast_eval(workers, window),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        write_state(
            {
                "status": "running_fast_eval",
                "updated_at": now_iso(),
                "pid": proc.pid,
                "workers": workers,
                "window": window,
                "log": str(LOG),
            }
        )
        rc = proc.wait()
    append_log(f"fast finish exited rc={rc}")
    write_state(
        {
            "status": "completed" if rc == 0 else "failed",
            "updated_at": now_iso(),
            "return_code": rc,
            "workers": workers,
            "window": window,
            "log": str(LOG),
        }
    )
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for MPTS-52 shards, then run high-parallel CPU evaluation.")
    parser.add_argument("--workers", type=int, default=120)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    append_log("watcher started")
    while not all_shards_completed():
        plan = load_plan()
        counts = shard_counts(plan)
        write_state(
            {
                "status": "waiting_for_shards",
                "updated_at": now_iso(),
                "counts": counts,
                "workers": int(args.workers),
                "window": int(args.window),
                "log": str(LOG),
            }
        )
        append_log(f"waiting_for_shards counts={counts}")
        time.sleep(max(5, int(args.poll_seconds)))

    write_state(
        {
            "status": "shards_complete_starting_fast_eval",
            "updated_at": now_iso(),
            "workers": int(args.workers),
            "window": int(args.window),
            "log": str(LOG),
        }
    )
    raise SystemExit(run_fast_eval(int(args.workers), int(args.window)))


if __name__ == "__main__":
    main()
