#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PY = Path("/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python")
CONTROLLER = ROOT / "scripts/run_opentry10.py"
PLAN_PATH = ROOT / "state/validation_anchor_symprec0p1_shards_mp20.json"
STATUS_PATH = ROOT / "state/mp20_until_done_status.json"
LOCK_PATH = ROOT / "state/mp20_until_done.lock"
LOG_PATH = ROOT / "logs/mp20_until_done.log"
CONTROLLER_LOG_PATH = ROOT / "logs/mp20_until_done_controller.log"
STAGE = "generate_validation_anchor_symprec0p1_mp20_shards"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_plan() -> dict[str, Any]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def summarize_plan() -> dict[str, Any]:
    plan = read_plan()
    shards = plan["shards"]
    counts: dict[str, int] = {}
    for shard in shards:
        status = str(shard.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    completed = counts.get("completed", 0)
    total = len(shards)
    expected_total = sum(int(s.get("expected_cifs", 0)) for s in shards)
    expected_completed = sum(
        int(s.get("expected_cifs", 0))
        for s in shards
        if s.get("status") == "completed"
    )
    running = [
        {
            "shard_index": int(s["shard_index"]),
            "started_at": s.get("started_at"),
            "expected_cifs": int(s.get("expected_cifs", 0)),
            "run_dir": s.get("run_dir"),
        }
        for s in shards
        if s.get("status") == "running"
    ]
    return {
        "updated_at": now_iso(),
        "total_shards": total,
        "counts": counts,
        "completed_shards": completed,
        "remaining_shards": total - completed,
        "expected_cifs_completed": expected_completed,
        "expected_cifs_total": expected_total,
        "fraction_complete": round(expected_completed / expected_total, 6)
        if expected_total
        else None,
        "running": running,
        "complete": completed == total,
    }


def write_status(status: str, **extra: Any) -> None:
    payload = {
        "status": status,
        "pid": os.getpid(),
        "updated_at": now_iso(),
        "root": str(ROOT),
        "stage": STAGE,
        "controller": str(CONTROLLER),
        "controller_log": str(CONTROLLER_LOG_PATH),
        "summary": summarize_plan() if PLAN_PATH.exists() else None,
        **extra,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {message}\n")


def active_external_controller_pids() -> list[int]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,pcpu,pmem,cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    pids: list[int] = []
    self_pid = os.getpid()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[5]
        if pid == self_pid:
            continue
        if str(CONTROLLER) in cmd and STAGE in cmd:
            pids.append(pid)
    return sorted(set(pids))


def run_controller_once() -> int:
    cmd = [
        str(PY),
        str(CONTROLLER),
        "--resume",
        "--only",
        STAGE,
        "--max-shards-per-stage",
        "0",
    ]
    append_log("starting controller: " + " ".join(cmd))
    with CONTROLLER_LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now_iso()}] starting controller\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        write_status("controller_running", controller_pid=proc.pid)
        return_code = proc.wait()
        log.write(f"[{now_iso()}] controller exited rc={return_code}\n")
    append_log(f"controller exited rc={return_code}")
    return return_code


def main() -> int:
    ROOT.joinpath("state").mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another supervisor already holds {LOCK_PATH}", file=sys.stderr)
            return 2
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "locked_at": now_iso()}) + "\n")
        lock.flush()

        append_log("supervisor started")
        failures = 0
        while True:
            summary = summarize_plan()
            if summary["complete"]:
                write_status("complete")
                append_log("MP-20 shards complete")
                return 0

            pids = active_external_controller_pids()
            if pids:
                write_status("waiting_for_existing_controller", external_controller_pids=pids)
                append_log(f"waiting for existing controller pids={pids}")
                time.sleep(300)
                continue

            return_code = run_controller_once()
            summary = summarize_plan()
            if summary["complete"]:
                write_status("complete", last_controller_return_code=return_code)
                append_log("MP-20 shards complete after controller")
                return 0
            if return_code != 0:
                failures += 1
                write_status(
                    "controller_failed_retrying",
                    last_controller_return_code=return_code,
                    consecutive_failures=failures,
                )
                append_log(f"controller failure {failures}; sleeping before retry")
                if failures >= 3:
                    write_status(
                        "failed",
                        last_controller_return_code=return_code,
                        consecutive_failures=failures,
                    )
                    append_log("stopping after 3 consecutive controller failures")
                    return return_code
                time.sleep(600)
            else:
                failures = 0
                write_status("controller_exited_not_complete_retrying", last_controller_return_code=return_code)
                time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
