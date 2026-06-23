#!/usr/bin/env python3
"""Run a training command without shell wrapping and persist stdout/stderr."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        raise SystemExit("missing command after --")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n===== run_train_with_log start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.write("command: " + " ".join(args.cmd) + "\n")
        proc = subprocess.Popen(args.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
        log.write(f"===== run_train_with_log exit rc={rc} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
