#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-help}"

ALIGNN_PY="${ALIGNN_PY:-/root/autodl-tmp/miniconda3/envs/alignn-score/bin/python}"
SERVER_PY="${SERVER_PY:-/root/autodl-tmp/model/CrystaLLM/resources/alignn_zmq_server_multi.py}"

ALIGNN_HOST="${ALIGNN_HOST:-127.0.0.1}"
ALIGNN_PORT="${ALIGNN_PORT:-5555}"
ALIGNN_PROPERTIES="${ALIGNN_PROPERTIES:-formation_energy bandgap}"
ALIGNN_CUTOFF="${ALIGNN_CUTOFF:-8}"
ALIGNN_MAX_NEIGHBORS="${ALIGNN_MAX_NEIGHBORS:-12}"

RUN_DIR="${RUN_DIR:-/root/autodl-tmp/run}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/logs}"
# Default caches/temp back on system disk (under /root and /tmp).
# NOTE: If your system disk is small, you may want to override these back to /root/autodl-tmp.
TMPDIR="${TMPDIR:-/tmp/alignn_zmq_tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache}"
TORCH_HOME="${TORCH_HOME:-/root/.cache/torch}"

PID_FILE="${PID_FILE:-$RUN_DIR/alignn_zmq_server_${ALIGNN_PORT}.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/alignn_zmq_server_${ALIGNN_PORT}.log}"

_usage() {
  cat <<EOF
Usage:
  $(basename "$0") start|stop|restart|status

Environment overrides:
  ALIGNN_PY=...               (python in alignn env)
  SERVER_PY=...               (alignn_zmq_server_multi.py)
  ALIGNN_HOST=127.0.0.1
  ALIGNN_PORT=5555
  ALIGNN_PROPERTIES="formation_energy bandgap"
  ALIGNN_CUTOFF=8
  ALIGNN_MAX_NEIGHBORS=12
  TMPDIR=/root/autodl-tmp/tmp         (avoid filling /)
  XDG_CACHE_HOME=/root/autodl-tmp/xdg-cache
  TORCH_HOME=/root/autodl-tmp/torch-cache
  RUN_DIR=/root/autodl-tmp/run
  LOG_DIR=/root/autodl-tmp/logs
EOF
}

_pid_from_file() {
  if [ -f "$PID_FILE" ]; then
    cat "$PID_FILE" 2>/dev/null || true
  fi
}

_is_running_pid() {
  local pid="$1"
  [ -n "${pid:-}" ] && ps -p "$pid" >/dev/null 2>&1
}

_status() {
  local pid
  pid="$(_pid_from_file || true)"
  if _is_running_pid "$pid"; then
    echo "RUNNING pid=$pid host=${ALIGNN_HOST} port=${ALIGNN_PORT}"
    ps -p "$pid" -o etime,cmd --no-headers 2>/dev/null || true
    return 0
  fi
  echo "STOPPED host=${ALIGNN_HOST} port=${ALIGNN_PORT}"
  return 1
}

_stop() {
  local pid
  pid="$(_pid_from_file || true)"
  if ! _is_running_pid "$pid"; then
    rm -f "$PID_FILE" 2>/dev/null || true
    echo "Already stopped."
    return 0
  fi
  echo "Stopping pid=$pid ..."
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if _is_running_pid "$pid"; then
      sleep 1
    else
      break
    fi
  done
  if _is_running_pid "$pid"; then
    echo "Still running; sending SIGKILL..."
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE" 2>/dev/null || true
  echo "Stopped."
}

_start() {
  if _status >/dev/null 2>&1; then
    _status
    return 0
  fi

  mkdir -p "$RUN_DIR" "$LOG_DIR" "$TMPDIR"

  if [ ! -x "$ALIGNN_PY" ]; then
    echo "ERROR: ALIGNN_PY not found/executable: $ALIGNN_PY" >&2
    exit 2
  fi
  if [ ! -f "$SERVER_PY" ]; then
    echo "ERROR: SERVER_PY not found: $SERVER_PY" >&2
    exit 2
  fi

  # PyTorch 2.6+ may default torch.load(weights_only=True), which breaks ALIGNN pretrained loading.
  # Monkeypatch torch.load to ensure weights_only=False by default (safe for local checkpoints).
  local py_bootstrap
  py_bootstrap=$(
    cat <<'PY'
import os, runpy, torch
_real = torch.load
def _load(*a, **k):
    if "weights_only" not in k:
        k["weights_only"] = False
    return _real(*a, **k)
torch.load = _load
server_py = os.environ.get("SERVER_PY")
if not server_py:
    raise SystemExit("SERVER_PY env var is required")
runpy.run_path(server_py, run_name="__main__")
PY
  )

  echo "Starting ALIGNN ZMQ on ${ALIGNN_HOST}:${ALIGNN_PORT} ..."
  nohup env \
    TMPDIR="$TMPDIR" \
    XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    TORCH_HOME="$TORCH_HOME" \
    SERVER_PY="$SERVER_PY" \
    "$ALIGNN_PY" -c "$py_bootstrap" \
      --host "$ALIGNN_HOST" \
      --port "$ALIGNN_PORT" \
      --cutoff "$ALIGNN_CUTOFF" \
      --max-neighbors "$ALIGNN_MAX_NEIGHBORS" \
      --properties $ALIGNN_PROPERTIES \
    >>"$LOG_FILE" 2>&1 &

  echo $! >"$PID_FILE"
  sleep 0.2
  _status || true
  echo "log: $LOG_FILE"
  echo "pid_file: $PID_FILE"
}

case "$CMD" in
  start) _start ;;
  stop) _stop ;;
  restart) _stop; _start ;;
  status) _status ;;
  help|-h|--help) _usage ;;
  *)
    echo "Unknown command: $CMD" >&2
    _usage >&2
    exit 2
    ;;
esac
