#!/usr/bin/env bash
set -euo pipefail

# Simple process manager for ALIGNN ZMQ scoring server (resources/alignn_zmq_server_multi.py)
#
# Usage:
#   bin/alignn_zmq_server_ctl.sh start   [--host 127.0.0.1] [--port 5555] [--properties "formation_energy bandgap"] [--conda-prefix /root/autodl-tmp/miniconda3] [--env alignn-score]
#   bin/alignn_zmq_server_ctl.sh stop
#   bin/alignn_zmq_server_ctl.sh status
#   bin/alignn_zmq_server_ctl.sh restart
#   bin/alignn_zmq_server_ctl.sh pause   # SIGSTOP
#   bin/alignn_zmq_server_ctl.sh resume  # SIGCONT
#
# Notes:
# - We intentionally do NOT delete any files. Logs rotate by timestamped filename.
# - This script is meant for container environments without systemd.

ACTION="${1:-}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_PY="$ROOT_DIR/resources/alignn_zmq_server_multi.py"

# Prefer a writable data root:
# - If AUTODL_ROOT is set, use it.
# - Otherwise infer it from the repo location: <AUTODL_ROOT>/model/CrystaLLM -> <AUTODL_ROOT>
AUTODL_ROOT_INFERRED="$(cd "$ROOT_DIR/../.." && pwd)"
DATA_ROOT_DEFAULT="${AUTODL_ROOT:-$AUTODL_ROOT_INFERRED}"

RUN_DIR_DEFAULT="$DATA_ROOT_DEFAULT/run"
LOG_DIR_DEFAULT="$DATA_ROOT_DEFAULT/logs/alignn_zmq"
PIDFILE_DEFAULT="$RUN_DIR_DEFAULT/alignn_zmq_server_multi.pid"
MODEL_ZIP_SEED_DIR_DEFAULT="$DATA_ROOT_DEFAULT/alignn_model_zips"

HOST="127.0.0.1"
PORT="5555"
PROPERTIES="formation_energy bandgap"
CONDA_PREFIX="$DATA_ROOT_DEFAULT/miniforge3"
CONDA_ENV="alignn-score"
PIDFILE="$PIDFILE_DEFAULT"
LOG_DIR="$LOG_DIR_DEFAULT"
MODEL_ZIP_SEED_DIR="$MODEL_ZIP_SEED_DIR_DEFAULT"
FORCE_CPU="${ALIGNN_FORCE_CPU:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2;;
    --port) PORT="${2:-}"; shift 2;;
    --properties) PROPERTIES="${2:-}"; shift 2;;
    --conda-prefix) CONDA_PREFIX="${2:-}"; shift 2;;
    --env) CONDA_ENV="${2:-}"; shift 2;;
    --pidfile) PIDFILE="${2:-}"; shift 2;;
    --log-dir) LOG_DIR="${2:-}"; shift 2;;
    --model-zip-seed-dir) MODEL_ZIP_SEED_DIR="${2:-}"; shift 2;;
    --force-cpu) FORCE_CPU="${2:-}"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$PIDFILE")" "$LOG_DIR" "$MODEL_ZIP_SEED_DIR"

_pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

_read_pid() {
  if [[ -f "$PIDFILE" ]]; then
    tr -d ' \n\t' < "$PIDFILE" || true
  fi
}

_status() {
  local pid
  pid="$(_read_pid || true)"
  if [[ -z "${pid:-}" ]]; then
    echo "[alignn_ctl] status: not running (no pidfile) pidfile=$PIDFILE"
    return 1
  fi
  if _pid_alive "$pid"; then
    echo "[alignn_ctl] status: running pid=$pid pidfile=$PIDFILE"
    ps -p "$pid" -o pid,stat,etime,args | sed -n '1,2p' || true
    return 0
  fi
  echo "[alignn_ctl] status: stale pidfile pid=$pid (process not alive) pidfile=$PIDFILE"
  return 2
}

case "$ACTION" in
  start)
    if _status >/dev/null 2>&1; then
      echo "[alignn_ctl] already running; refusing to start a second instance."
      _status || true
      exit 0
    fi

    # Prefer running with the conda env's python directly (no need for interactive conda activate).
    PY="$CONDA_PREFIX/envs/$CONDA_ENV/bin/python"
    if [[ ! -x "$PY" ]]; then
      echo "[alignn_ctl] ERROR: python not found at $PY (set --conda-prefix/--env accordingly)" >&2
      exit 1
    fi

    TS="$(date +%Y%m%dT%H%M%S)"
    LOG_FILE="$LOG_DIR/alignn_zmq_server_multi.$TS.log"

    # Keep consistent with existing workaround used in your running process (weights_only default).
    # NOTE: Build the wrapper code without fragile mixed quoting.
    WRAPPER_CODE="$(cat <<PY
import runpy
import torch

_real = torch.load

def _load(*a, **k):
    if "weights_only" not in k:
        k["weights_only"] = False
    return _real(*a, **k)

torch.load = _load
runpy.run_path(r"""$SERVER_PY""", run_name="__main__")
PY
)"

    echo "[alignn_ctl] starting..."
    echo "[alignn_ctl] server=$SERVER_PY"
    echo "[alignn_ctl] bind=tcp://$HOST:$PORT"
    echo "[alignn_ctl] properties=$PROPERTIES"
    echo "[alignn_ctl] log=$LOG_FILE"
    echo "[alignn_ctl] model_zip_seed_dir=$MODEL_ZIP_SEED_DIR"
    echo "[alignn_ctl] ALIGNN_FORCE_CPU=$FORCE_CPU"

    # shellcheck disable=SC2086
    nohup env PYTHONUNBUFFERED=1 DGL_DISABLE_GRAPHBOLT=1 ALIGNN_FORCE_CPU="$FORCE_CPU" ALIGNN_MODEL_ZIP_SEED_DIR="$MODEL_ZIP_SEED_DIR" "$PY" -u -c "$WRAPPER_CODE" --host "$HOST" --port "$PORT" --properties $PROPERTIES \
      >>"$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PIDFILE"
    echo "[alignn_ctl] started pid=$PID pidfile=$PIDFILE"
    ;;

  stop)
    pid="$(_read_pid || true)"
    if [[ -z "${pid:-}" ]]; then
      echo "[alignn_ctl] stop: no pidfile at $PIDFILE"
      exit 1
    fi
    if _pid_alive "$pid"; then
      echo "[alignn_ctl] stopping pid=$pid"
      kill "$pid"
      exit 0
    fi
    echo "[alignn_ctl] stop: pidfile exists but process not alive pid=$pid"
    exit 2
    ;;

  pause)
    pid="$(_read_pid || true)"
    if [[ -z "${pid:-}" ]]; then
      echo "[alignn_ctl] pause: no pidfile at $PIDFILE"
      exit 1
    fi
    if _pid_alive "$pid"; then
      echo "[alignn_ctl] pausing (SIGSTOP) pid=$pid"
      kill -STOP "$pid"
      exit 0
    fi
    echo "[alignn_ctl] pause: pidfile exists but process not alive pid=$pid"
    exit 2
    ;;

  resume)
    pid="$(_read_pid || true)"
    if [[ -z "${pid:-}" ]]; then
      echo "[alignn_ctl] resume: no pidfile at $PIDFILE"
      exit 1
    fi
    if _pid_alive "$pid"; then
      echo "[alignn_ctl] resuming (SIGCONT) pid=$pid"
      kill -CONT "$pid"
      exit 0
    fi
    echo "[alignn_ctl] resume: pidfile exists but process not alive pid=$pid"
    exit 2
    ;;

  status)
    _status
    ;;

  restart)
    "$0" stop --pidfile "$PIDFILE" || true
    sleep 0.5
    "$0" start --host "$HOST" --port "$PORT" --properties "$PROPERTIES" --conda-prefix "$CONDA_PREFIX" --env "$CONDA_ENV" --pidfile "$PIDFILE" --log-dir "$LOG_DIR" --model-zip-seed-dir "$MODEL_ZIP_SEED_DIR"
    ;;

  *)
    echo "Usage: $0 {start|stop|pause|resume|status|restart} [args...]" >&2
    exit 2
    ;;
esac
