#!/usr/bin/env bash
set -u

ROOT=/data/users/xsw/autodlmini
PY=${ROOT}/miniforge3/envs/crystallm_env/bin/python
SCRIPT=${ROOT}/model/New_model/opentry_14/scripts/run_exp0_exp1_alignment.py
LOG_DIR=${ROOT}/model/New_model/opentry_14/artifacts/exp1_predicted_skeleton_noise_pairs/logs
N_SHARDS=${N_SHARDS:-8}
TOP_K=${TOP_K:-20}

mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

pids=()
for idx in $(seq 0 $((N_SHARDS - 1))); do
  nice -n 5 "${PY}" -u "${SCRIPT}" \
    --skip-exp0 \
    --top-k "${TOP_K}" \
    --num-shards "${N_SHARDS}" \
    --shard-index "${idx}" \
    > "${LOG_DIR}/exp1_shard_${idx}_of_${N_SHARDS}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [ "${status}" -ne 0 ]; then
  echo "one or more shards failed; see ${LOG_DIR}" >&2
  exit "${status}"
fi

nice -n 5 "${PY}" -u "${SCRIPT}" \
  --top-k "${TOP_K}" \
  --aggregate-shards "${N_SHARDS}" \
  > "${LOG_DIR}/exp1_aggregate_${N_SHARDS}.log" 2>&1
