#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/xsw/autodlmini"
OP="$ROOT/model/New_model/opentry_10"
PROMPTS="$OP/generations/mpts52_test_gt_sg_k50/prompts/data_atomtype_gt_sg"
RAW="$OP/generations/mpts52_test_gt_sg_k50/cifs_raw/data_atomtype_gt_sg"
POST="$OP/generations/mpts52_test_gt_sg_k50/cifs_post/data_atomtype_gt_sg"
LOGDIR="$OP/logs/mpts52_k50_generation"
PY="$ROOT/miniforge3/envs/crystallm_env/bin/python"
GEN="$ROOT/model/scp_task/CrystaLLM/bin/generate_cifs_from_prompts_dir.py"
POSTPROCESS="$ROOT/model/scp_task/CrystaLLM/bin/postprocess.py"
MODEL="$ROOT/model/scp_task/CrystaLLM/crystallm_benchmarkmodel/cif_model_mpts_52_b"
EXPECTED_PER_BLOCK=80950

mkdir -p "$LOGDIR"

count_block() {
  local start="$1"
  local end="$2"
  "$PY" - "$RAW" "$start" "$end" <<'PY'
import re
import sys
from pathlib import Path

raw = Path(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
pat = re.compile(r"__(\d+)\.cif$")
count = 0
if raw.exists():
    for path in raw.glob("*.cif"):
        m = pat.search(path.name)
        if m and start <= int(m.group(1)) <= end:
            count += 1
print(count)
PY
}

wait_for_block() {
  local start="$1"
  local end="$2"
  local offset="$3"
  while true; do
    local count
    count="$(count_block "$start" "$end")"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) block ${start}-${end}: ${count}/${EXPECTED_PER_BLOCK}" | tee -a "$LOGDIR/watchdog.log"
    if [[ "$count" -ge "$EXPECTED_PER_BLOCK" ]]; then
      while pgrep -f "generate_cifs_from_prompts_dir.py.*sample-index-offset ${offset}" >/dev/null; do
        sleep 15
      done
      break
    fi
    sleep 60
  done
}

run_postprocess() {
  local tag="$1"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) postprocess ${tag}" | tee -a "$LOGDIR/watchdog.log"
  "$PY" "$POSTPROCESS" "$RAW" "$POST" --workers 80 --resume > "$LOGDIR/postprocess_${tag}.log" 2>&1
}

run_generation_block() {
  local offset="$1"
  local seed="$2"
  local temp="$3"
  local start_rank="$4"
  local end_rank="$5"
  local tag="r${start_rank}_${end_rank}"
  local existing
  existing="$(count_block "$start_rank" "$end_rank")"
  if [[ "$existing" -ge "$EXPECTED_PER_BLOCK" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) skip generation ${tag}; already complete" | tee -a "$LOGDIR/watchdog.log"
    return
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start generation ${tag}" | tee -a "$LOGDIR/watchdog.log"
  CUDA_VISIBLE_DEVICES=0,1 PYTHONDONTWRITEBYTECODE=1 "$PY" "$GEN" \
    --model-dir "$MODEL" \
    --prompts-dir "$PROMPTS" \
    --out-dir "$RAW" \
    --start-index 0 --num-prompts 8095 \
    --num-samples-per-prompt 10 --sample-index-offset "$offset" \
    --sample-seed-stride 100000 --seed "$seed" \
    --temperature "$temp" --top-k 10 --max-new-tokens 2048 \
    --device cuda:auto --dtype bfloat16 --workers 8 \
    --batch-samples --retry-missing-single-worker 2>&1 | tee "$LOGDIR/generate_${tag}.log"
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog start" | tee -a "$LOGDIR/watchdog.log"
wait_for_block 21 30 20
run_postprocess "r21_30"
run_generation_block 30 8537 0.85 31 40
wait_for_block 31 40 30
run_postprocess "r31_40"
run_generation_block 40 10037 1.0 41 50
wait_for_block 41 50 40
run_postprocess "r41_50"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog complete" | tee -a "$LOGDIR/watchdog.log"
