#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/xsw/autodlmini"
OP="$ROOT/model/New_model/opentry_10"
RAW="$OP/generations/mpts52_test_gt_sg_k50/cifs_raw/data_atomtype_gt_sg"
POST="$OP/generations/mpts52_test_gt_sg_k50/cifs_post/data_atomtype_gt_sg"
LOGDIR="$OP/logs/mpts52_k30_generation"
PY="$ROOT/miniforge3/envs/crystallm_env/bin/python"
POSTPROCESS="$ROOT/model/scp_task/CrystaLLM/bin/postprocess.py"
EXPECTED=80950

mkdir -p "$LOGDIR"

count_block() {
  "$PY" - "$RAW" <<'PY'
import re
import sys
from pathlib import Path

raw = Path(sys.argv[1])
pat = re.compile(r"__(\d+)\.cif$")
count = 0
if raw.exists():
    for path in raw.glob("*.cif"):
        m = pat.search(path.name)
        if m and 21 <= int(m.group(1)) <= 30:
            count += 1
print(count)
PY
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) k30 watcher start" | tee -a "$LOGDIR/watch.log"
while true; do
  count="$(count_block)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) block 21-30: ${count}/${EXPECTED}" | tee -a "$LOGDIR/watch.log"
  if [[ "$count" -ge "$EXPECTED" ]]; then
    while pgrep -f "generate_cifs_from_prompts_dir.py.*sample-index-offset 20" >/dev/null; do
      sleep 15
    done
    break
  fi
  sleep 60
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) final postprocess r21_30" | tee -a "$LOGDIR/watch.log"
"$PY" "$POSTPROCESS" "$RAW" "$POST" --workers 80 --resume > "$LOGDIR/postprocess_r21_30.log" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) k30 watcher complete" | tee -a "$LOGDIR/watch.log"
