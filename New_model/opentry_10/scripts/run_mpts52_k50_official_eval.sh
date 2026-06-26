#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/xsw/autodlmini"
OP="$ROOT/model/New_model/opentry_10"
PY="$ROOT/miniforge3/envs/crystallm_env/bin/python"
SYSTEM_ID="${1:-mpts52_k50_rf_seed1_margin_route}"

GEN_TAR="$OP/generations/${SYSTEM_ID}_test_k20/tars/generated_data_atomtype_gt_sg.tar.gz"
TRUE_TAR="$OP/generations/mpts52_official_test/tars/true.tar.gz"
OUT="$OP/metrics/official_test"
ROWS_DIR="$OUT/${SYSTEM_ID}_rows_ge7_subset"

if [[ ! -f "$GEN_TAR" ]]; then
  echo "missing generated tar: $GEN_TAR" >&2
  exit 2
fi
if [[ ! -f "$TRUE_TAR" ]]; then
  echo "missing true tar: $TRUE_TAR" >&2
  exit 2
fi

mkdir -p "$OUT"

run_metric() {
  local k="$1"
  local gen_tar="$2"
  local true_tar="$3"
  local raw="$4"
  local workers="$5"
  local diag="$6"
  local diag_dir="$7"
  PYTHONDONTWRITEBYTECODE=1 "$PY" "$OP/scripts/benchmark_metrics_opentry10.py" \
    "$gen_tar" "$true_tar" \
    --num-gens "$k" \
    --max-sites 512 \
    --rmsd-timeout-seconds 20 \
    --workers "$workers" \
    --hard-timeout-seconds 90 \
    --unmatched-diagnostics "$diag" \
    --unmatched-diagnostics-dir "$diag_dir" \
    > "$raw" 2>&1
}

run_metric 1 "$GEN_TAR" "$TRUE_TAR" "$OUT/${SYSTEM_ID}_k1.raw.txt" 40 summary "$OUT/${SYSTEM_ID}_k1_unmatched"
run_metric 5 "$GEN_TAR" "$TRUE_TAR" "$OUT/${SYSTEM_ID}_k5.raw.txt" 40 summary "$OUT/${SYSTEM_ID}_k5_unmatched"
run_metric 20 "$GEN_TAR" "$TRUE_TAR" "$OUT/${SYSTEM_ID}_k20.raw.txt" 80 off "$OUT/${SYSTEM_ID}_k20_unmatched"

PYTHONDONTWRITEBYTECODE=1 "$PY" "$OP/scripts/build_rows_ge7_subset_tars.py" \
  --generated-tar "$GEN_TAR" \
  --true-tar "$TRUE_TAR" \
  --out-dir "$ROWS_DIR"

run_metric 1 "$ROWS_DIR/generated_rows_ge7.tar.gz" "$ROWS_DIR/true_rows_ge7.tar.gz" "$OUT/${SYSTEM_ID}_rows_ge7_k1.raw.txt" 40 off "$OUT/${SYSTEM_ID}_rows_ge7_k1_unmatched"
run_metric 5 "$ROWS_DIR/generated_rows_ge7.tar.gz" "$ROWS_DIR/true_rows_ge7.tar.gz" "$OUT/${SYSTEM_ID}_rows_ge7_k5.raw.txt" 40 off "$OUT/${SYSTEM_ID}_rows_ge7_k5_unmatched"
run_metric 20 "$ROWS_DIR/generated_rows_ge7.tar.gz" "$ROWS_DIR/true_rows_ge7.tar.gz" "$OUT/${SYSTEM_ID}_rows_ge7_k20.raw.txt" 80 off "$OUT/${SYSTEM_ID}_rows_ge7_k20_unmatched"

PYTHONDONTWRITEBYTECODE=1 "$PY" "$OP/scripts/summarize_mpts52_official_eval.py" \
  --system-id "$SYSTEM_ID" \
  --k1-raw "$OUT/${SYSTEM_ID}_k1.raw.txt" \
  --k5-raw "$OUT/${SYSTEM_ID}_k5.raw.txt" \
  --k20-raw "$OUT/${SYSTEM_ID}_k20.raw.txt" \
  --rows-ge7-k1-raw "$OUT/${SYSTEM_ID}_rows_ge7_k1.raw.txt" \
  --rows-ge7-k5-raw "$OUT/${SYSTEM_ID}_rows_ge7_k5.raw.txt" \
  --rows-ge7-k20-raw "$OUT/${SYSTEM_ID}_rows_ge7_k20.raw.txt" \
  --out-json "$OUT/${SYSTEM_ID}_summary.json" \
  --out-report "$OP/reports/${SYSTEM_ID}_official_test.md"
