#!/bin/bash
# Calibrate WhiFlash-Neural thresholds per task category, as in the paper: sweep the threshold grid on
# CALIBRATION_SAMPLES random samples per task (--split calibration), disjoint from benchmark.sh's eval split.
#
# Usage: [CUDA_VISIBLE_DEVICES=<GPUs, default all>] bash scripts/release/benchmark/calibrate_thresholds.sh
# Then:  uv run python analysis/calibration_select_thresholds.py <LOG_ROOT>/sdpa
#        and paste the printed table into benchmark.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WHIFLASH_SPLIT=calibration WHIFLASH_THRESHOLD_MODE=sweep
export WHIFLASH_LOG_ROOT="${WHIFLASH_LOG_ROOT:-logs/calibration_$(date +%d%b%Y)}"
source "$SCRIPT_DIR/benchmark.sh"


echo "Calibration sweep (${WHIFLASH_ROUTER_THRESHOLDS[*]}) on ${CALIBRATION_SAMPLES} samples per task -> ${LOG_ROOT}"
run_whiflash_benchmark
run_queue
print_summary
echo "Select: uv run python analysis/calibration_select_thresholds.py ${LOG_ROOT}/${ATTN_IMPL}"
