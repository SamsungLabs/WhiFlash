#!/bin/bash
# Benchmark NoSpec, DFlash, DDTree, EAGLE-3 and WhiFlash-Neural on every task below.
# One directory per run under ${LOG_ROOT}; finished runs are skipped, so re-running resumes.
# Tabulate: uv run python analysis/read_benchmark_results.py <LOG_ROOT>/sdpa --full-table --model-type qwen8b
#
# Env: WHIFLASH_LOG_ROOT, WHIFLASH_CKPT_ROOT, WHIFLASH_THRESHOLD_MODE (calibrated|sweep),
#      WHIFLASH_SPLIT (eval|all|calibration), WHIFLASH_CALIBRATION_SAMPLES (default 10; AIME: half),
#      WHIFLASH_MAX_SAMPLES (cap per task), CUDA_VISIBLE_DEVICES (default: all GPUs), ATTN_IMPL, HF_HUB_CACHE.

set -e

export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
BASE_DIR="${WHIFLASH_CKPT_ROOT:-$HOME/checkpoints/whiflash}"   # read for bare router names only
LOG_ROOT="${WHIFLASH_LOG_ROOT:-logs/$(date +%d%b%Y)_split-${WHIFLASH_SPLIT:-eval}}"   # calibrate_thresholds.sh overrides it

# Plain autoregressive decoding, the speedup reference (draft path unused).
NOSPEC_MODELS=(
  "Qwen/Qwen3-8B:z-lab/Qwen3-8B-DFlash-b16:NOSPEC:default"
)

DFLASH_MODELS=(
  "Qwen/Qwen3-8B:z-lab/Qwen3-8B-DFlash-b16:DFLASH:default"
)

# DDTREE reuses the DFlash draft model; only verification differs (tree instead of linear block).
DDTREE_MODELS=(
  "Qwen/Qwen3-8B:z-lab/Qwen3-8B-DFlash-b16:DDTREE:default"
)
# Tree-node budgets for the standalone DDTREE benchmark.
DDTREE_TREE_BUDGETS=(32 128)

EAGLE3_MODELS=(
  "Qwen/Qwen3-8B:WhiFlash/Qwen3-8B-EAGLE3:EAGLE3:default"
)

WHIFLASH_MODELS=(
  "Qwen/Qwen3-8B:z-lab/Qwen3-8B-DFlash-b16:WhiFlash/Qwen3-8B-EAGLE3:WHIFLASH:default"
)

WHIFLASH_ROUTER="neural"
# "<target model>:<router checkpoint>:<drafter pair>:<ddtree tree budget, or ->". The checkpoint is a Hub
# repo id or a bare name for a local export at ${BASE_DIR}/neural_routers/<name>.safetensors.
WHIFLASH_ROUTER_MODELS=(
    "Qwen/Qwen3-8B:WhiFlash/Qwen3-8B-NeuralRouter-df-e3:df-e3:-"
    "Qwen/Qwen3-8B:WhiFlash/Qwen3-8B-NeuralRouter-ddt-e3-b32:ddt-e3:32"
    "Qwen/Qwen3-8B:WhiFlash/Qwen3-8B-NeuralRouter-ddt-e3-b128:ddt-e3:128"
)

# sweep = every threshold below; calibrated = one per (model, category, pair) from WHIFLASH_THRESHOLDS.
WHIFLASH_THRESHOLD_MODE="${WHIFLASH_THRESHOLD_MODE:-calibrated}"
WHIFLASH_ROUTER_THRESHOLDS=(-2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0)

# Per-category thresholds ("<target model>:<category>:<drafter pair>") maximising throughput on the
# calibration split (H100), from calibrate_thresholds.sh + analysis/calibration_select_thresholds.py.
declare -A WHIFLASH_THRESHOLDS_TPS=(
    ["Qwen3-8B:chat:df-e3"]="0.5"
    ["Qwen3-8B:math:df-e3"]="1.0"
    ["Qwen3-8B:coding:df-e3"]="1.0"
    ["Qwen3-8B:agentic:df-e3"]="0.5"
    ["Qwen3-8B:chat:ddt-e3-b32"]="1.0"
    ["Qwen3-8B:math:ddt-e3-b32"]="0.5"
    ["Qwen3-8B:coding:ddt-e3-b32"]="1.5"
    ["Qwen3-8B:agentic:ddt-e3-b32"]="1.5"
    ["Qwen3-8B:chat:ddt-e3-b128"]="1.0"
    ["Qwen3-8B:math:ddt-e3-b128"]="1.0"
    ["Qwen3-8B:coding:ddt-e3-b128"]="1.0"
    ["Qwen3-8B:agentic:ddt-e3-b128"]="0.5"
    ["Qwen3-4B:chat:df-e3"]="1.0"
    ["Qwen3-4B:math:df-e3"]="0.5"
    ["Qwen3-4B:coding:df-e3"]="0.5"
    ["Qwen3-4B:agentic:df-e3"]="0.0"
    ["Qwen3-4B:chat:ddt-e3-b32"]="0.5"
    ["Qwen3-4B:math:ddt-e3-b32"]="1.5"
    ["Qwen3-4B:coding:ddt-e3-b32"]="0.5"
    ["Qwen3-4B:agentic:ddt-e3-b32"]="1.0"
    ["Qwen3-4B:chat:ddt-e3-b128"]="1.0"
    ["Qwen3-4B:math:ddt-e3-b128"]="2.0"
    ["Qwen3-4B:coding:ddt-e3-b128"]="0.5"
    ["Qwen3-4B:agentic:ddt-e3-b128"]="1.0"
)
declare -n WHIFLASH_THRESHOLDS=WHIFLASH_THRESHOLDS_TPS

# Thresholds are calibrated on CALIBRATION_SAMPLES random (fixed-seed) samples per task; every method is
# evaluated on the rest (--split eval), so the two never overlap. WHIFLASH_SPLIT=all uses all samples.
WHIFLASH_SPLIT="${WHIFLASH_SPLIT:-eval}"
CALIBRATION_SAMPLES="${WHIFLASH_CALIBRATION_SAMPLES:-10}"
AIME_CALIBRATION_SAMPLES=$(( CALIBRATION_SAMPLES > 1 ? CALIBRATION_SAMPLES / 2 : 1 ))   # 30-sample tasks: half
declare -A CALIBRATION_SAMPLES_OF=([aime24]=$AIME_CALIBRATION_SAMPLES [aime25]=$AIME_CALIBRATION_SAMPLES)
MAX_SAMPLES_CAP="${WHIFLASH_MAX_SAMPLES:-}"   # caps every task's sample count, for quick tests
if [[ -n "$MAX_SAMPLES_CAP" ]] && (( CALIBRATION_SAMPLES >= MAX_SAMPLES_CAP )); then
  echo "WHIFLASH_CALIBRATION_SAMPLES ($CALIBRATION_SAMPLES) must be below WHIFLASH_MAX_SAMPLES ($MAX_SAMPLES_CAP)." >&2
  exit 1
fi

TASKS=(
  "mt-bench:80"
  "alpaca:128"
  "ultrachat-test:80"
  "wildchat:80"
  "gsm8k:128"
  "math500:128"
  "aime24:30"
  "aime25:30"
  "humaneval:164"
  "mbpp:128"
  "livecodebench:128"
  "swe-bench:128"
  "tau-bench:162"
  "agentbench:103"
  "agent-instruct:80"
)

# Threshold category of each task.
declare -A TASK_CATEGORY=(
  [mt-bench]=chat  [alpaca]=chat    [ultrachat-test]=chat  [wildchat]=chat
  [gsm8k]=math     [math500]=math   [aime24]=math          [aime25]=math
  [humaneval]=coding  [mbpp]=coding  [livecodebench]=coding
  [swe-bench]=agentic  [tau-bench]=agentic  [agentbench]=agentic  [agent-instruct]=agentic
)

ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# Hub ids keep their org (z-lab/...); local paths are reduced to their last component.
tag_of() { if [[ "$1" == /* ]]; then basename "$1"; else echo "$1"; fi; }

SKIPPED=0
STATUS="$(mktemp)"   # one "ok|failed <seconds> <output dir>" line per finished run
START_EPOCH="$(date +%s)"
JOBS=()      # queued runs, each "<quoted output dir> <quoted command...>" (see run_queue)
JOB_OUT=()   # output dir of each queued run

# One job per GPU at a time, on every GPU (from nvidia-smi) unless CUDA_VISIBLE_DEVICES lists them.
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -E "^[0-9]+$" | paste -sd,)}"
NUM_SLOTS=$(( ${#GPUS[@]} > 1 ? ${#GPUS[@]} : 1 ))
declare -A GPU_PID=()

# _tqdm_time <seconds>: MM:SS, or H:MM:SS from one hour on.
_tqdm_time() {
  local s=$1
  if (( s >= 3600 )); then printf '%d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60))
  else printf '%02d:%02d' $((s/60)) $((s%60)); fi
}

# progress <job number> <status> <output dir>: one line per finished run, elapsed<remaining tqdm style.
# remaining = mean run time so far x unfinished runs / GPUs. Rough: runs differ a lot in length.
progress() {
  local idx=$1 status=$2 out=$3 tp="" finished failed secs eta prefill decode
  read -r finished failed secs < <(awk '{ n++; s += $2; f += ($1 == "failed") } END { print n + 0, f + 0, s + 0 }' "$STATUS")
  eta=$(( secs * (TOTAL - finished) / finished / NUM_SLOTS ))
  if [[ "$status" == ok ]]; then
    prefill=$(grep -oP '\[Prefill\] Average Throughput: \K[0-9.]+' "$out/transformers.log" || true)
    decode=$(grep -oP '\[Decoding\] Average Throughput: \K[0-9.]+' "$out/transformers.log" || true)
    tp=" (prefill ${prefill:-?} tok/s, decode ${decode:-?} tok/s)"
  fi
  echo "${RUN_GPU:+[GPU $RUN_GPU] }job $idx/$TOTAL status=$status $out$tp | finished $finished/$TOTAL, failed $failed, unfinished $((TOTAL - finished)) | $(_tqdm_time $(( $(date +%s) - START_EPOCH )))<$(_tqdm_time "$eta")"
}

# run_one <job number> <output dir> <command...>: runs one benchmark into <output dir>, records the outcome, reports progress.
run_one() {
  local idx=$1 out=$2 start status=ok; shift 2
  start=$(date +%s)
  "$@" > "$out/transformers.log" 2>&1 || status=failed
  echo "$status $(( $(date +%s) - start )) $out" >> "$STATUS"
  progress "$idx" "$status" "$out"
}

# run_benchmark <output dir> <task> <max samples> <benchmark.py args...>: queues one run, unless already done.
# Nothing runs until run_queue, so every progress line knows the total.
run_benchmark() {
  local out="$1" task="$2" n="$3"; shift 3
  [[ -z "$MAX_SAMPLES_CAP" ]] || (( n <= MAX_SAMPLES_CAP )) || n="$MAX_SAMPLES_CAP"
  if [[ -s "$out/transformers.jsonl" ]] && grep -qs "Wrote .* records to" "$out/transformers.log"; then
    echo "SKIP (done): $out"; SKIPPED=$((SKIPPED + 1)); return 0
  fi
  local split=()
  [[ "$WHIFLASH_SPLIT" == all ]] || \
    split=(--split "$WHIFLASH_SPLIT" --calibration-samples "${CALIBRATION_SAMPLES_OF[$task]:-$CALIBRATION_SAMPLES}")
  mkdir -p "$out"
  local cmd=(uv run benchmark.py --dataset "$task" --max-samples "$n" "${split[@]}" --attn-impl "$ATTN_IMPL" "$@"
             --max-new-tokens 2048 --temperature 0.0 --skip-nospec --output-dir "$out")
  JOB_OUT+=("$out")
  JOBS+=("$(printf '%q ' "$out" "${cmd[@]}")")
}

# start_on_free_gpu <job index>: starts the job on the next GPU that is free, waiting until one is.
start_on_free_gpu() {
  local i=$1 gpu pid
  while true; do
    for gpu in "${GPUS[@]}"; do
      pid="${GPU_PID[$gpu]:-}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        [[ -n "$pid" ]] || sleep 10   # stagger first launches (shared HF cache downloads)
        echo "RUN $((i+1))/$TOTAL [GPU $gpu]: ${JOB_OUT[$i]}"
        ( export CUDA_VISIBLE_DEVICES="$gpu" RUN_GPU="$gpu"; eval "run_one $((i+1)) ${JOBS[$i]}" ) &
        GPU_PID[$gpu]=$!
        return 0
      fi
    done
    wait -n 2>/dev/null || sleep 1   # until any job finishes
  done
}

# run_queue: runs every queued job, one per GPU at a time (one at a time with a single GPU).
run_queue() {
  local i
  TOTAL=${#JOBS[@]}
  echo "Queued ${TOTAL} runs (${SKIPPED} already done) -> ${LOG_ROOT}"
  for ((i = 0; i < TOTAL; i++)); do
    if (( ${#GPUS[@]} <= 1 )); then
      echo "RUN $((i+1))/$TOTAL: ${JOB_OUT[$i]}"
      eval "run_one $((i+1)) ${JOBS[$i]}"
    else
      start_on_free_gpu "$i"
    fi
  done
}

# Standalone methods: once per task (DDTree: once per task and tree budget).
run_drafter_benchmarks() {
  local spec task budget out
  for spec in "${NOSPEC_MODELS[@]}" "${DFLASH_MODELS[@]}" "${DDTREE_MODELS[@]}" "${EAGLE3_MODELS[@]}"; do
    IFS=':' read -r MODEL_NAME DRAFT_NAME ALGO SYS_PROMPT <<< "$spec"
    local budgets=(-)
    if [[ "$ALGO" == DDTREE ]]; then budgets=("${DDTREE_TREE_BUDGETS[@]}"); fi
    for task in "${TASKS[@]}"; do
      IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"
      for budget in "${budgets[@]}"; do
        out="${LOG_ROOT}/${ATTN_IMPL}/$(tag_of "$DRAFT_NAME")/${ALGO}-sysprompt-${SYS_PROMPT}/${DATASET_NAME}"
        local extra=()
        if [[ "$budget" != - ]]; then out+="/budget_${budget}"; extra=(--ddtree-tree-budget "$budget"); fi
        run_benchmark "$out" "$DATASET_NAME" "$MAX_SAMPLES" \
          --model-name-or-path "$MODEL_NAME" --draft-name-or-path "$DRAFT_NAME" \
          --speculative-algorithm "$ALGO" --system-prompt "$SYS_PROMPT" "${extra[@]}"
      done
    done
  done
}

# select_thresholds <model> <dataset> <pair tag>: sets THRESHOLDS to the sweep grid or the calibrated value
# (a missing calibrated value is fatal).
select_thresholds() {
  case "$WHIFLASH_THRESHOLD_MODE" in
    sweep) THRESHOLDS=("${WHIFLASH_ROUTER_THRESHOLDS[@]}") ;;
    calibrated)
      local key="$1:${TASK_CATEGORY[$2]:-?}:$3"
      if [[ -z "${WHIFLASH_THRESHOLDS[$key]:-}" ]]; then
        echo "No calibrated threshold for '$key'; add it to WHIFLASH_THRESHOLDS (and TASK_CATEGORY)," \
             "or use WHIFLASH_THRESHOLD_MODE=sweep." >&2
        exit 1
      fi
      THRESHOLDS=("${WHIFLASH_THRESHOLDS[$key]}") ;;
    *) echo "Unknown WHIFLASH_THRESHOLD_MODE '$WHIFLASH_THRESHOLD_MODE' (expected sweep or calibrated)." >&2
       exit 1 ;;
  esac
}

run_whiflash_benchmark() {
  local spec router_spec task thr
  for spec in "${WHIFLASH_MODELS[@]}"; do
    IFS=':' read -r MODEL_NAME DRAFT_NAME EAGLE_DRAFT ALGO SYS_PROMPT <<< "$spec"
    for router_spec in "${WHIFLASH_ROUTER_MODELS[@]}"; do
      IFS=':' read -r ROUTER_TARGET ROUTER PAIR BUDGET <<< "$router_spec"
      [[ "$ROUTER_TARGET" == "$MODEL_NAME" ]] || continue
      # A name with a '/' is a Hub repo id; a bare name is a local export (export_router_models.sh).
      local router_path="$ROUTER"
      [[ "$ROUTER" == */* ]] || router_path="${BASE_DIR}/neural_routers/${ROUTER}.safetensors"
      # '-' marks a pair without DDTree, so no budget is passed.
      local budget_args=() budget_tag="" pair_tag="$PAIR"
      if [[ "$BUDGET" != - ]]; then
        budget_args=(--ddtree-tree-budget "$BUDGET"); budget_tag="_b${BUDGET}"; pair_tag+="-b${BUDGET}"
      fi
      for task in "${TASKS[@]}"; do
        IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"
        select_thresholds "$(basename "$MODEL_NAME")" "$DATASET_NAME" "$pair_tag"
        for thr in "${THRESHOLDS[@]}"; do
          run_benchmark "${LOG_ROOT}/${ATTN_IMPL}/$(tag_of "$DRAFT_NAME")/eagle-$(basename "$EAGLE_DRAFT")/${ALGO}-sysprompt-${SYS_PROMPT}/${DATASET_NAME}/neural_${PAIR}${budget_tag}_${ROUTER##*/}_${thr}" \
            "$DATASET_NAME" "$MAX_SAMPLES" \
            --model-name-or-path "$MODEL_NAME" --draft-name-or-path "$DRAFT_NAME" \
            --speculative-algorithm "$ALGO" --system-prompt "$SYS_PROMPT" \
            --whiflash-eagle-draft "$EAGLE_DRAFT" --whiflash-drafters "$PAIR" "${budget_args[@]}" \
            --whiflash-router "$WHIFLASH_ROUTER" --whiflash-neural-router-path "$router_path" \
            --whiflash-neural-router-threshold "$thr"
        done
      done
    done
  done
}

# Prints the outcome; non-zero if any run failed.
print_summary() {
  wait   # for parallel runs
  echo "Finished: $(grep -c '^ok ' "$STATUS") run, ${SKIPPED} already done, $(grep -c '^failed ' "$STATUS") failed." \
       "Results: ${LOG_ROOT}"
  grep '^failed ' "$STATUS" | sed 's|^failed [0-9]* \(.*\)|  FAILED: \1/transformers.log|'
  echo "Tabulate: uv run python analysis/read_benchmark_results.py ${LOG_ROOT}/${ATTN_IMPL} --full-table --model-type qwen8b"
  ! grep -q '^failed ' "$STATUS"
}

# Guarded so other scripts can source this file without starting a run.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  run_drafter_benchmarks
  run_whiflash_benchmark
  run_queue
  print_summary
fi
