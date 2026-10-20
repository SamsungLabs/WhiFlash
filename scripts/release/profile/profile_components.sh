#!/usr/bin/env bash
set -euo pipefail

# WhiFlash component-level latency profiling.
#
# Measures (sweeping over base_kv_lens):
#   - _eagle3_round absolute cost (steady state)
#   - _dflash_round absolute cost across block sizes
#   - _flush_eagle3_backlog absolute cost across backlog sizes
#   - target prefill (optional, --profile-prefill)
#
# Informs:
#   - Switching overhead as % of an EAGLE3 round (is fusion/CUDA-graph worth it?)
#   - DFlash vs EAGLE3 per-round cost ratio (utility-oracle weights)
#   - kv_len-axis behaviour of all components (the paper story)
#
# Usage:
#   bash scripts/release/profile/profile_components.sh
#   bash scripts/release/profile/profile_components.sh --model llama
#   bash scripts/release/profile/profile_components.sh --measure-e2d-switch --profile-prefill --ablate-logits-to-keep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

OUTPUT_DIR="results/components"
MODEL_FILTER=""
NUM_STEPS=8
TOP_K=10
NUM_DRAFT_TOKENS=60
BLOCK_SIZES="8 16 32"
BACKLOG_SIZES="1 16 32 64 128 256 512 1024"
BASE_KV_LENS="512 1024 2048 4096 8192 16384"
# EAGLE3 hyperparam sweep — each entry is "depth:topk:ndt".
# Default covers a small/fast preset and the paper's high-acceptance preset.
# EAGLE_NUM_STEPS=(4 5 6 7 8), EAGLE_TOP_K=(4 8 12), EAGLE_NUM_DRAFT_TOKENS=(16 32 64 128)
EAGLE_CONFIGS="4:4:16 4:12:128 5:12:64 5:12:128 6:8:32 6:12:64 6:12:128 7:12:64 7:12:128 8:10:60 8:12:128"
N_REPEATS=30
# Measurement-extending flags default ON — gives the full paper-ready profile in one run.
# Pass --no-* to opt out per measurement.
MEASURE_E2D_SWITCH=1
ABLATE_LOGITS_TO_KEEP=1
PROFILE_PREFILL=1
# Hidden-state hooks default ON (only opt out for sanity-check vs end-to-end benchmark.py).
NO_HIDDEN_STATE_HOOKS=0
RUN_REPORT=1                     # by default, aggregate JSONs into CSV / md / plots after the sweep
REPORT_DIR=""                    # default: $OUTPUT_DIR/report
# Default sweep: 'auto' picks FA2 if installed, then 'sdpa' for the comparison.
ATTN_IMPLS=("sdpa")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)             OUTPUT_DIR="$2"; shift 2 ;;
        --model)                  MODEL_FILTER="$2"; shift 2 ;;
        --num-steps)              NUM_STEPS="$2"; shift 2 ;;
        --top-k)                  TOP_K="$2"; shift 2 ;;
        --num-draft-tokens)       NUM_DRAFT_TOKENS="$2"; shift 2 ;;
        --block-sizes)            BLOCK_SIZES="$2"; shift 2 ;;
        --backlog-sizes)          BACKLOG_SIZES="$2"; shift 2 ;;
        --base-kv-lens)           BASE_KV_LENS="$2"; shift 2 ;;
        --eagle-configs)          EAGLE_CONFIGS="$2"; shift 2 ;;
        --n-repeats)              N_REPEATS="$2"; shift 2 ;;
        --measure-e2d-switch)        MEASURE_E2D_SWITCH=1; shift ;;
        --no-measure-e2d-switch)     MEASURE_E2D_SWITCH=0; shift ;;
        --ablate-logits-to-keep)     ABLATE_LOGITS_TO_KEEP=1; shift ;;
        --no-ablate-logits-to-keep)  ABLATE_LOGITS_TO_KEEP=0; shift ;;
        --profile-prefill)           PROFILE_PREFILL=1; shift ;;
        --no-profile-prefill)        PROFILE_PREFILL=0; shift ;;
        --no-hidden-state-hooks)     NO_HIDDEN_STATE_HOOKS=1; shift ;;
        --no-report)                 RUN_REPORT=0; shift ;;
        --report-dir)             REPORT_DIR="$2"; shift 2 ;;
        --attn-impl)              ATTN_IMPLS=("$2"); shift 2 ;;
        --attn-impls)             IFS=' ' read -ra ATTN_IMPLS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Format: target : dflash_draft : eagle3_draft : ALGO : short_name
declare -a MODELS_2DRAFTS_ALGO_NAMES=(
    "Qwen/Qwen3-4B:z-lab/Qwen3-4B-DFlash-b16:WhiFlash/Qwen3-4B-EAGLE3:WHIFLASH:qwen3-4b"
    "Qwen/Qwen3-8B:z-lab/Qwen3-8B-DFlash-b16:WhiFlash/Qwen3-8B-EAGLE3:WHIFLASH:qwen3-8b"
    "meta-llama/Llama-3.1-8B-Instruct:z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat:yuhuili/EAGLE3-LLaMA3.1-Instruct-8B:WHIFLASH:llama3.1-8b"
)

echo "=== WhiFlash Component Profile ==="
echo "  EAGLE3 params:           step=$NUM_STEPS, top_k=$TOP_K, num_draft_tokens=$NUM_DRAFT_TOKENS"
echo "  DFlash block sizes:      $BLOCK_SIZES"
echo "  Backlog sizes:           $BACKLOG_SIZES"
echo "  Base kv_lens:            $BASE_KV_LENS"
echo "  EAGLE3 configs:          $EAGLE_CONFIGS"
echo "  Repeats/config:          $N_REPEATS"
echo "  E→D switch measure:      $MEASURE_E2D_SWITCH"
echo "  Ablate logits-to-keep:   $ABLATE_LOGITS_TO_KEEP"
echo "  Profile prefill:         $PROFILE_PREFILL"
echo "  Hidden-state hooks off:  $NO_HIDDEN_STATE_HOOKS"
echo "  Attn impls:              ${ATTN_IMPLS[*]}"
echo "  Output:                  $OUTPUT_DIR/"
echo ""

for ATTN_IMPL in "${ATTN_IMPLS[@]}"; do
    for config_str in "${MODELS_2DRAFTS_ALGO_NAMES[@]}"; do
        IFS=':' read -r TARGET DFLASH_DRAFT EAGLE3_DRAFT _ALGO MODEL_NAME <<< "$config_str"

        if [[ -n "$MODEL_FILTER" ]]; then
            case "$MODEL_FILTER" in
                llama*) [[ "$MODEL_NAME" != llama* ]] && continue ;;
                qwen*)  [[ "$MODEL_NAME" != qwen* ]] && continue ;;
                *)      [[ "$MODEL_NAME" != *"$MODEL_FILTER"* ]] && continue ;;
            esac
        fi

        echo "============================================"
        echo "Components: $MODEL_NAME (attn=$ATTN_IMPL)"
        echo "============================================"

        RUN_DIR="$OUTPUT_DIR/$MODEL_NAME/$ATTN_IMPL"
        mkdir -p "$RUN_DIR"

        EXTRA_FLAGS=()
        [[ "$MEASURE_E2D_SWITCH" == "1" ]] && EXTRA_FLAGS+=(--measure-e2d-switch)
        [[ "$ABLATE_LOGITS_TO_KEEP" == "1" ]] && EXTRA_FLAGS+=(--ablate-logits-to-keep)
        [[ "$PROFILE_PREFILL" == "1" ]] && EXTRA_FLAGS+=(--profile-prefill)
        [[ "$NO_HIDDEN_STATE_HOOKS" == "1" ]] && EXTRA_FLAGS+=(--no-hidden-state-hooks)

        uv run "$PROJECT_ROOT/analysis/profile_components.py" \
            --model-name-or-path "$TARGET" \
            --whiflash-eagle-draft "$EAGLE3_DRAFT" \
            --draft-name-or-path "$DFLASH_DRAFT" \
            --eagle-num-steps "$NUM_STEPS" \
            --eagle-topk "$TOP_K" \
            --eagle-num-draft-tokens "$NUM_DRAFT_TOKENS" \
            --block-sizes $BLOCK_SIZES \
            --backlog-sizes $BACKLOG_SIZES \
            --base-kv-lens $BASE_KV_LENS \
            --eagle-configs $EAGLE_CONFIGS \
            --n-repeats "$N_REPEATS" \
            --attn-impl "$ATTN_IMPL" \
            --output-dir "$RUN_DIR" \
            --model-name "$MODEL_NAME" \
            "${EXTRA_FLAGS[@]}" \
            || echo "WARNING: profiling failed for $MODEL_NAME (attn=$ATTN_IMPL)"
    done
done

echo ""
echo "=== Component Profile Complete ==="
echo "Results: $OUTPUT_DIR/"

# Aggregate the per-run components.json files into CSV / markdown / plots.
# Skip with --no-report; fall through quietly if pandas/matplotlib aren't
# installed so the data-collection step is still considered successful.
if [[ "$RUN_REPORT" == "1" ]]; then
    REPORT_DIR="${REPORT_DIR:-$OUTPUT_DIR/report}"
    echo ""
    echo "=== Aggregating into report at $REPORT_DIR ==="
    uv run "$PROJECT_ROOT/analysis/report_components.py" \
        --root "$OUTPUT_DIR" \
        --out "$REPORT_DIR" \
        || echo "WARNING: report aggregation failed (pandas/matplotlib installed?)"
    echo "Report: $REPORT_DIR/"
fi
