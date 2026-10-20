#!/bin/bash

set -e

MODEL_PATH="$1"   # arg 1: target model HF path/name (e.g. Qwen/Qwen3-8B)
DATASET="$2"      # arg 2: dataset key for load_and_process_dataset (e.g. dolci, dolci-test)
shift 2           # drop the two required positionals; whatever remains is the drafter list
DRAFTERS=("$@")   # args 3+: drafters to run together (>=2 from scratch, or 1 with BASELINE_DIR)
[[ ${#DRAFTERS[@]} -eq 0 ]] && DRAFTERS=(eagle3 dflash)   # default: the EAGLE3-vs-DFlash pair

MODEL_NAME=$(basename "$MODEL_PATH")

BASE_PATH="${WHIFLASH_CKPT_ROOT:-$HOME/checkpoints/whiflash}"

# Drafter pairs per target. The EAGLE-3 drafters we trained are published under the WhiFlash
# org; point EAGLE_MODEL at a local directory instead to train a router for your own drafter.
case "$MODEL_NAME" in
    Qwen3-8B)
        DFLASH_MODEL="z-lab/Qwen3-8B-DFlash-b16"
        EAGLE_MODEL="WhiFlash/Qwen3-8B-EAGLE3"
        ;;
    Qwen3-4B)
        DFLASH_MODEL="z-lab/Qwen3-4B-DFlash-b16"
        EAGLE_MODEL="WhiFlash/Qwen3-4B-EAGLE3"
        ;;
    Llama-3.1-8B-Instruct)
        DFLASH_MODEL="z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
        EAGLE_MODEL="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        ;;
    *)
        echo "Usage: $0 <model_path> <dataset> [drafter ...]"
        echo "Supported models: Qwen3-8B, Qwen3-4B, Llama-3.1-8B-Instruct"
        exit 1
        ;;
esac

TREE_BUDGETS="${TREE_BUDGETS:-32 64 128 256 512}"  # ddtree only

TAG=$(IFS=-; echo "${DRAFTERS[*]}")
if [[ "$TAG" == "eagle3-dflash" ]]; then
    OUTPUT_DIR="${BASE_PATH}/training_data/${MODEL_NAME}_${DATASET}"
else
    OUTPUT_DIR="${BASE_PATH}/training_data/${MODEL_NAME}_${DATASET}_${TAG}"
fi

# Incremental: set BASELINE_DIR to an existing dataset dir to ADD these drafters to it (gated on
# its stored output_ids; writes AL-only columns). Unset = build a self-contained dataset.
BASELINE_ARGS=()
[[ -n "$BASELINE_DIR" ]] && BASELINE_ARGS=(--baseline-dir "$BASELINE_DIR")

export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

uv run torchrun \
    --nproc_per_node gpu \
    -m neural_router.generate_training_data \
    --target-model-path "$MODEL_PATH" \
    --dflash-model-path "$DFLASH_MODEL" \
    --eagle-model-path "$EAGLE_MODEL" \
    "${BASELINE_ARGS[@]}" \
    --dataset "$DATASET" \
    --drafters "${DRAFTERS[@]}" \
    --tree-budgets $TREE_BUDGETS \
    --output-dir "$OUTPUT_DIR"
