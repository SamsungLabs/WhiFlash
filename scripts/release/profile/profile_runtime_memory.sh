#!/bin/bash

set -e

# ============================================================================
# Configuration (mirrors benchmark.sh)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

OUTPUT_DIR="${PROJECT_ROOT}/results/memory_profile"

mkdir -p "$OUTPUT_DIR"

# Local checkpoint root for EAGLE3 drafts and neural-router weights that are not
# hosted on the Hub. Override CKPT_ROOT to point at your own checkpoint directory.
CKPT_ROOT="${CKPT_ROOT:-${HOME}/checkpoints/whiflash}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

# ============================================================================
# Model definitions (from benchmark.sh)
# ============================================================================

# Each entry: MODEL_NAME:DFLASH_DRAFT:EAGLE3_DRAFT:TAG
MODEL_CONFIGS=(
    "Qwen/Qwen3-8B:z-lab/Qwen3-8B-DFlash-b16:WhiFlash/Qwen3-8B-EAGLE3:qwen3_8b"
    "Qwen/Qwen3-4B:z-lab/Qwen3-4B-DFlash-b16:WhiFlash/Qwen3-4B-EAGLE3:qwen3_4b"
)

# Neural router models per model type. A name with a '/' is a Hub repo id; a bare name is a
# local export at ${CKPT_ROOT}/neural_routers/<name>.safetensors.
NEURAL_ROUTERS_Qwen3_8B=("WhiFlash/Qwen3-8B-NeuralRouter-df-e3")
NEURAL_ROUTERS_Qwen3_4B=("WhiFlash/Qwen3-4B-NeuralRouter-df-e3")

# Token lengths to profile
MAX_NEW_TOKENS_LIST=(2048)


# ============================================================================
# Run memory profiles
# ============================================================================

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    IFS=':' read -r MODEL_NAME DFLASH_DRAFT EAGLE3_DRAFT MODEL_TAG <<< "$MODEL_CONFIG"

    # Select neural routers based on MODEL_NAME
    if [[ "$MODEL_NAME" == *"Qwen3-8B"* ]]; then
        NEURAL_ROUTERS=("${NEURAL_ROUTERS_Qwen3_8B[@]}")
    elif [[ "$MODEL_NAME" == *"Qwen3-4B"* ]]; then
        NEURAL_ROUTERS=("${NEURAL_ROUTERS_Qwen3_4B[@]}")
    else
        echo "WARNING: No neural routers defined for MODEL_NAME=$MODEL_NAME, skipping neural." >&2
        NEURAL_ROUTERS=()
    fi

    for MAX_NEW_TOKENS in "${MAX_NEW_TOKENS_LIST[@]}"; do

        # NOSPEC (target-only autoregressive baseline)
        echo "========================================================"
        echo "Memory Profile: NOSPEC (${MODEL_TAG}, max_new_tokens=${MAX_NEW_TOKENS})"
        echo "========================================================"
        uv run analysis/profile_memory.py \
            --speculative-algorithm NOSPEC \
            --model-name-or-path "$MODEL_NAME" \
            --output-prefix "${OUTPUT_DIR}/${MODEL_TAG}_nospec_${MAX_NEW_TOKENS}" \
            --max-new-tokens "$MAX_NEW_TOKENS"

        # DFlash
        echo "========================================================"
        echo "Memory Profile: DFLASH (${MODEL_TAG}, max_new_tokens=${MAX_NEW_TOKENS})"
        echo "========================================================"
        uv run analysis/profile_memory.py \
            --speculative-algorithm DFLASH \
            --model-name-or-path "$MODEL_NAME" \
            --draft-name-or-path "$DFLASH_DRAFT" \
            --output-prefix "${OUTPUT_DIR}/${MODEL_TAG}_dflash_${MAX_NEW_TOKENS}" \
            --max-new-tokens "$MAX_NEW_TOKENS"

        # EAGLE3
        echo "========================================================"
        echo "Memory Profile: EAGLE3 (${MODEL_TAG}, max_new_tokens=${MAX_NEW_TOKENS})"
        echo "========================================================"
        uv run analysis/profile_memory.py \
            --speculative-algorithm EAGLE3 \
            --model-name-or-path "$MODEL_NAME" \
            --draft-name-or-path "$EAGLE3_DRAFT" \
            --output-prefix "${OUTPUT_DIR}/${MODEL_TAG}_eagle3_${MAX_NEW_TOKENS}" \
            --max-new-tokens "$MAX_NEW_TOKENS"

        # WhiFlash-Entropy
        echo "========================================================"
        echo "Memory Profile: WHIFLASH-entropy (${MODEL_TAG}, max_new_tokens=${MAX_NEW_TOKENS})"
        echo "========================================================"
        uv run analysis/profile_memory.py \
            --speculative-algorithm WHIFLASH \
            --whiflash-router entropy \
            --model-name-or-path "$MODEL_NAME" \
            --draft-name-or-path "$DFLASH_DRAFT" \
            --whiflash-eagle-draft "$EAGLE3_DRAFT" \
            --output-prefix "${OUTPUT_DIR}/${MODEL_TAG}_whiflash_entropy_${MAX_NEW_TOKENS}" \
            --max-new-tokens "$MAX_NEW_TOKENS"

        # WhiFlash-Neural (one profile per router model)
        for WHIFLASH_ROUTER_MODEL in "${NEURAL_ROUTERS[@]}"; do
            if [[ "$WHIFLASH_ROUTER_MODEL" == */* ]]; then
                WHIFLASH_ROUTER_PATH="$WHIFLASH_ROUTER_MODEL"
            else
                WHIFLASH_ROUTER_PATH="${CKPT_ROOT}/neural_routers/${WHIFLASH_ROUTER_MODEL}.safetensors"
            fi
            ROUTER_TAG="${WHIFLASH_ROUTER_MODEL##*/}"   # keeps output file names flat for repo ids

            echo "========================================================"
            echo "Memory Profile: WHIFLASH-neural (${MODEL_TAG}, ${ROUTER_TAG}, max_new_tokens=${MAX_NEW_TOKENS})"
            echo "========================================================"
            uv run analysis/profile_memory.py \
                --speculative-algorithm WHIFLASH \
                --whiflash-router neural \
                --whiflash-neural-router-path "$WHIFLASH_ROUTER_PATH" \
                --model-name-or-path "$MODEL_NAME" \
                --draft-name-or-path "$DFLASH_DRAFT" \
                --whiflash-eagle-draft "$EAGLE3_DRAFT" \
                --output-prefix "${OUTPUT_DIR}/${MODEL_TAG}_whiflash_neural_${ROUTER_TAG}_${MAX_NEW_TOKENS}" \
                --max-new-tokens "$MAX_NEW_TOKENS"
        done

    done
done

echo "========================================================"
echo "Memory profiling complete. Results in: ${OUTPUT_DIR}/"
echo "Run ./analysis/read_memory_results.py to generate comparison table."
echo "========================================================"

echo "Memory profile summary for Qwen3-4B"
uv run analysis/read_memory_results.py "$OUTPUT_DIR"/qwen3_4b*

echo "Memory profile summary for Qwen3-8B"
uv run analysis/read_memory_results.py "$OUTPUT_DIR"/qwen3_8b*
