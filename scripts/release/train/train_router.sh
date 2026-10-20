#!/bin/bash

set -e

TARGET_MODEL_PATH=${1:-"Qwen/Qwen3-4B"}            # arg 1: target model HF path/name
DATASET_NAME=${2:-"dolci"}                         # arg 2: dataset key
DATA_SUFFIX=${3:-""}                               # arg 3: dataset dir suffix ("" -> <model>_<dataset>; else <model>_<dataset>_<suffix>)
LABEL_POS_COL=${4:-"eagle_acceptance_lengths"}     # arg 4: '+' column of the label (AL[pos] - AL[neg])
LABEL_NEG_COL=${5:-"dflash_acceptance_lengths"}    # arg 5: '-' column of the label
EXTRA_SUFFIXES=("${@:6}")  # args 6+ (everything after the 5 named args): AL-only dirs (incremental drafters) to stitch via --extra-train-dirs

BASE_PATH="${WHIFLASH_CKPT_ROOT:-$HOME/checkpoints/whiflash}"
DATASET_DIR="${BASE_PATH}/training_data"

TARGET_MODEL_NAME=$(basename "$TARGET_MODEL_PATH")
TRAIN_DATA_DIR="${DATASET_DIR}/${TARGET_MODEL_NAME}_${DATASET_NAME}${DATA_SUFFIX:+_$DATA_SUFFIX}"
TEST_DATA_DIR="${DATASET_DIR}/${TARGET_MODEL_NAME}_${DATASET_NAME}-test${DATA_SUFFIX:+_$DATA_SUFFIX}"

EXTRA_TRAIN_DIRS=()
EXTRA_TEST_DIRS=()
for S in "${EXTRA_SUFFIXES[@]}"; do
    EXTRA_TRAIN_DIRS+=("${DATASET_DIR}/${TARGET_MODEL_NAME}_${DATASET_NAME}_${S}")
    EXTRA_TEST_DIRS+=("${DATASET_DIR}/${TARGET_MODEL_NAME}_${DATASET_NAME}-test_${S}")
done

# Name the run by the routed columns (drop the shared _acceptance_lengths suffix) so distinct
# routers never collide and export can stay table-free.
POS_SHORT="${LABEL_POS_COL%_acceptance_lengths}"
NEG_SHORT="${LABEL_NEG_COL%_acceptance_lengths}"
if [[ "$POS_SHORT" == "eagle" && "$NEG_SHORT" == "dflash" ]]; then
    RUN_NAME="${TARGET_MODEL_NAME}_${DATASET_NAME}"  # back-compat name for the default router
else
    RUN_NAME="${TARGET_MODEL_NAME}_${DATASET_NAME}_${POS_SHORT}-vs-${NEG_SHORT}"
fi

OUTPUT_DIR="${BASE_PATH}/router_models/${RUN_NAME}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

EXTRA_TRAIN_ARGS=()
[[ ${#EXTRA_TRAIN_DIRS[@]} -gt 0 ]] && EXTRA_TRAIN_ARGS=(--extra-train-dirs "${EXTRA_TRAIN_DIRS[@]}")
EXTRA_TEST_ARGS=()
[[ ${#EXTRA_TEST_DIRS[@]} -gt 0 ]] && EXTRA_TEST_ARGS=(--extra-test-dirs "${EXTRA_TEST_DIRS[@]}")

uv run -m neural_router.train \
    --train-dir "$TRAIN_DATA_DIR" \
    --test-dir "$TEST_DATA_DIR" \
    --label-pos-col "$LABEL_POS_COL" \
    --label-neg-col "$LABEL_NEG_COL" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    "${EXTRA_TEST_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --num-train-examples 24_165_824 \
    --num-eval-examples 32_768 \
    --num-test-examples 1_048_576
