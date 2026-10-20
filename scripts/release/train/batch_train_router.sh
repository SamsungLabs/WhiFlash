#!/bin/bash
# Train a batch of WhiFlash-Neural routers over {targets} x {datasets} x {router pairs} from a
# unified dataset (generate_training_data.sh --drafters). All pairs are read from one dataset dir
# (DATA_SUFFIX) by selecting two columns — no stitching.
#
# Prereq: generate_training_data.sh <target> <dataset> $DRAFTERS for each (target, dataset) and
# its -test split, where the drafters cover every column used below (default: eagle3 dflash ddtree).
#
# Env: DATA_SUFFIX (which dataset dir, default eagle3-dflash-ddtree), TREE_BUDGETS (default 32..512).

set -e

TARGET_MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-4B"
    "meta-llama/Llama-3.1-8B-Instruct"
)

DATASET_NAMES=(
    "dolci"
)

DATA_SUFFIX="${DATA_SUFFIX:-eagle3-dflash-ddtree}"  # the unified dataset holding all the columns
TREE_BUDGETS="${TREE_BUDGETS:-32 64 128 256 512}"

# Each entry is "<pos_col> <neg_col>".
ROUTERS=("eagle_acceptance_lengths dflash_acceptance_lengths")  # EAGLE3 vs DFlash
for B in $TREE_BUDGETS; do
    ROUTERS+=("eagle_acceptance_lengths ddtree_b${B}_acceptance_lengths")   # EAGLE3 vs DDTree@B
    ROUTERS+=("ddtree_b${B}_acceptance_lengths dflash_acceptance_lengths")  # DFlash  vs DDTree@B
done

for TARGET_MODEL in "${TARGET_MODEL_PATHS[@]}"; do
    for DATASET in "${DATASET_NAMES[@]}"; do
        for SPEC in "${ROUTERS[@]}"; do
            echo "========================================================"
            echo "Router: TARGET=$TARGET_MODEL DATASET=$DATASET SUFFIX=$DATA_SUFFIX PAIR=[$SPEC]"
            echo "========================================================"
            bash scripts/release/train/train_router.sh "$TARGET_MODEL" "$DATASET" "$DATA_SUFFIX" $SPEC
        done
    done
done
