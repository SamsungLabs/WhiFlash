#!/bin/bash
# Export every trained router to neural_routers/<run_name>_<version>.safetensors.
# Generic: copies each router_models/*/model.safetensors regardless of which drafter
# pair it routes between (the run name already encodes that), so new drafters need no
# changes here. Pass to WhiFlash via --whiflash-neural-router-path.

set -e

BASE_PATH="${WHIFLASH_CKPT_ROOT:-$HOME/checkpoints/whiflash}"
MODEL_DIR="${BASE_PATH}/router_models"
TARGET_DIR="${BASE_PATH}/neural_routers"
VERSION="$1"   # arg 1: version tag appended to each exported filename (neural_routers/<run_name>_<VERSION>.safetensors)

if [[ -z "${VERSION}" ]]; then
    echo "Error: version not specified." >&2
    echo "Usage: $0 <version>" >&2
    exit 1
fi

mkdir -p "${TARGET_DIR}"

for SRC_FILE in "${MODEL_DIR}"/*/model.safetensors; do
    [[ -f "${SRC_FILE}" ]] || continue
    RUN_NAME=$(basename "$(dirname "${SRC_FILE}")")
    cp -p "${SRC_FILE}" "${TARGET_DIR}/${RUN_NAME}_${VERSION}.safetensors"
done

ls -lh "${TARGET_DIR}"
