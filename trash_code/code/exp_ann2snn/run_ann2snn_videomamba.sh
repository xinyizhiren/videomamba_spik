#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CUDA indexes start from 0, so the third physical GPU is index 2.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_PATH="${DATA_PATH:-/data/users/ouyangys/data/multiview_action_videos/}"
PREFIX="${PREFIX:-${DATA_PATH}}"

# The ANN checkpoint is expected next to the training log.txt.
ANN_OUTPUT_DIR="${ANN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_test3_ann_clean_full}"
ANN_CHECKPOINT="${ANN_CHECKPOINT:-${ANN_OUTPUT_DIR}/best.pth}"
ANN_LOG="${ANN_LOG:-${ANN_OUTPUT_DIR}/log.txt}"

# Default run is a no-spike sanity check. Override these for real SNN ablations:
#   SPIKE_PATCH=0 SPIKE_BLOCK_INDICES=0
#   SPIKE_PATCH=0 SPIKE_BLOCK_INDICES=0,1
#   SPIKE_PATCH=1 SPIKE_BLOCK_INDICES=0,1
RUN_NAME="${RUN_NAME:-no_spike_sanity}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/ann2snn_videomamba/${RUN_NAME}}"
SPIKE_PATCH="${SPIKE_PATCH:-0}"
SPIKE_BLOCK_INDICES="${SPIKE_BLOCK_INDICES-}"

BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-256}"
CALIBRATION_STEPS="${CALIBRATION_STEPS:-200}"
TIMESTEPS="${TIMESTEPS:-16}"
DELAY="${DELAY:--1}"
THRESHOLD_SCALE="${THRESHOLD_SCALE:-1.0}"
SAVE_SNN_CHECKPOINT="${SAVE_SNN_CHECKPOINT:-0}"

if [ ! -f "${ANN_CHECKPOINT}" ]; then
        echo "Missing ANN checkpoint: ${ANN_CHECKPOINT}" >&2
        echo "Set ANN_OUTPUT_DIR to the folder that contains log.txt and best.pth, or set ANN_CHECKPOINT directly." >&2
        exit 1
fi

if [ ! -f "${ANN_LOG}" ]; then
        echo "Warning: ANN log file not found at ${ANN_LOG}" >&2
fi

mkdir -p "${OUTPUT_DIR}"

SPIKE_PATCH_ARGS=(--no_spike_patch)
if [ "${SPIKE_PATCH}" != "0" ]; then
        SPIKE_PATCH_ARGS=(--spike_patch)
fi

{
        echo "RUN_NAME=${RUN_NAME}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "ANN_CHECKPOINT=${ANN_CHECKPOINT}"
        echo "OUTPUT_DIR=${OUTPUT_DIR}"
        echo "SPIKE_PATCH=${SPIKE_PATCH}"
        echo "SPIKE_BLOCK_INDICES=${SPIKE_BLOCK_INDICES}"
        echo "THRESHOLD_SCALE=${THRESHOLD_SCALE}"
        echo "SAVE_SNN_CHECKPOINT=${SAVE_SNN_CHECKPOINT}"

        SAVE_ARGS=(--skip_save_checkpoint)
        if [ "${SAVE_SNN_CHECKPOINT}" != "0" ]; then
                SAVE_ARGS=()
        fi

        python "${PROJECT_DIR}/ann2snn/convert_videomamba_ann_to_snn.py" \
                --checkpoint "${ANN_CHECKPOINT}" \
                --data_path "${DATA_PATH}" \
                --prefix "${PREFIX}" \
                --output_dir "${OUTPUT_DIR}" \
                --calib_view_csv 'aligned_v01_1.csv' \
                --val_view_csv 'v03_val_set.csv' \
                --test_view_csv 'v03_test_set.csv' \
                --batch_size "${BATCH_SIZE}" \
                --num_workers "${NUM_WORKERS}" \
                --calibration_samples "${CALIBRATION_SAMPLES}" \
                --calibration_steps "${CALIBRATION_STEPS}" \
                --timesteps "${TIMESTEPS}" \
                --delay "${DELAY}" \
                --spike_block_indices "${SPIKE_BLOCK_INDICES}" \
                --threshold_scale "${THRESHOLD_SCALE}" \
                --device cuda \
                --dump_layer_order \
                "${SAVE_ARGS[@]}" \
                "${SPIKE_PATCH_ARGS[@]}"
} 2>&1 | tee "${OUTPUT_DIR}/run_log.txt"
