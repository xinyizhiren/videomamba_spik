#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CUDA indexes start from 0, so the third physical GPU is index 2.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29505}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_PATH="${DATA_PATH:-/data/users/ouyangys/data/multiview_action_videos/}"
PREFIX="${PREFIX:-${DATA_PATH}}"
ANN_OUTPUT_DIR="${ANN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_test3_ann_clean_full}"
MODEL_PATH="${MODEL_PATH:-${ANN_OUTPUT_DIR}/best.pth}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${ANN_OUTPUT_DIR}/best.pth}"

SNN_BLOCK_INDICES="${SNN_BLOCK_INDICES:-0}"
SNN_BLOCK_TAG="${SNN_BLOCK_INDICES//,/-}"
SNN_BLOCK_TAG="${SNN_BLOCK_TAG:-none}"
SNN_TIMESTEPS="${SNN_TIMESTEPS:-4}"
SNN_SPIKE_PATCH="${SNN_SPIKE_PATCH:-0}"
SNN_SIGNED_SPIKES="${SNN_SIGNED_SPIKES:-1}"
SNN_TRAIN_THRESHOLD="${SNN_TRAIN_THRESHOLD:-1}"
SNN_THRESHOLD_INIT="${SNN_THRESHOLD_INIT:-1.0}"
SNN_THRESHOLD_PERCENTILE="${SNN_THRESHOLD_PERCENTILE:-0.99}"
SNN_SURROGATE_ALPHA="${SNN_SURROGATE_ALPHA:-4.0}"
SPIKE_LR_MULTIPLIER="${SPIKE_LR_MULTIPLIER:-5.0}"

JOB_NAME="${JOB_NAME:-videomamba_small_trainable_snn_b${SNN_BLOCK_TAG}_t${SNN_TIMESTEPS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${JOB_NAME}}"

MODEL_NAME="${MODEL_NAME:-videomamba_small_trainable_snn}"
NUM_FRAMES="${NUM_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-3e-5}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
DROP_PATH="${DROP_PATH:-0.05}"
UPDATE_FREQ="${UPDATE_FREQ:-2}"
PRINT_FREQ="${PRINT_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEBUG_OVERFIT_SAMPLES="${DEBUG_OVERFIT_SAMPLES:-0}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
DUMP_MODEL_SUMMARY="${DUMP_MODEL_SUMMARY:-1}"
SUMMARY_DEPTH="${SUMMARY_DEPTH:-5}"

DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
FUSED_CE_LOSS_WEIGHT="${FUSED_CE_LOSS_WEIGHT:-1.0}"
VIEW_CE_LOSS_WEIGHT="${VIEW_CE_LOSS_WEIGHT:-1.0}"

TRAIN_CROP_MIN_SCALE="${TRAIN_CROP_MIN_SCALE:-0.50}"
TRAIN_CROP_MAX_SCALE="${TRAIN_CROP_MAX_SCALE:-1.0}"
TRAIN_CROP_MIN_RATIO="${TRAIN_CROP_MIN_RATIO:-0.75}"
TRAIN_CROP_MAX_RATIO="${TRAIN_CROP_MAX_RATIO:-1.3333}"
DISABLE_TRAIN_FLIP="${DISABLE_TRAIN_FLIP:-1}"

if [ ! -f "${MODEL_PATH}" ]; then
        echo "Missing ANN checkpoint: ${MODEL_PATH}" >&2
        echo "Set ANN_OUTPUT_DIR or MODEL_PATH to the folder/file containing the trained ANN best.pth." >&2
        exit 1
fi

RUN_CMD=(python)
if [ "${NPROC_PER_NODE}" -gt 1 ] || [ "${NNODES}" -gt 1 ]; then
        RUN_CMD=(
                torchrun
                --nnodes "${NNODES}"
                --nproc_per_node "${NPROC_PER_NODE}"
                --node_rank "${NODE_RANK}"
                --master_addr "${MASTER_ADDR}"
                --master_port "${MASTER_PORT}"
        )
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SNN_BLOCK_INDICES=${SNN_BLOCK_INDICES}"
echo "SNN_TIMESTEPS=${SNN_TIMESTEPS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "UPDATE_FREQ=${UPDATE_FREQ}"
echo "DISTILL_WEIGHT=${DISTILL_WEIGHT}"
echo "CUDNN_BENCHMARK=${CUDNN_BENCHMARK}"
echo "DUMP_MODEL_SUMMARY=${DUMP_MODEL_SUMMARY}"

CMD=(
        "${RUN_CMD[@]}"
        "${PROJECT_DIR}/run_class_finetuning_et_clean.py"
        --model "${MODEL_NAME}" \
        --finetune "${MODEL_PATH}" \
        --teacher_checkpoint "${TEACHER_CHECKPOINT}" \
        --distill_weight "${DISTILL_WEIGHT}" \
        --distill_temperature "${DISTILL_TEMPERATURE}" \
        --data_path "${DATA_PATH}" \
        --prefix "${PREFIX}" \
        --train_view1_csv 'aligned_v01_1.csv' \
        --train_view2_csv 'aligned_v02_2.csv' \
        --val_view_csv 'v03_val_set.csv' \
        --csv_delimiter ',' \
        --nb_classes 12 \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --num_frames "${NUM_FRAMES}" \
        --sampling_rate 4 \
        --input_size 224 \
        --short_side_size 224 \
        --tubelet_size 1 \
        --num_workers "${NUM_WORKERS}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --min_lr "${MIN_LR}" \
        --warmup_epochs "${WARMUP_EPOCHS}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --drop_path "${DROP_PATH}" \
        --update_freq "${UPDATE_FREQ}" \
        --fused_ce_loss_weight "${FUSED_CE_LOSS_WEIGHT}" \
        --view_ce_loss_weight "${VIEW_CE_LOSS_WEIGHT}" \
        --snn_block_indices "${SNN_BLOCK_INDICES}" \
        --snn_timesteps "${SNN_TIMESTEPS}" \
        --snn_threshold_init "${SNN_THRESHOLD_INIT}" \
        --snn_threshold_percentile "${SNN_THRESHOLD_PERCENTILE}" \
        --snn_surrogate_alpha "${SNN_SURROGATE_ALPHA}" \
        --spike_lr_multiplier "${SPIKE_LR_MULTIPLIER}" \
        --train_crop_min_scale "${TRAIN_CROP_MIN_SCALE}" \
        --train_crop_max_scale "${TRAIN_CROP_MAX_SCALE}" \
        --train_crop_min_ratio "${TRAIN_CROP_MIN_RATIO}" \
        --train_crop_max_ratio "${TRAIN_CROP_MAX_RATIO}" \
        --print_freq "${PRINT_FREQ}" \
        --bf16
)

if [ "${CUDNN_BENCHMARK}" = "0" ]; then
        CMD+=(--disable_cudnn_benchmark)
fi

if [ "${DUMP_MODEL_SUMMARY}" != "0" ]; then
        CMD+=(--dump_model_summary --summary_depth "${SUMMARY_DEPTH}")
fi

if [ "${SNN_SPIKE_PATCH}" != "0" ]; then
        CMD+=(--snn_spike_patch)
fi

if [ "${SNN_SIGNED_SPIKES}" = "0" ]; then
        CMD+=(--snn_unsigned_spikes)
fi

if [ "${SNN_TRAIN_THRESHOLD}" = "0" ]; then
        CMD+=(--snn_fixed_threshold)
fi

if [ "${DISABLE_TRAIN_FLIP}" != "0" ]; then
        CMD+=(--disable_train_flip)
fi

if [ "${DEBUG_OVERFIT_SAMPLES}" != "0" ]; then
        CMD+=(--debug_overfit_samples "${DEBUG_OVERFIT_SAMPLES}")
fi

"${CMD[@]}"
