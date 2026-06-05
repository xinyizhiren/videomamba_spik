#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CUDA indexes start from 0, so the third physical GPU is index 2.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29502}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

JOB_NAME="${JOB_NAME:-videomamba_small_cv_train12_valtest_ann_clean_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${JOB_NAME}}"

DATA_PATH="${DATA_PATH:-/data/users/ouyangys/data/multiview_action_videos/}"
PREFIX="${PREFIX:-${DATA_PATH}}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"
RESUME_PATH="${RESUME_PATH:-}"
MODEL_NAME="${MODEL_NAME:-videomamba_small_clean}"
NUM_FRAMES="${NUM_FRAMES:-16}"

VAL_VIEW_CSV="${VAL_VIEW_CSV:-v03_val_set.csv}"
TEST_VIEW_CSV="${TEST_VIEW_CSV:-v03_test_set.csv}"
MERGED_VIEW_CSV="${MERGED_VIEW_CSV:-v03_val_test_set.csv}"
CREATE_MERGED_CSV="${CREATE_MERGED_CSV:-1}"

BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
DROP_PATH="${DROP_PATH:-0.1}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
PRINT_FREQ="${PRINT_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEBUG_OVERFIT_SAMPLES="${DEBUG_OVERFIT_SAMPLES:-0}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"

TRAIN_CROP_MIN_SCALE="${TRAIN_CROP_MIN_SCALE:-0.50}"
TRAIN_CROP_MAX_SCALE="${TRAIN_CROP_MAX_SCALE:-1.0}"
TRAIN_CROP_MIN_RATIO="${TRAIN_CROP_MIN_RATIO:-0.75}"
TRAIN_CROP_MAX_RATIO="${TRAIN_CROP_MAX_RATIO:-1.3333}"
DISABLE_TRAIN_FLIP="${DISABLE_TRAIN_FLIP:-1}"
FUSED_CE_LOSS_WEIGHT="${FUSED_CE_LOSS_WEIGHT:-1.0}"
VIEW_CE_LOSS_WEIGHT="${VIEW_CE_LOSS_WEIGHT:-1.0}"

if [[ "${MERGED_VIEW_CSV}" = /* ]]; then
        MERGED_CSV_PATH="${MERGED_VIEW_CSV}"
else
        MERGED_CSV_PATH="${DATA_PATH%/}/${MERGED_VIEW_CSV}"
fi

if [ "${CREATE_MERGED_CSV}" != "0" ] && [ ! -f "${MERGED_CSV_PATH}" ]; then
        python "${PROJECT_DIR}/tools/merge_val_test_csv.py" \
                --data-path "${DATA_PATH}" \
                --val-csv "${VAL_VIEW_CSV}" \
                --test-csv "${TEST_VIEW_CSV}" \
                --output-csv "${MERGED_VIEW_CSV}" \
                --delimiter ','
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
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "RESUME_PATH=${RESUME_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "VAL_VIEW_CSV=${VAL_VIEW_CSV}"
echo "TEST_VIEW_CSV=${TEST_VIEW_CSV}"
echo "MERGED_VIEW_CSV=${MERGED_VIEW_CSV}"

CMD=(
        "${RUN_CMD[@]}"
        "${PROJECT_DIR}/run_class_finetuning_et_clean.py"
        --model "${MODEL_NAME}"
        --finetune "${MODEL_PATH}"
        --data_path "${DATA_PATH}"
        --prefix "${PREFIX}"
        --train_view1_csv 'aligned_v01_1.csv'
        --train_view2_csv 'aligned_v02_2.csv'
        --val_view_csv "${MERGED_VIEW_CSV}"
        --test_view_csv "${MERGED_VIEW_CSV}"
        --csv_delimiter ','
        --nb_classes 12
        --output_dir "${OUTPUT_DIR}"
        --batch_size "${BATCH_SIZE}"
        --num_frames "${NUM_FRAMES}"
        --sampling_rate 4
        --input_size 224
        --short_side_size 224
        --tubelet_size 1
        --num_workers "${NUM_WORKERS}"
        --epochs "${EPOCHS}"
        --lr "${LR}"
        --min_lr "${MIN_LR}"
        --warmup_epochs "${WARMUP_EPOCHS}"
        --weight_decay "${WEIGHT_DECAY}"
        --drop_path "${DROP_PATH}"
        --update_freq "${UPDATE_FREQ}"
        --fused_ce_loss_weight "${FUSED_CE_LOSS_WEIGHT}"
        --view_ce_loss_weight "${VIEW_CE_LOSS_WEIGHT}"
        --train_crop_min_scale "${TRAIN_CROP_MIN_SCALE}"
        --train_crop_max_scale "${TRAIN_CROP_MAX_SCALE}"
        --train_crop_min_ratio "${TRAIN_CROP_MIN_RATIO}"
        --train_crop_max_ratio "${TRAIN_CROP_MAX_RATIO}"
        --print_freq "${PRINT_FREQ}"
        --bf16
)

if [ "${CUDNN_BENCHMARK}" = "0" ]; then
        CMD+=(--disable_cudnn_benchmark)
fi

if [ "${DISABLE_TRAIN_FLIP}" != "0" ]; then
        CMD+=(--disable_train_flip)
fi

if [ "${DEBUG_OVERFIT_SAMPLES}" != "0" ]; then
        CMD+=(--debug_overfit_samples "${DEBUG_OVERFIT_SAMPLES}")
fi

if [ -n "${RESUME_PATH}" ]; then
        CMD+=(--resume "${RESUME_PATH}")
fi

"${CMD[@]}"
