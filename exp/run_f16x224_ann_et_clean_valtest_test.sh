#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CUDA indexes start from 0, so the third physical GPU is index 2.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29503}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_valtest_ann_clean_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TRAIN_OUTPUT_DIR}/best.pth}"

DATA_PATH="${DATA_PATH:-/data/users/ouyangys/data/multiview_action_videos/}"
PREFIX="${PREFIX:-${DATA_PATH}}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"
MODEL_NAME="${MODEL_NAME:-videomamba_small_clean}"
NUM_FRAMES="${NUM_FRAMES:-16}"

VAL_VIEW_CSV="${VAL_VIEW_CSV:-v03_val_set.csv}"
TEST_VIEW_CSV="${TEST_VIEW_CSV:-v03_test_set.csv}"
MERGED_VIEW_CSV="${MERGED_VIEW_CSV:-v03_val_test_set.csv}"
CREATE_MERGED_CSV="${CREATE_MERGED_CSV:-1}"

BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"

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
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL_SPLIT=${EVAL_SPLIT}"
echo "MERGED_VIEW_CSV=${MERGED_VIEW_CSV}"

CMD=(
        "${RUN_CMD[@]}"
        "${PROJECT_DIR}/run_class_finetuning_et_clean.py"
        --eval
        --model "${MODEL_NAME}"
        --eval_split "${EVAL_SPLIT}"
        --eval_checkpoint "${CHECKPOINT_PATH}"
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
        --bf16
        --disable_train_flip
)

if [ "${CUDNN_BENCHMARK}" = "0" ]; then
        CMD+=(--disable_cudnn_benchmark)
fi

"${CMD[@]}"
