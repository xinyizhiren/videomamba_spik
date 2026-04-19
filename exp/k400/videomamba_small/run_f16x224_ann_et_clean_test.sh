#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_test3_ann_clean_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TRAIN_OUTPUT_DIR}/best.pth}"

PREFIX="${PREFIX:-/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/}"
DATA_PATH="${DATA_PATH:-/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"

BATCH_SIZE="${BATCH_SIZE:-6}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
TEST_VIEW_CSV="${TEST_VIEW_CSV:-v03_test_set.csv}"
VAL_VIEW_CSV="${VAL_VIEW_CSV:-v03_val_set.csv}"

python "${PROJECT_DIR}/run_class_finetuning_et_clean.py" \
        --eval \
        --eval_split "${EVAL_SPLIT}" \
        --eval_checkpoint "${CHECKPOINT_PATH}" \
        --finetune "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --prefix "${PREFIX}" \
        --train_view1_csv 'aligned_v01_1.csv' \
        --train_view2_csv 'aligned_v02_2.csv' \
        --val_view_csv "${VAL_VIEW_CSV}" \
        --test_view_csv "${TEST_VIEW_CSV}" \
        --csv_delimiter ',' \
        --nb_classes 12 \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --num_frames 16 \
        --sampling_rate 4 \
        --input_size 224 \
        --short_side_size 224 \
        --tubelet_size 1 \
        --num_workers 4 \
        --bf16 \
        --disable_train_flip
