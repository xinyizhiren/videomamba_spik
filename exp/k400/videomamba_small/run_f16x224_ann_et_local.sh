#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES="0"

JOB_NAME='videomamba_small_cv_train12_test3_ann'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/${JOB_NAME}}"

PREFIX="${PREFIX:-/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/}"
DATA_PATH="${DATA_PATH:-/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"

python "${PROJECT_DIR}/run_class_finetuning_et.py" \
        --model videomamba_small \
        --finetune "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --prefix "${PREFIX}" \
        --data_set 'Kinetics_sparse_et' \
        --train_view1_csv 'aligned_v01_1.csv' \
        --train_view2_csv 'aligned_v02_2.csv' \
        --val_view_csv 'v03_val_set.csv' \
        --test_view_csv 'v03_test_set.csv' \
        --split ',' \
        --nb_classes 12 \
        --log_dir "${LOG_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size 4 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 20 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_workers 4 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 80 \
        --lr 1e-4 \
        --layer_decay 0.8 \
        --fc_drop_rate 0.1 \
        --drop_path 0.2 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16 \
        --update_freq 2
