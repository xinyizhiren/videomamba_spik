#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

JOB_NAME='videomamba_small_cv_train12_test3_ann'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/${JOB_NAME}}"

PREFIX="${PREFIX:-/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/}"
DATA_PATH="${DATA_PATH:-/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"
RESUME_PATH="${RESUME_PATH:-}"
AUTO_AUG="${AUTO_AUG:-none}"
TRAIN_CROP_MIN_SCALE="${TRAIN_CROP_MIN_SCALE:-0.50}"
TRAIN_CROP_MAX_SCALE="${TRAIN_CROP_MAX_SCALE:-1.0}"
DISABLE_TRAIN_FLIP="${DISABLE_TRAIN_FLIP:-1}"
BASE_LR="${BASE_LR:-3.2e-3}"
MIN_LR="${MIN_LR:-3.2e-5}"
WARMUP_LR="${WARMUP_LR:-3.2e-6}"
EPOCHS="${EPOCHS:-80}"
LAYER_DECAY="${LAYER_DECAY:-0.8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
DROP_PATH="${DROP_PATH:-0.1}"
FUSED_CE_LOSS_WEIGHT="${FUSED_CE_LOSS_WEIGHT:-0.0}"
VIEW_CE_LOSS_WEIGHT="${VIEW_CE_LOSS_WEIGHT:-1.0}"
ET_AUX_LOSS_WEIGHT="${ET_AUX_LOSS_WEIGHT:-0.0}"
PRINT_FREQ="${PRINT_FREQ:-10}"
DEBUG_OVERFIT_SAMPLES="${DEBUG_OVERFIT_SAMPLES:-0}"
LOG_PRED_HIST="${LOG_PRED_HIST:-1}"

TRAIN_FLIP_ARGS=()
if [ "${DISABLE_TRAIN_FLIP}" != "0" ]; then
        TRAIN_FLIP_ARGS=(--disable_train_flip)
fi

RESUME_ARGS=(--no_auto_resume)
if [ -n "${RESUME_PATH}" ]; then
        RESUME_ARGS=(--resume "${RESUME_PATH}")
fi

PRED_HIST_ARGS=()
if [ "${LOG_PRED_HIST}" = "0" ]; then
        PRED_HIST_ARGS=(--no_log_pred_hist)
fi

OVERFIT_ARGS=()
if [ "${DEBUG_OVERFIT_SAMPLES}" != "0" ]; then
        OVERFIT_ARGS=(--debug_overfit_samples "${DEBUG_OVERFIT_SAMPLES}")
fi

python "${PROJECT_DIR}/run_class_finetuning_et.py" \
        --model videomamba_small \
        --finetune "${MODEL_PATH}" \
        --delete_head \
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
        --batch_size 6 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 20 \
        --num_frames 16 \
        --sampling_rate 4 \
        --aa "${AUTO_AUG}" \
        --train_crop_min_scale "${TRAIN_CROP_MIN_SCALE}" \
        --train_crop_max_scale "${TRAIN_CROP_MAX_SCALE}" \
        --num_workers 4 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs "${EPOCHS}" \
        --lr "${BASE_LR}" \
        --min_lr "${MIN_LR}" \
        --warmup_lr "${WARMUP_LR}" \
        --fused_ce_loss_weight "${FUSED_CE_LOSS_WEIGHT}" \
        --view_ce_loss_weight "${VIEW_CE_LOSS_WEIGHT}" \
        --et_aux_loss_weight "${ET_AUX_LOSS_WEIGHT}" \
        --layer_decay "${LAYER_DECAY}" \
        --smoothing 0.0 \
        --fc_drop_rate 0.0 \
        --drop_path "${DROP_PATH}" \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay "${WEIGHT_DECAY}" \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --print_freq "${PRINT_FREQ}" \
        --bf16 \
        --update_freq 2 \
        "${TRAIN_FLIP_ARGS[@]}" \
        "${PRED_HIST_ARGS[@]}" \
        "${OVERFIT_ARGS[@]}" \
        "${RESUME_ARGS[@]}"
