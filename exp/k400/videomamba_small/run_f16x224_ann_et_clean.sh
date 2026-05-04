#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# 默认使用第 1 张 GPU；临时换卡可在命令前覆盖 CUDA_VISIBLE_DEVICES。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# 正式 clean 训练实验名，同时决定默认输出目录名。
JOB_NAME='videomamba_small_cv_train12_test3_ann_clean_full'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# 训练日志和 checkpoint 输出目录；可用 OUTPUT_DIR=... 覆盖。
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${JOB_NAME}}"

# DATA_PATH 放 CSV 标注文件，PREFIX 是 CSV 中相对视频路径的根目录。
PREFIX="${PREFIX:-/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/}"
DATA_PATH="${DATA_PATH:-/data_hdd/oyys/VIT_4090/dataset/data/multiview_action_videos/}"
# MODEL_PATH 是 K400 预训练权重；RESUME_PATH 是继续上一次 clean 训练的断点。
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"
RESUME_PATH="${RESUME_PATH:-}"

# BATCH_SIZE 是单次 optimizer step 前每个 mini-batch 的样本数；UPDATE_FREQ 用于梯度累积。
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-40}"
# clean 入口中 LR 就是实际 AdamW 学习率，不再按 batch size 做线性缩放。
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
# DROP_PATH 是 VideoMamba stochastic depth 强度，数据小的时候不宜过大。
DROP_PATH="${DROP_PATH:-0.1}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
PRINT_FREQ="${PRINT_FREQ:-10}"
# 设为非 0 时只取均衡小样本训练，用于快速 overfit/sanity check。
DEBUG_OVERFIT_SAMPLES="${DEBUG_OVERFIT_SAMPLES:-0}"

# 训练随机裁剪范围；默认保守，避免小数据集上裁掉动作主体。
TRAIN_CROP_MIN_SCALE="${TRAIN_CROP_MIN_SCALE:-0.50}"
TRAIN_CROP_MAX_SCALE="${TRAIN_CROP_MAX_SCALE:-1.0}"
TRAIN_CROP_MIN_RATIO="${TRAIN_CROP_MIN_RATIO:-0.75}"
TRAIN_CROP_MAX_RATIO="${TRAIN_CROP_MAX_RATIO:-1.3333}"
# 默认关闭水平翻转，因为跨视角动作数据里左右关系可能有语义。
DISABLE_TRAIN_FLIP="${DISABLE_TRAIN_FLIP:-1}"
# clean 训练同时监督双视角平均 logits 和两个单视角 logits。
FUSED_CE_LOSS_WEIGHT="${FUSED_CE_LOSS_WEIGHT:-1.0}"
VIEW_CE_LOSS_WEIGHT="${VIEW_CE_LOSS_WEIGHT:-1.0}"

TRAIN_FLIP_ARGS=()
if [ "${DISABLE_TRAIN_FLIP}" != "0" ]; then
        TRAIN_FLIP_ARGS=(--disable_train_flip)
fi

RESUME_ARGS=()
if [ -n "${RESUME_PATH}" ]; then
        RESUME_ARGS=(--resume "${RESUME_PATH}")
fi

OVERFIT_ARGS=()
if [ "${DEBUG_OVERFIT_SAMPLES}" != "0" ]; then
        OVERFIT_ARGS=(--debug_overfit_samples "${DEBUG_OVERFIT_SAMPLES}")
fi

# 训练用 view1+view2，验证默认使用未参与训练的 view3。
python "${PROJECT_DIR}/run_class_finetuning_et_clean.py" \
        --finetune "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --prefix "${PREFIX}" \
        --train_view1_csv 'aligned_v01_1.csv' \
        --train_view2_csv 'aligned_v02_2.csv' \
        --val_view_csv 'v03_val_set.csv' \
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
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --min_lr "${MIN_LR}" \
        --warmup_epochs "${WARMUP_EPOCHS}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --drop_path "${DROP_PATH}" \
        --update_freq "${UPDATE_FREQ}" \
        --fused_ce_loss_weight "${FUSED_CE_LOSS_WEIGHT}" \
        --view_ce_loss_weight "${VIEW_CE_LOSS_WEIGHT}" \
        --train_crop_min_scale "${TRAIN_CROP_MIN_SCALE}" \
        --train_crop_max_scale "${TRAIN_CROP_MAX_SCALE}" \
        --train_crop_min_ratio "${TRAIN_CROP_MIN_RATIO}" \
        --train_crop_max_ratio "${TRAIN_CROP_MAX_RATIO}" \
        --print_freq "${PRINT_FREQ}" \
        --bf16 \
        "${TRAIN_FLIP_ARGS[@]}" \
        "${OVERFIT_ARGS[@]}" \
        "${RESUME_ARGS[@]}"
