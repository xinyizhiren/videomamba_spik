#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# 单卡默认使用第 3 张 GPU；多卡时用 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29504}"

# 从 0 开始训练的实验名，避免覆盖预训练微调输出。
JOB_NAME='videomamba_small_cv_train12_test3_ann_clean_scratch'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${JOB_NAME}}"

# DATA_PATH 放 CSV 标注文件，PREFIX 是 CSV 中相对视频路径的根目录。
PREFIX="${PREFIX:-/data/users/ouyangys/data/multiview_action_videos/}"
DATA_PATH="${DATA_PATH:-/data/users/ouyangys/data/multiview_action_videos/}"
# scratch 默认不加载预训练；如需临时对照，可显式传 MODEL_PATH=/path/to/xxx.pth。
MODEL_PATH="${MODEL_PATH-}"
RESUME_PATH="${RESUME_PATH:-}"
MODEL_NAME="${MODEL_NAME:-videomamba_small}"
NUM_FRAMES="${NUM_FRAMES:-16}"
SPIK_TIME_STEPS="${SPIK_TIME_STEPS:-1}"
SPIK_PATCH_SIZE="${SPIK_PATCH_SIZE:-14}"
SPIK_EMBED_DIMS="${SPIK_EMBED_DIMS:-384}"

# BATCH_SIZE 是每张卡每个 mini-batch 的样本数；多卡时全局 batch 会变大。
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-120}"
# scratch 比微调给更长 warmup 和略高 LR；如不稳定可先降到 1e-4。
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
DROP_PATH="${DROP_PATH:-0.05}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
PRINT_FREQ="${PRINT_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEBUG_OVERFIT_SAMPLES="${DEBUG_OVERFIT_SAMPLES:-0}"

TRAIN_CROP_MIN_SCALE="${TRAIN_CROP_MIN_SCALE:-0.50}"
TRAIN_CROP_MAX_SCALE="${TRAIN_CROP_MAX_SCALE:-1.0}"
TRAIN_CROP_MIN_RATIO="${TRAIN_CROP_MIN_RATIO:-0.75}"
TRAIN_CROP_MAX_RATIO="${TRAIN_CROP_MAX_RATIO:-1.3333}"
DISABLE_TRAIN_FLIP="${DISABLE_TRAIN_FLIP:-1}"
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

"${RUN_CMD[@]}" "${PROJECT_DIR}/run_class_finetuning_et_clean.py" \
        --model "${MODEL_NAME}" \
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
        --num_frames "${NUM_FRAMES}" \
        --sampling_rate 4 \
        --input_size 224 \
        --short_side_size 224 \
        --tubelet_size 1 \
        --spik_time_steps "${SPIK_TIME_STEPS}" \
        --spik_patch_size "${SPIK_PATCH_SIZE}" \
        --spik_embed_dims "${SPIK_EMBED_DIMS}" \
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
        --train_crop_min_scale "${TRAIN_CROP_MIN_SCALE}" \
        --train_crop_max_scale "${TRAIN_CROP_MAX_SCALE}" \
        --train_crop_min_ratio "${TRAIN_CROP_MIN_RATIO}" \
        --train_crop_max_ratio "${TRAIN_CROP_MAX_RATIO}" \
        --print_freq "${PRINT_FREQ}" \
        --bf16 \
        "${TRAIN_FLIP_ARGS[@]}" \
        "${OVERFIT_ARGS[@]}" \
        "${RESUME_ARGS[@]}"
