#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# 单卡默认使用第 1 张 GPU；多卡时用 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29502}"

# 这是新增的 val+test 合并验证链路，不替代原 clean 训练脚本。
JOB_NAME='videomamba_small_cv_train12_valtest_ann_clean_full'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/${JOB_NAME}}"

# 新服务器默认数据目录；DATA_PATH 放 CSV，PREFIX 是 CSV 中相对视频路径的根目录。
PREFIX="${PREFIX:-/data/users/ouyangys/data/multiview_action_videos/}"
DATA_PATH="${DATA_PATH:-/data/users/ouyangys/data/multiview_action_videos/}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"
RESUME_PATH="${RESUME_PATH:-}"

# 原始验证/测试 CSV，以及合并后给训练阶段做 validation 的 CSV。
VAL_VIEW_CSV="${VAL_VIEW_CSV:-v03_val_set.csv}"
TEST_VIEW_CSV="${TEST_VIEW_CSV:-v03_test_set.csv}"
MERGED_VIEW_CSV="${MERGED_VIEW_CSV:-v03_val_test_set.csv}"
# 默认为 1：如果合并 CSV 不存在，启动训练前自动生成。
CREATE_MERGED_CSV="${CREATE_MERGED_CSV:-1}"

BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-40}"
# clean 入口中 LR 就是实际 AdamW 学习率，不再按 batch size 做线性缩放。
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
DROP_PATH="${DROP_PATH:-0.1}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
PRINT_FREQ="${PRINT_FREQ:-10}"
DEBUG_OVERFIT_SAMPLES="${DEBUG_OVERFIT_SAMPLES:-0}"

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

# 训练仍使用 view1+view2；validation 改为 view3 的 val+test 合并集。
"${RUN_CMD[@]}" "${PROJECT_DIR}/run_class_finetuning_et_clean.py" \
        --finetune "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --prefix "${PREFIX}" \
        --train_view1_csv 'aligned_v01_1.csv' \
        --train_view2_csv 'aligned_v02_2.csv' \
        --val_view_csv "${MERGED_VIEW_CSV}" \
        --test_view_csv "${MERGED_VIEW_CSV}" \
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
