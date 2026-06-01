#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# 单卡默认使用第 1 张 GPU；多卡评估时用 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# TRAIN_OUTPUT_DIR 指向 clean 训练输出；默认从其中读取 best.pth。
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_test3_ann_clean_full}"
# OUTPUT_DIR 默认复用训练目录，评估日志会写到 test_log.txt 或 validation_log.txt。
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TRAIN_OUTPUT_DIR}/best.pth}"

# DATA_PATH 放 CSV 标注文件，PREFIX 是 CSV 中相对视频路径的根目录。
PREFIX="${PREFIX:-/data/users/ouyangys/data/multiview_action_videos/}"
DATA_PATH="${DATA_PATH:-/data/users/ouyangys/data/multiview_action_videos/}"
# MODEL_PATH 只在未显式提供评估 checkpoint 时作为 fallback 初始化来源。
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"
MODEL_NAME="${MODEL_NAME:-videomamba_small_clean}"
NUM_FRAMES="${NUM_FRAMES:-16}"
SPIK_TIME_STEPS="${SPIK_TIME_STEPS:-1}"
SPIK_PATCH_SIZE="${SPIK_PATCH_SIZE:-14}"
SPIK_EMBED_DIMS="${SPIK_EMBED_DIMS:-384}"

BATCH_SIZE="${BATCH_SIZE:-6}"
# EVAL_SPLIT 可选 test 或 validation；正式测试默认读取 v03_test_set.csv。
EVAL_SPLIT="${EVAL_SPLIT:-test}"
TEST_VIEW_CSV="${TEST_VIEW_CSV:-v03_test_set.csv}"
VAL_VIEW_CSV="${VAL_VIEW_CSV:-v03_val_set.csv}"

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

# 评估复用 clean 训练入口，仅通过 --eval 切换到单视角验证/测试流程。
"${RUN_CMD[@]}" "${PROJECT_DIR}/run_class_finetuning_et_clean.py" \
        --eval \
        --model "${MODEL_NAME}" \
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
        --num_frames "${NUM_FRAMES}" \
        --sampling_rate 4 \
        --input_size 224 \
        --short_side_size 224 \
        --tubelet_size 1 \
        --spik_time_steps "${SPIK_TIME_STEPS}" \
        --spik_patch_size "${SPIK_PATCH_SIZE}" \
        --spik_embed_dims "${SPIK_EMBED_DIMS}" \
        --num_workers 4 \
        --bf16 \
        --disable_train_flip
