#!/bin/bash

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
# 默认使用第 1 张 GPU；临时换卡可在命令前覆盖 CUDA_VISIBLE_DEVICES。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 对应 run_f16x224_ann_et_clean_valtest.sh 的输出目录。
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_valtest_ann_clean_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TRAIN_OUTPUT_DIR}/best.pth}"

# 新服务器默认数据目录；DATA_PATH 放 CSV，PREFIX 是 CSV 中相对视频路径的根目录。
PREFIX="${PREFIX:-/data/users/ouyangys/data/multiview_action_videos/}"
DATA_PATH="${DATA_PATH:-/data/users/ouyangys/data/multiview_action_videos/}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/videomamba_s16_k400_f16_res224.pth}"

VAL_VIEW_CSV="${VAL_VIEW_CSV:-v03_val_set.csv}"
TEST_VIEW_CSV="${TEST_VIEW_CSV:-v03_test_set.csv}"
MERGED_VIEW_CSV="${MERGED_VIEW_CSV:-v03_val_test_set.csv}"
CREATE_MERGED_CSV="${CREATE_MERGED_CSV:-1}"

BATCH_SIZE="${BATCH_SIZE:-6}"
# validation 或 test 都会读取同一个合并 CSV，默认记为 test 日志。
EVAL_SPLIT="${EVAL_SPLIT:-test}"

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

# 评估复用 clean 入口，val/test CSV 均指向合并后的 held-out view3。
python "${PROJECT_DIR}/run_class_finetuning_et_clean.py" \
        --eval \
        --eval_split "${EVAL_SPLIT}" \
        --eval_checkpoint "${CHECKPOINT_PATH}" \
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
        --bf16 \
        --disable_train_flip
