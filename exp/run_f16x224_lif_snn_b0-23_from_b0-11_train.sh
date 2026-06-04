#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_LAUNCHER="${PROJECT_DIR}/exp/run_f16x224_trainable_snn.sh"

# Active workflow: train the 24-block unsigned LIF SNN.  Default to two GPUs;
# override with CUDA_VISIBLE_DEVICES=2 NPROC_PER_NODE=1 for a single-card run.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29505}"

export SNN_SPIKE_LAYER="${SNN_SPIKE_LAYER:-lif}"
export SNN_LIF_TAU="${SNN_LIF_TAU:-2.0}"
export SNN_LIF_BACKEND="${SNN_LIF_BACKEND:-torch}"
export SNN_SIGNED_SPIKES="${SNN_SIGNED_SPIKES:-0}"
export SNN_BLOCK_INDICES="${SNN_BLOCK_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}"
export SNN_SPIKE_POSITION="${SNN_SPIKE_POSITION:-post}"
export SNN_SPIKE_PATCH="${SNN_SPIKE_PATCH:-0}"
export SNN_TIMESTEPS="${SNN_TIMESTEPS:-4}"

ANN_OUTPUT_DIR="${ANN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_test3_ann_clean_scratch}"
PREV_OUTPUT_DIR="${PREV_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_test3_ann_clean_scratch}"
export MODEL_PATH="${MODEL_PATH:-${PREV_OUTPUT_DIR}/best.pth}"
export TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${ANN_OUTPUT_DIR}/best.pth}"

SNN_BLOCK_TAG="${SNN_BLOCK_INDICES//,/-}"
if [ "${SNN_SIGNED_SPIKES}" = "0" ]; then
        SNN_SIGN_TAG="unsigned"
else
        SNN_SIGN_TAG="signed"
fi
export JOB_NAME="${JOB_NAME:-videomamba_small_lif_${SNN_SIGN_TAG}_snn_b${SNN_BLOCK_TAG}_t${SNN_TIMESTEPS}_from_b0-11}"

export EPOCHS="${EPOCHS:-30}"
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.7}"
export DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export UPDATE_FREQ="${UPDATE_FREQ:-2}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
export DUMP_MODEL_SUMMARY="${DUMP_MODEL_SUMMARY:-1}"
export DUMP_SPIKE_STATS="${DUMP_SPIKE_STATS:-1}"

echo "Training full-scope VideoMamba LIF SNN from b0-11"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NNODES=${NNODES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT}"
echo "SNN_BLOCK_INDICES=${SNN_BLOCK_INDICES}"
echo "SNN_SPIKE_LAYER=${SNN_SPIKE_LAYER}"
echo "SNN_SIGNED_SPIKES=${SNN_SIGNED_SPIKES}"
echo "SNN_TIMESTEPS=${SNN_TIMESTEPS}"
echo "EPOCHS=${EPOCHS}"
echo "LR=${LR}"

bash "${BASE_LAUNCHER}"
