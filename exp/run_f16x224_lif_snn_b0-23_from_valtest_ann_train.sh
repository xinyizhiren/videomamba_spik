#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_LAUNCHER="${PROJECT_DIR}/exp/run_f16x224_trainable_snn.sh"

RUN_MODE="${1:-train}"
case "${RUN_MODE}" in
        train|resume|resume-latest)
                ;;
        *)
                echo "Usage: bash $0 [train|resume]" >&2
                exit 2
                ;;
esac

# Train/resume a full-scope 24-block unsigned LIF SNN from the valtest ANN.
# This launcher ignores generic inherited variables such as MODEL_PATH,
# RESUME_PATH, DISTILL_WEIGHT, USE_CHECKPOINT, and CHECKPOINT_NUM. Use VALTEST_*
# overrides for intentional changes so stale shell exports cannot alter a run.
export CUDA_VISIBLE_DEVICES="${VALTEST_CUDA_VISIBLE_DEVICES:-1}"
export NPROC_PER_NODE="${VALTEST_NPROC_PER_NODE:-1}"
export NNODES="${VALTEST_NNODES:-1}"
export NODE_RANK="${VALTEST_NODE_RANK:-0}"
export MASTER_ADDR="${VALTEST_MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${VALTEST_MASTER_PORT:-29506}"

ANN_OUTPUT_DIR="${ANN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_valtest_ann_clean_full}"
export MODEL_PATH="${MODEL_PATH:-${ANN_OUTPUT_DIR}/best.pth}"
export TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${ANN_OUTPUT_DIR}/best.pth}"

SNN_BLOCK_TAG="${SNN_BLOCK_INDICES//,/-}"
if [ "${SNN_SIGNED_SPIKES}" = "0" ]; then
        SNN_SIGN_TAG="unsigned"
else
        SNN_SIGN_TAG="signed"
fi
export JOB_NAME="${VALTEST_JOB_NAME:-videomamba_small_lif_${SNN_SIGN_TAG}_snn_b${SNN_BLOCK_TAG}_t${SNN_TIMESTEPS}_from_valtest_ann}"
export OUTPUT_DIR="${VALTEST_OUTPUT_DIR:-${PROJECT_DIR}/outputs/${JOB_NAME}}"

RESUME_FROM_LATEST="${VALTEST_RESUME_FROM_LATEST:-0}"
if [ "${RUN_MODE}" = "resume" ] || [ "${RUN_MODE}" = "resume-latest" ]; then
        RESUME_FROM_LATEST=1
fi

RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
DEFAULT_RESUME_PATH="${OUTPUT_DIR}/latest.pth"
unset RESUME_PATH
if [ -n "${VALTEST_RESUME_PATH:-}" ]; then
        export RESUME_PATH="${VALTEST_RESUME_PATH}"
elif [ "${RESUME_FROM_LATEST}" != "0" ]; then
        export RESUME_PATH="${DEFAULT_RESUME_PATH}"
fi

export EPOCHS="${EPOCHS:-30}"
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.0}"
export DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export UPDATE_FREQ="${UPDATE_FREQ:-2}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
export DUMP_MODEL_SUMMARY="${DUMP_MODEL_SUMMARY:-0}"
export DUMP_SPIKE_STATS="${DUMP_SPIKE_STATS:-0}"
export USE_CHECKPOINT="${USE_CHECKPOINT:-0}"
export CHECKPOINT_NUM="${CHECKPOINT_NUM:-0}"

echo "Training full-scope VideoMamba LIF SNN from valtest ANN"
echo "RUN_MODE=${RUN_MODE}"
echo "Override namespace: VALTEST_*"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NNODES=${NNODES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESUME_FROM_LATEST=${RESUME_FROM_LATEST}"
echo "RESUME_PATH=${RESUME_PATH:-}"
echo "SNN_BLOCK_INDICES=${SNN_BLOCK_INDICES}"
echo "SNN_SPIKE_LAYER=${SNN_SPIKE_LAYER}"
echo "SNN_SIGNED_SPIKES=${SNN_SIGNED_SPIKES}"
echo "SNN_TIMESTEPS=${SNN_TIMESTEPS}"
echo "EPOCHS=${EPOCHS}"
echo "LR=${LR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "DISTILL_WEIGHT=${DISTILL_WEIGHT}"
echo "USE_CHECKPOINT=${USE_CHECKPOINT}"
echo "CHECKPOINT_NUM=${CHECKPOINT_NUM}"

bash "${BASE_LAUNCHER}"
