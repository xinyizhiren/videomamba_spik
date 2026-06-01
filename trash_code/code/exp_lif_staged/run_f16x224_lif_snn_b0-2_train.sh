#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_LAUNCHER="${PROJECT_DIR}/exp/run_f16x224_trainable_snn.sh"

export SNN_SPIKE_LAYER="${SNN_SPIKE_LAYER:-lif}"
export SNN_LIF_TAU="${SNN_LIF_TAU:-2.0}"
export SNN_LIF_BACKEND="${SNN_LIF_BACKEND:-torch}"
export SNN_SIGNED_SPIKES="${SNN_SIGNED_SPIKES:-0}"
export SNN_BLOCK_INDICES="${SNN_BLOCK_INDICES:-0,1,2}"
export SNN_SPIKE_POSITION="${SNN_SPIKE_POSITION:-post}"
export SNN_SPIKE_PATCH="${SNN_SPIKE_PATCH:-0}"
export SNN_TIMESTEPS="${SNN_TIMESTEPS:-4}"

SNN_BLOCK_TAG="${SNN_BLOCK_INDICES//,/-}"
if [ "${SNN_SIGNED_SPIKES}" = "0" ]; then
        SNN_SIGN_TAG="unsigned"
else
        SNN_SIGN_TAG="signed"
fi
export JOB_NAME="${JOB_NAME:-videomamba_small_lif_${SNN_SIGN_TAG}_snn_b${SNN_BLOCK_TAG}_t${SNN_TIMESTEPS}_from_ann}"
export EPOCHS="${EPOCHS:-30}"
export LR="${LR:-2e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.7}"
export DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export UPDATE_FREQ="${UPDATE_FREQ:-2}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
export DUMP_MODEL_SUMMARY="${DUMP_MODEL_SUMMARY:-1}"
export DUMP_SPIKE_STATS="${DUMP_SPIKE_STATS:-1}"

echo "Training selected VideoMamba LIF SNN"
echo "SNN_BLOCK_INDICES=${SNN_BLOCK_INDICES}"
echo "SNN_SPIKE_LAYER=${SNN_SPIKE_LAYER}"
echo "SNN_SIGNED_SPIKES=${SNN_SIGNED_SPIKES}"
echo "SNN_TIMESTEPS=${SNN_TIMESTEPS}"
echo "EPOCHS=${EPOCHS}"

bash "${BASE_LAUNCHER}"
