#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_LAUNCHER="${PROJECT_DIR}/exp/run_f16x224_trainable_snn.sh"

ANN_OUTPUT_DIR="${ANN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_test3_ann_clean_full}"
ANN_CHECKPOINT="${ANN_CHECKPOINT:-${ANN_OUTPUT_DIR}/best.pth}"

BLOCK0_CHECKPOINT="${BLOCK0_CHECKPOINT:-${PROJECT_DIR}/outputs/videomamba_small_trainable_snn_b0_t4/best.pth}"
BLOCK01_CHECKPOINT="${BLOCK01_CHECKPOINT:-${PROJECT_DIR}/outputs/videomamba_small_trainable_snn_b0-1_t4_from_b0/best.pth}"

ABLATION_JOBS="${ABLATION_JOBS:-block01_from_block0}"

export BATCH_SIZE="${BATCH_SIZE:-1}"
export UPDATE_FREQ="${UPDATE_FREQ:-2}"
export SNN_TIMESTEPS="${SNN_TIMESTEPS:-4}"
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"

run_scope_job() {
        local job="$1"
        local model_path=""
        local output_dir=""
        local job_name=""
        local block_indices=""

        case "${job}" in
                block01_from_block0)
                        model_path="${BLOCK0_CHECKPOINT}"
                        block_indices="0,1"
                        job_name="videomamba_small_trainable_snn_b0-1_t${SNN_TIMESTEPS}_from_b0"
                        output_dir="${PROJECT_DIR}/outputs/${job_name}"
                        ;;
                block0123_from_block01)
                        model_path="${BLOCK01_CHECKPOINT}"
                        block_indices="0,1,2,3"
                        job_name="videomamba_small_trainable_snn_b0-1-2-3_t${SNN_TIMESTEPS}_from_b0-1"
                        output_dir="${PROJECT_DIR}/outputs/${job_name}"
                        ;;
                *)
                        echo "Unknown ABLATION job: ${job}" >&2
                        echo "Supported jobs: block01_from_block0 block0123_from_block01" >&2
                        exit 1
                        ;;
        esac

        if [ ! -f "${model_path}" ]; then
                echo "Missing init checkpoint for ${job}: ${model_path}" >&2
                echo "Run the previous scope first, or override BLOCK0_CHECKPOINT/BLOCK01_CHECKPOINT." >&2
                exit 1
        fi

        if [ ! -f "${ANN_CHECKPOINT}" ]; then
                echo "Missing ANN teacher checkpoint: ${ANN_CHECKPOINT}" >&2
                echo "Set ANN_OUTPUT_DIR or ANN_CHECKPOINT." >&2
                exit 1
        fi

        echo "===== Running ${job} ====="
        echo "MODEL_PATH=${model_path}"
        echo "TEACHER_CHECKPOINT=${ANN_CHECKPOINT}"
        echo "SNN_BLOCK_INDICES=${block_indices}"
        echo "OUTPUT_DIR=${output_dir}"

        MODEL_PATH="${model_path}" \
        TEACHER_CHECKPOINT="${ANN_CHECKPOINT}" \
        SNN_BLOCK_INDICES="${block_indices}" \
        JOB_NAME="${job_name}" \
        OUTPUT_DIR="${output_dir}" \
        bash "${BASE_LAUNCHER}"
}

echo "Trainable SNN scope ablation jobs: ${ABLATION_JOBS}"
for job in ${ABLATION_JOBS}; do
        run_scope_job "${job}"
done
