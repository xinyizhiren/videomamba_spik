#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_LAUNCHER="${PROJECT_DIR}/exp/run_f16x224_lif_snn_b0-23_from_valtest_ann_train.sh"

SNN_OUTPUT_DIR="${SNN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_lif_unsigned_snn_b0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_t4_from_valtest_ann}"
export RESUME_PATH="${RESUME_PATH:-${SNN_OUTPUT_DIR}/latest.pth}"

if [ ! -f "${RESUME_PATH}" ]; then
        echo "Missing resume checkpoint: ${RESUME_PATH}" >&2
        echo "Set RESUME_PATH=/path/to/latest.pth or SNN_OUTPUT_DIR=/path/to/output_dir." >&2
        exit 1
fi

echo "Resume full-scope VideoMamba LIF SNN"
echo "RESUME_PATH=${RESUME_PATH}"

bash "${BASE_LAUNCHER}"
