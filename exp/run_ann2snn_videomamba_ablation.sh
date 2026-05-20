#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Job format: run_name:spike_patch:spike_block_indices:threshold_scale
# Keep patch spike off first; add it only after block-boundary spike is stable.
DEFAULT_JOBS="block0:0:0:1.0 block01:0:0,1:1.0 block0123:0:0,1,2,3:1.0"

if [ "${RUN_PATCH_ABLATION:-0}" != "0" ]; then
        DEFAULT_JOBS="${DEFAULT_JOBS} patch_block01:1:0,1:1.0"
fi

ABLATION_JOBS="${ABLATION_JOBS:-${DEFAULT_JOBS}}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${PROJECT_DIR}/outputs/ann2snn_videomamba/ablation_summary.txt}"

echo "ANN2SNN ablation jobs: ${ABLATION_JOBS}"
echo "Summary will be written to: ${SUMMARY_OUTPUT}"

for job in ${ABLATION_JOBS}; do
        IFS=':' read -r RUN_NAME_VALUE SPIKE_PATCH_VALUE SPIKE_BLOCK_INDICES_VALUE THRESHOLD_SCALE_VALUE <<< "${job}"

        if [ -z "${RUN_NAME_VALUE}" ]; then
                echo "Invalid ablation job: ${job}" >&2
                exit 1
        fi

        SPIKE_PATCH_VALUE="${SPIKE_PATCH_VALUE:-0}"
        SPIKE_BLOCK_INDICES_VALUE="${SPIKE_BLOCK_INDICES_VALUE-}"
        THRESHOLD_SCALE_VALUE="${THRESHOLD_SCALE_VALUE:-1.0}"

        echo
        echo "===== Running ${RUN_NAME_VALUE} ====="
        echo "SPIKE_PATCH=${SPIKE_PATCH_VALUE}"
        echo "SPIKE_BLOCK_INDICES=${SPIKE_BLOCK_INDICES_VALUE}"
        echo "THRESHOLD_SCALE=${THRESHOLD_SCALE_VALUE}"

        RUN_NAME="${RUN_NAME_VALUE}" \
        SPIKE_PATCH="${SPIKE_PATCH_VALUE}" \
        SPIKE_BLOCK_INDICES="${SPIKE_BLOCK_INDICES_VALUE}" \
        THRESHOLD_SCALE="${THRESHOLD_SCALE_VALUE}" \
        bash "${SCRIPT_DIR}/run_ann2snn_videomamba.sh"

        python "${PROJECT_DIR}/ann2snn/summarize_ablation_results.py" \
                --root "${PROJECT_DIR}/outputs/ann2snn_videomamba" \
                --output "${SUMMARY_OUTPUT}"
done

echo
echo "Ablation complete."
cat "${SUMMARY_OUTPUT}"
