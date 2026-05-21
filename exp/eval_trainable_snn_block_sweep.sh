#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_LAUNCHER="${PROJECT_DIR}/exp/run_f16x224_trainable_snn.sh"

ANN_OUTPUT_DIR="${ANN_OUTPUT_DIR:-${PROJECT_DIR}/outputs/videomamba_small_cv_train12_test3_ann_clean_full}"
SWEEP_MODEL_PATH="${MODEL_PATH:-${ANN_OUTPUT_DIR}/best.pth}"

MIN_BLOCKS="${MIN_BLOCKS:-1}"
MAX_BLOCKS="${MAX_BLOCKS:-24}"
SNN_TIMESTEPS="${SNN_TIMESTEPS:-4}"
SNN_SPIKE_POSITION="${SNN_SPIKE_POSITION:-post}"
SNN_SPIKE_PATCH="${SNN_SPIKE_PATCH:-0}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-0}"
SWEEP_DISTILL_WEIGHT="${SWEEP_DISTILL_WEIGHT:-0}"
SWEEP_DUMP_MODEL_SUMMARY="${SWEEP_DUMP_MODEL_SUMMARY:-0}"
SWEEP_SKIP_INITIAL_BEST_CHECKPOINT="${SWEEP_SKIP_INITIAL_BEST_CHECKPOINT:-1}"
SWEEP_TAG="${SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/trainable_snn_block_sweep_${SWEEP_TAG}}"
SUMMARY_TXT="${OUTPUT_ROOT}/block_sweep_summary.txt"

if [ ! -f "${SWEEP_MODEL_PATH}" ]; then
        echo "Missing sweep checkpoint: ${SWEEP_MODEL_PATH}" >&2
        echo "Set MODEL_PATH to the checkpoint used for all no-train block-sweep evaluations." >&2
        exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
{
        echo "# Trainable SNN Block Sweep"
        echo
        echo "- checkpoint: ${SWEEP_MODEL_PATH}"
        echo "- block range: ${MIN_BLOCKS}..${MAX_BLOCKS}"
        echo "- snn_timesteps: ${SNN_TIMESTEPS}"
        echo "- spike_position: ${SNN_SPIKE_POSITION}"
        echo "- spike_patch: ${SNN_SPIKE_PATCH}"
        echo "- epochs per run: ${SWEEP_EPOCHS}"
        echo "- distill_weight: ${SWEEP_DISTILL_WEIGHT}"
        echo "- nproc_per_node: ${NPROC_PER_NODE:-1}"
        echo "- cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-2}"
        echo
        echo "| blocks | indices | spike_position | patch | initial_acc1 | final_acc1 | best_acc1 | best_epoch | final_loss | output_dir |"
        echo "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
} > "${SUMMARY_TXT}"

for block_count in $(seq "${MIN_BLOCKS}" "${MAX_BLOCKS}"); do
        last_block=$((block_count - 1))
        block_indices="$(seq -s, 0 "${last_block}")"
        run_name="block_count_${block_count}_${SNN_SPIKE_POSITION}"
        if [ "${SNN_SPIKE_PATCH}" != "0" ]; then
                run_name="${run_name}_patch"
        fi
        output_dir="${OUTPUT_ROOT}/${run_name}"

        echo "===== Evaluating ${block_count} block spike(s): ${block_indices} ====="
        MODEL_PATH="${SWEEP_MODEL_PATH}" \
        OUTPUT_DIR="${output_dir}" \
        JOB_NAME="${run_name}" \
        SNN_BLOCK_INDICES="${block_indices}" \
        SNN_SPIKE_POSITION="${SNN_SPIKE_POSITION}" \
        SNN_SPIKE_PATCH="${SNN_SPIKE_PATCH}" \
        SNN_TIMESTEPS="${SNN_TIMESTEPS}" \
        EPOCHS="${SWEEP_EPOCHS}" \
        DISTILL_WEIGHT="${SWEEP_DISTILL_WEIGHT}" \
        DUMP_MODEL_SUMMARY="${SWEEP_DUMP_MODEL_SUMMARY}" \
        SKIP_INITIAL_BEST_CHECKPOINT="${SWEEP_SKIP_INITIAL_BEST_CHECKPOINT}" \
        bash "${BASE_LAUNCHER}"

        python - "${output_dir}/log.txt" "${SUMMARY_TXT}" "${block_count}" "${block_indices}" "${SNN_SPIKE_POSITION}" "${SNN_SPIKE_PATCH}" "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
block_count = sys.argv[3]
block_indices = sys.argv[4]
spike_position = sys.argv[5]
spike_patch = sys.argv[6]
output_dir = sys.argv[7]

rows = []
for line in log_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        rows.append(json.loads(line))

initial_rows = [row for row in rows if row.get("mode") == "initial_eval"]
if not initial_rows:
    raise SystemExit(f"No initial_eval row found in {log_path}")
initial = initial_rows[-1]
epoch_rows = [
    row
    for row in rows
    if row.get("mode") is None and "epoch" in row and "val_acc1" in row
]
if epoch_rows:
    final = epoch_rows[-1]
    best = max(epoch_rows, key=lambda row: row["val_acc1"])
else:
    final = initial
    best = initial

best_epoch = best.get("epoch", -1)

with summary_path.open("a", encoding="utf-8") as handle:
    handle.write(
        f"| {block_count} | `{block_indices}` | `{spike_position}` | {spike_patch} | "
        f"{initial['val_acc1']:.4f} | {final['val_acc1']:.4f} | {best['val_acc1']:.4f} | "
        f"{best_epoch} | {final['val_loss']:.4f} | `{output_dir}` |\n"
    )
PY
done

echo "Saved sweep summary to ${SUMMARY_TXT}"
