#!/bin/bash
set -euo pipefail

: "${REMOTE:?Set REMOTE, for example user@server}"
: "${REMOTE_DIR:?Set REMOTE_DIR, for example /path/to/video_sm}"

DRY_RUN="${DRY_RUN:-1}"
DRY_RUN_ARGS=()
if [ "${DRY_RUN}" != "0" ]; then
        DRY_RUN_ARGS=(--dry-run)
fi

rsync -avz --delete "${DRY_RUN_ARGS[@]}" \
        --exclude data \
        --exclude local_data \
        --exclude outputs \
        --exclude outputs_server \
        --exclude checkpoints \
        --exclude logs \
        --exclude runs \
        --exclude wandb \
        --exclude tensorboard \
        --exclude __pycache__ \
        --exclude "*.pyc" \
        --exclude "*.pth" \
        --exclude "*.pt" \
        --exclude "*.ckpt" \
        ./ "${REMOTE}:${REMOTE_DIR}/"
