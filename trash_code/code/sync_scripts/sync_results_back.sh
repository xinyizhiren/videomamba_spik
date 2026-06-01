#!/bin/bash
set -euo pipefail

: "${REMOTE:?Set REMOTE, for example user@server}"
: "${REMOTE_DIR:?Set REMOTE_DIR, for example /path/to/video_sm}"

LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-outputs_server}"
mkdir -p "${LOCAL_RESULTS_DIR}"

rsync -avz "${REMOTE}:${REMOTE_DIR%/}/outputs/" "${LOCAL_RESULTS_DIR}/"
