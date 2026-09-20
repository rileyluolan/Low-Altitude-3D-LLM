#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
export TRAIN_PROFILE=train
[[ "${RESUME:-0}" == 1 || ! -f "${EXPERIMENT_ROOT}/artifacts/checkpoints/${RUN_ID}/config.full.yaml" ]] || die 'Run already exists. Choose a new RUN_ID.'
# Real-sample forward/backward; no optimizer update and no checkpoint writes.
bash "${EXPERIMENT_ROOT}/scripts/preflight.sh"
exec bash "${EXPERIMENT_ROOT}/scripts/train.sh" "$@"
