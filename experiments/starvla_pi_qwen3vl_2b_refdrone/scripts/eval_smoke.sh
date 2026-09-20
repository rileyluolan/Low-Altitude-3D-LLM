#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
# Explicitly label plumbing checks of the available initialization checkpoint.
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BASE_CKPT}}"
export EVAL_TAG="${EVAL_TAG:-smoke_$(date -u +%Y%m%dT%H%M%S)}"
export NUM_TRAJS="${NUM_TRAJS:-2}"
export MAX_STEPS="${MAX_STEPS:-3}"
exec "${EXPERIMENT_ROOT}/scripts/eval.sh" "$@"
