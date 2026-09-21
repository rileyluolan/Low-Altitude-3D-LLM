#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

STEP="${CHECKPOINT_STEP:-$(latest_checkpoint_step)}"
SMOKE_ROOT="${RESULTS_ROOT}/smoke/${RUN_NAME}/step_${STEP}/seed_${SEED}"
export NUM_TRAJS="${NUM_TRAJS:-4}"
export NUM_SHARDS="${NUM_SHARDS:-1}"

"${SCRIPT_DIR}/eval_rollout.sh" "${SPLITS:-test_seen}" "${SMOKE_ROOT}/rollouts"
"${SCRIPT_DIR}/eval_metrics.sh" "${SMOKE_ROOT}/rollouts" "${SMOKE_ROOT}/metrics"
