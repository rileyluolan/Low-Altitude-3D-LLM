#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

RUN_ROOT="$(evaluation_run_root)"
"${SCRIPT_DIR}/eval_rollout.sh" "${SPLITS:-test_seen,test_unseen}" "${RUN_ROOT}/rollouts"
"${SCRIPT_DIR}/eval_metrics.sh" "${RUN_ROOT}/rollouts" "${RUN_ROOT}/metrics"

