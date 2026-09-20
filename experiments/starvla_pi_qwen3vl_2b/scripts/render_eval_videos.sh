#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
exec "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/render_saved_rollouts.py" "$@"
