#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/run_eval.py" "$@"
