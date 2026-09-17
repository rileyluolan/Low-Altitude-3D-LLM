#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
"${EXPERIMENT_ROOT}/scripts/prepare_data.sh"
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/validate_base.py" --verify-hash
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/validate_integration.py"
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/validate_runtime.py"
