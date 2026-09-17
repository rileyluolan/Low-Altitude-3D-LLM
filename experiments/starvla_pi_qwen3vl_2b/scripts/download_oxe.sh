#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/pretraining/download_oxe_metadata.py"
exec "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/pretraining/download_oxe.py" "$@"
