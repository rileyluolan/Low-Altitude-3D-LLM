#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
exec "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/pretraining/build_oxe_base_checkpoint.py" "$@"
