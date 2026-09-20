#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
bash "${EXPERIMENT_ROOT}/scripts/prepare_data.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
gpu_list="${TRAIN_GPUS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${gpu_list%%,*}"
exec "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/preflight_training.py"
