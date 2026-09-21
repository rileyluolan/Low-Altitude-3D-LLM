#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

prepare_data_links
require_dir "${OPENPI_ROOT}/src/openpi"
has_openpi_runtime || die \
  "OpenPI runtime not found. Restore ${OPENPI_ROOT}/.venv or install uv."

grep -q "name=\"${TRAIN_CONFIG}\"" "${OPENPI_ROOT}/src/openpi/training/config.py" \
  || die "Training config ${TRAIN_CONFIG} is missing"
grep -q 'gs://openpi-assets/checkpoints/pi05_base/params' \
  "${OPENPI_ROOT}/src/openpi/training/config.py" \
  || die "Training config does not declare the expected pi0.5-base weights"
grep -q 'repo_id="task_overall/train"' "${OPENPI_ROOT}/src/openpi/training/config.py" \
  || die "Training config does not target HUGE-Bench task_overall/train"

mkdir -p "${CHECKPOINTS_ROOT}" "${ASSETS_ROOT}" "${LOGS_ROOT}/train"
TRAIN_LOG="${LOGS_ROOT}/train/${RUN_NAME}.log"
MODE_ARGS=()
if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
  MODE_ARGS+=(--overwrite)
elif [[ -d "${TRAINED_CHECKPOINT_ROOT}" ]] \
  && find "${TRAINED_CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit | grep -q .; then
  MODE_ARGS+=(--resume)
elif [[ -d "${TRAINED_CHECKPOINT_ROOT}" ]]; then
  MODE_ARGS+=(--overwrite)
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

note "base weights: ${BASE_WEIGHTS_URI}"
note "training: ${TRAIN_CONFIG}/${RUN_NAME} (${MODE_ARGS[*]:-new run})"
note "checkpoint root: ${TRAINED_CHECKPOINT_ROOT}"
note "log: ${TRAIN_LOG}"
run_openpi scripts/train.py "${TRAIN_CONFIG}" \
  --exp-name="${RUN_NAME}" \
  --num-workers="${NUM_WORKERS:-16}" \
  "${MODE_ARGS[@]}" 2>&1 | tee -a "${TRAIN_LOG}"
