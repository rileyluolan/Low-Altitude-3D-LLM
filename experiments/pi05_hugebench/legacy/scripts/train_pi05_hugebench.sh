#!/usr/bin/env bash
# Train XPolicyLab's pi05_base on HUGE-Bench task_overall/train.
#
# Usage:
#   bash train_pi05_hugebench.sh [config_name] [experiment_name]
#
# The XPolicyLab Pi_05 config supplies the canonical base-weight URI, dataset
# repo id, model shape, and checkpoint root. This wrapper handles first-run
# versus resume selection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pi05env.sh
source "${SCRIPT_DIR}/pi05env.sh"

CONFIG="${1:-${PI05_CONFIG}}"
EXP="${2:-${CONFIG}_run1}"
CHECKPOINT_DIR="${PI05_CKPT}/${CONFIG}/${EXP}"

require_openpi_runtime
prepare_huge_lerobot
[[ -d "${OPENPI}" ]] || { echo "[ERR] XPolicyLab openpi is missing: ${OPENPI}" >&2; exit 1; }
grep -Fq "${PI05_BASE_WEIGHTS}" "${OPENPI}/src/openpi/training/config.py" || {
  echo "[ERR] XPolicyLab config does not reference ${PI05_BASE_WEIGHTS}." >&2
  exit 1
}
grep -Fq 'repo_id="task_overall/train"' "${OPENPI}/src/openpi/training/config.py" || {
  echo "[ERR] XPolicyLab config is not targeting HUGE-Bench task_overall/train." >&2
  exit 1
}

MODE_ARGS=()
if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
  MODE_ARGS+=(--overwrite)
elif [[ -d "${CHECKPOINT_DIR}" ]] && find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name '[0-9]*' -print -quit | grep -q .; then
  MODE_ARGS+=(--resume)
elif [[ -d "${CHECKPOINT_DIR}" ]]; then
  MODE_ARGS+=(--overwrite)
fi

echo "[INFO] implementation : ${PI05_BASE_WEIGHTS_OWNER}"
echo "[INFO] base weights   : ${PI05_BASE_WEIGHTS}"
echo "[INFO] config         : ${CONFIG}"
echo "[INFO] dataset        : ${DATA}/data_traj/train (task_overall/train)"
echo "[INFO] experiment     : ${EXP}"
echo "[INFO] checkpoint dir : ${CHECKPOINT_DIR}"
echo "[INFO] mode           : ${MODE_ARGS[*]:-new run}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1

cd "${OPENPI}"
if [[ -n "${UV_BIN}" ]]; then
  exec "${UV_BIN}" run scripts/train.py "${CONFIG}" \
    --exp-name="${EXP}" \
    --num-workers="${NUM_WORKERS:-16}" \
    "${MODE_ARGS[@]}"
else
  exec "${OPENPI_PYTHON}" scripts/train.py "${CONFIG}" \
    --exp-name="${EXP}" \
    --num-workers="${NUM_WORKERS:-16}" \
    "${MODE_ARGS[@]}"
fi
