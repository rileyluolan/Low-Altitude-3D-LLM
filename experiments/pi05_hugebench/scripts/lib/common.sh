#!/usr/bin/env bash

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../configs/experiment.env
source "${COMMON_DIR}/../../configs/experiment.env"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '[pi05-hugebench] %s\n' "$*"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

prepare_data_links() {
  local split source target
  mkdir -p "${LEROBOT_HOME}/task_overall"
  for split in train test_seen test_unseen; do
    source="${HUGE_DATA_ROOT}/data_traj/${split}"
    target="${LEROBOT_HOME}/task_overall/${split}"
    require_dir "${source}"
    if [[ -L "${target}" ]]; then
      [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] \
        || die "Dataset link points elsewhere: ${target}"
    elif [[ -e "${target}" ]]; then
      die "Refusing to replace non-link dataset path: ${target}"
    else
      ln -s "${source}" "${target}"
    fi
  done
}

find_openpi_python() {
  local candidate
  if [[ -n "${OPENPI_PYTHON:-}" && -x "${OPENPI_PYTHON}" ]]; then
    printf '%s\n' "${OPENPI_PYTHON}"
    return 0
  fi
  for candidate in \
    "${OPENPI_ROOT}/.venv/bin/python" \
    "${XPOLICYLAB_ROOT}/.venv/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

has_openpi_runtime() {
  command -v uv >/dev/null 2>&1 || find_openpi_python >/dev/null
}

run_openpi() {
  if command -v uv >/dev/null 2>&1; then
    (cd "${OPENPI_ROOT}" && uv run "$@")
    return
  fi
  local python
  python="$(find_openpi_python)" || die \
    "OpenPI runtime not found. Restore ${OPENPI_ROOT}/.venv or install uv."
  (cd "${OPENPI_ROOT}" && "${python}" "$@")
}

find_renderer_python() {
  local candidate
  if [[ -n "${RENDERER_PYTHON:-}" && -x "${RENDERER_PYTHON}" ]]; then
    printf '%s\n' "${RENDERER_PYTHON}"
    return 0
  fi
  for candidate in \
    "${PROJECT_ROOT}/.venv/gaussian-splatting/bin/python" \
    "/home/aiscuser/miniconda3/envs/gaussian_splatting/bin/python" \
    "/opt/conda/envs/gaussian_splatting/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

latest_checkpoint_step() {
  local step
  require_dir "${TRAINED_CHECKPOINT_ROOT}"
  step="$(find "${TRAINED_CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+$/' | sort -n | tail -n 1)"
  [[ -n "${step}" ]] || die "No numeric checkpoint found in ${TRAINED_CHECKPOINT_ROOT}"
  printf '%s\n' "${step}"
}

checkpoint_path() {
  local step="${CHECKPOINT_STEP:-$(latest_checkpoint_step)}"
  local path="${TRAINED_CHECKPOINT_ROOT}/${step}"
  require_dir "${path}/params"
  require_dir "${path}/assets"
  printf '%s\n' "${path}"
}

evaluation_run_root() {
  local step="${CHECKPOINT_STEP:-$(latest_checkpoint_step)}"
  printf '%s/trained/%s/step_%s/seed_%s\n' \
    "${RESULTS_ROOT}" "${RUN_NAME}" "${step}" "${SEED}"
}

require_port_free() {
  local port="$1"
  if command -v ss >/dev/null 2>&1 && ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    die "TCP port ${port} is already in use"
  fi
}

print_identity() {
  local checkpoint
  checkpoint="$(checkpoint_path)"
  note "base weights: ${BASE_WEIGHTS_URI}"
  note "trained checkpoint: ${checkpoint}"
  note "dataset: ${LEROBOT_HOME}/task_overall"
}

