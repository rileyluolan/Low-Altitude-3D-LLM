#!/usr/bin/env bash
# Shared environment for the XPolicyLab Pi_05 + HUGE-Bench experiment.
#
# This file deliberately lives in workspace-ll.  Do not replace it with a
# home-directory copy: all paths used by the train/eval entry points are
# resolved from this file so that the base checkpoint and the output model are
# unambiguous.

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  WORKSPACE_ROOT="/home/aiscuser/workspace-ll"
fi

export WORKSPACE_ROOT
export XPL_ROOT="${WORKSPACE_ROOT}/XPolicyLab"
export OPENPI="${XPL_ROOT}/policy/Pi_05/openpi"
export HUGE="${WORKSPACE_ROOT}/HUGE-Bench"
export DATA="${WORKSPACE_ROOT}/HUGE_data"

# This is the XPolicyLab Pi_05 base checkpoint used to initialize training.
# It is intentionally recorded as a single canonical value in the workspace.
export PI05_BASE_WEIGHTS="gs://openpi-assets/checkpoints/pi05_base/params"
export PI05_BASE_WEIGHTS_OWNER="XPolicyLab/policy/Pi_05"

# The new model trained on HUGE-Bench task_overall/train.
export PI05_CKPT="${WORKSPACE_ROOT}/pi05_ckpts"
export PI05_TRAINED_CKPT="${PI05_CKPT}/pi05_overall/pi05_overall_run1"
export PI05_CONFIG="pi05_overall"

# LeRobot v2.1 resolves local datasets below HF_LEROBOT_HOME/<repo_id>.
export HF_LEROBOT_HOME="${WORKSPACE_ROOT}/lerobot_home"
export HF_HUB_CACHE="${WORKSPACE_ROOT}/hf_cache/hub"
export HF_DATASETS_CACHE="${WORKSPACE_ROOT}/openpi_cache/hf-datasets"
export JAX_COMPILATION_CACHE_DIR="${WORKSPACE_ROOT}/openpi_cache/jax"
export PI05_ASSETS="${WORKSPACE_ROOT}/pi05_assets"
export PATH="${WORKSPACE_ROOT}/.tools/bin:${PATH}"

mkdir -p "${HF_LEROBOT_HOME}" "${HF_DATASETS_CACHE}" "${JAX_COMPILATION_CACHE_DIR}" "${PI05_ASSETS}" "${PI05_CKPT}"

# Prefer uv when it is installed.  The fallback is useful on nodes where the
# uv-managed interpreter has not been recreated yet.
if command -v uv >/dev/null 2>&1; then
  export UV_BIN="$(command -v uv)"
else
  export UV_BIN=""
fi
if [[ -x "${OPENPI}/.venv/bin/python" ]]; then
  export OPENPI_PYTHON="${OPENPI}/.venv/bin/python"
else
  export OPENPI_PYTHON=""
fi

# Set GS_PYTHON explicitly when the 3DGS environment is not in the historical
# miniconda location.  Leaving it empty makes eval_pi05_4gpu.sh fail early with
# an actionable message instead of starting incomplete renderer processes.
if [[ -z "${GS_PYTHON:-}" ]]; then
  for candidate in \
    "${WORKSPACE_ROOT}/.venv/bin/python" \
    "/home/aiscuser/miniconda3/envs/gaussian_splatting/bin/python" \
    "/opt/conda/envs/gaussian_splatting/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      export GS_PYTHON="${candidate}"
      break
    fi
  done
fi

prepare_huge_lerobot() {
  local split source target
  mkdir -p "${HF_LEROBOT_HOME}/task_overall"
  for split in train test_seen test_unseen; do
    source="${DATA}/data_traj/${split}"
    target="${HF_LEROBOT_HOME}/task_overall/${split}"
    [[ -d "${source}" ]] || {
      echo "[ERR] HUGE-Bench split is missing: ${source}" >&2
      return 1
    }
    if [[ -e "${target}" && ! -L "${target}" ]]; then
      echo "[ERR] Refusing to replace non-symlink dataset path: ${target}" >&2
      return 1
    fi
    ln -sfn "${source}" "${target}"
  done
}

require_openpi_runtime() {
  if [[ -n "${UV_BIN}" ]]; then
    return 0
  fi
  if [[ -n "${OPENPI_PYTHON}" ]]; then
    return 0
  fi
  cat >&2 <<EOF
[ERR] No runnable XPolicyLab Pi_05 Python environment found.
      Expected uv or ${OPENPI}/.venv/bin/python.
      Recreate the environment in ${OPENPI}, then rerun this entry point.
EOF
  return 1
}

openpi_run() {
  require_openpi_runtime
  if [[ -n "${UV_BIN}" ]]; then
    (cd "${OPENPI}" && "${UV_BIN}" run "$@")
  else
    (cd "${OPENPI}" && "${OPENPI_PYTHON}" "$@")
  fi
}
