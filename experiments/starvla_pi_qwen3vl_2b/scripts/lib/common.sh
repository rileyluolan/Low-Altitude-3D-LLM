#!/usr/bin/env bash
# Source this file before running the Python tools directly.
COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
if [[ -f "${EXPERIMENT_ENV_FILE:-${EXPERIMENT_ROOT}/env.local}" ]]; then
  source "${EXPERIMENT_ENV_FILE:-${EXPERIMENT_ROOT}/env.local}"
fi
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${EXPERIMENT_ROOT}/../.." && pwd)}"
export STARVLA_ROOT="${STARVLA_ROOT:-${EXPERIMENT_ROOT}/third_party/starVLA}"
export HUGEBENCH_ROOT="${HUGEBENCH_ROOT:-${EXPERIMENT_ROOT}/third_party/HUGE-Bench}"
export GAUSSIAN_SPLATTING_ROOT="${GAUSSIAN_SPLATTING_ROOT:-${EXPERIMENT_ROOT}/third_party/gaussian-splatting}"
export HUGE_DATA_ROOT="${HUGE_DATA_ROOT:-${EXPERIMENT_ROOT}/data/HUGE_data}"
export MODEL_ROOT="${MODEL_ROOT:-${EXPERIMENT_ROOT}/models/Qwen3-VL-2B-Instruct}"
export BASE_CKPT="${BASE_CKPT:-${EXPERIMENT_ROOT}/artifacts/base/qwen3vl_2b_pi_v3_hugebench_mature_base/checkpoints/base_pytorch_model.pt}"
export VENV_ROOT="${VENV_ROOT:-${EXPERIMENT_ROOT}/.venv}"
export PATH="${VENV_ROOT}/bin:${CUDA_HOME:+${CUDA_HOME}/bin:}${PATH}"
export PYTHONPATH="${EXPERIMENT_ROOT}/scripts/tools:${STARVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${EXPERIMENT_ROOT}/.cache"
export TMPDIR="${EXPERIMENT_ROOT}/.tmp"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip" HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TORCH_HOME="${XDG_CACHE_HOME}/torch" TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton" CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib" CONDA_PKGS_DIRS="${XDG_CACHE_HOME}/conda"
export CONDA_REGISTER_ENVS=false WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export RUN_ID="${RUN_ID:-qwen3vl_2b_pi_v3_hugebench_run1}" SEED="${SEED:-42}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LOGS_ROOT="${EXPERIMENT_ROOT}/logs" RESULTS_ROOT="${EXPERIMENT_ROOT}/results"
mkdir -p "${TMPDIR}" "${CUDA_CACHE_PATH}" "${LOGS_ROOT}"
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_env() { [[ -x "${VENV_ROOT}/bin/python" ]] || die 'Run scripts/setup_environment.sh first.'; }
resolve_profile() {
  RUNTIME_CONFIG_DIR="${EXPERIMENT_ROOT}/runtime/${1}/${RUN_ID}"
  export RUNTIME_CONFIG_DIR
  "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/resolve_config.py" "$1" --output "${RUNTIME_CONFIG_DIR}" "${@:2}"
}
