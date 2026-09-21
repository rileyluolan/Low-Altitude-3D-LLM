#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
OPENPI_ROOT="${WORKSPACE_ROOT}/XPolicyLab/policy/Pi_05/openpi"
GS_ROOT="${WORKSPACE_ROOT}/gaussian-splatting"
TOOLS_BIN="${WORKSPACE_ROOT}/.tools/bin"
UV_BIN="${TOOLS_BIN}/uv"
UV_CACHE_DIR="${PROJECT_ROOT}/artifacts/cache/uv"
PYTHON_INSTALL_DIR="${PROJECT_ROOT}/.python"
OPENPI_ENV="${OPENPI_ROOT}/.venv"
GS_ENV="${PROJECT_ROOT}/.venv/gaussian-splatting"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

mkdir -p "${TOOLS_BIN}" "${UV_CACHE_DIR}" "${PYTHON_INSTALL_DIR}" "${PROJECT_ROOT}/.venv"

if [[ ! -x "${UV_BIN}" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/pi05-install-uv.sh
  UV_INSTALL_DIR="${TOOLS_BIN}" sh /tmp/pi05-install-uv.sh
fi

env UV_CACHE_DIR="${UV_CACHE_DIR}" \
  "${UV_BIN}" python install 3.11 --install-dir "${PYTHON_INSTALL_DIR}" --no-bin

MANAGED_PYTHON="${PYTHON_INSTALL_DIR}/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
[[ -x "${MANAGED_PYTHON}" ]] || die "Managed CPython 3.11 was not installed"

if [[ -d "${OPENPI_ENV}/lib/python3.11/site-packages" ]]; then
  [[ -e "${OPENPI_ENV}/bin/python3.11" ]] || ln -s "${MANAGED_PYTHON}" "${OPENPI_ENV}/bin/python3.11"
  [[ -e "${OPENPI_ENV}/bin/python3" ]] || ln -s python3.11 "${OPENPI_ENV}/bin/python3"
  [[ -e "${OPENPI_ENV}/bin/python" ]] || ln -s python3.11 "${OPENPI_ENV}/bin/python"
else
  env UV_CACHE_DIR="${UV_CACHE_DIR}" \
    UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}" \
    UV_PROJECT_ENVIRONMENT="${OPENPI_ENV}" \
    "${UV_BIN}" sync --frozen --group lerobot --project "${OPENPI_ROOT}"
fi

if [[ ! -x "${GS_ENV}/bin/python" ]]; then
  [[ -x /opt/conda/envs/ptca/bin/python ]] \
    || die "Missing /opt/conda/envs/ptca; cannot seed the CUDA 12.6 renderer environment"
  /opt/conda/bin/conda create --prefix "${GS_ENV}" --clone /opt/conda/envs/ptca -y
fi

env UV_CACHE_DIR="${UV_CACHE_DIR}" \
  "${UV_BIN}" pip install \
  --python "${GS_ENV}/bin/python" \
  --requirement "${PROJECT_ROOT}/configs/gaussian-splatting-requirements.txt"

install_extension() {
  local import_name="$1"
  local source_dir="$2"
  # Import torch first so its shared libraries (libc10/libtorch) are visible to
  # extension modules such as simple_knn._C.
  if PYTHONPATH="${GS_ROOT}" "${GS_ENV}/bin/python" -c "import torch; import ${import_name}" >/dev/null 2>&1; then
    return
  fi
  env CUDA_HOME=/usr/local/cuda-12.6 \
    TORCH_CUDA_ARCH_LIST=8.0 \
    MAX_JOBS="${MAX_JOBS:-8}" \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
    "${UV_BIN}" pip install \
    --python "${GS_ENV}/bin/python" \
    --no-build-isolation \
    --reinstall "${source_dir}"
}

install_extension diff_gaussian_rasterization \
  "${GS_ROOT}/submodules/diff-gaussian-rasterization"
install_extension simple_knn._C \
  "${GS_ROOT}/submodules/simple-knn"
install_extension fused_ssim \
  "${GS_ROOT}/submodules/fused-ssim"

bash "${SCRIPT_DIR}/verify_runtime.sh"
