#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  [[ -n "${CONDA_BIN}" ]] || die 'Install conda or set CONDA_BIN; Python 3.10 is required.'
  "${CONDA_BIN}" create --yes --prefix "${VENV_ROOT}" python=3.10 pip
fi
python="${VENV_ROOT}/bin/python"
# These compilers are also needed by small Python dependencies built from source.
export CUDA_HOME="${CUDA_HOME:?Set CUDA_HOME to a CUDA 12.4 toolkit}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export PATH="${VENV_ROOT}/bin:${CUDA_HOME}/bin:${PATH}"
"${python}" -m pip install -c "${EXPERIMENT_ROOT}/configs/environment_lock.txt" 'setuptools==80.9.0' wheel ninja
"${python}" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"${python}" -m pip install -c "${EXPERIMENT_ROOT}/configs/environment_lock.txt" \
  -r "${EXPERIMENT_ROOT}/configs/requirements_train.txt" -r "${EXPERIMENT_ROOT}/configs/requirements_eval.txt"
# This experiment imports only PyTorch3D's pure-Python rotation transforms.
# Its renderer uses the separately compiled Gaussian Splatting extensions.
PYTORCH3D_NO_EXTENSION=1 "${python}" -m pip install -c "${EXPERIMENT_ROOT}/configs/environment_lock.txt" --no-build-isolation "${EXPERIMENT_ROOT}/third_party/pytorch3d"
"${python}" -m pip install --no-deps --no-build-isolation -e "${STARVLA_ROOT}"
# Reuse the installed CUDA 12.4 compiler read-only; generated files stay here.
export CUDA_HOME="${CUDA_HOME:?Set CUDA_HOME to a CUDA 12.4 toolkit}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] && command -v "${CXX}" >/dev/null || die 'Set CUDA_HOME, CC and CXX to a CUDA 12.4 toolkit and compiler.'
for module in diff-gaussian-rasterization simple-knn; do
  "${python}" -m pip install --no-deps --no-build-isolation "${GAUSSIAN_SPLATTING_ROOT}/submodules/${module}"
done
"${python}" -m pip check
mkdir -p "${EXPERIMENT_ROOT}/runtime"
"${python}" -m pip freeze > "${EXPERIMENT_ROOT}/runtime/environment_installed.txt"
printf 'Environment ready: %s\n' "${VENV_ROOT}"
