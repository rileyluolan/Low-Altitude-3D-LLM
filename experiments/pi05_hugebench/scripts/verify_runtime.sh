#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

OPENPI_PY="$(find_openpi_python)" || die "OpenPI Python is unavailable"
GS_PY="$(find_renderer_python)" || die "Gaussian Splatting Python is unavailable"

if env | grep -q '^LEROBOT_HOME='; then
  die "Deprecated LEROBOT_HOME is exported; only HF_LEROBOT_HOME may be exported"
fi

"${OPENPI_PY}" -c '
import jax, jaxlib, lerobot, numpy, openpi, pyarrow, torch
from openpi.training import config
c = config.get_config("pi05_overall")
assert c.model.action_horizon == 20
print("OpenPI OK", jax.__version__, jaxlib.__version__, torch.__version__, numpy.__version__)
'

PYTHONPATH="${GAUSSIAN_SPLATTING_ROOT}:${HUGEBENCH_ROOT}:${HUGEBENCH_ROOT}/gaussian_splatting" \
"${GS_PY}" -c '
import diff_gaussian_rasterization, fused_ssim, plyfile, rtree, torch, trimesh
from simple_knn import _C
print("3DGS OK", torch.__version__, torch.version.cuda, trimesh.__version__)
'

note "runtime imports and experiment configuration are valid"
