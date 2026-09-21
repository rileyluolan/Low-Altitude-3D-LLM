#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

RUN_ROOT="$(evaluation_run_root)"
ROLLOUT_DIR="${1:-${RUN_ROOT}/rollouts}"
METRICS_DIR="${2:-${RUN_ROOT}/metrics}"
METRIC_PY="${METRIC_PYTHON:-}"

require_dir "${ROLLOUT_DIR}"
if [[ -z "${METRIC_PY}" ]]; then
  METRIC_PY="$(find_renderer_python)" || die \
    "Metric runtime not found. Export METRIC_PYTHON to a Python with numpy, tqdm and trimesh."
fi
[[ -x "${METRIC_PY}" ]] || die "Metric Python is not executable: ${METRIC_PY}"

mkdir -p "${METRICS_DIR}"
GENERIC_JSON="${METRICS_DIR}/metric.json"
PAPER_JSON="${METRICS_DIR}/paper_aligned.json"

"${METRIC_PY}" "${HUGEBENCH_ROOT}/metric.py" \
  --out_dir "${ROLLOUT_DIR}" \
  --tasks overall \
  --mesh_root "${HUGE_DATA_ROOT}/data_3d" \
  --mesh_rel terra_ply/simplified_mesh.obj \
  --tcr_thresholds 1,2,5 \
  --json_out "${GENERIC_JSON}" \
  2>&1 | tee "${METRICS_DIR}/metric.log"

"${METRIC_PY}" "${HUGEBENCH_ROOT}/metric_paper_aligned.py" \
  --out_dir "${ROLLOUT_DIR}" \
  --tasks overall \
  --mesh_root "${HUGE_DATA_ROOT}/data_3d" \
  --mesh_rel terra_ply/simplified_mesh.obj \
  --json_out "${PAPER_JSON}" \
  2>&1 | tee "${METRICS_DIR}/paper_aligned.log"

note "generic metrics: ${GENERIC_JSON}"
note "paper-aligned metrics: ${PAPER_JSON}"

