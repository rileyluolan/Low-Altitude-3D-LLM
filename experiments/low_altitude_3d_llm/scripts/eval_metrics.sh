#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
[[ $# -ge 1 ]] || die 'Usage: scripts/eval_metrics.sh /absolute/path/to/rollouts [official metric options]'
out_dir="$1"
shift
exec "${VENV_ROOT}/bin/python" "${HUGEBENCH_ROOT}/metric.py" \
  --out_dir "${out_dir}" --mesh_root "${HUGE_DATA_ROOT}/data_3d" \
  --mesh_rel terra_ply/simplified_mesh.obj --tcr_thresholds 1,2,5 \
  --json_out "${out_dir}/../metrics.json" "$@"
