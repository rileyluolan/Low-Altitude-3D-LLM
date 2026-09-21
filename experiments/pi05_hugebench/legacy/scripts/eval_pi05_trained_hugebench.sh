#!/usr/bin/env bash
# Evaluate the newly trained XPolicyLab Pi_05 model on HUGE-Bench.
#
# Usage:
#   bash eval_pi05_trained_hugebench.sh [out_dir] [splits]
#
# The checkpoint is intentionally not a positional argument.  This entry point
# always evaluates PI05_TRAINED_CKPT from pi05env.sh, never the official
# HUGE_PI05 reference checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pi05env.sh
source "${SCRIPT_DIR}/pi05env.sh"

OUT="${1:-${DATA}/rollout_pi05_overall_run1}"
SPLITS="${2:-test_seen,test_unseen}"
CHECKPOINT="${PI05_TRAINED_CKPT}"

[[ -d "${CHECKPOINT}" ]] || {
  echo "[ERR] Trained checkpoint root is missing: ${CHECKPOINT}" >&2
  exit 1
}
find "${CHECKPOINT}" -maxdepth 1 -type d -name '[0-9]*' -print -quit | grep -q . || {
  echo "[ERR] No numeric checkpoint step exists under: ${CHECKPOINT}" >&2
  exit 1
}
[[ -f "${HUGE}/metric.py" && -f "${HUGE}/metric_paper_aligned.py" ]] || {
  echo "[ERR] HUGE-Bench metric scripts are missing under: ${HUGE}" >&2
  exit 1
}
prepare_huge_lerobot

echo "[INFO] model        : ${PI05_TRAINED_CKPT} (latest numeric step selected by action_infer.py)"
echo "[INFO] initialized  : ${PI05_BASE_WEIGHTS}"
echo "[INFO] implementation: ${PI05_BASE_WEIGHTS_OWNER}"
echo "[INFO] config       : ${PI05_CONFIG}"
echo "[INFO] dataset      : ${DATA}/data_traj/{${SPLITS}}"
echo "[INFO] rollout out  : ${OUT}"

"${SCRIPT_DIR}/eval_pi05_4gpu.sh" \
  "${CHECKPOINT}" \
  "${OUT}" \
  "${PI05_CONFIG}" \
  "${SPLITS}"

episodes="$(find "${OUT}" -type f -name traj_gt_pred_xyzk.npz | wc -l)"
(( episodes > 0 )) || {
  echo "[ERR] Rollout produced no trajectory files under: ${OUT}" >&2
  exit 1
}

if [[ -n "${METRIC_PYTHON:-}" ]]; then
  METRIC_RUNNER="${METRIC_PYTHON}"
elif [[ -n "${GS_PYTHON:-}" && -x "${GS_PYTHON}" ]]; then
  METRIC_RUNNER="${GS_PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  METRIC_RUNNER="$(command -v python3)"
else
  echo "[ERR] No Python interpreter available for HUGE-Bench metrics." >&2
  exit 1
fi

MESH_ROOT="${MESH_ROOT:-${DATA}/data_3d}"
MESH_REL="${MESH_REL:-terra_ply/simplified_mesh.obj}"
METRIC_JSON="${METRIC_JSON:-${OUT}/metric.json}"
PAPER_JSON="${PAPER_JSON:-${OUT}/metric_paper_aligned.json}"

echo "[INFO] metric python: ${METRIC_RUNNER}"
"${METRIC_RUNNER}" "${HUGE}/metric.py" \
  --out_dir "${OUT}" \
  --tasks overall \
  --mesh_root "${MESH_ROOT}" \
  --mesh_rel "${MESH_REL}" \
  --tcr_thresholds "${TCR_THRESHOLDS:-1,2,5}" \
  --json_out "${METRIC_JSON}"

"${METRIC_RUNNER}" "${HUGE}/metric_paper_aligned.py" \
  --out_dir "${OUT}" \
  --tasks overall \
  --json_out "${PAPER_JSON}"

echo "[OK] Evaluated ${episodes} trajectories from the newly trained checkpoint."
echo "[OK] Metrics: ${METRIC_JSON}"
echo "[OK] Paper metrics: ${PAPER_JSON}"
