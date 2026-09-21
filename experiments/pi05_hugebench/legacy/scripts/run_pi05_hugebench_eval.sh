#!/usr/bin/env bash
# Roll out the newly trained XPolicyLab Pi_05 checkpoint on HUGE-Bench and
# calculate benchmark metrics.
#
# Usage:
#   bash run_pi05_hugebench_eval.sh [out_dir] [splits]
#
# There is intentionally no checkpoint positional argument: this entry point
# always evaluates PI05_TRAINED_CKPT from pi05env.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pi05env.sh
source "${SCRIPT_DIR}/pi05env.sh"

OUT="${1:-${DATA}/rollout_pi05_overall_run1}"
SPLITS="${2:-test_seen,test_unseen}"
CHECKPOINT="${PI05_TRAINED_CKPT}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-0}"
EXEC_STEPS="${EXEC_STEPS:-10}"
BASE_PORT="${BASE_PORT:-5550}"
NUM_TRAJS="${NUM_TRAJS:-0}"
GS_ROOT="${WORKSPACE_ROOT}/gaussian-splatting"
GS_RUNTIME="${GS_PYTHON:-${GS_PY:-}}"

require_openpi_runtime
prepare_huge_lerobot
[[ -d "${CHECKPOINT}" ]] || { echo "[ERR] Trained checkpoint is missing: ${CHECKPOINT}" >&2; exit 1; }
find "${CHECKPOINT}" -maxdepth 1 -type d -name '[0-9]*' -print -quit | grep -q . || {
  echo "[ERR] No numeric step checkpoint under: ${CHECKPOINT}" >&2
  exit 1
}
[[ -f "${GS_ROOT}/3dgs_renderer.py" ]] || {
  echo "[ERR] 3DGS renderer is missing: ${GS_ROOT}/3dgs_renderer.py" >&2
  exit 1
}
[[ -n "${GS_RUNTIME}" && -x "${GS_RUNTIME}" ]] || {
  echo "[ERR] Set GS_PYTHON to a Python environment with 3DGS dependencies." >&2
  echo "      Example: GS_PYTHON=/path/to/gaussian_splatting/bin/python bash $0" >&2
  exit 1
}
(( NUM_SHARDS >= 1 && NUM_SHARDS <= 4 )) || {
  echo "[ERR] NUM_SHARDS must be in [1,4] on the 8-GPU layout." >&2
  exit 1
}

LOG_DIR="${OUT}_logs"
mkdir -p "${OUT}" "${LOG_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

echo "[INFO] model         : ${CHECKPOINT} (latest numeric step selected)"
echo "[INFO] initialized   : ${PI05_BASE_WEIGHTS}"
echo "[INFO] implementation: ${PI05_BASE_WEIGHTS_OWNER}"
echo "[INFO] config        : ${PI05_CONFIG}"
echo "[INFO] data          : ${DATA}/data_traj/{test_seen,test_unseen}"
echo "[INFO] output        : ${OUT}"
echo "[INFO] shards/seed   : ${NUM_SHARDS}/${SEED}; exec_steps=${EXEC_STEPS}"

for worker in $(seq 0 $((NUM_SHARDS - 1))); do
  port=$((BASE_PORT + worker))
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "[ERR] Port ${port} is already in use." >&2
    exit 1
  fi
done

pids=()
cleanup() {
  trap - INT TERM EXIT
  if ((${#pids[@]})); then
    echo "[INFO] stopping child processes..."
    kill -TERM "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

for worker in $(seq 0 $((NUM_SHARDS - 1))); do
  port=$((BASE_PORT + worker))
  (
    export CUDA_VISIBLE_DEVICES="${worker}"
    "${GS_RUNTIME}" "${GS_ROOT}/3dgs_renderer.py" \
      --host 127.0.0.1 \
      --port "${port}" \
      --ply_template "${DATA}/data_3d/{env_id}/3dgs_ply/point_cloud_utm50.ply"
  ) >"${LOG_DIR}/renderer_gpu${worker}.log" 2>&1 &
  pids+=("$!")
done

echo "[INFO] waiting for ${NUM_SHARDS} renderers..."
for worker in $(seq 0 $((NUM_SHARDS - 1))); do
  port=$((BASE_PORT + worker))
  ready=0
  for _ in $(seq 1 120); do
    if ! kill -0 "${pids[$worker]}" 2>/dev/null; then
      echo "[ERR] renderer ${worker} exited; see ${LOG_DIR}/renderer_gpu${worker}.log" >&2
      exit 1
    fi
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  (( ready == 1 )) || {
    echo "[ERR] renderer ${worker} did not listen on ${port}." >&2
    exit 1
  }
done

eval_pids=()
for worker in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((NUM_SHARDS + worker))
  port=$((BASE_PORT + worker))
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    cd "${OPENPI}"
    infer_args=(
      scripts/action_infer.py
      --task_id overall
      --config_name "${PI05_CONFIG}"
      --checkpoint_dir "${CHECKPOINT}"
      --splits "${SPLITS}"
      --exec_steps "${EXEC_STEPS}"
      --num_trajs "${NUM_TRAJS}"
      --num_shards "${NUM_SHARDS}"
      --shard_index "${worker}"
      --seed "${SEED}"
      --out_dir "${OUT}"
      --host 127.0.0.1
      --port "${port}"
    )
    if [[ -n "${UV_BIN}" ]]; then
      "${UV_BIN}" run "${infer_args[@]}"
    else
      "${OPENPI_PYTHON}" "${infer_args[@]}"
    fi
  ) >"${LOG_DIR}/eval_shard${worker}_gpu${gpu}.log" 2>&1 &
  eval_pids+=("$!")
  pids+=("$!")
done

status=0
for pid in "${eval_pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || {
  echo "[ERR] rollout failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
}

episodes="$(find "${OUT}" -type f -name traj_gt_pred_xyzk.npz | wc -l)"
(( episodes > 0 )) || { echo "[ERR] No rollout trajectories found under ${OUT}" >&2; exit 1; }

if [[ -n "${METRIC_PYTHON:-}" ]]; then
  METRIC_RUNTIME="${METRIC_PYTHON}"
elif [[ -x "${GS_RUNTIME}" ]]; then
  METRIC_RUNTIME="${GS_RUNTIME}"
elif command -v python3 >/dev/null 2>&1; then
  METRIC_RUNTIME="$(command -v python3)"
else
  echo "[ERR] No Python interpreter for HUGE-Bench metrics." >&2
  exit 1
fi

MESH_ROOT="${MESH_ROOT:-${DATA}/data_3d}"
MESH_REL="${MESH_REL:-terra_ply/simplified_mesh.obj}"
METRIC_JSON="${METRIC_JSON:-${OUT}/metric.json}"
PAPER_JSON="${PAPER_JSON:-${OUT}/metric_paper_aligned.json}"

"${METRIC_RUNTIME}" "${HUGE}/metric.py" \
  --out_dir "${OUT}" \
  --tasks overall \
  --mesh_root "${MESH_ROOT}" \
  --mesh_rel "${MESH_REL}" \
  --tcr_thresholds "${TCR_THRESHOLDS:-1,2,5}" \
  --json_out "${METRIC_JSON}"

"${METRIC_RUNTIME}" "${HUGE}/metric_paper_aligned.py" \
  --out_dir "${OUT}" \
  --tasks overall \
  --json_out "${PAPER_JSON}"

echo "[OK] evaluated ${episodes} trajectories from ${CHECKPOINT}"
echo "[OK] metrics       : ${METRIC_JSON}"
echo "[OK] paper metrics : ${PAPER_JSON}"
