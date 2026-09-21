#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

SPLITS="${1:-test_seen,test_unseen}"
RUN_ROOT="$(evaluation_run_root)"
OUTPUT_DIR="${2:-${RUN_ROOT}/rollouts}"
OUTPUT_PARENT="$(dirname "${OUTPUT_DIR}")"
NUM_TRAJS="${NUM_TRAJS:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
EXEC_STEPS="${EXEC_STEPS:-10}"
RESUME_EVAL="${RESUME_EVAL:-0}"
CHECKPOINT="$(checkpoint_path)"

[[ "${RESUME_EVAL}" == "0" || "${RESUME_EVAL}" == "1" ]] \
  || die "RESUME_EVAL must be 0 or 1"

((NUM_SHARDS >= 1 && NUM_SHARDS <= 4)) \
  || die "NUM_SHARDS must be between 1 and 4 for the 8-GPU layout"

MIN_FREE_MB="${MIN_FREE_MB:-24000}"
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_FREE_MB < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  (( ${#GPU_FREE_MB[@]} >= 2 * NUM_SHARDS )) \
    || die "Need at least $((2 * NUM_SHARDS)) visible GPUs, found ${#GPU_FREE_MB[@]}"
  for ((gpu = 0; gpu < 2 * NUM_SHARDS; gpu++)); do
    if (( GPU_FREE_MB[gpu] < MIN_FREE_MB )); then
      die "GPU ${gpu} has ${GPU_FREE_MB[gpu]} MiB free; require at least ${MIN_FREE_MB} MiB before rollout"
    fi
  done
fi

prepare_data_links
has_openpi_runtime || die \
  "OpenPI runtime not found. Restore it before rollout."
RENDERER_PY="$(find_renderer_python)" || die \
  "Gaussian-splatting runtime not found. Restore it or export RENDERER_PYTHON."
require_dir "${HUGE_DATA_ROOT}/data_3d"
[[ -f "${GAUSSIAN_SPLATTING_ROOT}/3dgs_renderer.py" ]] \
  || die "Missing renderer: ${GAUSSIAN_SPLATTING_ROOT}/3dgs_renderer.py"

for ((worker = 0; worker < NUM_SHARDS; worker++)); do
  require_port_free "$((RENDERER_START_PORT + worker))"
done

existing_count=0
if [[ -d "${OUTPUT_DIR}" ]]; then
  existing_count="$(find "${OUTPUT_DIR}" -type f -name 'traj_gt_pred_xyzk.npz' | wc -l)"
fi
if ((existing_count > 0)) && [[ "${RESUME_EVAL}" != "1" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  die "${OUTPUT_DIR} already contains ${existing_count} trajectories; choose a new output or set ALLOW_OVERWRITE=1"
fi

resume_args=()
if [[ "${RESUME_EVAL}" == "1" ]]; then
  resume_args+=(--skip_complete_episodes)
  note "resume enabled: validating and skipping complete episodes; existing trajectories=${existing_count}"
fi

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_PARENT}/logs"
pids=()

cleanup() {
  trap - EXIT INT TERM
  if ((${#pids[@]})); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

print_identity
note "rollouts: ${OUTPUT_DIR}"
note "render GPUs: 0-$((NUM_SHARDS - 1)); inference GPUs: ${NUM_SHARDS}-$((2 * NUM_SHARDS - 1))"

for ((worker = 0; worker < NUM_SHARDS; worker++)); do
  port="$((RENDERER_START_PORT + worker))"
  CUDA_VISIBLE_DEVICES="${worker}" "${RENDERER_PY}" \
    "${GAUSSIAN_SPLATTING_ROOT}/3dgs_renderer.py" \
    --host "${RENDERER_HOST}" \
    --port "${port}" \
    --ply_template "${HUGE_DATA_ROOT}/data_3d/{env_id}/3dgs_ply/point_cloud_utm50.ply" \
    >"${OUTPUT_PARENT}/logs/renderer_gpu${worker}.log" 2>&1 &
  pids+=("$!")
done

note "waiting for ${NUM_SHARDS} renderer processes"
for ((worker = 0; worker < NUM_SHARDS; worker++)); do
  port="$((RENDERER_START_PORT + worker))"
  ready=0
  for ((attempt = 0; attempt < 60; attempt++)); do
    if ! kill -0 "${pids[$worker]}" 2>/dev/null; then
      die "Renderer ${worker} exited; inspect ${OUTPUT_PARENT}/logs/renderer_gpu${worker}.log"
    fi
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 5
  done
  ((ready == 1)) || die "Renderer ${worker} did not listen on port ${port}"
done

eval_pids=()
for ((worker = 0; worker < NUM_SHARDS; worker++)); do
  gpu="$((NUM_SHARDS + worker))"
  port="$((RENDERER_START_PORT + worker))"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    run_openpi scripts/action_infer.py \
      --task_id overall \
      --config_name "${TRAIN_CONFIG}" \
      --checkpoint_dir "${CHECKPOINT}" \
      --splits "${SPLITS}" \
      --exec_steps "${EXEC_STEPS}" \
      --num_trajs "${NUM_TRAJS}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${worker}" \
      --seed "${SEED}" \
      --out_dir "${OUTPUT_DIR}" \
      --host "${RENDERER_HOST}" \
      --port "${port}" \
      "${resume_args[@]}"
  ) >"${OUTPUT_PARENT}/logs/eval_shard${worker}_gpu${gpu}.log" 2>&1 &
  eval_pids+=("$!")
  pids+=("$!")
done

status=0
for pid in "${eval_pids[@]}"; do
  wait "${pid}" || status=1
done
((status == 0)) || die "Rollout failed; inspect ${OUTPUT_PARENT}/logs"

trajectory_count="$(find "${OUTPUT_DIR}" -type f -name 'traj_gt_pred_xyzk.npz' | wc -l)"
((trajectory_count > 0)) || die "No rollout trajectories were generated"
if [[ "${SPLITS// /}" == "test_seen,test_unseen" && "${NUM_TRAJS}" == "0" ]]; then
  ((trajectory_count == 993)) \
    || die "Full evaluation produced ${trajectory_count} trajectories; expected 993"
  for required_name in instruction.txt compare_gt_vs_pred_3d.png gt_pred_side_by_side.mp4; do
    required_count="$(find "${OUTPUT_DIR}" -type f -name "${required_name}" -size +0c | wc -l)"
    ((required_count == 993)) \
      || die "Full evaluation has ${required_count}/993 non-empty ${required_name} files"
  done
fi
{
  printf 'model_source: trained\n'
  printf 'base_weights: %s\n' "${BASE_WEIGHTS_URI}"
  printf 'checkpoint: %s\n' "${CHECKPOINT}"
  printf 'splits: %s\n' "${SPLITS}"
  printf 'seed: %s\n' "${SEED}"
  printf 'exec_steps: %s\n' "${EXEC_STEPS}"
  printf 'num_shards: %s\n' "${NUM_SHARDS}"
  printf 'resume_eval: %s\n' "${RESUME_EVAL}"
  printf 'existing_trajectories_at_start: %s\n' "${existing_count}"
  printf 'trajectory_count: %s\n' "${trajectory_count}"
} >"${OUTPUT_PARENT}/run_manifest.yaml"
note "rollout complete: ${trajectory_count} trajectories"
