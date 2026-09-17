#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
phase="${1:?Usage: scripts/train_oxe_phase.sh align|joint}"
shift
[[ "${phase}" == align || "${phase}" == joint ]] || die 'Expected align or joint.'
export RUN_ID="qwen3vl_2b_pi_v3_bridge_rt1_${phase}"
count="${OXE_NUM_PROCESSES:-8}"
resolve_profile "oxe_${phase}" --num-processes "${count}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'Dry run: %s\n' "${RUNTIME_CONFIG_DIR}/train.yaml"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-$(seq -s, 0 "$((count - 1))")}"
IFS=, read -ra gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#gpu_ids[@]}" == "${count}" ]] || die 'TRAIN_GPUS count must match OXE_NUM_PROCESSES.'
for gpu in "${gpu_ids[@]}"; do
  free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
  ((free_mb >= ${MIN_FREE_MB:-30000})) || die "GPU ${gpu} has only ${free_mb} MiB free."
done
if [[ "${phase}" == joint ]]; then
  "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/verify_stage.py" align
fi
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/pretraining/prepare_oxe.py" --require-full-data
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/pretraining/validate_oxe_recipe.py"
run="${EXPERIMENT_ROOT}/artifacts/checkpoints/${RUN_ID}"
resume_args=()
if [[ "${RESUME:-0}" == 1 ]]; then
  compgen -G "${run}/checkpoints/steps_*_pytorch_model.pt" >/dev/null || die 'No checkpoint for RESUME=1.'
  printf 'Resume restores weights and step count only; optimizer state is not restored.\n' >&2
  resume_args+=(--trainer.is_resume true)
elif [[ -f "${run}/config.full.yaml" ]]; then
  die 'Run exists. Use explicit RESUME=1 for weights-only continuation, or a separate experiment checkout.'
fi
mkdir -p "${run}"
printf '{"num_processes":%s,"seed":%s,"weights_only_resume":%s}\n' "${count}" "${SEED}" "${RESUME:-0}" > "${run}/launcher_recipe.json"
cp "${RUNTIME_CONFIG_DIR}/train.yaml" "${run}/source_config.yaml"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${STARVLA_ROOT}"
"${VENV_ROOT}/bin/accelerate" launch --config_file "${RUNTIME_CONFIG_DIR}/accelerate.yaml" \
  --num_processes "${count}" --main_process_port "${MASTER_PORT:-29532}" \
  starVLA/training/train_starvla.py --config_yaml "${RUNTIME_CONFIG_DIR}/train.yaml" \
  --run_id "${RUN_ID}" --seed "${SEED}" "${resume_args[@]}" "$@"
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/verify_stage.py" "${phase}"
