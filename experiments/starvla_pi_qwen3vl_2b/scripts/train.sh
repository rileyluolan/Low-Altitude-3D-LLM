#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
PROFILE="${TRAIN_PROFILE:-train}"
[[ "${PROFILE}" == train || "${PROFILE}" == smoke ]] || die 'TRAIN_PROFILE must be train or smoke.'
resolve_profile "${PROFILE}"
CONFIG="${RUNTIME_CONFIG_DIR}/train.yaml"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'Dry run: config=%s accelerate=%s\n' "${CONFIG}" "${RUNTIME_CONFIG_DIR}/accelerate.yaml"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3}"
IFS=, read -ra gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#gpu_ids[@]}" == 4 ]] || die 'This configuration requires four GPU IDs in TRAIN_GPUS.'
for gpu in "${gpu_ids[@]}"; do
  free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
  ((free_mb >= ${MIN_FREE_MB:-55000})) || die "GPU ${gpu} has only ${free_mb} MiB free."
done
[[ -f "${BASE_CKPT}" ]] || die "Missing VLA checkpoint: ${BASE_CKPT}"
"${EXPERIMENT_ROOT}/scripts/prepare_data.sh"
RUN_DIR="${EXPERIMENT_ROOT}/artifacts/checkpoints/${RUN_ID}"
mkdir -p "${RUN_DIR}"
resume_args=()
if [[ "${RESUME:-0}" == 1 ]]; then
  compgen -G "${RUN_DIR}/checkpoints/steps_*_pytorch_model.pt" >/dev/null || die 'No saved checkpoint exists for RESUME=1.'
  printf 'Resuming model weights and step count; original trainer does not restore optimizer moments.\n' >&2
  resume_args+=(--trainer.is_resume true)
elif [[ -f "${RUN_DIR}/config.full.yaml" ]]; then
  die 'Run already exists. Set a new RUN_ID or RESUME=1 to continue from its checkpoint.'
fi
cp "${CONFIG}" "${RUN_DIR}/source_config.yaml"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ACCELERATE_LOG_LEVEL=info
cd "${STARVLA_ROOT}"
exec "${VENV_ROOT}/bin/accelerate" launch \
  --config_file "${RUNTIME_CONFIG_DIR}/accelerate.yaml" \
  --num_processes 4 --main_process_port "${MASTER_PORT:-29531}" \
  starVLA/training/train_starvla.py --config_yaml "${CONFIG}" \
  --run_id "${RUN_ID}" --seed "${SEED}" "${resume_args[@]}" "$@"
