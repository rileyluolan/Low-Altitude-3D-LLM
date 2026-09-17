#!/usr/bin/env bash
# Downloads are separate and explicit. This trains 5k + 50k optimizer updates.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
init="${EXPERIMENT_ROOT}/artifacts/base/qwen3vl_2b_pi_v3_bridge_rt1_init"
[[ -f "${init}/checkpoints/base_pytorch_model.pt" ]] || "${EXPERIMENT_ROOT}/scripts/build_oxe_base.sh"
for phase in align joint; do
  if ! "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/verify_stage.py" "${phase}"; then
    "${EXPERIMENT_ROOT}/scripts/train_oxe_phase.sh" "${phase}"
  fi
done
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/pretraining/promote_oxe_base.py"
base="${EXPERIMENT_ROOT}/artifacts/base/qwen3vl_2b_pi_v3_hugebench_mature_base"
if [[ ! -f "${base}/checkpoints/base_pytorch_model.pt" ]]; then
  CUDA_VISIBLE_DEVICES="" "${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/pretraining/convert_mature_oxe_to_huge.py"
fi
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/validate_base.py" --run-dir "${base}" --verify-hash
