#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
command -v tmux >/dev/null || die 'Install tmux first.'
session="${SESSION:-hugebench_reproduce}"
[[ "${session}" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'SESSION may contain letters, numbers, underscores and hyphens.'
! tmux has-session -t "${session}" 2>/dev/null || die "tmux session ${session} already exists."
if [[ $# == 0 ]]; then set -- "${EXPERIMENT_ROOT}/scripts/train.sh"; fi
mkdir -p "${EXPERIMENT_ROOT}/runtime/tmux"
launch="${EXPERIMENT_ROOT}/runtime/tmux/${session}.sh"
# Environment values may be private; launcher is ignored and owner-readable only.
umask 077
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  for name in EXPERIMENT_ROOT WORKSPACE_ROOT VENV_ROOT STARVLA_ROOT HUGEBENCH_ROOT GAUSSIAN_SPLATTING_ROOT HUGE_DATA_ROOT MODEL_ROOT BASE_CKPT CUDA_HOME CC CXX RUN_ID SEED TRAIN_GPUS OXE_NUM_PROCESSES MIN_FREE_MB RESUME MASTER_PORT EXPERIMENT_ENV_FILE; do
    if [[ -v "${name}" ]]; then printf 'export %s=%q\n' "${name}" "${!name}"; fi
  done
  printf 'cd %q\n' "${EXPERIMENT_ROOT}"
  printf 'exec'; printf ' %q' "$@"; printf '\n'
} > "${launch}"
chmod 700 "${launch}"
# Start behind a gate so logging is attached before training emits output.
gate="${launch}.gate"
rm -f "${gate}"
printf -v cmd 'while [[ ! -f %q ]]; do sleep 0.2; done; exec bash %q' "${gate}" "${launch}"
pane="$(tmux new-session -d -P -F '#{pane_id}' -s "${session}" -n train -c "${EXPERIMENT_ROOT}" "bash -c $(printf %q "${cmd}")")"
trap 'tmux kill-session -t "${session}"' ERR
log="${LOGS_ROOT}/${session}_$(date -u +%Y%m%dT%H%M%SZ).log"
printf -v pipe 'cat >> %q' "${log}"
tmux pipe-pane -t "${pane}" -o "${pipe}"
tmux set-option -t "${session}" remain-on-exit on
touch "${gate}"
trap - ERR
printf 'Started tmux session: %s\nLog: %s\nAttach: tmux attach -t %s\nDetach: Ctrl+b, then d\n' "${session}" "${log}" "${session}"
