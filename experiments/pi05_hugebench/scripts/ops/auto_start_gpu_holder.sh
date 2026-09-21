#!/usr/bin/env bash
set -euo pipefail

EVAL_PID="${EVAL_PID:-4093621}"
POLL_SECONDS="${POLL_SECONDS:-15}"
IDLE_MAX_USED_MB="${IDLE_MAX_USED_MB:-512}"
HOLDER_PYTHON="${HOLDER_PYTHON:-/opt/conda/envs/ptca/bin/python}"
HOLDER_SCRIPT="${HOLDER_SCRIPT:-/blob/thinking.py}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/logs/runtime"
LOCK_FILE="${RUNTIME_DIR}/auto_start_gpu_holder.lock"

mkdir -p "${RUNTIME_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[gpu-holder-watcher] another watcher/holder owns %s; exiting\n' "${LOCK_FILE}"
  exit 0
fi

[[ -x "${HOLDER_PYTHON}" ]] || {
  printf '[gpu-holder-watcher] Python is not executable: %s\n' "${HOLDER_PYTHON}" >&2
  exit 1
}
[[ -f "${HOLDER_SCRIPT}" ]] || {
  printf '[gpu-holder-watcher] holder script is missing: %s\n' "${HOLDER_SCRIPT}" >&2
  exit 1
}

holder_running() {
  pgrep -f "^${HOLDER_PYTHON} ${HOLDER_SCRIPT}$" >/dev/null 2>&1
}

all_gpus_idle() {
  local used
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  ((${#used[@]} > 0)) || return 1
  for value in "${used[@]}"; do
    ((value <= IDLE_MAX_USED_MB)) || return 1
  done
}

if holder_running; then
  printf '[gpu-holder-watcher] holder is already running; exiting\n'
  exit 0
fi

printf '[gpu-holder-watcher] waiting: eval_pid=%s idle_max_used_mb=%s poll_seconds=%s\n' \
  "${EVAL_PID}" "${IDLE_MAX_USED_MB}" "${POLL_SECONDS}"

while kill -0 "${EVAL_PID}" 2>/dev/null; do
  if all_gpus_idle; then
    printf '[gpu-holder-watcher] all GPUs are idle while eval shell is still alive\n'
    break
  fi
  sleep "${POLL_SECONDS}"
done

if ! kill -0 "${EVAL_PID}" 2>/dev/null; then
  printf '[gpu-holder-watcher] evaluation process %s has exited\n' "${EVAL_PID}"
fi

if holder_running; then
  printf '[gpu-holder-watcher] holder was started elsewhere; exiting\n'
  exit 0
fi

printf '[gpu-holder-watcher] starting: %s %s\n' "${HOLDER_PYTHON}" "${HOLDER_SCRIPT}"
exec "${HOLDER_PYTHON}" "${HOLDER_SCRIPT}"
