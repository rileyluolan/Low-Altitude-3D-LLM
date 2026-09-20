#!/usr/bin/env bash
# Run the actual training process in a tmux terminal; its output is captured
# by tmux pipe-pane without redirecting the trainer away from the terminal.
set -euo pipefail
EXPERIMENT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -n "${TRAIN_TMUX_GATE:-}" ]]; then
  tmux wait-for "${TRAIN_TMUX_GATE}"
fi
exec bash "${EXPERIMENT_ROOT}/scripts/train.sh"
