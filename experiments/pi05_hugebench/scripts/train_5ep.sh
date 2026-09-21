#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_CONFIG="pi05_overall_5ep"
export RUN_NAME="pi05_overall_5ep_run1"

exec "${SCRIPT_DIR}/train.sh"
