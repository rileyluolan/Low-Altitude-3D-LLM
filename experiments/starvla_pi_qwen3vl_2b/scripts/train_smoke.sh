#!/usr/bin/env bash
set -euo pipefail
export TRAIN_PROFILE=smoke
export RUN_ID="${RUN_ID:-qwen3vl_2b_hugebench_smoke_$(date -u +%Y%m%dT%H%M%S)}"
exec "$(dirname "$0")/train.sh" "$@"
