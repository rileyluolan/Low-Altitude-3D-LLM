#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
require_env
SOURCE_DATA_ROOT="${HUGE_DATA_ROOT}/data_traj/train"
DATASET_VIEW="${EXPERIMENT_ROOT}/data/lerobot/hugebench_train"
REGISTRY_DIR="${STARVLA_ROOT}/examples/simBenchmarks/HUGEBench/train_files/data_registry"
for required in meta/info.json meta/episodes.jsonl meta/tasks.jsonl data; do
  [[ -e "${SOURCE_DATA_ROOT}/${required}" ]] || die "Missing data: ${SOURCE_DATA_ROOT}/${required}"
done
mkdir -p "${DATASET_VIEW}/meta" "${REGISTRY_DIR}"
ln -sfn "${SOURCE_DATA_ROOT}/data" "${DATASET_VIEW}/data"
for metadata in info.json episodes.jsonl episodes_stats.jsonl tasks.jsonl; do
  [[ ! -e "${SOURCE_DATA_ROOT}/meta/${metadata}" ]] || ln -sfn "${SOURCE_DATA_ROOT}/meta/${metadata}" "${DATASET_VIEW}/meta/${metadata}"
done
cp "${EXPERIMENT_ROOT}/integration/train_files/"{modality.json,stats_gr00t.json} "${DATASET_VIEW}/meta/"
"${VENV_ROOT}/bin/python" "${EXPERIMENT_ROOT}/scripts/tools/build_steps_cache.py" "${DATASET_VIEW}"
for name in __init__.py data_config.py; do
  ln -sfn "${EXPERIMENT_ROOT}/integration/train_files/data_registry/${name}" "${REGISTRY_DIR}/${name}"
done
printf 'Prepared HUGE-Bench training data: %s\n' "${DATASET_VIEW}"
