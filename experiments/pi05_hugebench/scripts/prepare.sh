#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_dir "${OPENPI_ROOT}"
require_dir "${HUGEBENCH_ROOT}"
require_dir "${GAUSSIAN_SPLATTING_ROOT}"
require_dir "${HUGE_DATA_ROOT}/data_3d"
prepare_data_links
mkdir -p "${RESULTS_ROOT}" "${LOGS_ROOT}/train" "${LOGS_ROOT}/eval"

print_identity
if ! find_openpi_python >/dev/null; then
  note "OpenPI runtime is unavailable; see docs/runtime.md"
fi
if ! find_renderer_python >/dev/null; then
  note "Gaussian-splatting runtime is unavailable; see docs/runtime.md"
fi
note "layout validation complete"

