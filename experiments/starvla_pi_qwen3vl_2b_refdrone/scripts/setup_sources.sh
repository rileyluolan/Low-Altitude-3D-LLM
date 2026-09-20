#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
huge_commit=e80e8ac6c39b7503aea31d9c288d24591c0a0a79
gaussian_commit=54c035f7834b564019656c3e3fcc3646292f727d
restore_repo() {
  local url="$1" target="$2" commit="$3"
  if [[ ! -d "${target}/.git" ]]; then
    mkdir -p "$(dirname "${target}")"
    git clone --no-checkout "${url}" "${target}"
    git -C "${target}" checkout --detach "${commit}"
  fi
  [[ "$(git -C "${target}" rev-parse HEAD)" == "${commit}" ]] || die "Unexpected revision in ${target}; expected ${commit}."
}
restore_repo https://github.com/starVLA/starVLA.git "${STARVLA_ROOT}" 2f17402a5ccaa09907516ae5e542b0fa6ee5d155
for name in 0001-starvla-refdrone-runtime.patch; do
  patch="${EXPERIMENT_ROOT}/patches/${name}"
  if ! git -C "${STARVLA_ROOT}" apply --reverse --check "${patch}" 2>/dev/null; then
    git -C "${STARVLA_ROOT}" apply --check "${patch}"
    git -C "${STARVLA_ROOT}" apply "${patch}"
  fi
done
restore_repo https://github.com/jingyu198/HUGE-Bench.git "${HUGEBENCH_ROOT}" "${huge_commit}"
restore_repo https://github.com/graphdeco-inria/gaussian-splatting.git "${GAUSSIAN_SPLATTING_ROOT}" "${gaussian_commit}"
restore_repo https://github.com/facebookresearch/pytorch3d.git "${EXPERIMENT_ROOT}/third_party/pytorch3d" f34104cf6ebefacd7b7e07955ee7aaa823e616ac
git -C "${GAUSSIAN_SPLATTING_ROOT}" submodule update --init --recursive -- submodules/diff-gaussian-rasterization submodules/simple-knn
patch="${EXPERIMENT_ROOT}/patches/0003-hugebench-lazy-openpi.patch"
if git -C "${HUGEBENCH_ROOT}" apply --reverse --check "${patch}" 2>/dev/null; then
  : # Already applied.
else
  git -C "${HUGEBENCH_ROOT}" apply --check "${patch}"
  git -C "${HUGEBENCH_ROOT}" apply "${patch}"
fi
for file in 3dgs_renderer.py my_render_traj.py utils/graphics_utils.py; do
  cp "${HUGEBENCH_ROOT}/gaussian_splatting/${file}" "${GAUSSIAN_SPLATTING_ROOT}/${file}"
done
printf 'Official HUGE-Bench and Gaussian Splatting sources are ready.\n'
