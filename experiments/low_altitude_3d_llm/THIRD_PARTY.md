# Upstream sources and assets

`setup_sources.sh` downloads the complete pinned repositories, including their original license files. Their code is not relicensed by this repository's root LICENSE.

- [StarVLA](https://github.com/starVLA/starVLA/tree/2f17402a5ccaa09907516ae5e542b0fa6ee5d155): QwenPI_v3 framework, action model, trainer, datasets and policy protocol. Local content changes are provided as patches.
- [HUGE-Bench](https://github.com/jingyu198/HUGE-Bench/tree/e80e8ac6c39b7503aea31d9c288d24591c0a0a79): official rollout helpers, renderer adapters and metrics.
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/54c035f7834b564019656c3e3fcc3646292f727d): renderer and pinned CUDA submodules. Consult its upstream LICENSE and each submodule's license.
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d/tree/f34104cf6ebefacd7b7e07955ee7aaa823e616ac): pure Python rotation transforms.
- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and [StarVLA source checkpoint](https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1): downloaded separately at the revisions in `SOURCE.lock.yaml`.
- Bridge, RT-1, HUGE trajectories and 3D scene assets are downloaded from the dataset repositories in `SOURCE.lock.yaml`; their own dataset terms apply. Committed normalization statistics describe the pinned HUGE training split.
