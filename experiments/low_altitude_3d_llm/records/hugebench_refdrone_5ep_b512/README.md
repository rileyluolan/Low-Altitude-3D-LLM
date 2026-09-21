# RefDrone 实验启动记录

历史正式 run：`hugebench_qwen3vl2b_refdrone_5ep_b512_20260919`。
本目录记录 **2026-09-19 启动时**的检查，不表示当前实时训练状态或最终评测结果。

| 文件 | 内容 |
|---|---|
| `setup_provenance.json` | 原 base、新 VLM 和新输出的来源 |
| `vlm_download_manifest.json` | 原始下载的文件大小、SHA256、Blob 校验 |
| `base_conversion_manifest.json` | 626 个 VLM 张量替换、528 个非 VLM 张量保持不变的检查 |
| `original_weights_verified.json` | 原 base 和此前 HUGE final checkpoint 的哈希未变 |
| `training_setup.json` | 与原 5 epoch 配方的路径差异、源码版本 |
| `launch_source_config.yaml` | 本机正式训练原始配置，保留历史路径 |
| `preflight.json` | 实际 GPU 上的载入、真实数据前向/反向、有限动作输出检查 |

前向 loss 为 1.0437064；三个模块组均有有限非零梯度，动作输出为 `[1, 20, 4]`。
这次检查没有执行 optimizer update。原训练与模型文件未在发布过程中改动。
