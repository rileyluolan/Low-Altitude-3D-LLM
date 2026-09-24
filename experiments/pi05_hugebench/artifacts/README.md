# pi05 模型权重

## 5 epoch 最终 checkpoint

完整的 `67189/` 已放入 [Hugging Face 仓库目录](https://huggingface.co/lld-koi/Low-Altitude-3D-LLM/tree/main/experiments/pi05_hugebench/artifacts/checkpoints/pi05_overall_5ep/pi05_overall_5ep_run1/67189)，可直接浏览、下载全部 68 个文件。
下载命令见 [三版最终 checkpoint 与视频](../../../model_assets/FINAL_CHECKPOINTS.md)。这是原无状态输入的 pi05 复现版本。

## 历史 base、基线与产物备份

实际二进制文件位于 [GitHub Release](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/pi05-hugebench-5ep-20260921)。下载和还原步骤见 [IMPORT.md](../IMPORT.md)。

仅包含原始 pi0.5 base、官方 HUGE_PI05 基线和 5 epoch 最终 checkpoint `67189`。
最终 checkpoint 保留模型参数、训练状态、归一化统计和 Orbax 元数据。
1 epoch 和中间 checkpoint 不在本次发布范围。
