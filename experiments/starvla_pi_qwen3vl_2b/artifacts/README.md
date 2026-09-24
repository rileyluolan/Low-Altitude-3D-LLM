# 实际权重与训练产物

## 最终 checkpoint

官方 Qwen3-VL-2B 版 HUGE-Bench 5 epoch 最终模型与必要配置可从
[Hugging Face 仓库目录](https://huggingface.co/lld-koi/Low-Altitude-3D-LLM/tree/main/experiments/starvla_pi_qwen3vl_2b/artifacts/checkpoints/hugebench_qwen3vl2b_5ep_b512_20260916)
直接下载。三版权重及自有 VLA 视频的下载命令见 [统一入口](../../../model_assets/FINAL_CHECKPOINTS.md)。

## 历史 base 与中间 checkpoint 快照

本目录的大文件通过本 fork 的 [GitHub Release](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/starvla-artifacts-20260921) 分片保存。
Release 提供实际权重本体；Git 源码树中仅保留本说明。

下载 Release 中的 `manifest.json` 和 `restore.py` 后，在目标工作目录运行：

```bash
python3 restore.py manifest.json --workspace /path/to/workspace
```

会还原两个实验的 `experiments/*/artifacts/`，包括本目录的 base、已保存 checkpoint 和配套配置。
每个分片和还原后的完整文件均校验 SHA256。不会覆盖内容不同的现有文件。

2026-09-21 快照中，原实验包含 base、16 个中间 checkpoint 和 final model；
`low_altitude_3d_llm` 包含替换 VLM 后的 base 及 1000–5000 步 checkpoint。
该历史快照记录的是后者训练中途的状态；后来完成的最终 checkpoint 和视频见上方统一入口。

该 Release 的文件清单、准确快照时间及每个权重的校验值见其 `manifest.json`。
