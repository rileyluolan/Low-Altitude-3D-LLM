# 实际权重与训练产物

## 最终 checkpoint 与全部视频

自有 VLA 的 HUGE-Bench 5 epoch 最终 checkpoint（16,798 updates）及全部 993 个评测视频已发布到
[low-altitude-vla-final-20260924](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/low-altitude-vla-final-20260924)。
逐版本下载清单及还原方法见 [最终 checkpoint 与视频](../../../model_assets/FINAL_CHECKPOINTS.md)。

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
该日期的快照仅覆盖后者训练中途的产物；最终 checkpoint 和视频使用上方的新 Release。

该 Release 的文件清单、准确快照时间及每个权重的校验值见其 `manifest.json`。
