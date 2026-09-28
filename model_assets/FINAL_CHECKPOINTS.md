# 三版最终 checkpoint 与自有 VLA 的完整评测结果

实际权重与自有 VLA 的完整 `results/` 目录保存在 Hugging Face 仓库
[lld-koi/Low-Altitude-3D-LLM](https://huggingface.co/lld-koi/Low-Altitude-3D-LLM/tree/main)，
可直接浏览、下载完整文件，保留原来的 `experiments/` 路径。无需 Release 还原或 Blob SAS。
本 GitHub 分支保存代码、配置、结果记录和下载入口。

HF 上我们模型统一位于 `experiments/low_altitude_3d_llm/`，训练目录名为
`hugebench_low_altitude_3d_llm_5ep_b512_20260920`。清单中的 `source_path` 记录本机原始路径，
配置、轨迹和权重内容保持原样。目录改名及逐文件校验见 [记录](huggingface/results-directory-rename-20260928.json)。

## 最终 checkpoint

三版均为 HUGE-Bench 5 epoch 训练后的最终 checkpoint。

| 实验 | 最终 checkpoint | 仓库目录 | 大小（十进制 GB） |
|---|---|---|---:|
| pi05 复现（原无状态输入版本） | `pi05_overall_5ep/pi05_overall_5ep_run1/67189/`，含 params、train_state、assets 和 Orbax 元数据 | [pi05 最终权重](https://huggingface.co/lld-koi/Low-Altitude-3D-LLM/tree/main/experiments/pi05_hugebench/artifacts/checkpoints/pi05_overall_5ep/pi05_overall_5ep_run1/67189) | 44.75 |
| StarVLA-PI + 官方 Qwen3-VL-2B | `hugebench_qwen3vl2b_5ep_b512_20260916/final_model/pytorch_model.pt`，16,798 updates | [官方 2B 版最终权重及配置](https://huggingface.co/lld-koi/Low-Altitude-3D-LLM/tree/main/experiments/starvla_pi_qwen3vl_2b/artifacts/checkpoints/hugebench_qwen3vl2b_5ep_b512_20260916) | 5.84 |
| StarVLA-PI + 自有 Low-Altitude-3D-LLM VLM | `hugebench_low_altitude_3d_llm_5ep_b512_20260920/final_model/pytorch_model.pt`，16,798 updates | [自有 VLM 版最终权重及配置](https://huggingface.co/lld-koi/Low-Altitude-3D-LLM/tree/main/experiments/low_altitude_3d_llm/artifacts/checkpoints/hugebench_low_altitude_3d_llm_5ep_b512_20260920) | 5.84 |

开启当前状态输入的 pi05 对照实验仍暂停，尚无最终 checkpoint，本表的 pi05 是原无状态版本。
本次上传范围为最终 checkpoint 和配套配置，未扩展到 base 或中间 checkpoint。

## 自有 VLA：完整 results 目录与 993 个视频

[浏览视频与完整指标目录](https://huggingface.co/lld-koi/Low-Altitude-3D-LLM/tree/main/experiments/low_altitude_3d_llm/results/hugebench_low_altitude_3d_llm_5ep_b512_20260920/final_model/ll0923_full_seed_42)

- Seen 576、Unseen 417，共 **993 个独立 MP4**，约 4.31 GB。
- 画面左侧为 GT，右侧为模型预测。
- 包含逐视频完成记录、运行配置、`metrics.json` 和 `metrics_table.md`。
- 原 StarVLA 2B 的视频不在此次新增上传范围内。

2026-09-28 已补齐本机 `experiments/low_altitude_3d_llm/results/` 的全部内容，
共 **6,985 个文件、4,784,899,596 字节（约 4.78 GB）**，保留原始路径和内容：

| 文件类型 | 数量 |
|---|---:|
| 视频 MP4 | 993 |
| 轨迹 NPZ | 993 |
| 图片 PNG | 1,994 |
| 指标、配置、逐条记录等 JSON | 1,994 |
| TXT | 993 |
| 日志 LOG | 15 |
| 指标 Markdown 表格 | 1 |
| 完成标记 | 2 |

本次新增 4,996 个文件，约 472 MB；完整目录逐文件核对路径、大小和内容哈希，
并匿名下载轨迹、图片和日志样本校验。记录见 [完整结果上传校验](huggingface/results-verification-20260928.json)。

## 下载

安装 Hugging Face CLI：`python -m pip install -U huggingface_hub`。
在目标工作目录运行所需命令，文件自动保留对应的 `experiments/` 路径：

```bash
# pi05 最终 checkpoint
hf download lld-koi/Low-Altitude-3D-LLM --local-dir . \
  --include 'experiments/pi05_hugebench/artifacts/checkpoints/**'

# 官方 Qwen3-VL-2B 版最终 checkpoint
hf download lld-koi/Low-Altitude-3D-LLM --local-dir . \
  --include 'experiments/starvla_pi_qwen3vl_2b/artifacts/checkpoints/**'

# 自有 VLM 版最终 checkpoint
hf download lld-koi/Low-Altitude-3D-LLM --local-dir . \
  --include 'experiments/low_altitude_3d_llm/artifacts/checkpoints/**'

# 自有 VLA 的完整 results：轨迹、图片、视频、指标、记录与日志
hf download lld-koi/Low-Altitude-3D-LLM --local-dir . \
  --include 'experiments/low_altitude_3d_llm/results/**'

# 下载该仓库全部文件，约 61.21 GB
hf download lld-koi/Low-Altitude-3D-LLM --local-dir .
```

请使用合适的目标目录，避免与其他实验版本混用同一路径。

## 校验与来源

[完整文件清单](huggingface/manifest.json) 记录 7,063 个源文件的路径、大小及 SHA256。
Hugging Face 仓库另提供 `SHA256SUMS`，完整下载后可执行 `sha256sum -c SHA256SUMS`。
上传后逐文件核对远端大小和内容哈希，并实际匿名下载一个视频校验。

两个 StarVLA 最终模型均为 5,839,846,728 字节，内容不同：

| 模型 | 最终文件 SHA256 |
|---|---|
| 官方 2B VLM 版本 | `8ad5b7ef8a0f4a0cce299b0a9a69741a7c94786ee7b5a750e18d6fec7ea4825d` |
| 自有 Low-Altitude-3D-LLM VLM 版本 | `aa7a180744dddbe47b599347ab8bbfa38f606cd0734af6cf950333c0364e7e99` |

以往 Release 及 `releases/` 清单保留为历史备份；本页的最终权重和完整评测结果均从上述 Hugging Face 仓库下载。
