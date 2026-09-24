# 三版最终 checkpoint 与自有 VLA 的全部视频

三版均为 HUGE-Bench 5 epoch 训练后的最终 checkpoint。下面的清单只选择最终权重及必要配置。
实际二进制保存在本 fork 的 GitHub Releases；下载不需要 Blob SAS。

| 实验 | 最终 checkpoint | 下载清单 | 二进制所在 Release |
|---|---|---|---|
| pi05 复现（原无状态输入版本） | `pi05_overall_5ep/pi05_overall_5ep_run1/67189/`，含 params、train_state、assets 和 Orbax 元数据 | [finals-pi05.json](releases/finals-pi05.json) | [pi05-hugebench-5ep-20260921](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/pi05-hugebench-5ep-20260921) |
| StarVLA-PI + 官方 Qwen3-VL-2B | `hugebench_qwen3vl2b_5ep_b512_20260916/final_model/pytorch_model.pt`，16,798 updates | [finals-starvla-2b.json](releases/finals-starvla-2b.json) | [starvla-artifacts-20260921](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/starvla-artifacts-20260921) |
| StarVLA-PI + 自有 RefDrone VLM | `hugebench_qwen3vl2b_refdrone_5ep_b512_20260920/final_model/pytorch_model.pt`，16,798 updates | [finals-low-altitude.json](releases/finals-low-altitude.json) | [low-altitude-vla-final-20260924](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/low-altitude-vla-final-20260924) |

pi05 和原 StarVLA 2B 的最终权重沿用已发布附件。自有 VLA 的最终权重与视频于本次补充发布。
这些清单可直接复用 `restore_release.py`；每个分片和完整文件都校验 SHA256。
源代码、JSON 清单随 git clone 下载；权重和视频需要执行以下还原命令。

## 分别下载最终 checkpoint

在仓库根目录执行所需的一条：

```bash
python3 model_assets/restore_release.py model_assets/releases/finals-pi05.json --workspace .
python3 model_assets/restore_release.py model_assets/releases/finals-starvla-2b.json --workspace .
python3 model_assets/restore_release.py model_assets/releases/finals-low-altitude.json --workspace .
```

文件还原到对应的 `experiments/<实验名>/artifacts/checkpoints/`，并保留原始相对路径。
工具会核验已有文件，不覆盖内容不同的文件。

两个 StarVLA 最终模型均为 5,839,846,728 字节，内容不同：

| 模型 | 最终文件 SHA256 |
|---|---|
| 官方 2B VLM 版本 | `8ad5b7ef8a0f4a0cce299b0a9a69741a7c94786ee7b5a750e18d6fec7ea4825d` |
| 自有 RefDrone VLM 版本 | `aa7a180744dddbe47b599347ab8bbfa38f606cd0734af6cf950333c0364e7e99` |

## 自有 VLA：993 个完整评测视频

视频为同一最终 checkpoint 的完整 HUGE-Bench 评测：Seen 576、Unseen 417，共 993 个 MP4。
画面左侧 GT、右侧模型预测。归档附逐视频 SHA256、视频完成记录、运行配置和已计算的指标。

```bash
python3 model_assets/restore_release.py model_assets/releases/videos-low-altitude.json --workspace .
tar -xf experiments/low_altitude_3d_llm/artifacts/archives/final-videos-20260924.tar -C .
```

视频还原到：

```text
experiments/low_altitude_3d_llm/results/
  hugebench_qwen3vl2b_refdrone_5ep_b512_20260920/final_model/ll0923_full_seed_42/
  rollouts/task_*/test_*/*/episode_*/gt_pred_side_by_side.mp4
```

可在空目标目录中还原、解包，保留现有本机评测目录。
也可使用 [完整清单](releases/low-altitude-vla-final-20260924.json) 一次下载自有 VLA 最终 checkpoint、配置和视频归档。

本次视频发布范围是 `low_altitude_3d_llm`。原 StarVLA 2B 视频仍以既有实验存储及完成清单为准；
pi05 的已有视频保存在原 pi05 Release 的 `results-and-logs.tar` 中。
