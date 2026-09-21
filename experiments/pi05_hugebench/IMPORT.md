# 2026-09-21 Blob 导入

来源：`output/liyan/ll_0909/pi05_hugebench/`。导入位置为本仓库的 `experiments/pi05_hugebench/`。

## 本次选定的权重

按用户指定，仅同步以下三组，合计 **69,629,284,055 字节**：

| 组别 | 原目录 | 大小 |
|---|---|---:|
| 原始 pi0.5 base | `artifacts/cache/openpi-assets/checkpoints/pi05_base/` | 12.44 GB |
| 官方 HUGE_PI05 基线 | `artifacts/baselines/official_pi05/` | 12.44 GB |
| 5 epoch 最终 checkpoint | `artifacts/checkpoints/pi05_overall_5ep/pi05_overall_5ep_run1/67189/` | 44.75 GB |

最终 checkpoint 保留完整目录，包括 `params`、`train_state`、`assets` 和 Orbax 元数据。
1 epoch checkpoint 和所有中间 checkpoint 不在此次发布范围内。

实际权重分片已发布到 [GitHub Release](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/pi05-hugebench-5ep-20260921)，全部通过 SHA256 校验。
实验代码、配置、来源记录保存在 Git；完整评测轨迹、图片、视频和日志将随 Release 归档。
数据缓存、JAX/uv 缓存和已安装的 `.venv`、`.python` 环境不上传。
原始 base 虽位于 `artifacts/cache/openpi-assets/`，属于本次保留的模型资产。

## 旧文档和实际产物

原 README 和 `manifests/artifacts.yaml` 描述的是早期 1 epoch 整理状态。
后续 Blob 快照实际还包含 5 epoch 的最终 step `67189`，以及三套正式结果：

| 结果来源 | 目录 | 已发现的轨迹文件数 |
|---|---|---:|
| 官方基线 | `results/baselines/official_pi05/full/` | 993 |
| 1 epoch 训练模型 | `results/trained/pi05_overall_run1/step_13437/seed_0/` | 993 |
| 5 epoch 训练模型 | `results/trained/pi05_overall_5ep_run1/step_67189/seed_0/` | 993 |

这里记录的是 Blob 中实际保存的产物，没有在本机重新运行 pi0.5 训练或评测。
保留 1 epoch 评测记录用于追溯，不代表同时发布它的 checkpoint。
原 README 的逐字备份位于 `records/blob_20260909/README.original.md`。

## 在此目录选择 5 epoch 权重

原脚本默认值仍对应历史 1 epoch 实验。使用本次选定权重时，显式设置：

```bash
export TRAIN_CONFIG=pi05_overall_5ep
export RUN_NAME=pi05_overall_5ep_run1
export CHECKPOINT_STEP=67189
```

运行脚本前还需恢复数据链接、相应运行环境及外部源码。Blob 此前没有复制符号链接目标，
因此本次前缀中不包含 `external/`、`data/` 的目标内容，也没有 XPolicyLab 工作树。
`docs/provenance.md` 记录了外部源码 commit 和三个必要的未提交修改文件；仅下载上游 commit
不足以还原这些修改。此次导入保留这一限制，不将目录同步视为已完成运行环境复现。

## 本机导入记录

`records/blob_20260909/source_files.json` 记录代码、配置、历史文档和日志在下载时的 SHA256 与 Blob ETag。
其中的原始绝对路径用于来源追溯；不要据此覆盖当前工作目录配置。

## 下载与还原

下载 [Release](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/pi05-hugebench-5ep-20260921) 中的 `manifest.json` 和 `restore.py`，在目标工作目录执行：

```bash
python3 restore.py manifest.json --workspace /path/to/workspace
tar -xf /path/to/workspace/experiments/pi05_hugebench/artifacts/archives/results-and-logs.tar -C /path/to/workspace
```

还原器检查分片和完整文件 SHA256。归档解包会恢复原来的 `results/` 和 `logs/` 路径；
请在新的目标目录解包，避免覆盖已有评测产物。
完整的 Release 文件清单位于仓库的 `model_assets/releases/pi05-hugebench-5ep-20260921.json`。
