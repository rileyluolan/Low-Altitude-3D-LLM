# 模型权重下载

## 两个实验的实际 artifacts

2026-09-21 的两个 `artifacts/` 快照（约 138.92 GB）以实际二进制分片保存在
[本 fork 的 GitHub Release](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/starvla-artifacts-20260921)，无需 Blob SAS。
包含原 base、原 HUGE 完整训练权重、全部已保存中间 checkpoint，以及新的 VLM 替换 base 和新训练的 1000–5000 步 checkpoint。
新训练仍在进行，后续产物需要另行同步。

在目标工作目录中执行：

```bash
mkdir -p .tmp
curl -fL https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/download/starvla-artifacts-20260921/manifest.json -o .tmp/artifacts-manifest.json
python3 model_assets/restore_release.py .tmp/artifacts-manifest.json --workspace .
```

下载器将文件还原到 `experiments/starvla_pi_qwen3vl_2b/artifacts/` 与
`experiments/low_altitude_3d_llm/artifacts/`，检查分片和完整文件 SHA256，并拒绝覆盖不同内容。
Git clone 和源码 ZIP 不包括 Release 附件。

## 现有 Blob 来源

`manifest.json` 为 2026-09-20 本机现有权重的清单：包含原 mature base、HUGE 五 epoch 最终模型、16 个中间 checkpoint、官方 Qwen3-VL-2B 资源及 RefDrone VLM。每项记录对应的本地路径、Blob 对象名、字节数和实际计算的 SHA256。

权重二进制保存在现有 Azure Blob 中，Git 保存此清单和下载器。私有 SAS 凭据由使用者自行提供，不包含在仓库或下载 URL 中。访问需要相应容器权限。

在仓库根目录执行：

```bash
python3 model_assets/fetch.py --list
python3 model_assets/fetch.py mature-base original-vlm \
  --workspace /root/workspace-ll --config /root/workspace-ll/.blob_config.json
python3 model_assets/fetch.py huge-final \
  --workspace /root/workspace-ll --config /root/workspace-ll/.blob_config.json
```

中间 checkpoint 按 `huge-step-1000` 到 `huge-step-16000` 单独选择。下载器不覆盖内容不同的现有文件；支持 `.partial` 续传，完成后校验 SHA256。`--verify-only` 仅检查已有文件；`--metadata-only` 只取配置、统计和来源记录。

## RefDrone base

RefDrone base 由原 mature base 与固定的 RefDrone VLM 重建。先获取 `mature-base` 和 `original-vlm`，然后按 [RefDrone 实验说明](../experiments/low_altitude_3d_llm/README.md) 执行 `download_vlm.py` 和 `assemble_base.py`。下载器还会生成组装所需的 `DOWNLOAD_MANIFEST.json`。

本机于 2026-09-20 重建的 base 为 5,217,525,867 字节，SHA256：

```text
e016e41b329f4b16ce8a927d38990f69f1b63a0950c7b9efe26cdf10d55df162
```

它与历史 RefDrone base 完全一致：替换 626 个 VLM 张量，保留 112 个 projector 张量和 416 个 action 张量。原 HUGE 最终模型单独保存于 `huge-final` 组。

数据集、场景压缩包、运行环境和缓存由各实验的现有准备脚本管理；此清单聚焦模型文件及其必要配套配置。
