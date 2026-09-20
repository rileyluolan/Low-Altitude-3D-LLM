# 2026-09-20 本机正式训练启动记录

本机目录：`/root/workspace-ll/experiments/starvla_pi_qwen3vl_2b_refdrone`。
源码从分支 `feat/starvla-qwen3vl2b-hugebench` 的 `6ba394fd51a0b91ee54033cc3547a2229ff3f854` 同步。

新 run：`hugebench_qwen3vl2b_refdrone_5ep_b512_20260920`。正式训练已启动，启动记录中已观察到连续的 optimizer updates。`launch_status.json` 是归档时的状态快照；本目录不表示训练已完成，也没有本次新训练的最终评测成绩。

## 初始化与检查

- 从固定 Blob 来源下载 RefDrone Stage 2 VLM，14 个文件 SHA256 全部匹配。
- 使用原 HUGE 训练前 mature base 组装，输出 SHA256 与历史 RefDrone base 一致：`e016e41b329f4b16ce8a927d38990f69f1b63a0950c7b9efe26cdf10d55df162`。
- 新权重替换 626 个 VLM 张量，逐元素保留 112 个 projector 和 416 个 action 张量。
- 实际完整 VLA 载入后，626 个 VLM 张量与候选模型一致；真实 HUGE 样本前向/反向通过，三个模块组均有有限非零梯度，预测为 `[1,20,4]`。
- 首步训练 loss 为 `0.6647538542747498`；每步 loss、epoch、学习率和吞吐保存在运行目录的 `metrics.jsonl`。

四张 A100 80GB、每卡 batch 8、累积 16、全局 batch 512、5 epoch、16,798 步、seed=42。从 RefDrone base 的 step 0 开始，继续联合更新 VLM、projector 和动作模块。

环境为 Python 3.10.21、PyTorch 2.6.0+cu124、Transformers 4.57.0、Accelerate 1.5.2、DeepSpeed 0.16.9。独立环境和源码在本实验目录，CUDA 12.4 / GCC 12.4 工具链位于 `/root/workspace-ll/.tools/cuda124`。本机 `env.local` 保存路径配置。

## 查看本机训练

```bash
TMUX_TMPDIR=/root/workspace-ll/.tmp tmux attach -t hugebench_refdrone_20260920
```

退出查看并保留训练：按 `Ctrl+b`，再按 `d`。

```bash
tail -f /root/workspace-ll/experiments/starvla_pi_qwen3vl_2b_refdrone/artifacts/checkpoints/hugebench_qwen3vl2b_refdrone_5ep_b512_20260920/metrics.jsonl
```

checkpoint 每 1,000 步保存，全部完成后写入该 run 的 `final_model/pytorch_model.pt`。源 base、此前 HUGE 最终权重和历史实验输出分别保留。

## 归档证据

- `base_conversion_manifest.json`：本次组装的来源、哈希和逐张量验证。
- `vlm_download_manifest.json`：Blob VLM 下载与校验记录，无凭据。
- `preflight.json`：本机 GPU 检查。
- `source_config.yaml` / `config.full.yaml`：本机正式配置。
- `launch_status.json`：启动后观察到的训练进度。

权重索引与恢复命令见仓库根目录 [model_assets](../../../../model_assets/README.md)。
