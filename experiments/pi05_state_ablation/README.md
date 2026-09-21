# pi0.5：当前状态输入对照

本实验比较 **首帧 RGB + 当前 RGB + 任务指令** 与 **相同输入 + 当前四维状态**。
两组分别从官方 `pi05_base` 初始化，用同一份 HUGE-Bench train 数据和归一化统计训练。
本目录发布的是配方、源码恢复补丁和启动入口；**尚未执行新的配对训练，也没有状态版训练成绩或 checkpoint**。

## 输入与唯一处理变量

| 项目 | `no_state` | `state` |
|---|---|---|
| 图像 | 轨迹首帧、当前帧，256×256 → 224×224 | 相同 |
| 指令 | 数据集 task 文本 | 相同 |
| 当前状态 | 数据加载但不进入模型 | `[x, y, z, yaw_rad]` 进入文本 |
| 动作标签 | 当前时刻起连续 20 步 `[dx, dy, dz, dyaw]` | 相同 |
| `discrete_state_input` | `False` | `True` |

状态先按训练集 q01/q99 归一化，再用 256 档量化，形成
`Task: <任务>, State: <四个编号>;\nAction: `。每个样本只给当前状态，不提供未来真实状态。
pi0.5 的连续 state encoder 在两个分支中均不存在；`state` 分支经语言 token 使用状态。
这与两组 StarVLA 的**状态信息和离散化思路**一致，文本模板和 tokenizer 不同。
OpenPI 动作头内部将四维动作补齐为 32 维；StarVLA 是原生四维头，因此也不是完全相同的架构。

## 配对预算

| 参数 | 两组共用 |
|---|---|
| 数据 | `task_overall/train`，5,175 episodes / 1,720,096 frames，5 Hz |
| 初始化 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| 默认预算 | 13,438 次更新，沿用历史约 1 epoch 配方 |
| `--epochs 5` | 67,190 次更新，沿用历史约 5 epoch 配方 |
| 全局 batch / FSDP | 128 / 默认 8 卡；`--fsdp-devices 4` 保持全局 batch 128 |
| 随机种子 | 42 |
| 学习率 | warmup 1,000 步，cosine 2.5e-5 → 2.5e-6 |
| 冻结 / EMA | 沿用同一 `pi05_overall` 配方，两组一致 |
| 动作预测 / 执行 | horizon=20；闭环每次执行 10 步 |

这里的 1/5 epoch 是历史名称：`steps × 128` 分别为 1,720,064 / 8,600,320 个样本，
与数据帧数的 1/5 倍略有取整差异。它也不同于 StarVLA 的 batch 512、16,798 步预算。
公平的状态消融应运行本目录的**两组新训练**，固定预算、拓扑、seed、统计和评测分片；
已有无状态 checkpoint 可作为历史参考，但不能替代对训练条件的核对。

## 源码与运行环境

所有操作默认写入本目录的 `third_party/` 和 `runtime/`，不改动正在运行的 StarVLA。
源码来自固定的 XPolicyLab commit，补丁恢复 Blob 中实际使用的 HUGE 输入适配器、训练配置、
闭环推理脚本、依赖声明和 `uv.lock`。不需要 Blob 凭据即可恢复这些代码。
[SOURCE.lock.json](SOURCE.lock.json) 记录上游 commit、补丁和五个恢复文件的 SHA256。
上游代码沿用 Apache-2.0，许可证见 [UPSTREAM_LICENSE](UPSTREAM_LICENSE)。

```bash
cd experiments/pi05_state_ablation
python3 scripts/setup_sources.py

# 需要 uv、Python 3.11–3.13，以及与原配方兼容的 NVIDIA CUDA 环境。
# 安装仅发生在新建的实验源码目录中。
OPENPI="$PWD/third_party/XPolicyLab/policy/Pi_05/openpi"
UV_CACHE_DIR="$PWD/runtime/cache/uv" GIT_LFS_SKIP_SMUDGE=1 \
  uv sync --project "$OPENPI" --frozen --group lerobot
PY="$OPENPI/.venv/bin/python"
```

LeRobot 固定为 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，`datasets==3.6.0`，
用于读取现有 LeRobot v2.1 数据。不要直接替换为 LeRobot v3 数据加载器。
归一化统计为两个历史配置共用的训练集统计，包含 SHA256；本脚本不扫描测试集估计统计。

## 查看和执行训练

无需安装 ML 依赖即可查看计划；`plan` 不创建目录、不启动训练：

```bash
python3 scripts/run.py plan --variant no_state --epochs 1
python3 scripts/run.py plan --variant state --epochs 1
python3 -B scripts/test_contract.py
```

`DATA` 指向已有 `HUGE_data/data_traj`，其下为 `train/`、`test_seen/`、`test_unseen/`。
两组使用相同 GPU 数；四卡示例：

```bash
DATA=/absolute/path/to/HUGE_data/data_traj

# 准备只创建数据链接和复制统计；check 解析真实 OpenPI 配置与变换，不更新权重。
python3 scripts/run.py prepare --variant state --data-root "$DATA" --fsdp-devices 4
CUDA_VISIBLE_DEVICES='' "$PY" scripts/run.py check \
  --variant state --data-root "$DATA" --fsdp-devices 4

# 在 GPU 空闲时分别执行。两个命令各自从 pi05_base 初始化。
"$PY" scripts/run.py train --variant no_state --data-root "$DATA" --fsdp-devices 4
"$PY" scripts/run.py train --variant state --data-root "$DATA" --fsdp-devices 4
```

五轮对照给**两个命令同时**增加 `--epochs 5`。`--base-weights /path/to/pi05_base/params`
可使用已有原始 base；不要把无状态 HUGE final 权重冒充共同初始化。
`--work-dir` 可指定独立输出位置，`--run-name` 可区分重复实验。

新运行拒绝覆盖已有目录。恢复需显式 `--resume`，并要求记录中的配方完全一致；
优化器及 checkpoint 恢复交给原 OpenPI。每组配方保存在
`runtime/records/<config>/<run>/recipe.json`，权重在
`runtime/checkpoints/<config>/<run>/<step>/`。
默认关闭 W&B，训练仍打印 loss，并保存原生 checkpoint。

## 闭环评测

评测需要已有 HUGE-Bench 3DGS renderer 和完整测试场景；本目录不自动占用 GPU 或启动渲染器。
启动方式可参考 [原 pi05 脚本](../pi05_hugebench/scripts/eval_rollout.sh)，
其中渲染器部分可复用，策略部分用下面的入口，以确保按**该训练分支**恢复状态输入。

```bash
"$PY" scripts/run.py rollout --variant state --epochs 1 --fsdp-devices 4 \
  --data-root "$DATA" \
  --checkpoint "$PWD/runtime/checkpoints/pi05_overall_state/paired_1ep_seed42/13437" \
  --host 127.0.0.1 --port 5550 --num-trajs 2 --eval-tag smoke
```

`--num-trajs 2 --eval-tag smoke` 是冒烟测试；正式结果使用 `--num-trajs 0 --eval-tag full`。
`--split test_seen,test_unseen` 共 993 条轨迹。多进程时设置 `--num-shards`、`--shard-index`，
每个进程连接各自 renderer，结果按 shard 分开写入；汇总时合并轨迹，再运行固定版本的
HUGE-Bench `metric.py` / `metric_paper_aligned.py`。两组固定相同 shard 数、seed、exec_steps=10、
轨迹范围和指标版本。入口拒绝用另一个分支的 checkpoint 评测，也拒绝覆盖已有轨迹。

## 历史证据与验证范围

[历史开环测试](records/open_loop_probe_20260825.json) 是从已有文字记录转录的汇总：
两个输入配置各加载**同一份官方 HUGE_PI05 权重**，在相同 200 个 test_seen 样本上推理。
它没有训练更新，不能作为“带状态重新训练”的成绩，也不能证明状态输入本身有害。
原始逐样本文件没有恢复，不能从本摘要重算统计。

在已检查的本地、Blob `ll_0831` 和 `ll_0909` 中，状态版只有配置、统计及该推理测试记录，
没有找到正式训练日志或 checkpoint。原实验说明见
[历史记录](../pi05_hugebench/legacy/docs/PI05_XPOLICYLAB_BRINGUP.md)。

本次验证覆盖源码补丁在固定 commit 上应用、恢复文件 SHA256、Python 语法、配对计划和数据准备保护。
未安装新的 GPU 训练环境、未运行真实模型前向、训练或闭环评测；这些运行验证需在环境恢复后完成。
