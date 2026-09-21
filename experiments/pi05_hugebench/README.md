> **2026-09-21 导入说明：** 本文保留了早期 1 epoch 实验说明。Blob 后续已有 5 epoch 训练和完整评测产物；本次只发布 base、官方基线和 5 epoch 最终 checkpoint `67189`。请先阅读 [IMPORT.md](IMPORT.md)，不要把下方旧状态当作当前发布范围。

# XPolicyLab pi0.5-base × HUGE-Bench

本目录是该实验的唯一规范入口，用于：

1. 从 XPolicyLab 的 `pi0.5-base` 权重初始化；
2. 在 HUGE-Bench `task_overall/train` 上训练 `pi05_overall_run1`；
3. 对训练后权重执行 3DGS 闭环 rollout；
4. 分别保存通用指标和论文对齐指标。

## 模型身份

- 初始化权重：`gs://openpi-assets/checkpoints/pi05_base/params`
- 训练配置：`pi05_overall`
- 训练运行：`pi05_overall_run1`
- 已保存步数：`4000`、`8000`、`12000`、`13437`
- 默认评测权重：`artifacts/checkpoints/pi05_overall/pi05_overall_run1/13437`

`13437` 是 13,438 次训练迭代的最后一个零基编号。脚本默认解析最大数字 checkpoint，也可通过 `CHECKPOINT_STEP=12000` 显式选择。

## 目录

```text
pi05_hugebench/
├── artifacts/              # checkpoint、训练资产、缓存和审计文件
├── configs/                # 唯一实验参数与路径定义
├── data/                   # LeRobot 数据入口（链接，不复制原始数据）
├── docs/                   # 来源、结果与运行环境说明
├── external/               # XPolicyLab/HUGE-Bench/3DGS 仓库链接
├── legacy/                 # 历史脚本、日志和文档，只供追溯
├── logs/                   # 规范化训练/评测日志
├── manifests/              # 机器可读的实验和资产清单
├── results/
│   ├── baselines/          # 官方 HUGE_PI05 对照结果
│   ├── smoke/              # 新训练权重冒烟结果
│   └── trained/            # 新训练权重正式结果
└── scripts/                # prepare/train/rollout/metrics 入口
```

大型目录均保留单份实体。工作区原有路径使用兼容链接，因此 XPolicyLab 配置中的旧绝对路径仍然有效。

## 使用

```bash
cd /home/aiscuser/workspace-ll/experiments/pi05_hugebench

# 路径和数据链接预检
bash scripts/prepare.sh

# 继续训练（已有 checkpoint 时自动 --resume）
bash scripts/train.sh

# 先用 4 条 seen episode 做冒烟
bash scripts/smoke.sh

# 默认评测 step 13437，seen + unseen 共 993 条
bash scripts/eval.sh

# rollout 和指标也可分开执行
bash scripts/eval_rollout.sh
bash scripts/eval_metrics.sh
```

常用覆盖参数：

```bash
CHECKPOINT_STEP=12000 SEED=0 bash scripts/eval.sh
NUM_SHARDS=2 NUM_TRAJS=8 bash scripts/eval_rollout.sh test_seen
RENDERER_PYTHON=/path/to/gs/python bash scripts/smoke.sh
METRIC_PYTHON=/path/to/metric/python bash scripts/eval_metrics.sh
```

所有横向对比必须固定 `NUM_SHARDS`、`SEED` 和 `EXEC_STEPS`；flow-matching 采样随机序列会受分片方式影响。

## 当前状态

- 训练：已完成 1 epoch，最终 checkpoint 为 step `13437`。
- 官方基线评测：已完成 993 条轨迹，详见 `results/baselines/official_pi05/`。
- 新训练权重评测：尚未运行，`results/trained/` 当前不应出现正式指标。
- 运行环境：已于 2026-08-30 恢复并通过 JAX GPU、3DGS CUDA 单帧渲染及单轨迹指标验证；见 `docs/runtime.md`。

结果归属和已知指标见 [docs/results.md](docs/results.md)，代码与权重来源见 [docs/provenance.md](docs/provenance.md)。
