# 当前脚本入口

在实验根目录运行以下入口；参数与验证结果见上一级 `README.md`。

| 入口 | 用途 |
|---|---|
| `setup_sources.sh` | 固定版本的官方 HUGE-Bench 与 Gaussian Splatting 源码及适配 |
| `setup_environment.sh` | 本实验独立 Python、训练依赖与渲染 CUDA 扩展 |
| `prepare_data.sh` | 连接现有 HUGE-Bench 训练集并生成索引 |
| `check.sh` | 权重、数据映射及运行环境检查 |
| `train_smoke.sh` | 四卡短训练，2 次优化器更新，梯度累积 2 |
| `train.sh` | 四卡正式训练，每卡 batch 8、梯度累积 16，全局批量 512，5 epoch |
| `eval_smoke.sh` | 四卡短闭环验证；默认使用现有初始化权重 |
| `eval.sh` | 测试集闭环推理和官方指标计算 |
| `eval_metrics.sh` | 对已保存的轨迹重新计算官方指标 |

`lib/` 是公共路径和环境配置，`tools/` 是这些入口调用的 Python 工具。

OXE 预训练、权重构建和下载入口见上一级 `README.md`；当前仓库也提供
`build_oxe_base.sh`、`train_oxe.sh`、`download_weights.sh` 等复现入口。
