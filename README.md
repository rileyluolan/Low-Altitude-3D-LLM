# Low-Altitude-3D-LLM

## StarVLA-PI + Qwen3-VL-2B on HUGE-Bench

[完整复现说明](experiments/starvla_pi_qwen3vl_2b/README.md)：从官方 StarVLA-PI 和 Qwen3-VL-2B 权重出发，构建 2B VLA，进行 Bridge/RT-1 对齐与联合训练，再迁移到 HUGE-Bench 四维无人机动作空间，完成训练和官方闭环评测。

模型改动、训练配方、数据与源码版本、tmux 启动器均包含在实验目录中。权重、数据、环境和运行输出通过脚本生成，不进入 Git。

### 已完成的训练与测试结果

[5 epoch 完整结果](experiments/starvla_pi_qwen3vl_2b/records/hugebench_5ep_b512/README.md)：16,798 步全量 loss、训练配置、993 条正式评测指标、逐轨迹 CSV 和视频生成清单。

### VLM backbone 替换实验

[实验复现说明](experiments/low_altitude_3d_llm/README.md)：下载微调过的 VLM，将其接入 HUGEBENCH 训练前的 mature base，保留原 projector/action 权重，在独立目录中进行四卡 5 epoch 训练、官方评测与视频生成。

两个实验的代码和已完成结果位于 `feat/starvla-qwen3vl2b-hugebench` 分支。
权重、原始数据、运行环境、原始轨迹及视频留在实验存储中；已完成实验的轻量结果归档在 `records/`。

## 权重文件

两个实验的实际权重分片保存在 [GitHub Release](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/starvla-artifacts-20260921)。
下载和还原说明见 [model_assets](model_assets/README.md)；源码树里的 `artifacts/README.md` 提供对应入口。

## pi0.5 / HUGE-Bench 实验

[pi05_hugebench](experiments/pi05_hugebench/IMPORT.md)：从 Blob `ll_0909` 导入的代码、配置和实验记录。
本次权重范围为 pi0.5 base、官方基线及 5 epoch 最终 checkpoint `67189`；实际二进制已发布到 [pi05 Release](https://github.com/rileyluolan/Low-Altitude-3D-LLM/releases/tag/pi05-hugebench-5ep-20260921)。

[状态输入对照](experiments/pi05_state_ablation/README.md)：固定共同初始化、数据和训练预算，比较
`discrete_state_input=False/True`；包含源码恢复补丁、配对训练/评测入口与历史开环测试说明。
该状态对照尚未完成正式训练，已有无状态模型成绩不代表状态版成绩。
