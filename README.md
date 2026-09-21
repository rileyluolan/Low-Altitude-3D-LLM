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
