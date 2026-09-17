# Low-Altitude-3D-LLM

## StarVLA-PI + Qwen3-VL-2B on HUGE-Bench

[完整复现说明](experiments/starvla_pi_qwen3vl_2b/README.md)：从官方 StarVLA-PI 和 Qwen3-VL-2B 权重出发，构建 2B VLA，进行 Bridge/RT-1 对齐与联合训练，再迁移到 HUGE-Bench 四维无人机动作空间，完成训练和官方闭环评测。

模型改动、训练配方、数据与源码版本、tmux 启动器均包含在实验目录中。权重、数据、环境和运行输出通过脚本生成，不进入 Git。
