# Low-Altitude-3D-LLM：VLA 联合训练

将含 RefDrone 微调的 Qwen3-VL-2B 接入既有 mature VLA base，再进行 HUGE-Bench 5 epoch 联合训练。
本目录对应本机实验 `low_altitude_3d_llm`；Git 中的启动器已改为可配置路径。

## 已完成的最终权重与视频

实际发布 run 为 `hugebench_qwen3vl2b_refdrone_5ep_b512_20260920`，已完成 5 epoch / 16,798 次更新，
最终模型为 `artifacts/checkpoints/<run>/final_model/pytorch_model.pt`。
完整 Seen 576、Unseen 417 评测及全部 993 个视频已完成。
最终 checkpoint、必要配置和全部逐条视频见 [Hugging Face 仓库](https://huggingface.co/lld-koi/Low-Altitude-3D-LLM/tree/main/experiments/low_altitude_3d_llm)；
独立下载命令见 [三版最终 checkpoint 与视频](../../model_assets/FINAL_CHECKPOINTS.md)。

## 模型替换

- 起点是 **HUGE 训练前**的 `qwen3vl_2b_pi_v3_hugebench_mature_base`，不是原实验的 HUGE final checkpoint。
- 原 base 的 SHA256 为 `87e38d987fd2d1b7fd50b77529717d4fc19111b1c0ca58096f24e933b79b6c4b`。
- 新 VLM 来自 Blob `output/liyan/lnj/images/0729/qwen3vl_stage2/`，即含 RefDrone 版本。
  [文件与 SHA256 清单](assets/refdrone_vlm.lock.json)固定这次实验实际使用的 14 个文件。
- 组装脚本替换 `qwen_vl_interface.model.*` 全部 626 个张量；其中 597 个数值与原 base 不同。
  112 个 projector 张量和 416 个动作模块张量逐元素保留；动作、状态维度均为 4，horizon 为 20。
- VLM 原始配置由 Transformers 5.2 保存。脚本核对架构、RoPE、词表、BPE merges、特殊 token 和模板后，
  使用原官方 2B 模型的 4.57 兼容输入配置，并链接新的 VLM 权重。加载禁止忽略尺寸不匹配。
- 不额外进行 OXE 或单独的 projector 对齐；正式训练从 step 0 开始更新 VLM、projector 和动作模块。

仅修改 `base_vlm` 路径不能完成替换，因为完整 VLA checkpoint 会重新覆盖 VLM；必须先运行组装脚本。
输入模型只读，组装拒绝覆盖目标 checkpoint 或非空的兼容模型目录。

## 复现步骤

需要先获得原 mature base（包含 checkpoint、`conversion_manifest.json`、`dataset_statistics.json`），
以及官方 `Qwen3-VL-2B-Instruct` 的输入配置。原 2B VLA 的构建、OXE 对齐及联合训练过程见
[原实验](../starvla_pi_qwen3vl_2b/README.md)。本实验锁定已审计 base 的哈希；重新训练得到的权重
通常不会逐位一致，应另建实验记录，不能声称是同一初始化。

从 Blob 下载原始 VLM 时需要有权访问该容器的凭据；SAS 凭据不提交 Git。默认读取 `~/.blob_config.json`
中的 `sas_url`、`sas_token`，也可通过 `BLOB_CONFIG` 或环境变量设置。约下载 4.9 GB。

在本目录执行：

```bash
cp env.example env.local
# 编辑 env.local：设置 CUDA_HOME、原 base、官方 2B 资源和已有 HUGE 数据路径。
bash scripts/setup_sources.sh
bash scripts/setup_environment.sh  # 如复用兼容 .venv，可跳过；不需要复制环境文件
source scripts/lib/common.sh
"$VENV_ROOT/bin/python" scripts/tools/download_vlm.py
"$VENV_ROOT/bin/python" scripts/tools/assemble_base.py
# 若没有 HUGE 数据：bash scripts/download_hugebench.sh
DRY_RUN=1 bash scripts/train.sh
bash scripts/train_tmux.sh
```

源码按 [SOURCE.lock.yaml](SOURCE.lock.yaml) 固定版本并应用本目录补丁，使用本实验独立源码目录。
请勿将 `STARVLA_ROOT` 指向正在运行的训练源码。
默认 tmux 启动先做真实 HUGE 样本前向/反向验证和完整 VLM 载入一致性检查，再启动四卡训练。
检查不执行 optimizer step，结果写入 `runtime/preflight.json`。

```bash
tmux attach -t hugebench_refdrone_reproduce
# Ctrl+b 后按 d，离开终端但继续训练。
```

历史本机 run 的 tmux 名是 `hugebench_refdrone_train`；上面的默认名字用于新复现。
`RUN_ID` 已存在时启动器拒绝覆盖。`RESUME=1` 只恢复模型和步数，不恢复 optimizer moments。

## 训练设置

| GPU | 每卡 batch | 梯度累积 | 全局 batch | 更新次数 | 数据帧 | seed |
|---|---:|---:|---:|---:|---:|---:|
| 4 × A100 80GB | 8 | 16 | 512 | 16,798 | 1,720,096 | 42 |

5 epoch 按整数更新取整，多 96 个样本；warmup 840 步。
VLM / projector / action 学习率分别为 `1e-5 / 2.5e-5 / 1e-4`，每 1,000 步和最后保存模型。
训练量沿用原 5 epoch 实验；batch 和 horizon 对齐官方 pi05 配方，官方 25,000 步的总量并不相同。
除模型和输出路径外，本次正式训练超参数与原实验相同。

## 评测和视频

训练完成后，在 GPU 空闲时执行：

```bash
export RUN_ID=hugebench_qwen3vl2b_refdrone_5ep_b512_20260919
export CHECKPOINT_PATH="$EXPERIMENT_ROOT/artifacts/checkpoints/$RUN_ID/final_model/pytorch_model.pt"
bash scripts/eval.sh
# 默认完整 seen/unseen，不限制轨迹数或长度，并调用官方 metric.py。
# 可以为刚完成的评测补视频，视频保存在各自已有轨迹目录：
bash scripts/render_eval_videos.sh \
  --eval-dir "$EXPERIMENT_ROOT/results/$RUN_ID/pytorch_model/full_seed_42"
```

评测默认两组 renderer/policy 配对使用四卡，`exec_steps=10`、`seed=42`、`smooth_overlap=true`，
动作转换沿用官方 HUGE-Bench，障碍物角度模式为 `yaw_legacy`。`MODEL_ROOT` 指向兼容的 RefDrone VLM。
补视频从已有 GT/pred 轨迹重渲染，不重新跑 policy。

## 文件与证据

- `scripts/tools/download_vlm.py`：断点续传、ETag/长度/MD5 和固定 SHA256 校验。
- `scripts/tools/assemble_base.py`：权重替换、配置兼容、保存重载逐张量校验。
- `patches/`：实际运行 StarVLA 的内容补丁和官方评测依赖补丁。
- `configs/`、`integration/`：四卡训练、环境版本、四维动作数据接入。
- [启动审计记录](records/hugebench_refdrone_5ep_b512/README.md)：模型来源、旧权重不变校验、真实样本验证。
- [发布验证说明](VALIDATION.md)：移植脚本的检查范围。

归档记录保留历史绝对路径以便核对来源；执行脚本使用 `env.local` 和相对实验目录。
自有 VLM 最终评测的 `metrics.json`、`metrics_table.md` 随上述视频归档发布。原模型已完成的训练及 993 条评测结果见
[原实验结果](../starvla_pi_qwen3vl_2b/records/hugebench_5ep_b512/README.md)。
