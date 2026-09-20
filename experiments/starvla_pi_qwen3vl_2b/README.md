# StarVLA-PI → Qwen3-VL-2B → HUGE-Bench

本目录保存模型改造到训练、评测的完整代码路径。它以固定版本的上游源码和公开权重为输入，包含构建脚本、源码补丁、OXE 两阶段训练及 HUGE-Bench 适配器。

**已完成：**[完整训练 loss 和 993 条正式测试结果](records/hugebench_5ep_b512/README.md)。
后续含 RefDrone 的 VLM 替换实验见[独立实验目录](../starvla_pi_qwen3vl_2b_refdrone/README.md)。

当前 HUGE-Bench 实验采用 **4 × A100 80GB、5 epoch、global batch 512、16,798 次优化器更新**。5 epoch 是当前实验预算；官方 π0.5 的 25,000 次更新属于另一训练预算。两者的全局 batch 和动作 horizon=20 对齐，总更新数不同。

## 目录与入口

```text
SOURCE.lock.yaml        上游 Git/Hugging Face 版本、权重 SHA256
MODEL_ADAPTATION.md      backbone、动作模块、投影层和四维迁移的说明
configs/                OXE/HUGE 配方、DeepSpeed 和依赖版本
patches/                固定上游版本上的三份源码补丁
assets/                 HUGE 训练集归一化统计
integration/            HUGE 数据注册、图像/状态/动作字段映射
scripts/pretraining/    权重构建、OXE 数据准备、成熟权重导出
scripts/tools/          配置解析、迁移规则、校验和评测适配器
records/                当前实验配方和已有权重溯源，不是运行目录
tests/                  不占用 GPU 的权重迁移检查
```

常用入口都在 `scripts/`，辅助 Python 文件不需要逐个手动执行：

| 入口 | 用途 |
| --- | --- |
| `setup_sources.sh` | 获取并锁定 StarVLA、HUGE-Bench、Gaussian Splatting、PyTorch3D；应用补丁 |
| `setup_environment.sh` | 建立 Python 3.10 环境并安装训练和渲染依赖 |
| `download_weights.sh` | 下载固定版本的官方 4B VLA 源权重和 Qwen3-VL-2B |
| `download_oxe.sh` | 下载 Bridge/RT-1 元数据、有效语言轨迹及所需单路视频 |
| `build_oxe_base.sh` | 替换 backbone、映射动作层并初始化投影层，生成 7D/16-step 初始权重 |
| `train_oxe.sh` | 5k 对齐 → 50k 联合训练 → 成熟 7D 权重 → HUGE 4D/20-step 权重 |
| `download_hugebench.sh` | 下载官方训练/测试轨迹及七个渲染、碰撞场景 |
| `prepare_data.sh` / `check.sh` | 建立数据视图、索引和注册器；检查权重、数据与运行依赖 |
| `train_smoke.sh` / `train.sh` | HUGE 两步联通测试 / 正式五 epoch 训练 |
| `train_tmux.sh` | 在独立 tmux 会话中启动命令，保留进度条并记录日志 |
| `eval_smoke.sh` / `eval.sh` | 闭环联通测试 / 完整 seen、unseen 评测 |
| `eval_metrics.sh` | 使用固定版本的官方 `metric.py` 重算指标 |

## 1. 源码与环境

Linux、NVIDIA GPU、Conda、Git、tmux、CUDA 12.4 toolkit 和兼容的 C++ 编译器是前提。当前运行环境为 Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.57.0、Accelerate 1.5.2、DeepSpeed 0.16.9；完整工作环境约束见 `configs/environment_lock.txt`。

```bash
cd experiments/starvla_pi_qwen3vl_2b
cp env.example env.local
# 编辑 env.local 中的 CUDA_HOME、CONDA_BIN；按需指定 CC/CXX。
bash scripts/setup_sources.sh
bash scripts/setup_environment.sh
```

默认源码在本目录 `third_party/`；默认 Conda 环境在 `.venv/`。**不复制旧机器的 `.venv`**。PyTorch3D 只安装官方纯 Python transforms；实际渲染由 Gaussian Splatting 的两个 CUDA 扩展完成。

`env.local` 中可设置 `MODEL_ROOT`、`HUGE_DATA_ROOT`、`BASE_CKPT` 等绝对路径，共享已经下载的输入。路径统一从 `scripts/lib/common.sh` 和 `scripts/tools/experiment_paths.py` 解析，启动前生成的绝对路径配置保存在被 Git 忽略的 `runtime/` 中。

## 2. 从官方权重构建 2B VLA

```bash
bash scripts/download_weights.sh
bash scripts/download_oxe.sh --workers 16
bash scripts/build_oxe_base.sh
```

构建脚本校验两个源权重的 SHA256。输出：

```text
artifacts/base/qwen3vl_2b_pi_v3_bridge_rt1_init/
  checkpoints/base_pytorch_model.pt
  config.yaml / config.full.yaml
  dataset_statistics.json
  conversion_manifest.json
```

该权重包含官方 Qwen3-VL-2B、由源 VLA 迁移的全部 416 个目标动作张量，以及新初始化的 28 个投影层。它还需要下面的对齐训练。

### OXE 训练配方

| 阶段 | 更新次数 | 可训练模块 | 学习率 | 全局 batch |
| --- | ---: | --- | --- | ---: |
| 对齐 | 5,000 | 仅 `project_layers` | 1e-4 | 128 |
| 联合 | 50,000 | backbone、projectors、action model | 1e-5 / 1e-4 / 1e-4 | 128 |

Bridge 与 RT-1 的抽样权重各 0.5；只使用有效语言轨迹。Bridge 保留 38,660 条、1,305,714 帧；RT-1 保留 87,204 条、3,786,152 帧。相机、提示词、归一化、7D 动作和 horizon=16 在配置及数据准备脚本中固定。

历史配方为 **8 GPU × microbatch 1 × accumulation 16**。在四卡机器复现训练预算时使用 **4 × 1 × 32**：

```bash
OXE_NUM_PROCESSES=4 TRAIN_GPUS=0,1,2,3 SESSION=oxe_2b \
  bash scripts/train_tmux.sh bash scripts/train_oxe.sh
tmux attach -t oxe_2b
```

要采用历史八卡拓扑，设置 `OXE_NUM_PROCESSES=8 TRAIN_GPUS=0,1,2,3,4,5,6,7`。四卡配方保持全局 batch 和更新次数；GPU 数量变化会改变随机数消费和分布式数值行为，不保证逐位重现历史权重。

流水线检查两阶段保存的配置、最终 step 记录和 checkpoint 后才导出成熟权重。中断后的 `RESUME=1` 只恢复权重和步数，**不恢复优化器状态或数据迭代位置**；这不是无损续训。完成后会自动生成：

```text
artifacts/base/qwen3vl_2b_pi_v3_bridge_rt1_mature/
artifacts/base/qwen3vl_2b_pi_v3_hugebench_mature_base/
```

后者选取源动作通道 `[0,1,2,5]`，对应 `dx,dy,dz,dyaw`，使用四维输入/输出头，不做动作 padding。除了四个通道相关张量的切片，其余权重完整保留。动作语义仍需在 HUGE 训练中学习，详见 [模型适配说明](MODEL_ADAPTATION.md)。

## 3. HUGE-Bench 训练

已有成熟 4D 权重时，可通过 `BASE_CKPT` 指定它，并从本节开始；该方式不重做 OXE。需要保留权重目录的配置、统计和转换 manifest，不能只复制 `.pt`。

```bash
bash scripts/download_hugebench.sh --part all
bash scripts/prepare_data.sh
bash scripts/check.sh

# GPU 空闲时先验证训练联通性。
bash scripts/train_smoke.sh

RUN_ID=hugebench_qwen3vl2b_5ep_b512 SESSION=hugebench_2b \
  bash scripts/train_tmux.sh
tmux attach -t hugebench_2b
```

终端会显示优化步进度条、loss、epoch、学习率和吞吐。`Ctrl+b` 后按 `d` 离开会话；训练继续。tmux 日志在 `logs/`，数值日志在 `artifacts/checkpoints/<RUN_ID>/metrics.jsonl`。`watch -n 1 nvidia-smi` 查看显卡状态。

正式配置为每卡 batch=8、梯度累积=16、4 卡，全局 batch=512。`ceil(5 × 1,720,096 / 512) = 16,798`，满 batch 取整多处理 96 个样本。数据按帧打乱遍历；`action_mode=abs` 表示直接使用数据中已经存好的 delta action，不再计算一次差分。

离线监督训练直接读取轨迹中的首帧和当前 RGB，不需要启动实时渲染器。闭环评测需要渲染器和场景网格。

GPU 忙碌时可以只检查配置：

```bash
DRY_RUN=1 bash scripts/train.sh
DRY_RUN=1 OXE_NUM_PROCESSES=4 bash scripts/train_oxe_phase.sh joint
```

## 4. 官方闭环评测

请在四张卡空闲后运行评测。默认 GPU 0、1 渲染，GPU 2、3 运行策略；运行期间会创建独立子进程并在结束时回收。

```bash
# 少量轨迹、少量步数，仅验证评测管线；默认使用 BASE_CKPT。
bash scripts/eval_smoke.sh

# 同一 RUN_ID 下优先 final_model，否则选最新周期 checkpoint。
RUN_ID=hugebench_qwen3vl2b_5ep_b512 bash scripts/eval.sh

# 也可明确指定某一轮权重。
RUN_ID=hugebench_qwen3vl2b_5ep_b512 \
CHECKPOINT_PATH=/absolute/path/to/steps_16000_pytorch_model.pt \
EVAL_TAG=step16000 bash scripts/eval.sh
```

完整评测包含 576 条 test_seen 和 417 条 test_unseen 轨迹；采用官方状态更新、角度转换、动作平滑与 `metric.py`，并加载七个场景的碰撞网格。默认每次执行 10 步，预测 horizon=20，重叠平滑开启，obstacle 角度模式为 `yaw_legacy`。输出目录保存 checkpoint 身份、运行设置、轨迹、指标 JSON 和完成标记。

需要重算指标时：

```bash
bash scripts/eval_metrics.sh /absolute/path/to/rollouts
```

## 校验范围与历史记录

[VALIDATION.md](VALIDATION.md) 记录本次导出的实际检查。已验证补丁、配置、CPU 权重迁移、现有权重哈希、真实 HUGE 元数据及 tmux 启动；未在此提交副本重新运行完整 55k OXE + 5 epoch HUGE 或全量闭环评测。

`records/hugebench_5ep_b512/` 保存已完成训练的全量指标、正式闭环评测、训练配方和 base 溯源。历史 base manifest 记录了 5k+50k OXE 训练，但没有取回原始 OXE 训练日志，因此应区分“已有权重的溯源记录”和“在新环境中完成复现”。

权重、数据、`.venv`、缓存和原始运行日志被 `.gitignore` 排除。依赖项目、权重和数据保留各自的许可证，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
