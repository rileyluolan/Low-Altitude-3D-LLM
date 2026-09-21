# 在 HUGE-Bench 上训练 / 测试 π0.5 —— 详细方案

> 生成时间：2026-08-25　工作区：`/home/aiscuser/workspace-ll`
> 目标：用 π0.5（openpi 实现）在 HUGE-Bench 数据上做微调 + rollout 评测，复现论文 Table 2 中 `π0.5` 一行（Avg.TCR 0.581 / nDTW 0.467 / NTP 0.618 / CR 0.018 / CSPL 0.805）。

---

## 0. 工作区现状核对（已逐项扫描确认）

### 0.1 数据（全部已就位，无需再下载）

| 资产 | 路径 | 规模 | 状态 |
|---|---|---|---|
| LeRobot 轨迹数据集（= HF `yu781986168/HUGE_Dataset_v0`，即 `task_overall`） | `HUGE_data/data_traj/{train,test_seen,test_unseen}` | 181 G | ✅ 完整（`data/` + `meta/`） |
| 3DGS-Mesh 环境 | `HUGE_data/data_3d/{1_office,2_city,3_road,4_lake,no1_building,no3_door,overhead_bridge}` | 59 G | ✅ 每个 env 均有 `3dgs_ply/` + `terra_ply/` |
| **HUGE-Bench 官方微调好的 π0.5 权重** | `HUGE_data/HUGE_PI05/` | 12 G | ✅ `params/`（orbax OCDBT）+ `assets/task_overall/train/norm_stats.json` |

#### `HUGE_PI05` 是什么 —— 由 checkpoint 元数据实证

读 `params/_METADATA`（51 个叶子）与 `array_metadatas/process_0`，无需加载权重：

| 证据 | 结论 |
|---|---|
| 有 `time_mlp_in` / `time_mlp_out`；**无** `state_proj`；**无** `action_time_mlp_*` | **π0.5 架构**（adaRMS 注入 flow timestep），不是 π0 |
| 无任何 `lora` 键 | **全参微调**，非 LoRA |
| `action_in_proj.kernel` 分片 shape `[4,1024]`（8 卡切 axis-0）→ 完整 `[32,1024]`；`action_out_proj.kernel` `[128,32]` → `[1024,32]`；`action_out_proj.bias` `[4]` → `[32]` | **`action_dim = 32`** |
| action expert width = 1024 | `gemma_300m` ✓ |
| asset id = `task_overall/train`，与 config 的 `repo_id` 一致 | 训练数据 = HUGE 全任务 train split |

即 **`HUGE_PI05` = `pi05_base` 在 `task_overall/train` 上的全参微调**。无人机的 4 维动作并没有改变模型的动作维度 —— 它被 `transforms.PadStatesAndActions` **零填充**到 32 维送入模型，输出侧再由 `drone_policy.DroneOutputs` 取 `[:, :4]`。

> 推论：`pi05_base` 本身从未见过 4 维无人机动作，适配完全靠 "pad 到 32 + 微调"。因此**从哪个仓库取 π0.5 权重都一样**，见 §1。

LeRobot `meta/info.json` 关键字段（三个 split 一致）：

```
codebase_version v2.1 | robot_type "drone" | fps 5 | total_videos 0
features: image(256,256,3) + first_image(256,256,3) + state(4: x,y,z,yaw_rad) + actions(4)
train:       5175 episodes / 1,720,096 frames
test_seen:    576 episodes /   195,336 frames
test_unseen:  417 episodes /   144,058 frames
```

> ⚠️ **待验证项**：`info.json` 的 `total_tasks` 与 `meta/tasks.jsonl` 行数不一致（train 109 vs 1102；test_seen 13 vs 368；test_unseen 10 vs 259），`splits` 字段也写成了 `{"train": "0:109"}`。这是合并脚本留下的元数据未更新。`prompt_from_task=True` 会按 `task_index → tasks.jsonl` 取 prompt，实测前需先跑一次一致性检查（见 §3.3）。

### 0.2 代码

| 仓库 | 路径 | 状态 |
|---|---|---|
| HUGE-Bench（`jingyu198/HUGE-Bench`） | `HUGE-Bench/` | ✅ `metric.py` / `openpi/` 适配层 / `vlaser/` |
| openpi（`Physical-Intelligence/openpi` @ `15a9616`） | `openpi/` | ✅ 已合入 HUGE-Bench 适配层 |
| gaussian-splatting | `gaussian-splatting/` | ✅ `3dgs_renderer.py` 已拷入，submodule 源码在，**未编译** |

`openpi/` 相对上游的改动（`git status`）：
- `src/openpi/policies/drone_policy.py` —— 与 HUGE-Bench 版本**逐字节一致** ✅
- `scripts/action_infer.py` —— 已拷入 ✅
- `src/openpi/training/config.py` —— 已拷入 HUGE-Bench 版本，**并且额外新增了一个 `pi05_overall` config**（HUGE-Bench 上游仓库里没有这个 config，只有 8 个单任务 `pi05_*`）

`openpi/src/openpi/training/config.py` 中已有的 `pi05_overall`：

```python
TrainConfig(
    name="pi05_overall",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=20, discrete_state_input=False),
    data=MyDroneDataConfig(repo_id="task_overall/train",
                           base_config=DataConfig(prompt_from_task=True),
                           extra_delta_transform=False),
    weight_loader=weight_loaders.CheckpointWeightLoader("/path/to/pi05_base/params"),   # ← 待填
    num_train_steps=25000, batch_size=512, save_interval=5000, fsdp_devices=8,
)
```

### 0.3 环境（**这是最大的缺口**）

- ❌ 没有 `uv`；❌ 没有 jax；❌ openpi 无 `.venv`；❌ `openpi/assets/` 不存在
- ❌ `gaussian_splatting` / `vlaser` conda env 已丢失（`miniconda3` 于 08-24 14:26 被重装，`envs/` 只剩 `nyp-3dpipe`、`nyp-moge`、`nyp-vllm`），而 `run_huge_vlaser_*.sh` 仍硬编码指向 `envs/gaussian_splatting/bin/python`
- ✅ 可用底座：`/opt/conda/envs/ptca` = torch 2.8.0+cu126、CUDA 可用；`nvcc` 12.6 在 PATH
- ✅ 硬件：**8 × A100-SXM4-40GB**；`/scratch` 与 `/` 同一 7.0 T 卷，剩余 5.0 T
- ✅ 外网：`storage.googleapis.com/openpi-assets/...` 直连 200，GitHub / arXiv 可达

---

## 1. 关于 π0.5 的来源：XPolicyLab vs 本地 openpi

已拉取 `XPolicyLab/XPolicyLab`（org 仓库，43 个 policy，7742 个文件）核对 `policy/Pi_05/`：

```
policy/Pi_05/{README.md, install.sh, train.sh, process_data.sh, eval.sh,
              deploy.py, deploy.yml, model.py, setup_eval_*.sh, openpi/}
```

它就是 **vendored 的 openpi**（`src/openpi/models/pi0.py`、`pi0_config.py`、`training/config.py` 全在），额外加了三样东西：

1. `checkpoint_dir_override` 训练参数（上游 openpi 没有）
2. `PartialCheckpointWeightLoader`（给 action_dim 不匹配时用，如 54D 的 `pi05_wuji_marvin_54d`）
3. RoboDojo 基准的适配层：`wuji_policy.py`、`process_data.py`、ws/tcp policy server

**π0.5 权重加载方式（XPolicyLab 官方写法，与 openpi 上游一致）**：

```python
weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
```

### 1.1 实证：XPolicyLab 的 π0.5 与本地 openpi 是同一份模型

把 XPolicyLab 的 `policy/Pi_05/openpi/src/openpi/models/` 下的文件与本地 openpi@`15a9616` 逐行 diff：

| 文件 | diff 结果 |
|---|---|
| `models/pi0.py` | **0 行差异（逐字节一致）** |
| `models/pi0_config.py` | **0 行差异（逐字节一致）** |
| `training/weight_loaders.py` | 唯一差异：多一个 `PartialCheckpointWeightLoader` |

配合 §0.1 的 checkpoint 实证可以确认：**"XPolicyLab 的 π0.5 没见过 4 维无人机动作"不构成任何问题 —— HUGE-Bench 的 π0.5 同样没见过。** 两边都从同一个 `gs://openpi-assets/checkpoints/pi05_base/params` 出发，`action_dim` 都是 32，4 维动作都是零填充进去的。动作维度**不是**选择仓库的考量因素。

### 1.2 那么两个仓库的真实差异只在工程层

| | XPolicyLab `policy/Pi_05` | 本地 `workspace-ll/openpi` |
|---|---|---|
| 模型代码 | 与本地逐字节相同 | — |
| π0.5 权重来源 | `gs://.../pi05_base/params` | 同一个 |
| drone 适配层 | ❌ 无（是 RoboDojo/ARX-X5 的 `wuji_policy` + `process_data.py`） | ✅ `drone_policy.py` + 11 个 `pi05_*` config + `action_infer.py` 已就位 |
| 额外工具 | `PartialCheckpointWeightLoader`、`checkpoint_dir_override` | ❌ 无 |

**采用方案：继续用工作区里已有的 `workspace-ll/openpi`。** 理由只有一条 —— drone 适配层已经做好了，换仓库等于把 `drone_policy.py` / drone `TrainConfig` / `action_infer.py` 这三样再搬一次，而模型侧收益为零。

> 若后续要以 XPolicyLab 为主仓（例如要用它那 40+ policy 在 HUGE-Bench 上做横向对比），迁移成本就是上述 3 个文件，随时可做。

### 1.3 值得从 XPolicyLab 移植的一样东西：`PartialCheckpointWeightLoader`

它对应一个真实的改进方向：**把 `action_dim` 从 32 改成 4**。

现状是 32 维中 28 维恒为零（`PadStatesAndActions` 零填充），flow-matching 的回归 loss 有 7/8 打在恒零的填充维上。若改成 `action_dim=4`，梯度可全部落在真实动作上；但 `action_in_proj`（`[32,1024]`）与 `action_out_proj`（`[1024,32]`）的形状会与 `pi05_base` 不匹配，默认 `CheckpointWeightLoader` 会直接报错 —— `PartialCheckpointWeightLoader` 正是为此设计的（跳过形状不匹配的 action/state 投影层，其余照常加载，两个投影层保持随机初始化）。

⚠️ 这偏离论文"其余超参用 π0 默认"的声明，且丢弃了预训练的动作投影层，**建议作为消融而非主实验**。

---

## 2. HUGE-Bench 论文中关于 π0.5 训练的明确声明

来自 arXiv **2603.19822v4**（ECCV 2026），Appendix 0.A *Model and Training Details*，逐条摘录：

| # | 论文原文声明 | 落到 openpi 的配置项 |
|---|---|---|
| 1 | "we use both the first frame of the trajectory and the current observation as visual inputs" | `drone_policy.DroneInputs`：`base_0_rgb = first_image`，`left_wrist_0_rgb = 当前帧`，`right_wrist_0_rgb = zeros`（mask=False）—— 已实现 |
| 2 | "For π0 and π0.5, we set the **action horizon to 20**" | `Pi0Config(action_horizon=20)` |
| 3 | "each model predicts a 20-step action chunk, and we **execute the averaged actions over the first 10** predicted steps" | `action_infer.py --exec_steps 10` + `--smooth_overlap`（默认开），由 `ChunkMeanBuffer.get_mean()` 实现 |
| 4 | "We **initialize both models from their corresponding base checkpoints**" | π0.5 → `gs://openpi-assets/checkpoints/pi05_base/params` |
| 5 | "and finetune them for **5 epochs** with a **per-GPU batch size of 64**" | 见 §4.2 的步数换算 |
| 6 | "all remaining hyperparameters follow the **default settings of the original π0 implementation**" | openpi 默认：AdamW(b1=0.9, b2=0.95, wd=1e-10, clip=1.0) + CosineDecay(warmup 1000, peak 2.5e-5, end 2.5e-6, **decay_steps=30000**) + `ema_decay=0.99` |
| 7 | (Fig.11) "finetuning consistently yields better performance" than from-scratch | 说明必须加载 base 权重，`pi05_*_scratch` 只作对照 |

指标定义（§ Metrics）：TCR@{1,2,5}m、Avg.TCR、nDTW、NTP、CR（episode 级）、CSPL = `S·(1−1[C>0])·L/max(P,L)`。

> 论文**没有**说明的两点，需要我们自己定：
> (a) π0.5 的 state 是走离散 token 还是连续输入（见 §4.1，这是本方案的关键决策点）；
> (b) 5 epoch 对应的确切 step 数与 `decay_steps` 是否同步调整。

---

## 3. 环境搭建（阶段 A，约 1.5–3 h）

### 3.1 openpi（JAX）训练环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /home/aiscuser/workspace-ll/openpi
# 清掉之前 pip 误建的 "=0.1.6" 之类垃圾文件
rm -f '='*
GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot     # openpi 要求 py3.11 + jax[cuda12]==0.5.3
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run python -c "import jax; print(jax.__version__, jax.devices())"   # 期望 8 个 cuda device
```

### 3.2 3DGS 渲染环境（评测必需，训练不需要）

```bash
conda create -n gaussian_splatting python=3.11 -y
conda activate gaussian_splatting
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install plyfile tqdm opencv-python numpy scipy
cd /home/aiscuser/workspace-ll/gaussian-splatting
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
# 冒烟：起单个 renderer，确认能加载 1_office 的 ply
CUDA_VISIBLE_DEVICES=0 python 3dgs_renderer.py --host 127.0.0.1 --port 5550 \
  --ply_template /home/aiscuser/workspace-ll/HUGE_data/data_3d/{env_id}/3dgs_ply/point_cloud_utm50.ply
```

> 环境名必须叫 `gaussian_splatting`，这样现成的 `start_huge_renderers_4gpu.sh` 可以直接复用（它硬编码了 `envs/gaussian_splatting/bin/python`）。

### 3.3 数据挂载 + 元数据一致性检查

openpi/LeRobot 通过 `HF_LEROBOT_HOME` 解析 `repo_id`，需要 `task_overall/{train,test_seen,test_unseen}` 的目录形状：

```bash
export HF_LEROBOT_HOME=/home/aiscuser/workspace-ll/lerobot_home
mkdir -p "$HF_LEROBOT_HOME/task_overall"
ln -sfn /home/aiscuser/workspace-ll/HUGE_data/data_traj/train       "$HF_LEROBOT_HOME/task_overall/train"
ln -sfn /home/aiscuser/workspace-ll/HUGE_data/data_traj/test_seen   "$HF_LEROBOT_HOME/task_overall/test_seen"
ln -sfn /home/aiscuser/workspace-ll/HUGE_data/data_traj/test_unseen "$HF_LEROBOT_HOME/task_overall/test_unseen"
```

一致性检查脚本（**先跑这个，再谈训练**）：

```bash
uv run python - <<'PY'
import os, json
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
for split in ["train","test_seen","test_unseen"]:
    ds = LeRobotDataset(f"task_overall/{split}")
    s = ds[0]
    print(split, len(ds), sorted(s.keys()))
    print("  state", s["state"].shape, "actions", s["actions"].shape,
          "image", s["image"].shape, "first_image", s["first_image"].shape)
    print("  task ->", ds.meta.tasks[int(s['task_index'])] if 'task_index' in s else s.get('task'))
PY
```

要确认三件事：① 加载不报 `total_tasks` 断言错；② `first_image` 字段真实存在（`DroneInputs` 靠它做第一帧输入）；③ `task_index` 能正确索引到 1102 条 instruction。若 ① 报错，最小修复是按 `tasks.jsonl` 实际行数改写 `meta/info.json` 的 `total_tasks` / `splits`（**只改 `$HF_LEROBOT_HOME` 下的副本，不动 `data_traj` 原始数据**）。

---

## 4. 训练配置（阶段 B）

### 4.1 ⚠️ 关键决策：`discrete_state_input`

现有 `pi05_overall`（以及 HUGE-Bench 上游全部 8 个 `pi05_*` config）都写了 `discrete_state_input=False`。结合 openpi 源码，这会**让模型完全看不到无人机状态**：

- `pi0.py:93-97`：`if config.pi05: (time_mlp) else: self.state_proj = ...` → **π0.5 分支根本不创建 `state_proj`**
- `pi0.py:151`：`if not self.pi05:` 才把连续 state token 拼进 suffix → **π0.5 不走连续 state 通路**
- `config.py:126-137`（`ModelTransformFactory`，PI05 分支）：只有 `discrete_state_input=True` 时才把 state 离散化成 256 bin 塞进 prompt（`tokenizer.py:24-28`，`"Task: ..., State: 12 240 7 133;\nAction: "`）

也就是说 `pi05=True, discrete_state_input=False` ⇒ **两条 state 通路都关闭**。而 `Pi0Config.__post_init__` 的默认行为是 `discrete_state_input = pi05`（即默认 True），HUGE-Bench 是显式覆写成 False 的。

对 HUGE 任务而言 state 是 `(x, y, z, yaw)`，是相对位移预测的核心输入，丢掉它理论上应显著掉点。两种可能：(i) 作者确实这么训的，π0.5 靠 first-frame + 当前帧的视觉差隐式推断位置，0.581 的分数就是在这个设定下拿到的；(ii) 是笔误，实际跑的是别的设定。

**处理办法：两条腿走，用官方权重当裁判。**

| 分支 | 配置 | 用途 |
|---|---|---|
| `pi05_overall`（复现） | `discrete_state_input=False`（保持原样） | 严格对齐官方 config，与 `HUGE_PI05` 官方权重的评测结果对照 |
| `pi05_overall_state`（改进） | `discrete_state_input=True`, `max_token_len=200` | 消融：把 state 作为离散 token 喂进去 |

判定依据：§6 会先用官方 `HUGE_PI05` 权重在 `pi05_overall`（False）下跑 rollout。**如果能复现出 ≈0.581 的 Avg.TCR，说明 False 就是官方设定**；如果分数明显偏低，说明推理侧 config 与权重训练时的设定不匹配，改用 True 再测一次即可确定。这一步成本只有几小时 rollout，却能一次性锁死整个训练设定，务必先做。

### 4.2 步数 / batch 换算

训练集 1,720,096 帧，openpi 按帧采样（每个样本 = 1 帧 + 20 步 action chunk），故 1 epoch = 1,720,096 样本。

| global batch | 5 epoch 对应 step | 备注 |
|---|---|---|
| 512（论文的 8×64） | **16,798** | 论文设定 |
| 256 | 33,596 | A100-40G 推荐 |
| 128 | 67,191 | 保底 |

注意仓库里 `pi0_overall` / `pi05_overall` 写的是 `num_train_steps=25000` @ batch 512 ≈ **7.3 epoch**，比论文的 5 epoch 多。两者取舍：以复现论文数字为准则用 16,800；想对齐仓库 config 则用 25,000。**建议按论文走 16,800**，并在报告里注明差异。

另外 openpi 的 `CosineDecaySchedule` 默认 `decay_steps=30_000`。若 `num_train_steps=16800` 而不改 `decay_steps`，LR 只会衰减到 cos 曲线的一半（结束时约 1.3e-5）就停。论文说"其余超参用 π0 默认"，字面理解就是保留 30000；但更合理的做法是 `decay_steps = num_train_steps`。**建议：主实验保留默认（忠实复现），另跑一个 `decay_steps=num_train_steps` 的对照。**

### 4.3 显存策略（A100-40GB 的硬约束）

π0.5 ≈ 3.3 B 参数（PaliGemma 2B + Gemma-300M action expert）。openpi 官方给的全量微调门槛是 **> 70 GB 单卡**，即 A100-80G / H100 级别。我们只有 40 GB × 8，且 **openpi 不支持梯度累积**（`scripts/train.py` 里没有任何 accumulate 逻辑，且强制 `batch_size % jax.device_count() == 0`），所以只能靠 FSDP + 降 batch。

- `fsdp_devices=8`：params/Adam-state/EMA 全部切 8 份，≈ (3.3e9 × 12 B + EMA 3.3e9 × 4 B) / 8 ≈ 6.6 GB/卡，可控
- 激活才是大头：π0.5 的 `max_token_len=200`（π0 只有 48），加上 3 路 SigLIP 各 256 token，单样本序列长 ≈ 988 token。每卡 64 样本 ⇒ 63 k token 的激活，**几乎肯定 OOM**
- 因此：**per-GPU batch 从 32 起试，逐级降到 16 / 8**；`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`

递进式试探（每档只跑 20 步看是否 OOM 及 step 时间）：

```bash
cd /home/aiscuser/workspace-ll/openpi
for BS in 256 128 64; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_overall \
    --exp-name=probe_bs$BS --batch-size=$BS --num-train-steps=20 \
    --fsdp-devices=8 --no-wandb-enabled --overwrite 2>&1 | tail -20
done
```

选定能跑的最大 batch 后，按 §4.2 反推 `num_train_steps` 使总 epoch = 5。

> 若 256 都 OOM，退路（按优先级）：① `paligemma_variant="gemma_2b_lora"` + `action_expert_variant="gemma_300m_lora"` 走 LoRA（显存降到 ~20 GB/卡，但偏离论文的"全参微调"）；② 把 `right_wrist_0_rgb` 的全零图去掉，省 256 token —— 需同时改 `drone_policy.py` 和推理侧，**会与官方权重不兼容**，只能用于自训分支。

### 4.4 最终要落的 config

编辑 `openpi/src/openpi/training/config.py`：

```python
TrainConfig(
    name="pi05_overall",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=20,           # 论文声明 #2
        discrete_state_input=False,  # 保持官方原样，见 §4.1
    ),
    data=MyDroneDataConfig(
        repo_id="task_overall/train",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"   # ← 论文声明 #4
    ),
    num_train_steps=16_800,          # 5 epoch @ batch 512；按 §4.3 实测 batch 同步调整
    batch_size=512,                  # 按 §4.3 实测下调
    save_interval=2_000,
    keep_period=4_000,
    fsdp_devices=8,
    assets_base_dir="/home/aiscuser/workspace-ll/openpi_assets",
    checkpoint_base_dir="/home/aiscuser/workspace-ll/openpi_ckpts",
),
```

再复制一份 `pi05_overall_state`，仅把 `discrete_state_input` 改成 `True`。

`checkpoint_base_dir` 一定要指到 `/home/aiscuser/workspace-ll` 下（5.0 T 可用），别留在 openpi 仓库里。单个 checkpoint ≈ params 12 G + train_state（含 Adam m/v + EMA）≈ 40–50 G，`save_interval=2000` + `keep_period=4000` 下峰值占用约 250–300 G。

---

## 5. 归一化统计（norm stats）

`MyDroneDataConfig` 会去 `assets_base_dir/<config_name>/<repo_id>/norm_stats.json` 找统计量，缺失就只 warning 不报错 —— **一定要显式确认存在，否则会静默地在未归一化数据上训练**。

两条路：

**(a) 直接复用官方的（推荐，省一次 146 G 全量扫描）**
norm stats 只依赖数据和 `data_transforms`（π0 与 π0.5 在这一层完全相同），官方 `HUGE_PI05` 里那份就是在 `task_overall/train` 上算的：

```bash
export ASSETS=/home/aiscuser/workspace-ll/openpi_assets
mkdir -p $ASSETS/pi05_overall/task_overall/train
cp /home/aiscuser/workspace-ll/HUGE_data/HUGE_PI05/assets/task_overall/train/norm_stats.json \
   $ASSETS/pi05_overall/task_overall/train/
cp -r $ASSETS/pi05_overall $ASSETS/pi05_overall_state   # 两个 config 共用
```

（该文件内容已核对：`state` 4 维 mean/std/q01/q99，`actions` 4 维，数值合理 —— action std ≈ 0.52/0.51/0.39/0.035，q01/q99 ≈ ±1。）

**(b) 自己重算（作为交叉验证，可后台跑）**

```bash
cd /home/aiscuser/workspace-ll/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_overall
```

跑完后 diff 一下 (a)/(b) 两份 json，确认 mean/std 一致 —— 这同时也验证了我们的数据挂载和官方发布的是同一份。

---

## 6. 先做基线验证：官方 `HUGE_PI05` 权重直接跑评测（阶段 C）

**这一步优先级高于自己训练**，理由：① 端到端打通 renderer + rollout + metric 三段流水；② 确定 §4.1 的 `discrete_state_input` 到底该取哪个值；③ 得到一个已知答案（0.581）作为我们自训结果的对照锚点。

### 6.1 起 4 个 renderer（GPU 0-3，端口 5550-5553）

```bash
cd /home/aiscuser/workspace-ll
bash start_huge_renderers_4gpu.sh     # 现成脚本，ply_template 已指向 HUGE_data/data_3d
```

### 6.2 跑 rollout（GPU 4-7）

```bash
cd /home/aiscuser/workspace-ll/openpi
export HF_LEROBOT_HOME=/home/aiscuser/workspace-ll/lerobot_home
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
uv run scripts/action_infer.py \
  --task_id overall \
  --config_name pi05_overall \
  --checkpoint_dir /home/aiscuser/workspace-ll/HUGE_data/HUGE_PI05 \
  --splits test_seen,test_unseen \
  --exec_steps 10 \
  --out_dir /home/aiscuser/workspace-ll/HUGE_data/pi05_official_rollouts \
  --host 127.0.0.1 --port 5550
```

> `--exec_steps 10` 是默认值，`--smooth_overlap` 也默认开，正好对应论文声明 #3（20 步 chunk 取前 10 步平均）。先加 `--num_trajs 8` 冒烟一遍再放全量。

### 6.3 ⚠️ 需要补的代码：rollout 分片

`openpi/scripts/action_infer.py` 的 CLI 里**没有** `--num_shards` / `--shard_index`（只有 `--num_trajs`），是单进程串行。而 `HUGE-Bench/vlaser/action_infer_vlaser_dual_image.py` 里已经有这两个参数（`:159-160`）。全量 993 个测试 episode 串行跑会非常久。

**改法**：照搬 vlaser 版的分片逻辑到 `action_infer.py` —— 在 `scan_dataset_collect()` 收集完 episode 列表后，按 `episode_index % num_shards == shard_index` 过滤即可，改动约 10 行。然后写一个 `run_huge_pi05_inference_4gpu.sh`（直接以 `run_huge_vlaser_inference_4gpu.sh` 为模板，把 evaluator 换成 `uv run scripts/action_infer.py`，worker i 用 GPU 4+i、端口 5550+i、`--shard_index i --num_shards 4`）。

### 6.4 评测

```bash
cd /home/aiscuser/workspace-ll/HUGE-Bench
python metric.py \
  --out_dir /home/aiscuser/workspace-ll/HUGE_data/pi05_official_rollouts \
  --mesh_root /home/aiscuser/workspace-ll/HUGE_data/data_3d \
  --mesh_rel terra_ply/simplified_mesh.obj \
  --tcr_thresholds 1,2,5
```

对照论文 Table 2 的 π0.5 行。仓库里还有 `metric_paper_aligned.py`，若 `metric.py` 的口径与论文有出入，用它交叉验证。

**决策点**：若这一步 Avg.TCR ≈ 0.58 → `discrete_state_input=False` 确认无误，训练用 `pi05_overall`；若明显偏低（比如 < 0.4）→ 换 `pi05_overall_state`（True）重跑 6.2，看是否回到 0.58。

---

## 7. 训练执行（阶段 D）

```bash
cd /home/aiscuser/workspace-ll/openpi
export HF_LEROBOT_HOME=/home/aiscuser/workspace-ll/lerobot_home
export HF_DATASETS_CACHE=/scratch/openpi-cache/hf-datasets     # 避免 NFS 锁竞争
export JAX_COMPILATION_CACHE_DIR=/scratch/openpi-cache/jax
mkdir -p "$HF_DATASETS_CACHE" "$JAX_COMPILATION_CACHE_DIR"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_overall \
  --exp-name=pi05_overall_5ep_bs<BS> \
  --num-workers=8 \
  --overwrite            # 续跑时换成 --resume
```

监控：`wandb` 默认开启（`wandb_enabled=True`），若无 wandb 账号加 `--no-wandb-enabled`。重点看 loss 曲线和 `learning_rate`，前 200 步确认 loss 从 ~1.0 量级正常下降。

**时间估算**：8×A100-40G，π0.5 全参 FSDP，序列 ≈ 988 token。按 global batch 256 / 33.6 k step、每步约 2.0–2.8 s 估计 ⇒ **约 19–26 小时**。加上编译预热和 checkpoint I/O，预留 30 小时。

**中断保护**：这是 Singularity/AzureML 作业节点（conda env 已经丢过一次），务必：
- `checkpoint_base_dir` 放在 `/home/aiscuser/workspace-ll/openpi_ckpts`
- 训练用 `tmux` / `setsid` 挂后台，别绑在交互式 shell 上
- 每存一个 checkpoint 后用 `blob_manager.sh` 同步一份到 blob（`.blob_config.json` 里的 SAS token **2026-08-29 过期**，需要时提前续期）

---

## 8. 自训模型的评测（阶段 E）

与 §6 完全相同，只把 `--checkpoint_dir` 换成
`/home/aiscuser/workspace-ll/openpi_ckpts/pi05_overall/pi05_overall_5ep_bs<BS>`
（`action_infer.py:617 _find_largest_numeric_subdir` 会自动挑最大 step 的子目录，也可以直接指到某个 step 目录）。

最终产出对照表：

| 模型 | 来源 | Avg.TCR | nDTW | NTP | CR | CSPL |
|---|---|---|---|---|---|---|
| π0.5（论文） | Table 2 | 0.581 | 0.467 | 0.618 | 0.018 | 0.805 |
| π0.5（官方权重复跑） | `HUGE_PI05` | | | | | |
| π0.5（我们训练） | `pi05_overall` | | | | | |
| π0.5 + discrete state（消融） | `pi05_overall_state` | | | | | |

rollout 输出结构（`action_infer.py` 自动生成，可直接用于定性分析）：
`<out_dir>/task_<token>/<split>/<env_id>/episode_<idx>/{compare_gt_vs_pred_3d.png, traj_gt_pred_xyzk.npz, instruction.txt, gt_pred_side_by_side.mp4}`

---

## 9. 执行顺序与工时

| 阶段 | 内容 | 预计 | 阻塞关系 |
|---|---|---|---|
| A1 | uv + openpi `.venv`（§3.1） | 1 h | — |
| A2 | gaussian_splatting env + CUDA 编译（§3.2） | 1–2 h | 可与 A1 并行 |
| A3 | `HF_LEROBOT_HOME` 软链 + 元数据一致性检查（§3.3） | 0.5 h | A1 |
| B1 | norm stats 就位（§5a），可选后台重算（§5b） | 5 min / 后台 3–6 h | A3 |
| **C** | **官方 `HUGE_PI05` 权重跑通全流程 + 定 `discrete_state_input`（§6）** | 4–8 h | A2, A3, B1 |
| C-fix | `action_infer.py` 加分片 + 写 4-worker 脚本（§6.3） | 1 h | A1 |
| B2 | 改 config、试 batch 探显存（§4.3、§4.4） | 1–2 h | A1, B1 |
| **D** | **π0.5 全量微调（§7）** | **20–30 h** | B2, C 的结论 |
| E | 自训模型 rollout + metric（§8） | 4–8 h | D |
| F | 消融 `pi05_overall_state` 训练 + 评测（可选） | +30 h | E |

关键路径：A1 → A3 → B1 → C → B2 → D → E，**约 3–4 天**（含一次训练）。

---

## 10. 风险清单

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | `discrete_state_input=False` 使 π0.5 丢失 state 输入（§4.1） | 训出来的模型达不到 0.581 | §6 用官方权重先验证；备好 `pi05_overall_state` 分支 |
| 2 | A100-40G 装不下 per-GPU batch 64 | 无法严格复现论文 batch | FSDP + 降 batch 保持总 epoch=5；实在不行走 LoRA 并注明偏离 |
| 3 | LeRobot `info.json` 的 `total_tasks`/`splits` 与 `tasks.jsonl` 不符 | 数据加载报错或 prompt 错位 | §3.3 先验证；只改 `$HF_LEROBOT_HOME` 下副本 |
| 4 | conda env 已丢失过一次，节点可能再被重置 | 训练进度丢失 | checkpoint 落 `workspace-ll` 并同步 blob；**SAS token 08-29 过期需续** |
| 5 | `action_infer.py` 无分片，993 episode 串行 | 评测耗时数十小时 | §6.3 移植 vlaser 的分片逻辑 |
| 6 | `CosineDecaySchedule.decay_steps=30000` 与 16.8 k 步不匹配 | LR 未衰减完 | 主实验保留默认（忠实复现）+ 对照实验设为 `num_train_steps` |
| 7 | `norm_stats.json` 缺失时 openpi 只 warning 不报错 | 静默在未归一化数据上训练 | §5 显式确认文件存在；训练日志里 grep `Loaded norm stats` |
| 8 | 需从 GCS 拉 `pi05_base` params（约 14 G） | 首次训练启动慢 | 已验证 `storage.googleapis.com` 直连 200；可提前预热 `maybe_download` |

---

## 附：为何主仓仍留在 `workspace-ll/openpi`（备查）

模型侧两边**完全等价**（§1.1 已逐字节验证），所以这纯粹是工程选择。

`XPolicyLab/policy/Pi_05` 的 `train.sh` 默认 config 是 `pi05_base_aloha_full_sim_arx-x5_seed_0`（ARX-X5 双臂机器人，joint 动作），数据来自 `process_data.py` 转换的 RoboDojo 演示，`deploy.py` 走 ws policy server + `obs_transform_pipeline: xspark-v1.0`。这套观测/动作约定与 HUGE 的 4 维无人机动作、双帧 RGB 输入、3DGS render server 协议不同，需要重新接一遍 —— 而本地 openpi 里这层已经现成。

> 更正：本文档早先版本曾把"4D↔54D 动作空间"列为不采用 XPolicyLab 的理由，该说法有误。54D 只是其 `config.py` 里 `pi05_wuji_marvin_54d` 单条 config 的属性，与选用 `Pi_05` 这一 policy 无关。动作维度在两边都是 32（零填充），不构成差异。
