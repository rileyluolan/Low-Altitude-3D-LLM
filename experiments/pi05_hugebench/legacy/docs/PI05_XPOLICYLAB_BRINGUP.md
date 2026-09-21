# 用 XPolicyLab 的 π0.5 在 HUGE-Bench 上训练 / 评测 —— 调试方案

> 生成时间：2026-08-25
> 主仓：`/home/aiscuser/workspace-ll/XPolicyLab`（已 clone，HEAD `c07a096`，147 MB）
> 配套方案文档：[`PI05_ON_HUGEBENCH_PLAN.md`](PI05_ON_HUGEBENCH_PLAN.md)（数据核对、论文声明、超参、风险）
> 本文只讲**怎么一步步调通**：每个关卡给命令、通过判据、失败排查。

---

## 路径约定

把下面这段存成 `~/pi05env.sh`，每个新 shell `source` 一次：

```bash
export XPL_ROOT=/home/aiscuser/workspace-ll/XPolicyLab
export OPENPI=$XPL_ROOT/policy/Pi_05/openpi
export HUGE=/home/aiscuser/workspace-ll/HUGE-Bench
export LOCAL_OPENPI=/home/aiscuser/workspace-ll/openpi     # 只作参考/取文件用
export DATA=/home/aiscuser/workspace-ll/HUGE_data

export HF_LEROBOT_HOME=/home/aiscuser/workspace-ll/lerobot_home
export PI05_ASSETS=/home/aiscuser/workspace-ll/pi05_assets
export PI05_CKPT=/home/aiscuser/workspace-ll/pi05_ckpts
export HF_DATASETS_CACHE=/scratch/openpi-cache/hf-datasets
export JAX_COMPILATION_CACHE_DIR=/scratch/openpi-cache/jax
export PATH="$HOME/.local/bin:$PATH"                        # uv
mkdir -p "$HF_LEROBOT_HOME" "$PI05_ASSETS" "$PI05_CKPT" "$HF_DATASETS_CACHE" "$JAX_COMPILATION_CACHE_DIR"
```

**关卡总览**

| 关卡 | 内容 | 预计 | 阻塞后果 |
|---|---|---|---|
| G0 | 建分支、锁定基线 | 5 min | — |
| G1 | 依赖改造（lerobot 降版）+ `uv sync` | 40–90 min | 全线阻塞 |
| G2 | 移植 3 个 drone 文件 | 30 min | 全线阻塞 |
| G3 | 数据挂载 + LeRobot 加载冒烟 | 20 min | 训练/评测都跑不了 |
| G4 | norm stats 就位 | 5 min | 静默训坏 |
| G5 | **π0.5 权重加载冒烟** | 30 min | 权重没真正加载 |
| G6 | 单步训练冒烟 + batch 探显存 | 1–2 h | OOM |
| G7 | 3DGS 渲染器 | 1–2 h | 评测跑不了 |
| G8 | **官方 HUGE_PI05 跑 rollout（定 `discrete_state_input`）** | 4–8 h | 训练设定不确定 |
| G9 | metric 冒烟 + 全量 | 1 h | — |
| G10 | 正式训练 | 20–30 h | — |

---

## G0　建分支、锁定基线

```bash
source ~/pi05env.sh
cd $XPL_ROOT
git switch -c huge-bench-pi05
git log --oneline -1        # 记下：c07a096
```

之后所有改动都在这个分支上，`git diff main` 随时能看清我们改了什么。

**判据**：`git status` 干净，分支为 `huge-bench-pi05`。

---

## G1　依赖改造：把 lerobot 降回 openpi 上游的 git pin

### 为什么必须改（证据）

| 事实 | 来源 |
|---|---|
| HUGE 数据集是 `"codebase_version": "v2.1"` | `$DATA/data_traj/*/meta/info.json` |
| XPolicyLab 锁的是 `lerobot==0.4.4`，其 `CODEBASE_VERSION = "v3.0"` | `$OPENPI/uv.lock:1640`；lerobot v0.4.4 `lerobot_dataset.py:83` |
| 0.4.4 加载 v2.1 会在 `check_version_compatibility()` 抛 `BackwardCompatibilityError`，并提示"用 v2.1→v3.0 转换脚本" | lerobot v0.4.4 `lerobot_dataset.py:165` / `:598` |
| openpi 上游 pin 的 rev `0cf8648…` 是 `CODEBASE_VERSION = "v2.1"` ✅ | `$LOCAL_OPENPI/pyproject.toml:70` |

**为什么降版是安全的（而不是反过来转 181 G 数据）**：

- XPolicyLab 的 `data_loader.py` 写的是 `try: import lerobot.datasets … except ModuleNotFoundError: import lerobot.common.datasets` —— 旧版会自动走 fallback 分支 ✅
- 旧 rev 的 `LeRobotDataset.__init__` **有** `video_backend` 形参（`lerobot_dataset.py:366`），XPolicyLab 的 `data_loader.py:152` 会传这个 kwarg，不会 `TypeError` ✅
- 旧 rev 的依赖约束是 `torch>=2.2.1` / `torchvision>=0.21.0` / `torchcodec>=0.2.1`（都是下界），与 XPolicyLab 锁的 `torch==2.10.0` **不冲突** ✅
- `action_infer.py:29` 硬编码 `import lerobot.common.datasets.lerobot_dataset` —— 只有旧版满足 ✅

### 操作

```bash
source ~/pi05env.sh
which uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)
cd $OPENPI
cp pyproject.toml pyproject.toml.orig
```

编辑 `$OPENPI/pyproject.toml`：

```toml
# ① dependency-groups 里，把 PyPI 版换成裸名（让 tool.uv.sources 接管）
[dependency-groups]
lerobot = [
    "lerobot",              # was: "lerobot==0.4.4"
]

# ② 新增/合并 tool.uv.sources
[tool.uv.sources]
lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" }
```

然后重新解析并安装（`uv.lock` 必须重生成，否则还是 0.4.4）：

```bash
cd $OPENPI
GIT_LFS_SKIP_SMUDGE=1 uv lock                       # 重新解析依赖图
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --group lerobot
GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv pip install -e .
```

> `$OPENPI/.python-version` 是 `3.11`，uv 会自动拉一个 3.11 解释器（系统 python 是 3.14，不要用）。

### 判据

```bash
cd $OPENPI
uv run python -c "
import lerobot.common.datasets.lerobot_dataset as m
print('lerobot CODEBASE_VERSION =', m.CODEBASE_VERSION)          # 期望 v2.1
import inspect; print('video_backend in sig:', 'video_backend' in inspect.signature(m.LeRobotDataset.__init__).parameters)
import jax; print('jax', jax.__version__, len(jax.devices()), 'devices')   # 期望 0.5.3 / 8
import openpi; print('openpi import ok')
"
```

三行都对才算过。

### 常见失败

| 现象 | 原因 | 处理 |
|---|---|---|
| `uv lock` 解析失败，torch/torchcodec 冲突 | 旧 lerobot 与 `torch==2.10.0` 的 torchcodec ABI 不匹配 | 把 `torch==2.10.0`/`torchvision==0.25.0`/`torchcodec==0.10.0` 放宽成 `>=`，或直接对齐本地 openpi 的 `torch==2.7.1` |
| `import lerobot.common` 失败但 `lerobot.datasets` 成功 | lock 没重生成，装的还是 0.4.4 | 删 `uv.lock` 重新 `uv lock` |
| `jax.devices()` 只有 CPU | jax cuda plugin 没装上 | 确认 `jax[cuda12]==0.5.3` 在依赖里；`uv run python -c "import jaxlib; print(jaxlib.__file__)"` |
| clone lerobot 超时 | 网络 | `GIT_LFS_SKIP_SMUDGE=1` 已设；必要时先手动 `git clone` 到本地再用 `path =` source |

---

## G2　移植 3 个 drone 文件

**原则：追加，不覆盖。** 绝不能把 HUGE-Bench 的 `config.py` 整个盖到 XPolicyLab 上 —— 那会丢掉 `checkpoint_dir_override`(`config.py:573,619`)、`DataConfig.video_backend`(`:76`)、`PartialCheckpointWeightLoader`、以及 wuji/droid 全部 config。

### ① `drone_policy.py`（整文件拷贝，零改动）

```bash
cp $HUGE/openpi/src/openpi/policies/drone_policy.py $OPENPI/src/openpi/policies/drone_policy.py
```

它只 `from openpi import transforms` + `from openpi.models import model as _model`，两边 API 一致，无需改。

### ② `config.py`：加 import + 加 `MyDroneDataConfig` + 加 `pi05_overall`

**(a) import**，在 `$OPENPI/src/openpi/training/config.py` 的 import 区加一行：

```python
import openpi.policies.drone_policy as drone_policy
```

**(b) `MyDroneDataConfig`**，从 HUGE-Bench 的 config.py 第 **226–282** 行整段插入（放在 `class LeRobotAlohaDataConfig` 之前）：

```bash
sed -n '226,282p' $HUGE/openpi/src/openpi/training/config.py > /tmp/MyDroneDataConfig.py
wc -l /tmp/MyDroneDataConfig.py     # 期望 57 行
```

它依赖的东西在 XPolicyLab 版里都在（已核对）：`DataConfigFactory`、`_transforms.{Group,RepackTransform,DeltaActions,AbsoluteActions,make_bool_mask}`、`ModelTransformFactory`、`DataConfig.action_sequence_keys`(`:97`)、`DataConfig.prompt_from_task`(`:100`)。

> 注意它做的 key 映射：`observation/image←image`、`observation/first_image←first_image`、`observation/state←state`、`actions←actions`、`prompt←task`。这正是论文声明 #1（首帧 + 当前帧双图输入）的落点。

**(c) `pi05_overall`**，在 `_CONFIGS = [` 列表里加：

```python
    TrainConfig(
        name="pi05_overall",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=20,            # 论文声明 #2
            discrete_state_input=False,   # ← G8 会判定这里到底该 False 还是 True
        ),
        data=MyDroneDataConfig(
            repo_id="task_overall/train",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"   # 论文声明 #4
        ),
        num_train_steps=16_800,           # 5 epoch @ global batch 512，G6 后按实测 batch 重算
        batch_size=512,                   # G6 后按实测下调
        save_interval=2_000,
        keep_period=4_000,
        fsdp_devices=8,
        assets_base_dir="/home/aiscuser/workspace-ll/pi05_assets",
        checkpoint_base_dir="/home/aiscuser/workspace-ll/pi05_ckpts",
    ),
```

再复制一份改名 `pi05_overall_state`，仅把 `discrete_state_input` 改成 `True`（G8 备用）。

### ③ `action_infer.py`

```bash
cp $LOCAL_OPENPI/scripts/action_infer.py $OPENPI/scripts/action_infer.py
```

（本地 openpi 那份与 HUGE-Bench 仓库里的一致，直接取用。）它用到的 `train_config.get_config` / `policy_config.create_trained_policy` 在 XPolicyLab 版里签名**逐字节相同**，已核对。

### 判据

```bash
cd $OPENPI
uv run python -c "
from openpi.training import config as C
c = C.get_config('pi05_overall')
print('name        :', c.name)
print('model       :', type(c.model).__name__, 'pi05=', c.model.pi05,
      'ah=', c.model.action_horizon, 'adim=', c.model.action_dim,
      'mtl=', c.model.max_token_len, 'dsi=', c.model.discrete_state_input)
print('model_type  :', c.model.model_type)
print('repo_id     :', c.data.repo_id)
print('weight_load :', c.weight_loader)
print('assets_dirs :', c.assets_dirs)
c2 = C.get_config('pi05_overall_state'); print('state variant dsi =', c2.model.discrete_state_input)
"
```

期望：`pi05=True`、`ah=20`、`adim=32`、`mtl=200`、`model_type=ModelType.PI05`、`repo_id=task_overall/train`。

**若 `mtl` 不是 200**：说明 `Pi0Config.__post_init__` 没走到（`max_token_len` 被显式指定了），检查有没有多写参数。

---

## G3　数据挂载 + LeRobot 加载冒烟

```bash
source ~/pi05env.sh
mkdir -p "$HF_LEROBOT_HOME/task_overall"
for s in train test_seen test_unseen; do
  ln -sfn $DATA/data_traj/$s "$HF_LEROBOT_HOME/task_overall/$s"
done
ls -l "$HF_LEROBOT_HOME/task_overall"
```

```bash
cd $OPENPI
uv run python - <<'PY'
import lerobot.common.datasets.lerobot_dataset as L
for split in ["train","test_seen","test_unseen"]:
    meta = L.LeRobotDatasetMetadata(f"task_overall/{split}")
    print(f"[{split}] episodes={meta.total_episodes} frames={meta.total_frames} "
          f"tasks_meta={meta.total_tasks} tasks_dict={len(meta.tasks)} fps={meta.fps}")
    ds = L.LeRobotDataset(f"task_overall/{split}")
    s = ds[0]
    print("   keys:", sorted(s.keys()))
    for k in ["image","first_image","state","actions","task","task_index"]:
        v = s.get(k, None)
        print(f"   {k:12s}", getattr(v,'shape',v))
PY
```

### 判据

- 三个 split 都能构造 `LeRobotDataset`，不抛 `BackwardCompatibilityError`
- 样本里 **`first_image` 存在**（`MyDroneDataConfig` 的 repack 依赖它；没有它 `DroneInputs` 会回退成用当前帧当首帧，静默降级为单图输入）
- `state` / `actions` 形状是 `(4,)`
- `task` 是字符串，且 `task_index` 能索引到

### 已知风险 & 排查

`info.json` 的 `total_tasks` 与 `meta/tasks.jsonl` 行数对不上（train 109 vs 1102；test_seen 13 vs 368；test_unseen 10 vs 259），`splits` 也写成 `{"train": "0:109"}`。这是合并脚本没更新元数据。

- 若上面脚本能跑通、`tasks_dict` 显示 1102/368/259 → **不用管**，`prompt_from_task` 走的是 `tasks` 字典
- 若抛断言错 → 只改 `$HF_LEROBOT_HOME` 下的副本（**先把软链换成真目录再改，别动 `data_traj` 原始数据**）：

```bash
# 仅在必要时执行
for s in train test_seen test_unseen; do
  rm "$HF_LEROBOT_HOME/task_overall/$s"
  mkdir -p "$HF_LEROBOT_HOME/task_overall/$s"
  cp -r $DATA/data_traj/$s/meta "$HF_LEROBOT_HOME/task_overall/$s/meta"   # meta 很小
  ln -sfn $DATA/data_traj/$s/data "$HF_LEROBOT_HOME/task_overall/$s/data" # data 仍软链
done
# 然后用 python 按 tasks.jsonl 实际行数修正 meta/info.json 的 total_tasks / splits
```

---

## G4　norm stats 就位

`MyDroneDataConfig` 找的是 `<assets_base_dir>/<config_name>/<repo_id>/norm_stats.json`。**找不到只 warning 不报错**（`config.py:198` `logging.info("Norm stats not found …, skipping.")`），会静默地在未归一化数据上训练 —— 必须显式确认。

```bash
source ~/pi05env.sh
for cfg in pi05_overall pi05_overall_state; do
  mkdir -p "$PI05_ASSETS/$cfg/task_overall/train"
  cp $DATA/HUGE_PI05/assets/task_overall/train/norm_stats.json "$PI05_ASSETS/$cfg/task_overall/train/"
done
find "$PI05_ASSETS" -name norm_stats.json
```

复用官方那份是安全的：norm stats 只取决于数据与 `data_transforms`，π0 / π0.5 在这一层完全相同；且该文件的 asset_id 就是 `task_overall/train`。内容已核对（state/actions 各 4 维，action std ≈ 0.52/0.51/0.39/0.035，q01/q99 ≈ ±1）。

**交叉验证（可后台跑，3–6 h）**：

```bash
cd $OPENPI && uv run scripts/compute_norm_stats.py --config-name pi05_overall
# 跑完 diff 一下新旧两份，mean/std 应一致 —— 同时也验证了我们挂的数据就是官方那份
```

**判据**：G6 训练日志里必须出现 `Loaded norm stats from …`。

---

## G5　π0.5 权重加载冒烟（关键关卡）

`CheckpointWeightLoader` 用的 `_merge_params(..., missing_regex=".*lora.*")` 是**严格**的：除 lora 外，加载的参数树必须与模型参数树完全对上，对不上直接抛错。所以这一步能一次性验证"`pi05=True` + `action_dim=32` + `pi05_base`"三者是否自洽。

先预热下载（≈14 G，`storage.googleapis.com` 已验证可直连）：

```bash
cd $OPENPI
uv run python -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params')
print('cached at', p)
"
```

再做纯结构校验（**不占显存**，用 `eval_shape`）：

```bash
cd $OPENPI
uv run python - <<'PY'
import jax, numpy as np
from openpi.training import config as C
import openpi.training.weight_loaders as WL

c = C.get_config("pi05_overall")
shapes = jax.eval_shape(c.model.create, jax.random.key(0))   # 只算 shape，不分配
import flax.nnx as nnx
params_shape = nnx.state(shapes, nnx.Param).to_pure_dict()

loaded = c.weight_loader.load(params_shape)
import flax.traverse_util as tu
fl_ref, fl_new = tu.flatten_dict(params_shape, sep="/"), tu.flatten_dict(loaded, sep="/")
missing  = [k for k in fl_ref if k not in fl_new]
mismatch = [k for k in fl_ref if k in fl_new and getattr(fl_ref[k],'shape',None) != getattr(fl_new[k],'shape',None)]
print("leaves:", len(fl_ref), "| missing:", len(missing), "| shape-mismatch:", len(mismatch))
print("has state_proj:", any("state_proj" in k for k in fl_ref))
print("has time_mlp  :", any("time_mlp"   in k for k in fl_ref))
for k in (missing+mismatch)[:10]: print("  !", k)
PY
```

### 判据

- `missing = 0`，`shape-mismatch = 0`
- `has state_proj: False`、`has time_mlp: True` → 确认走的是 π0.5 分支（与 `$DATA/HUGE_PI05/params/_METADATA` 的 51 个叶子结构一致）

### 常见失败

| 现象 | 原因 | 处理 |
|---|---|---|
| `state_proj` 出现在参数树里 | `pi05` 没生效，建成了 π0 | 检查 config 里 `pi0_config.Pi0Config(pi05=True)` |
| `action_in_proj` shape 不匹配 | `action_dim` 被改成 4 了 | 主实验保持默认 32；确要 4 维就换 `weight_loaders.PartialCheckpointWeightLoader`（XPolicyLab 自带） |
| 下载卡住 / 403 | GCS 网络 | `curl -sI https://storage.googleapis.com/openpi-assets/checkpoints/pi05_base/params/_METADATA` 应返回 200 |

---

## G6　单步训练冒烟 + batch 探显存

### ① 最小冒烟（1 步，小 batch）

```bash
cd $OPENPI && source ~/pi05env.sh
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_overall \
  --exp-name=smoke --batch-size=8 --num-train-steps=2 \
  --fsdp-devices=8 --no-wandb-enabled --overwrite 2>&1 | tee /tmp/smoke.log
```

**判据**（在 `/tmp/smoke.log` 里 grep）：
- `Loaded norm stats from /home/aiscuser/workspace-ll/pi05_assets/pi05_overall/task_overall/train` ← G4 的验收点
- 权重加载没报 missing/mismatch
- 至少打印一次 `step` + `loss`，loss 是有限值（1.0 量级），**不是 nan**
- `$PI05_CKPT/pi05_overall/smoke/` 下生成了 checkpoint

### ② 逐级探 batch（每档 20 步）

8 × A100-**40 GB**，而 openpi README 明写全参微调门槛是 **> 70 GB 单卡**；π0.5 的 `max_token_len=200`（π0 只有 48），加 3 路 SigLIP 各 256 token，单样本序列 ≈ 988 token。论文的 per-GPU 64 在 40 G 上大概率 OOM。openpi **不支持梯度累积**，且强制 `batch_size % jax.device_count() == 0`，只能靠 FSDP + 降 batch。

```bash
cd $OPENPI && source ~/pi05env.sh
for BS in 512 256 128 64; do
  echo "===== batch_size=$BS ====="
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 timeout 900 \
  uv run scripts/train.py pi05_overall \
    --exp-name=probe_bs$BS --batch-size=$BS --num-train-steps=20 \
    --fsdp-devices=8 --no-wandb-enabled --overwrite 2>&1 | tail -25
done
```

记录每档：是否 OOM、稳定后的 s/step。

### ③ 按选定 batch 反推步数

训练集 **1,720,096 帧**，1 epoch = 1,720,096 样本：

| global batch | 5 epoch 对应 `num_train_steps` |
|---|---|
| 512 | 16,798 |
| 256 | 33,596 |
| 128 | 67,191 |

把选定值写回 config 的 `batch_size` / `num_train_steps`。

> `CosineDecaySchedule` 默认 `decay_steps=30_000`。论文说"其余超参用 π0 默认"，字面理解是保留 30000；但若 `num_train_steps=16800`，LR 只衰减到曲线一半就停。**主实验保留默认（忠实复现），另跑一个 `lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=<num_train_steps>)` 的对照。**

### OOM 时的退路（按优先级）

1. 继续降 batch（保持总 epoch = 5，步数相应增加）
2. `paligemma_variant="gemma_2b_lora"` + `action_expert_variant="gemma_300m_lora"` 走 LoRA（openpi README: >22.5 GB），但偏离论文"全参微调"，须在报告中注明
3. 去掉 `right_wrist_0_rgb` 全零图省 256 token —— 需同时改 `drone_policy.py` 与推理侧，**会与官方 HUGE_PI05 权重不兼容**，只能用于自训分支

---

## G7　3DGS 渲染器（评测前置）

`gaussian_splatting` conda env 已随 08-24 重装 miniconda 丢失，需重建。环境名**必须叫 `gaussian_splatting`**，这样现成的 `start_huge_renderers_4gpu.sh` 能直接复用（它硬编码 `envs/gaussian_splatting/bin/python`）。

```bash
conda create -n gaussian_splatting python=3.11 -y
conda activate gaussian_splatting
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install plyfile tqdm opencv-python numpy scipy
cd /home/aiscuser/workspace-ll/gaussian-splatting
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
```

单渲染器冒烟：

```bash
CUDA_VISIBLE_DEVICES=0 python 3dgs_renderer.py --host 127.0.0.1 --port 5550 \
  --ply_template /home/aiscuser/workspace-ll/HUGE_data/data_3d/{env_id}/3dgs_ply/point_cloud_utm50.ply
```

**判据**：进程起来并监听 5550（`ss -ltn | grep 5550`），日志无报错。

四路启动（GPU 0-3 / 端口 5550-5553）：

```bash
cd /home/aiscuser/workspace-ll && bash start_huge_renderers_4gpu.sh
```

**常见失败**：`diff_gaussian_rasterization` 编译失败 → 确认 `nvcc --version` 是 12.6 且与 torch 的 cu126 匹配；必要时 `export TORCH_CUDA_ARCH_LIST="8.0"`（A100）。

---

## G8　用官方 `HUGE_PI05` 权重跑 rollout —— 并判定 `discrete_state_input`

**这是整个方案里最重要的一关**，先于自己训练做。它同时干三件事：打通评测流水线、拿到已知答案作锚点、定死训练设定。

### 背景：为什么要判定

现有 config 写的 `discrete_state_input=False`（HUGE-Bench 上游全部 8 个 `pi05_*` 都这么写）。但按 openpi 源码：

- `pi0.py:93-97`：`if config.pi05: (time_mlp) else: self.state_proj = …` → **π0.5 分支不创建 `state_proj`**
- `pi0.py:151`：`if not self.pi05:` 才拼连续 state token → **π0.5 不走连续通路**
- `config.py` PI05 分支：只有 `discrete_state_input=True` 才把 state 离散成 256 bin 塞进 prompt（`tokenizer.py:24-28`）

即 `pi05=True` + `discrete_state_input=False` ⇒ **两条 state 通路都关闭，模型完全看不到 (x,y,z,yaw)**。而 `Pi0Config.__post_init__` 的默认是 `discrete_state_input = pi05`（True），是被显式覆写掉的。

### 判定方法

先小规模跑（每 split 8 条），两个 config 各来一遍：

```bash
cd $OPENPI && source ~/pi05env.sh
for CFG in pi05_overall pi05_overall_state; do
  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
  uv run scripts/action_infer.py \
    --task_id overall --config_name $CFG \
    --checkpoint_dir $DATA/HUGE_PI05 \
    --splits test_seen --num_trajs 8 --exec_steps 10 \
    --out_dir $DATA/pi05_probe_$CFG \
    --host 127.0.0.1 --port 5550 2>&1 | tail -30
done
```

> `--exec_steps 10` 是默认值，`--smooth_overlap` 也默认开，正好对应论文声明 #3（20 步 chunk 取前 10 步平均，由 `ChunkMeanBuffer.get_mean()` 实现）。

然后跑 metric（G9）比两者的 nDTW / TCR。

**判定规则**：
- `pi05_overall`（False）明显更好，且接近论文量级 → **False 就是官方设定**，训练用 `pi05_overall`
- `pi05_overall_state`（True）明显更好 → 官方 config 写错了，训练改用 `pi05_overall_state`
- 两者都很差 → 不是 state 的问题，回头查 `first_image` 是否真的进模型了（G3）、渲染器坐标系是否对（看 `compare_gt_vs_pred_3d.png`）

定下来后，**全量跑一遍 test_seen + test_unseen**，目标复现论文 Table 2：Avg.TCR 0.581 / nDTW 0.467 / NTP 0.618 / CR 0.018 / CSPL 0.805。

### 必须补的代码：rollout 分片

`action_infer.py` 的 CLI 里**没有** `--num_shards` / `--shard_index`（只有 `--num_trajs`），是单进程串行；993 个测试 episode 会跑很久。而 `$HUGE/vlaser/action_infer_vlaser_dual_image.py:159-160` 已经有这两个参数。

改法：在 `scan_dataset_collect()` 收集完 episode 列表后加一行过滤 `episodes = [e for e in episodes if e.episode_index % num_shards == shard_index]`，并注册两个 argparse 参数，约 10 行。

再照 `run_huge_vlaser_inference_4gpu.sh` 写 `run_huge_pi05_inference_4gpu.sh`：worker `i` 用 GPU `4+i`、端口 `5550+i`、`--shard_index i --num_shards 4`。

**分片正确性判据**：4 个 shard 产出的 `episode_*` 目录并集 = 全量、交集 = 空。

---

## G9　评测

```bash
cd $HUGE
python metric.py \
  --out_dir $DATA/pi05_official_rollouts \
  --mesh_root $DATA/data_3d \
  --mesh_rel terra_ply/simplified_mesh.obj \
  --tcr_thresholds 1,2,5
```

**判据**：输出 Avg.TCR / nDTW / NTP / CR / CSPL 五项，且 CR/CSPL **不是 `nan`**（是 nan 说明 `--mesh_root` 没找到对应 env 的 mesh）。

仓库里还有 `metric_paper_aligned.py`，若 `metric.py` 口径与论文有出入，用它交叉验证。

`metric.py` 的依赖：`python -m pip install -r $HUGE/requirements-utils.txt`（在任一有 numpy/scipy/trimesh 的 env 里跑即可，不必是 openpi 的 venv）。

---

## G10　正式训练

```bash
cd $OPENPI && source ~/pi05env.sh
tmux new -s pi05train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_overall \
  --exp-name=pi05_overall_5ep_bs<BS> \
  --num-workers=8 2>&1 | tee $PI05_CKPT/train_pi05_overall.log
```

- 前 200 步盯 loss 是否从 ~1.0 量级正常下降，`learning_rate` 是否按 cosine 走
- 中断续跑：把 `--overwrite` 换成 `--resume`
- 这是 Singularity/AzureML 节点（conda env 已经丢过一次），务必：checkpoint 落 `$PI05_CKPT`（在 `workspace-ll` 下，5.0 T 可用）；用 tmux/setsid 挂后台；每存一个 checkpoint 用 `blob_manager.sh` 同步一份到 blob —— **`.blob_config.json` 的 SAS token 2026-08-29 过期，需提前续期**

存储估算：单 checkpoint ≈ params 12 G + train_state（Adam m/v + EMA）≈ 40–50 G；`save_interval=2000` + `keep_period=4000` 峰值约 250–300 G。

训练完回到 G8/G9，`--checkpoint_dir` 换成 `$PI05_CKPT/pi05_overall/pi05_overall_5ep_bs<BS>`（`action_infer.py:617 _find_largest_numeric_subdir` 会自动挑最大 step）。

---

## 故障速查表

| 现象 | 最可能原因 | 定位 |
|---|---|---|
| `BackwardCompatibilityError` / 提示转 v3.0 | lerobot 还是 0.4.4 | G1，检查 `CODEBASE_VERSION` |
| `ModuleNotFoundError: lerobot.common` | 同上 | G1 |
| `TypeError: __init__() got unexpected keyword 'video_backend'` | lerobot 版本更旧 | 确认 rev 是 `0cf8648…` |
| 训练日志没有 `Loaded norm stats` | norm stats 路径不对 | G4，路径必须是 `<assets_base_dir>/<config_name>/<repo_id>/` |
| 权重加载报 missing/shape mismatch | `pi05` / `action_dim` 不匹配 | G5 |
| loss = nan | LR 过大 / bf16 溢出 / 数据未归一化 | 先查 G4；再把 `peak_lr` 降到 1e-5 试 |
| OOM | batch 太大 | G6 逐级降 |
| rollout 轨迹原地打转 | state 没进模型，或 prompt 没取到 | G8 判定 `discrete_state_input`；查 `instruction.txt` 里 prompt 是否正确 |
| CR / CSPL = nan | `--mesh_root` 下缺 env 的 `terra_ply/simplified_mesh.obj` | G9 |
| 预测轨迹整体偏移/翻转 | 渲染器坐标系或 obstacle 的 yaw/phi 约定 | 看 `compare_gt_vs_pred_3d.png`；`action_infer.py` 有 `--obstacle_angle_mode` |

---

## 实测记录（2026-08-25 执行，G0–G5 + G7 全部通过）

### 两个必须打的补丁（方案原文没预料到）

**① `datasets` 必须 pin 到 3.6.0**，否则 G3 直接 `TypeError`。

uv 会把 `datasets` 解到 4.8.5（那是 lerobot 0.4.4 的需求，换掉 lerobot 后 uv 没有理由降版）。但旧 lerobot 在 `lerobot_dataset.py:508` 写的是 `torch.stack(self.hf_dataset["timestamp"])`，而 `datasets>=4` 的 `Dataset[col]` 返回惰性 `Column` 而非 tensor 列表：

```
TypeError: stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column
```

修法（已写进 `$OPENPI/pyproject.toml` 的 `[tool.uv] override-dependencies`）：

```toml
"datasets==3.6.0",     # 上游 openpi 对同一个 lerobot rev 锁的就是这个版本
```

**② 3DGS 子模块编译要加 `--no-build-isolation`**。pip 默认的 build isolation 让构建环境看不到 torch，而 `diff-gaussian-rasterization` 的 `setup.py` 需要 `import torch`：

```bash
export TORCH_CUDA_ARCH_LIST="8.0"          # A100
pip install --no-build-isolation submodules/diff-gaussian-rasterization submodules/simple-knn
```

### 实测数字

| 项 | 实测值 |
|---|---|
| π0.5 参数量 | **3.35 B**，51 个参数叶子 |
| `pi05_base` 下载 | 12 G，落在 `workspace-ll/openpi_cache`（`OPENPI_DATA_HOME` 重定向生效） |
| LeRobot train 首次加载 | **536 s**（建 arrow 缓存），**峰值 RSS 仅 3.7 GB** |
| train arrow 缓存 | **655 G** on `/scratch`（1,720,096 × 2 图 × 256²×3 ≈ 676 G，图像按裸 uint8 存）；arrow 走 mmap，不占进程内存 |
| test_seen / test_unseen arrow | 39 G / 28 G |
| 3DGS ply（1_office） | **7.4 G** |
| 渲染器显存 | **~7.5 GB/实例**（GPU0 由 27,987 → 35,483 MiB） |

排查提示：`uv run ... | grep ...` 会因 grep 的块缓冲而看不到进行中的输出，容易误判成"进程静默退出"。判断是否还在跑要看 `ps`，别看输出文件。

### G5 结构校验：与官方 checkpoint 逐项对齐

| 指标 | 我们的 `pi05_overall` | `HUGE_PI05/params/_METADATA` |
|---|---|---|
| 参数叶子数 | 51 | 51 ✅ |
| `state_proj` / `time_mlp` | 无 / 有 | 无 / 有 ✅ |
| `action_in_proj/kernel` | `(32, 1024)` | `(32, 1024)` ✅ |
| `action_out_proj/kernel` | `(1024, 32)` | `(1024, 32)` ✅ |

`pi05_base` 加载：missing = 0，shape-mismatch = 0，`at.check_pytree_equality`（含 dtype）通过。

### 对 XPolicyLab 的改动清单（分支 `huge-bench-pi05`，基线 `c07a096`）

```
 M  policy/Pi_05/openpi/pyproject.toml                        lerobot rev + datasets pin
 M  policy/Pi_05/openpi/src/openpi/training/config.py         +122 行：import + MyDroneDataConfig + 2 个 TrainConfig
 M  policy/Pi_05/openpi/uv.lock                               重新解析
 ?? policy/Pi_05/openpi/scripts/action_infer.py               新增
 ?? policy/Pi_05/openpi/src/openpi/policies/drone_policy.py   新增
```

### 占卡程序的影响

节点上 `/blob/thinking.py`（PID 901665，来自 `.nyp_048` 的终端）占着 8 卡各 ~27 GB，只剩 ~13 GB/卡。

- **训练（G6/G10）跑不了** —— FSDP 状态+梯度就要 8.3 GB/卡，剩余装不下激活
- **评测（G8）大概率跑得了** —— 渲染器实测 7.5 GB，π0.5 推理按 openpi README 是 > 8 GB，13 GB 余量勉强够，值得先小规模试

### `discrete_state_input` 已判定：**False**（2026-08-25 实测结论）

方案原文把这个留作 G8 的待定项。现在用一个比闭环 rollout 更便宜也更决定性的办法定了：**开环动作预测误差**（L2），不需要渲染器。

先排除了最便宜的一条路：官方 checkpoint **不含**任何配置记录 —— HF 仓库 `yu781986168/HUGE_PI05` 只有 `params/` + `assets/task_overall/train/norm_stats.json`，`_CHECKPOINT_METADATA` 只记 orbax handler。`discrete_state_input` 只影响 prompt 构造，不改变任何参数形状，所以从权重里查不出来。

**方法**：`test_seen` 固定种子取 200 条样本，两个 config 各加载一次官方 `HUGE_PI05` 权重做开环预测，与真值 20 步动作块比。控制变量干净 —— `create_trained_policy` 从 `checkpoint_dir/assets/<asset_id>` 读 norm stats，两个 config 的 `asset_id` 都是 `task_overall/train`，**用的是同一份归一化统计**，唯一差异就是 prompt 格式（17 token vs 39 token）。

| 指标 | `pi05_overall`(False) | `pi05_overall_state`(True) | 平凡基线（全零/训练均值） |
|---|---|---|---|
| MAE dx / dy / dz (m) | **0.065 / 0.064 / 0.019** | 0.099 / 0.090 / 0.032 | 0.332 / 0.392 / 0.181 |
| MAE dyaw (rad) | **0.0020** | 0.0040 | 0.0208 |
| 4 秒终点位移误差 mean (m) | **0.660** | 1.344 | 14.199 |
| 4 秒终点位移误差 median (m) | **0.198** | 0.890 | 13.490 |
| 终点偏航误差 mean (rad) | **0.0176** | 0.0498 | 0.3845 |

配对统计（同一批 200 条样本，逐样本比较）：

```
False 更优           : 162/200 = 81.0%
配对符号检验          : z = 8.77,  p ≈ 1.8e-18
中位数差 (True−False) : +0.327 m,  95% CI [+0.214, +0.390]   ← 不含 0
```

**结论：HUGE-Bench 发布的 config 不是笔误。** 官方 `HUGE_PI05` 确实是在"模型完全看不到 `(x,y,z,yaw)`"的设定下微调的，论文 Table 2 的 0.581 对应的就是这个。**正式训练用 `pi05_overall`**，`pi05_overall_state` 降级为消融。

顺带的发现：False 设定下开环中位数终点误差仅 **0.198 m**，比平凡基线好约 68×。原因大概是动作本身是**全局系增量** `(dx,dy,dz,dyaw)`，而首帧+当前帧这对视觉输入已隐含相对位移与任务进度，绝对坐标是冗余的。

**局限**：L2 是开环、喂真值观测的测试，只确定"哪个 prompt 格式匹配训练分布"，**不能证明闭环能复现 0.581**。L3（rollout + `metric.py`）仍需做，但现在只需跑一个 config，工作量减半。

复现脚本：`/tmp/l2_probe.py`（用法 `uv run python /tmp/l2_probe.py <config> <n>`，单卡约 40 s / 200 样本，显存 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.28` 即可）。

---

## 附：与"不换仓库"方案的差异（备查）

模型侧两边**完全等价** —— 已逐文件核对 14 个文件：`pi0.py`、`pi0_config.py`、`model.py`、`tokenizer.py`、`gemma.py`、`siglip.py`、`optimizer.py`、`sharding.py`、`checkpoints.py`、`normalize.py`、`download.py`、`compute_norm_stats.py` **全部 0 行差异**；`transforms.py`(48)、`data_loader.py`(12)、`train.py`(7)、`weight_loaders.py` 的差异**全是加法**（batched prompt、lerobot 双路 import、`JAX_COMPILATION_CACHE_DIR`、`PartialCheckpointWeightLoader`），不触及模型数学、loss、优化器、checkpoint 格式或归一化。

选 XPolicyLab 的额外成本只有 G1（依赖改造）；额外收益是 `PartialCheckpointWeightLoader`（`action_dim=4` 消融要用）、`checkpoint_dir_override`、以及 `PromptFromLeRobotTask` 对 DataFrame/Mapping 两种 `tasks` 形态的兼容。
