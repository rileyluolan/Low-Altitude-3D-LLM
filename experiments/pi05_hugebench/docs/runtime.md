# 运行环境状态

运行环境已于 2026-08-30 恢复，环境实体和缓存均位于 `workspace-ll`。

## OpenPI

- Python：`3.11.16`
- 环境：`XPolicyLab/policy/Pi_05/openpi/.venv`
- 工作区 CPython：`experiments/pi05_hugebench/.python/`
- uv：`workspace-ll/.tools/bin/uv` (`0.12.7`)
- JAX/JAXLIB：`0.5.3/0.5.3`，GPU backend 已验证
- Torch：`2.10.0+cu128`
- NumPy：`1.26.4`
- LeRobot：`0.1.0`，对应本实验所需的 v2.1 数据布局

原环境的约 9.7GB site-packages 完整保留，只有 uv 管理的 CPython 可执行文件丢失。本次在实验目录安装 CPython 3.11.16，并恢复 `.venv/bin/python{,3,3.11}`。`uv.lock` 继续作为依赖锁。

注意：只导出 `HF_LEROBOT_HOME`，不能导出已弃用的 `LEROBOT_HOME`；旧版 LeRobot 检测到后会直接报错。统一配置已处理这一点。

## Gaussian Splatting 与指标

- Python：`3.10.20`
- 环境：`experiments/pi05_hugebench/.venv/gaussian-splatting`
- Torch：`2.8.0+cu126`
- 系统 nvcc：`12.6`
- GPU 架构：A100 / compute capability `8.0`
- CUDA 扩展：`diff-gaussian-rasterization`、`simple-knn`、`fused-ssim`
- 指标依赖：`trimesh 5.0.0`、`rtree 1.4.1` 等

该环境由 `/opt/conda/envs/ptca` 克隆到工作区，再补齐 `configs/gaussian-splatting-requirements.txt` 并针对当前 Torch/CUDA/A100 重新编译三个本地扩展。

## 已完成验证

1. `openpi`、JAX、LeRobot、`pi05_overall` 配置导入；
2. JAX 识别 CUDA GPU；
3. `action_infer.py --help` 完整导入；
4. 在 CPU backend 完整恢复 step 13437 为 `Policy`；
5. `diff_gaussian_rasterization`、`simple_knn`、`fused_ssim` CUDA 扩展执行；
6. `no3_door` 的 13,230,531 个 Gaussian 单帧渲染，输出 343×257 PNG；
7. 一条既有 rollout 的 `metric.py`、`metric_paper_aligned.py` 和 mesh 碰撞检测。

验证产物位于 `artifacts/runtime-validation/metric_smoke/`。它们仅证明环境可用，不是新训练模型的评测结果。

## 当前 GPU 状态

恢复时 8 张 A100 均被 PID `4578` 的 `/blob/thinking.py` 占用约 27GB，每卡仅余约 12–13GB。未擅自终止该进程，因此尚未执行会加载完整 pi0.5 checkpoint 的模型级 smoke。释放显存后运行：

```bash
cd /home/aiscuser/workspace-ll/experiments/pi05_hugebench
NUM_SHARDS=1 bash scripts/smoke.sh
```

`scripts/smoke.sh` 当前默认就是 1 个 renderer + 1 个 inference shard，共 4 条 `test_seen` episode。
Rollout 默认要求所用 GPU 每卡至少有 24,000 MiB 空闲显存；不足时会在启动 renderer 前退出。

环境复核可运行：

```bash
bash scripts/verify_runtime.sh
```

若环境再次丢失或迁移到新节点，可执行幂等恢复入口：

```bash
bash scripts/setup_runtime.sh
```

它优先复用现有 OpenPI site-packages，只补 CPython；renderer 环境不存在时才从 CUDA 12.6 的 `ptca` 环境克隆，并按当前 A100 架构重编本地扩展。
