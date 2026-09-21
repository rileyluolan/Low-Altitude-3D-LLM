# 来源与可复现性

## 代码来源

| 组件 | 工作区位置 | 整理时 Git commit |
|---|---|---|
| XPolicyLab | `external/xpolicylab` | `c07a09614dd44cc4a67483bcb9a82e7439d99926` |
| HUGE-Bench | `external/huge_bench` | `d000b99794e9da85dff117db61ef874659b91ae8` |
| Gaussian Splatting | `external/gaussian_splatting` | `54c035f7834b564019656c3e3fcc3646292f727d` |

XPolicyLab 工作树包含本实验所需的未提交修改：

- `policy/Pi_05/openpi/src/openpi/training/config.py`
- `policy/Pi_05/openpi/scripts/action_infer.py`
- `policy/Pi_05/openpi/src/openpi/policies/drone_policy.py`

因此，仅记录 commit 仍不足以重建本实验；上述文件必须与 checkpoint 一起保留。完整工作树还有大量无关改动，本次整理未触碰它们。

## 训练来源

`pi05_overall` 明确使用：

- `Pi0Config(pi05=True, action_horizon=20, discrete_state_input=False)`；
- `MyDroneDataConfig(repo_id="task_overall/train", prompt_from_task=True)`；
- `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`；
- 8 卡 FSDP、batch size 128、13,438 个训练 step；
- 每 2,000 step 保存，每 4,000 step 保留。

训练保存了 `4000/8000/12000/13437` 四个 checkpoint。最终权重内同时保留 `params`、`assets` 和 `train_state`，归一化统计位于 checkpoint 的 `assets/task_overall/train/norm_stats.json`。

## 数据来源

原始 LeRobot v2.1 数据不复制，通过 `data/lerobot_home/task_overall/` 链接：

| split | episodes | frames |
|---|---:|---:|
| train | 5,175 | 1,720,096 |
| test_seen | 576 | 195,336 |
| test_unseen | 417 | 144,058 |

评测 3D 场景位于 `HUGE_data/data_3d`。训练输入包括 current/first image、4 维 state 和 4 维 action。

## 评测协议

- 4 个 3DGS renderer：GPU 0–3，端口 5550–5553；
- 4 个推理 shard：GPU 4–7；
- `exec_steps=10`，对应 20 步 action chunk 执行前 10 步；
- `seed=0`；
- 正式集合为 seen 576 + unseen 417，共 993 episodes；
- 产物计数以 `traj_gt_pred_xyzk.npz` 为准；
- `metric.py` 与 `metric_paper_aligned.py` 都显式使用 `--tasks overall`。

