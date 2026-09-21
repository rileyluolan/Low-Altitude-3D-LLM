# 结果登记

## 新训练权重

状态：**尚未评测**。

待评测模型为：

```text
artifacts/checkpoints/pi05_overall/pi05_overall_run1/13437
```

在 `results/trained/pi05_overall_run1/step_13437/seed_0/` 同时出现以下内容后，才可标记为完成：

- `rollouts/` 下 993 个 `traj_gt_pred_xyzk.npz`；
- `run_manifest.yaml` 明确记录 `model_source: trained` 和 checkpoint 路径；
- `metrics/metric.json`；
- `metrics/paper_aligned.json`；
- 四份 renderer 日志和四份 inference shard 日志。

不要把下述官方基线指标登记为新训练模型结果。

## 官方 HUGE_PI05 基线

模型来源：`HUGE_data/HUGE_PI05`，用于验证评测链路，不是 `pi05_overall_run1`。

已有全量 rollout：993 条（seen 576，unseen 417）。论文对齐指标：

| 范围 | episodes | Avg. TCR | Segment Avg. TCR | nDTW |
|---|---:|---:|---:|---:|
| Overall | 993 | 0.562776 | 0.566488 | 0.449724 |
| Seen | 576 | 0.598716 | 0.602945 | 0.495736 |
| Unseen | 417 | 0.513133 | 0.516130 | 0.386169 |

参考论文中的 pi0.5 数值为 Avg. TCR `0.581`、nDTW `0.467`。现有 JSON 未传 `mesh_root`，因此没有碰撞率 CR；这不影响已保存的 TCR/nDTW，但后续完整重算应使用 `scripts/eval_metrics.sh` 的显式 mesh 参数。

官方基线还保存了 4 条 seen episode 的 smoke rollout。历史日志归档在 `logs/eval/official_pi05/`。

