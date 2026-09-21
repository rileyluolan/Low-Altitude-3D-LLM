# XPolicyLab π0.5 on HUGE-Bench

The requested model is XPolicyLab Pi_05's `pi05_base` checkpoint fine-tuned on
HUGE-Bench `task_overall/train`.

Canonical paths:

- Base weights: `gs://openpi-assets/checkpoints/pi05_base/params`
- XPolicyLab implementation: `XPolicyLab/policy/Pi_05/openpi`
- Training config: `pi05_overall` in `XPolicyLab/policy/Pi_05/openpi/src/openpi/training/config.py`
- Newly trained checkpoint: `pi05_ckpts/pi05_overall/pi05_overall_run1` (latest step currently `13437`)
- Training entry point: `train_pi05_hugebench.sh`
- Rollout and metric entry point: `run_pi05_hugebench_eval.sh`

All paths and cache variables are defined by `pi05env.sh` in this workspace.
The official `HUGE_data/HUGE_PI05` checkpoint is a reference artifact and is
not used by the dedicated evaluation entry point.

## Train or resume

```bash
cd /home/aiscuser/workspace-ll
bash train_pi05_hugebench.sh pi05_overall pi05_overall_run1
```

The wrapper resumes when numeric checkpoints already exist. Set
`FORCE_RESTART=1` only when intentionally restarting the same experiment.

## Evaluate the newly trained checkpoint

Provide a Python environment containing the compiled 3DGS renderer
dependencies, then run:

```bash
cd /home/aiscuser/workspace-ll
GS_PYTHON=/path/to/gaussian_splatting/bin/python \
  bash run_pi05_hugebench_eval.sh
```

For a four-episode smoke test:

```bash
NUM_TRAJS=4 GS_PYTHON=/path/to/gaussian_splatting/bin/python \
  bash run_pi05_hugebench_eval.sh HUGE_data/rollout_pi05_smoke test_seen
```

With `NUM_SHARDS=4`, evaluation uses renderer GPUs 0-3 and policy GPUs 4-7.
`action_infer.py` selects the latest numeric step under the trained checkpoint
root. The entry point writes `traj_gt_pred_xyzk.npz` files and runs both
HUGE-Bench metric scripts with `--tasks overall`.
