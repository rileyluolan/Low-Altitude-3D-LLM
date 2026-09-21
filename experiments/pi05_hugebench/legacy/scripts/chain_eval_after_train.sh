#!/usr/bin/env bash
# 训练结束后自动接评测：等待 -> 冒烟验证 -> 全量 rollout -> 打印指标命令。
#
#   bash chain_eval_after_train.sh <train_pid>
#
# 先跑 NUM_TRAJS=4 的冒烟，通过了才跑全量 —— 无人值守时避免链条在半夜断掉、
# 白等 5 小时。评的是官方发布的 HUGE_PI05（对照组），用来确认整条评测链能
# 跑出论文量级的 nDTW，再拿我们自己训的模型去比。
set -uo pipefail

TPID="${1:?用法: $0 <train_pid>}"
source ~/pi05env.sh

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=/home/aiscuser/workspace-ll/logs/chain_eval_$STAMP.log
mkdir -p "$(dirname "$LOG")"
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

CKPT_OFFICIAL=$DATA/HUGE_PI05
OUT_SMOKE=$DATA/rollout_official_smoke
OUT_FULL=$DATA/rollout_official

say "守护启动，等待训练进程 $TPID 结束..."
while kill -0 "$TPID" 2>/dev/null; do sleep 60; done
say "训练进程已退出。"

# ---- 确认训练产出（不阻塞评测，仅记录）----
RUN_DIR=$PI05_CKPT/pi05_overall/pi05_overall_run1
if [[ -d "$RUN_DIR" ]]; then
  STEPS=$(find "$RUN_DIR" -maxdepth 1 -type d -regex '.*/[0-9]+$' -printf '%f\n' 2>/dev/null | sort -n | tr '\n' ' ')
  say "训练 checkpoint: $STEPS"
  LAST=$(echo "$STEPS" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | tail -1)
  [[ "${LAST:-0}" -ge 13000 ]] && say "最终 step=$LAST，训练已完成 1 epoch" \
                               || say "警告: 最终 step=${LAST:-none}，可能未跑满 13438 步"
else
  say "警告: 未找到 $RUN_DIR"
fi

sleep 30   # 等显存彻底释放
say "各卡空闲(MiB): $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | paste -sd, -)"

# ---- 阶段 1: 冒烟 ----
say "=== 阶段1: 冒烟评测 (每 split 4 条 episode) ==="
rm -rf "$OUT_SMOKE"
if NUM_TRAJS=4 bash /home/aiscuser/workspace-ll/eval_pi05_4gpu.sh \
      "$CKPT_OFFICIAL" "$OUT_SMOKE" pi05_overall test_seen >>"$LOG" 2>&1; then
  N=$(find "$OUT_SMOKE" -name traj_gt_pred_xyzk.npz 2>/dev/null | wc -l)
  say "冒烟完成，产出 $N 条轨迹"
  if [[ "$N" -lt 4 ]]; then
    say "冒烟产出过少（<4），中止全量。检查 ${OUT_SMOKE}_logs/"
    exit 1
  fi
else
  say "冒烟失败，中止全量。检查 ${OUT_SMOKE}_logs/"
  exit 1
fi

# ---- 阶段 2: 全量 ----
say "=== 阶段2: 全量 rollout (test_seen + test_unseen, 993 episodes) ==="
rm -rf "$OUT_FULL"
if bash /home/aiscuser/workspace-ll/eval_pi05_4gpu.sh \
      "$CKPT_OFFICIAL" "$OUT_FULL" pi05_overall test_seen,test_unseen >>"$LOG" 2>&1; then
  N=$(find "$OUT_FULL" -name traj_gt_pred_xyzk.npz 2>/dev/null | wc -l)
  say "全量 rollout 完成，产出 $N 条轨迹 (期望 993)"
else
  say "全量 rollout 异常退出，已产出的部分保留在 $OUT_FULL"
fi

# ---- 阶段 3: 指标 ----
# metric.py 只需 numpy/tqdm，碰撞指标额外需要 trimesh(+embreex 加速)。
# 这些装在 gaussian_splatting 环境里，openpi 的 venv 没有，别用错解释器。
say "=== 阶段3: 计算指标 ==="
METRIC_PY=/home/aiscuser/miniconda3/envs/gaussian_splatting/bin/python
cd "$HUGE"
"$METRIC_PY" metric.py --out_dir "$OUT_FULL" --mesh_root "$DATA/data_3d" \
  --mesh_rel terra_ply/simplified_mesh.obj --tcr_thresholds 1,2,5 \
  --json_out "$OUT_FULL/metric.json" >>"$LOG" 2>&1 \
  && say "metric.py 完成 -> $OUT_FULL/metric.json" || say "metric.py 失败"
"$METRIC_PY" metric_paper_aligned.py --out_dir "$OUT_FULL" \
  --json_out "$OUT_FULL/metric_paper_aligned.json" >>"$LOG" 2>&1 \
  && say "metric_paper_aligned.py 完成" || say "metric_paper_aligned.py 失败"

say "全部结束。日志: $LOG"
