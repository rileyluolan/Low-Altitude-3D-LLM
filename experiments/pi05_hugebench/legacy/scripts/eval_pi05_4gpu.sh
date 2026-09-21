#!/usr/bin/env bash
# HUGE-Bench rollout 评测：4 渲染器 + 4 推理，共 8 卡。
# 布局沿用 run_huge_vlaser_inference_4gpu.sh：渲染器 GPU 0-3 / 端口 5550-5553，
# 推理 GPU 4-7，按 episode 跨步分片 (shard i 取 eps[i::4])。
#
#   bash eval_pi05_4gpu.sh <checkpoint_dir> <out_dir> [config_name] [splits]
#
# 例：
#   # 官方发布的 HUGE_PI05 权重（对照组）
#   bash eval_pi05_4gpu.sh $HUGE_data/HUGE_PI05 $HUGE_data/rollout_official
#   # 我们自己训的
#   bash eval_pi05_4gpu.sh $PI05_CKPT/pi05_overall/pi05_overall_run1 $HUGE_data/rollout_run1
#
# 注意：flow-matching 采样器有随机性，num_shards 会影响噪声序列。
# 要横向比较的模型必须用**相同的 num_shards 和 seed**。
set -uo pipefail

CKPT="${1:?用法: $0 <checkpoint_dir> <out_dir> [config_name] [splits]}"
OUT="${2:?缺少 out_dir}"
CONFIG="${3:-pi05_overall}"
SPLITS="${4:-test_seen,test_unseen}"

NUM_SHARDS="${NUM_SHARDS:-4}"
SEED="${SEED:-0}"
EXEC_STEPS="${EXEC_STEPS:-10}"       # 论文声明 #3：20 步 chunk 取前 10 步平均
BASE_PORT="${BASE_PORT:-5550}"
NUM_TRAJS="${NUM_TRAJS:-0}"          # 0 = 全量；调试时设小值

source ~/pi05env.sh
GS_ROOT=/home/aiscuser/workspace-ll/gaussian-splatting
GS_PY=/home/aiscuser/miniconda3/envs/gaussian_splatting/bin/python
LOGDIR="${OUT}_logs"
mkdir -p "$OUT" "$LOGDIR"

# scan_dataset_collect 是单线程 for 循环, 每个样本做的都是极小的 tensor 操作。
# torch/OMP 默认按核数(96)开线程, 在这种负载上全是自旋等待: 实测
#   OMP_NUM_THREADS=1 -> 24 样本/秒 ; 不限 -> 3 样本/秒 (慢 8 倍)
# 4 个 shard 一起跑时更严重(各占 2200% CPU 却几乎不前进)。必须限制。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# action_infer.py 用裸 print(), 重定向到文件时会被 Python 缓冲住,
# 导致日志长时间是空的、无法判断是在跑还是卡死。
export PYTHONUNBUFFERED=1

for req in "$GS_PY" "$GS_ROOT/3dgs_renderer.py" "$CKPT"; do
  [[ -e "$req" ]] || { echo "[ERR] 缺少: $req" >&2; exit 1; }
done

echo "  checkpoint : $CKPT"
echo "  out_dir    : $OUT"
echo "  config     : $CONFIG      splits: $SPLITS"
echo "  分片       : $NUM_SHARDS 路   seed=$SEED   exec_steps=$EXEC_STEPS"
echo "  渲染器     : GPU 0-$((NUM_SHARDS-1))  端口 $BASE_PORT-$((BASE_PORT+NUM_SHARDS-1))"
echo "  推理       : GPU $NUM_SHARDS-$((2*NUM_SHARDS-1))"
echo "  日志       : $LOGDIR"
echo

pids=()
cleanup() { trap - INT TERM EXIT; ((${#pids[@]})) && { echo "[INFO] 收尾..."; kill -TERM "${pids[@]}" 2>/dev/null; wait "${pids[@]}" 2>/dev/null; }; }
trap cleanup INT TERM EXIT

# ---------- 渲染器 ----------
for w in $(seq 0 $((NUM_SHARDS-1))); do
  port=$((BASE_PORT + w))
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$" && { echo "[ERR] 端口 $port 已被占用" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES=$w "$GS_PY" "$GS_ROOT/3dgs_renderer.py" --host 127.0.0.1 --port "$port" \
    --ply_template "$DATA/data_3d/{env_id}/3dgs_ply/point_cloud_utm50.ply" \
    > "$LOGDIR/renderer_gpu${w}.log" 2>&1 &
  pids+=("$!")
done

echo "[INFO] 等待渲染器就绪..."
for w in $(seq 0 $((NUM_SHARDS-1))); do
  port=$((BASE_PORT + w)); ok=0
  for _ in $(seq 60); do
    sleep 5
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$" && { ok=1; break; }
  done
  [[ "$ok" == 1 ]] && echo "  渲染器 $w (port $port) 就绪" \
    || { echo "[ERR] 渲染器 $w 未就绪，见 $LOGDIR/renderer_gpu${w}.log" >&2; tail -20 "$LOGDIR/renderer_gpu${w}.log"; exit 1; }
done

# ---------- 推理 ----------
cd "$OPENPI"
eval_pids=()
for w in $(seq 0 $((NUM_SHARDS-1))); do
  gpu=$((NUM_SHARDS + w)); port=$((BASE_PORT + w))
  echo "[INFO] 推理 shard=$w  GPU=$gpu  port=$port"
  CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run scripts/action_infer.py \
    --task_id overall --config_name "$CONFIG" \
    --checkpoint_dir "$CKPT" \
    --splits "$SPLITS" \
    --exec_steps "$EXEC_STEPS" \
    --num_trajs "$NUM_TRAJS" \
    --num_shards "$NUM_SHARDS" --shard_index "$w" --seed "$SEED" \
    --out_dir "$OUT" \
    --host 127.0.0.1 --port "$port" \
    > "$LOGDIR/eval_shard${w}_gpu${gpu}.log" 2>&1 &
  eval_pids+=("$!")
  pids+=("$!")
done

echo
echo "[INFO] 4 路推理已启动。进度: tail -f $LOGDIR/eval_shard0_gpu${NUM_SHARDS}.log"
status=0
for p in "${eval_pids[@]}"; do wait "$p" || status=1; done
echo "[INFO] rollout 结束 (status=$status)"

echo
echo "下一步 —— 评测指标:"
echo "  cd $HUGE"
echo "  python metric.py --out_dir $OUT --mesh_root $DATA/data_3d \\"
echo "         --mesh_rel terra_ply/simplified_mesh.obj --tcr_thresholds 1,2,5"
echo "  python metric_paper_aligned.py --out_dir $OUT   # 论文口径(任务非加权平均)"
exit "$status"
