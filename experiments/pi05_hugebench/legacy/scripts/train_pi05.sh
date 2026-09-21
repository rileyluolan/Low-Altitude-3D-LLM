#!/usr/bin/env bash
# 在 tmux 中启动 / 恢复 pi05_overall 训练。
#
#   bash train_pi05.sh                      # 默认 pi05_overall, 自动判断首启还是续跑
#   bash train_pi05.sh pi05_overall my_exp  # 指定 config 与实验名
#   FORCE_RESTART=1 bash train_pi05.sh      # 丢弃已有 checkpoint 重头开始
#
# 启动后:  tmux attach -t pi05train        观察实时进度
#          Ctrl-b d                        脱离 (训练继续)
#          tail -f <log>                   看过滤后的文本日志
set -euo pipefail

CONFIG="${1:-pi05_overall}"
EXP="${2:-${CONFIG}_run1}"
SESSION="${TMUX_SESSION:-pi05train}"

source ~/pi05env.sh
CKPT_DIR="$PI05_CKPT/$CONFIG/$EXP"
LOG="$PI05_CKPT/${CONFIG}__${EXP}.log"

# ---------- 预检 1: tmux 会话是否已在跑 ----------
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[ERR] tmux 会话 '$SESSION' 已存在。attach 查看: tmux attach -t $SESSION" >&2
  echo "      若确认它已死, 先 tmux kill-session -t $SESSION" >&2
  exit 1
fi

# ---------- 预检 2: 显存 ----------
# 首启需要把 float32 的 pi05_base (13.4 GB) 复制到每张卡, 再加分片后的
# TrainState (~6.6 GB) 与临时量 -> 每卡约 22 GB 起步。续跑不加载 base 权重,
# 门槛低一些, 但仍需容纳 TrainState + 激活。
MIN_FREE_MB=24000
echo "[预检] 各卡空闲显存 (MiB):"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sed 's/^/        /'
LOW=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v m=$MIN_FREE_MB '$1<m' | wc -l)
if [[ "$LOW" -gt 0 ]]; then
  echo "[ERR] 有 $LOW 张卡空闲显存 < ${MIN_FREE_MB} MiB, 训练会在 init_train_state 阶段 OOM。" >&2
  echo "      占用进程:" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/        /' >&2
  [[ "${IGNORE_GPU_CHECK:-0}" == "1" ]] || exit 1
  echo "[WARN] IGNORE_GPU_CHECK=1, 继续。" >&2
fi

# ---------- 决定首启 / 续跑 ----------
# 注意: openpi 的 initialize_checkpoint_dir 只看目录是否存在, 不看里面有没有
# checkpoint。若首启在存盘前崩溃, 目录已建但为空, 此时带 --resume 会在
# restore_state 处炸掉。所以这里按"是否存在数字 step 子目录"判断。
HAS_CKPT=0
if [[ -d "$CKPT_DIR" ]] && find "$CKPT_DIR" -maxdepth 1 -type d -regex '.*/[0-9]+$' -print -quit | grep -q .; then
  HAS_CKPT=1
fi

FLAG=""
if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
  FLAG="--overwrite";  MODE="强制重头开始 (丢弃已有 checkpoint)"
elif [[ "$HAS_CKPT" == "1" ]]; then
  LAST=$(find "$CKPT_DIR" -maxdepth 1 -type d -regex '.*/[0-9]+$' -printf '%f\n' | sort -n | tail -1)
  FLAG="--resume";     MODE="续跑 (最新 checkpoint: step $LAST)"
elif [[ -d "$CKPT_DIR" ]]; then
  FLAG="--overwrite";  MODE="目录存在但无 checkpoint (上次可能启动即崩), 按首启处理"
else
  FLAG="";             MODE="首次启动"
fi

echo
echo "  config      : $CONFIG"
echo "  exp_name    : $EXP"
echo "  模式        : $MODE"
echo "  checkpoint  : $CKPT_DIR"
echo "  文本日志    : $LOG"
echo "  wandb       : $WANDB_MODE -> $WANDB_DIR"
echo "  tmux 会话   : $SESSION"
echo

mkdir -p "$(dirname "$LOG")"

CMD="source ~/pi05env.sh && cd \$OPENPI && \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py $CONFIG --exp-name=$EXP --num-workers=16 $FLAG"

tmux new-session -d -s "$SESSION" -x 200 -y 50
# tqdm 用 \r 原地刷新, 直接落盘会变成一坨。转成换行后只留有意义的行。
tmux pipe-pane -o -t "$SESSION" \
  "stdbuf -oL tr '\r' '\n' | stdbuf -oL grep -E 'Step [0-9]+:|INFO|WARNING|ERROR|Error|Traceback|Exception|it/s\]$' >> '$LOG'"
tmux send-keys -t "$SESSION" "$CMD" C-m

echo "[OK] 已在 tmux 会话 '$SESSION' 中启动。"
echo
echo "  实时进度 : tmux attach -t $SESSION        (脱离: Ctrl-b d)"
echo "  文本日志 : tail -f $LOG"
echo "  中断训练 : tmux send-keys -t $SESSION C-c  (checkpoint 已存的部分保留, 再跑本脚本即续跑)"
echo
echo "  tqdm 会显示: 已完成步数/总步数 [已用时<剩余时间, 速度]"
echo "  每 100 步另打印一行: Step N: loss=..., learning_rate=..."
