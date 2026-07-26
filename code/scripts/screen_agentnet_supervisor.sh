#!/bin/bash
# note (luojiaxuan): AgentNet screening 单分片 supervisor。score-cache 天然断点
# 续传,所以策略就是"崩了重跑同命令":非零退出重启(上限 --max-restarts),每次
# 重启记录退出码;正常退出(0)即收工。心跳文件由打分脚本每单元刷新,外部监视器
# (gpu-utilization-monitor 或 stale-progress 轮询)按 stale 阈值报警 —— 本脚本
# 不替代容器外监视,只兜进程级重启。
# 用法:
#   screen_agentnet_supervisor.sh <shard_index> <shard_count> <device> \
#       <manifest> <image_root> <model_dir> <snapshot_manifest> <cache> <logdir>
set -u
SHARD_INDEX=$1; SHARD_COUNT=$2; DEVICE=$3
MANIFEST=$4; IMAGE_ROOT=$5; MODEL_DIR=$6; SNAPSHOT=$7; CACHE=$8; LOGDIR=$9
MAX_RESTARTS=${MAX_RESTARTS:-30}
mkdir -p "$LOGDIR"
LOG="$LOGDIR/shard${SHARD_INDEX}.log"
HEARTBEAT="$LOGDIR/shard${SHARD_INDEX}.heartbeat"
for attempt in $(seq 1 "$MAX_RESTARTS"); do
  echo "[supervisor] shard=$SHARD_INDEX attempt=$attempt $(date -u +%FT%TZ)" >> "$LOG"
  python3 code/scripts/screen_agentnet_decision_points.py \
    --manifest "$MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --model-dir "$MODEL_DIR" \
    --snapshot-manifest "$SNAPSHOT" \
    --device "$DEVICE" \
    --score-cache "$CACHE" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --heartbeat "$HEARTBEAT" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] shard=$SHARD_INDEX SUCCESS $(date -u +%FT%TZ)" >> "$LOG"
    exit 0
  fi
  echo "[supervisor] shard=$SHARD_INDEX rc=$rc, restart in 20s" >> "$LOG"
  sleep 20
done
echo "[supervisor] shard=$SHARD_INDEX FAILED after $MAX_RESTARTS attempts" >> "$LOG"
exit 1
