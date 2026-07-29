#!/bin/bash
# 用法: osworld_worker_loop_h00.sh <arm> <shard> <count> <endpoint> <outroot> <memarm> <budget>
# note (luojiaxuan): hyper00 版(宿主同名路径 /data02/jaxan,provider 生成的
# 挂载路径对宿主 dockerd 有效)。逻辑同 hyper01 版:5 轮,failed 清零即收工。
set -u
ARM=$1; SHARD=$2; COUNT=$3; EP=$4; OUT=$5; MEMARM=$6; BUDGET=$7
LOG=/data02/jaxan/osworld/logs/worker-$ARM-$SHARD.log
cd /data02/jaxan/osworld/run
for attempt in 1 2 3 4 5; do
  PYTHONPATH=/data02/jaxan/CausalCache/code python3 \
    /data02/jaxan/CausalCache/code/scripts/run_osworld_benchmark_worker.py \
    --osworld-root /data02/jaxan/osworld/OSWorld \
    --meta-path evaluation_examples/test_nogdrive.json \
    --shard-index "$SHARD" --shard-count "$COUNT" \
    --output-root "$OUT" \
    --policy-endpoint "$EP" \
    --memory-arm "$MEMARM" --memory-budget "$BUDGET" \
    --cache-dir /data02/jaxan/osworld/cache \
    >> "$LOG" 2>&1
  code=$?
  echo "WORKER_LOOP_${ARM}_${SHARD}_ATTEMPT_${attempt}_EXIT_${code}" >> "$LOG"
  last=$(grep -h '"event": "WORKER_COMPLETE"' "$LOG" | tail -1)
  if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then
    echo "WORKER_LOOP_${ARM}_${SHARD}_CLEAN" >> "$LOG"
    break
  fi
  sleep 30
done
