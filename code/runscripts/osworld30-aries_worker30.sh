#!/bin/bash
set -u
SHARD=$1; EP=$2
B=/mnt/data6/jiaxuanluo/osworld30
LOG=$B/logs30/worker30-sel-$SHARD.log
cd $B/run30
for a in 1 2 3 4 5; do
  PYTHONPATH=$B/CausalCache/code:$B/OSWorld $B/venv/bin/python \
    $B/CausalCache/code/scripts/run_osworld_benchmark_worker.py \
    --osworld-root $B/OSWorld \
    --meta-path evaluation_examples/test_nogdrive.json \
    --shard-index "$SHARD" --shard-count 24 \
    --output-root $B/output30-sel \
    --policy-endpoint "$EP" \
    --memory-arm full --memory-budget 4 \
    --max-steps 30 \
    --cache-dir $B/cache30 >> "$LOG" 2>&1
  code=$?
  last=$(grep -h '"event": "WORKER_COMPLETE"' "$LOG" | tail -1)
  if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then break; fi
  sleep 30
done
