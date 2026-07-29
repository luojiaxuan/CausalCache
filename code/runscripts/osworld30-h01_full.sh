#!/bin/bash
set -u
B=/data04/jaxan/osworld
mkdir -p $B/output30-recent $B/logs30 $B/cache30 $B/run30 2>/dev/null || docker exec sglang-omni-jaxan bash -c 'mkdir -p /bigdata/osworld/output30-recent /bigdata/osworld/logs30 /bigdata/osworld/cache30 /bigdata/osworld/run30 && chmod -R 777 /bigdata/osworld'
cat > $B/worker30.sh <<'W30'
#!/bin/bash
set -u
SHARD=$1; EP=$2
B=/data04/jaxan/osworld
LOG=$B/logs30/worker30-recent-$SHARD.log
cd $B/run30
for a in 1 2 3 4 5; do
  PYTHONPATH=/data02/jaxan/CausalCache-mwhgkv/code:$B/OSWorld /usr/bin/python3 \
    /data02/jaxan/CausalCache-mwhgkv/code/scripts/run_osworld_benchmark_worker.py \
    --osworld-root $B/OSWorld \
    --meta-path evaluation_examples/test_nogdrive.json \
    --shard-index "$SHARD" --shard-count 16 \
    --output-root $B/output30-recent \
    --policy-endpoint "$EP" \
    --memory-arm full --memory-budget 4 \
    --max-steps 30 \
    --cache-dir $B/cache30 >> "$LOG" 2>&1
  code=$?
  last=$(grep -h '"event": "WORKER_COMPLETE"' "$LOG" | tail -1)
  if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then break; fi
  sleep 30
done
W30
chmod +x $B/worker30.sh
CIP=$(docker inspect sglang-omni-jaxan --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
for s in 8 9 10 11 12 13 14 15; do
  nohup $B/worker30.sh $s http://$CIP:19280/act > /dev/null 2>&1 &
done
sleep 12
ps aux | grep -c '[r]un_osworld_benchmark_worker'
