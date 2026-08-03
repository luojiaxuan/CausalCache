#!/bin/bash
# note (luojiaxuan): 135 任务闭环快测,h01 侧第三臂(didbase = 旧目标适配器)。
# h01 容器是 bridge 网络,endpoint 必须用容器 IP(README 第五条教训)。
set -u
B=/data04/jaxan/osworld
DSHA=$(cat /tmp/dsha.txt)

docker exec sglang-omni-jaxan bash -lc 'mkdir -p /bigdata/osworld/logs-fastdev /bigdata/osworld/out-fast-didbase /bigdata/osworld/run-fast /bigdata/osworld/cache-fast && chmod -R 777 /bigdata/osworld/logs-fastdev /bigdata/osworld/out-fast-didbase /bigdata/osworld/run-fast /bigdata/osworld/cache-fast && cat > /bigdata/serve_fast.sh <<"SRV"
#!/bin/bash
set -u
DEV=$1; PORT=$2; TAG=$3; CKPT=${4:-}; SHA=${5:-}
LOG=/bigdata/osworld/logs-fastdev/serve-$TAG.log
EXTRA=()
[ -n "$CKPT" ] && EXTRA+=(--adapter-checkpoint "$CKPT" --adapter-checkpoint-sha256 "$SHA")
cd /bigdata/osworld/CausalCache
while true; do
  CUDA_VISIBLE_DEVICES=$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \
    --model-dir /bigdata/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 \
    --memory-budget 4 \
    --selector-bundle /bigdata/mw/runs/selector-v4/arm-twotower/marginal_scorer.pt \
    --selector-arch two_tower --selector-beam 3 \
    "${EXTRA[@]}" >> "$LOG" 2>&1
  echo "{\"event\":\"POLICY_REPLICA_EXIT\",\"port\":$PORT,\"exit\":$?,\"at\":\"$(date -u +%FT%TZ)\"}" >> "$LOG"
  sleep 10
done
SRV
chmod +x /bigdata/serve_fast.sh'
docker exec -d sglang-omni-jaxan bash /bigdata/serve_fast.sh 5 19393 didbase \
  /bigdata/mw/runs/desktop-did-v7/didbase/lora-step300.pt "$DSHA"
echo "h01 server: didbase@19393(GPU5)"

cat > $B/fw.sh <<'W'
#!/bin/bash
set -u
SHARD=$1; EP=$2
B=/data04/jaxan/osworld
LOG=$B/logs-fastdev/worker-didbase-$SHARD.log
cd $B/run-fast 2>/dev/null || cd $B
for a in 1 2 3 4 5; do
  PYTHONPATH=/data02/jaxan/CausalCache-mwhgkv/code:$B/OSWorld /usr/bin/python3 \
    /data02/jaxan/CausalCache-mwhgkv/code/scripts/run_osworld_benchmark_worker.py \
    --osworld-root $B/OSWorld \
    --meta-path evaluation_examples/fast_devset_v1.json \
    --shard-index "$SHARD" --shard-count 8 \
    --output-root $B/out-fast-didbase \
    --policy-endpoint "$EP" \
    --memory-arm full --memory-budget 4 \
    --max-steps 50 \
    --cache-dir $B/cache-fast >> "$LOG" 2>&1
  code=$?
  last=$(grep -h '"event": "WORKER_COMPLETE"' "$LOG" | tail -1)
  if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then break; fi
  sleep 30
done
W
chmod +x $B/fw.sh
CIP=$(docker inspect sglang-omni-jaxan --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
echo "容器 IP: $CIP"
for s in 0 1 2 3 4 5 6 7; do
  nohup $B/fw.sh $s http://$CIP:19393/act > /dev/null 2>&1 &
done
sleep 5
echo "h01 workers: $(pgrep -cf '[r]un_osworld_benchmark_worker')"
