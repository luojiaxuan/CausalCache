#!/bin/bash
# note (luojiaxuan): 135 任务闭环快测,h00 侧两臂(poolrank + frozensel)。
# 三臂同批设计见 data/results/hgkv_v7_poolrank_v1/README.md;第三臂 didbase 在 h01。
# 所有臂:同一 selector bundle(v4 two_tower, beam 3)、budget 4、max_steps 50、
# 同一 meta(fast_devset_v1.json,135 任务)。唯一差异 = adapter checkpoint。
#
# server 在容器内(GPU),worker 在宿主(起 VM 要宿主 dockerd)。
# server 每臂独立、不跨臂复用(hyper00 RAM 泄漏教训:150-244GB RSS/replica)。
set -u
B=/data02/jaxan/osworld
PSHA=$(cat /tmp/psha.txt)

# ---- 容器内:两个 serve 循环 ----
docker exec sglang-omni-jaxan bash -lc 'mkdir -p /data/osworld/logs-fastdev && cat > /data/serve_fast.sh <<"SRV"
#!/bin/bash
# 用法: serve_fast.sh <gpu> <port> <tag> [adapter_ckpt] [adapter_sha]
set -u
DEV=$1; PORT=$2; TAG=$3; CKPT=${4:-}; SHA=${5:-}
LOG=/data/osworld/logs-fastdev/serve-$TAG.log
EXTRA=()
[ -n "$CKPT" ] && EXTRA+=(--adapter-checkpoint "$CKPT" --adapter-checkpoint-sha256 "$SHA")
cd /data/CausalCache
while true; do
  CUDA_VISIBLE_DEVICES=$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 \
    --memory-budget 4 \
    --selector-bundle /data/runs/selector-v4/arm-twotower/marginal_scorer.pt \
    --selector-arch two_tower --selector-beam 3 \
    "${EXTRA[@]}" >> "$LOG" 2>&1
  echo "{\"event\":\"POLICY_REPLICA_EXIT\",\"port\":$PORT,\"exit\":$?,\"at\":\"$(date -u +%FT%TZ)\"}" >> "$LOG"
  sleep 10
done
SRV
chmod +x /data/serve_fast.sh'
docker exec -d sglang-omni-jaxan bash /data/serve_fast.sh 4 19391 poolrank \
  /data/runs/desktop-did-v7/poolrank/lora-step250.pt "$PSHA"
docker exec -d sglang-omni-jaxan bash /data/serve_fast.sh 5 19392 frozensel
echo "h00 servers: poolrank@19391(GPU4) frozensel@19392(GPU5)"

# ---- 宿主:worker 循环 ----
mkdir -p $B/logs-fastdev 2>/dev/null || docker exec sglang-omni-jaxan bash -lc 'mkdir -p /data/osworld/logs-fastdev /data/osworld/out-fast-poolrank /data/osworld/out-fast-frozensel && chmod -R 777 /data/osworld/logs-fastdev /data/osworld/out-fast-poolrank /data/osworld/out-fast-frozensel'
cat > /tmp/fw.sh <<'W'
#!/bin/bash
set -u
ARM=$1; SHARD=$2; EP=$3
B=/data02/jaxan/osworld
LOG=$B/logs-fastdev/worker-$ARM-$SHARD.log
cd $B/run
for a in 1 2 3 4 5; do
  PYTHONPATH=/data02/jaxan/CausalCache/code:$B/OSWorld /usr/bin/python3 \
    /data02/jaxan/CausalCache/code/scripts/run_osworld_benchmark_worker.py \
    --osworld-root $B/OSWorld \
    --meta-path evaluation_examples/fast_devset_v1.json \
    --shard-index "$SHARD" --shard-count 8 \
    --output-root $B/out-fast-$ARM \
    --policy-endpoint "$EP" \
    --memory-arm full --memory-budget 4 \
    --max-steps 50 \
    --cache-dir $B/cache >> "$LOG" 2>&1
  code=$?
  last=$(grep -h '"event": "WORKER_COMPLETE"' "$LOG" | tail -1)
  if [ $code -eq 0 ] && echo "$last" | grep -q '"failed": 0'; then break; fi
  sleep 30
done
W
cp /tmp/fw.sh $B/fw.sh 2>/dev/null || docker exec sglang-omni-jaxan bash -lc 'cp /tmp/fw.sh /data/osworld/fw.sh' 2>/dev/null
chmod +x $B/fw.sh 2>/dev/null
# note (luojiaxuan): h00 容器**已重建为 bridge**(2026-08-03 实查 NetworkMode,
# 不是老教训里的 host)——endpoint 必须用容器 IP。第五条教训第四次生效:
# 每次发射前查 NetworkMode,别信记忆里的网络模式。
CIP=$(docker inspect sglang-omni-jaxan --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
echo "容器 IP: $CIP"
for s in 0 1 2 3 4 5 6 7; do
  nohup $B/fw.sh poolrank $s http://$CIP:19391/act > /dev/null 2>&1 &
  nohup $B/fw.sh frozensel $s http://$CIP:19392/act > /dev/null 2>&1 &
done
sleep 5
echo "h00 workers: $(pgrep -cf '[r]un_osworld_benchmark_worker')"
