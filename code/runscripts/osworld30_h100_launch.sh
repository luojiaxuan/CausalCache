#!/bin/bash
# note (luojiaxuan): OSWorld-30 扩展 horizon 诊断(h100,3 卡):
# 两臂 Recent-4 vs frozen+selector,--max-steps 30,361 官方名册,8 shard/臂。
# serve 在容器(GPU1/2=sel,GPU3=recent);worker 在宿主(需 dockerd)。
set -u
BASE=/data/jaxan/osworld-runner
until [ -f $BASE/ship/SHIP_DONE ]; do sleep 30; done
echo "SHIP_READY $(date -u +%H:%M)"

docker exec sglang-omni-jaxan bash -c 'cat > /data/osworld-runner/serve30.sh <<"EOF"
#!/usr/bin/env bash
set -uo pipefail
DEV=$1; PORT=$2; LOG=$3; SELECTOR=${4:-0}
REPO=/data/osworld-runner/CausalCache
EXTRA=()
if [ "$SELECTOR" = "1" ]; then
  EXTRA+=(--selector-bundle /data/osworld-runner/ship/runs/selector-v4/arm-twotower/marginal_scorer.pt
          --selector-arch two_tower --selector-beam 3)
fi
cd "$REPO"
while true; do
  CUDA_VISIBLE_DEVICES=$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \
    --model-dir /data/osworld-runner/ship/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 \
    --memory-budget 4 "${EXTRA[@]}" >> "$LOG" 2>&1
  echo "REPLICA_EXIT_$? $(date -u +%FT%TZ)" >> "$LOG"
  sleep 10
done
EOF
chmod +x /data/osworld-runner/serve30.sh; mkdir -p /data/osworld-runner/logs30'
docker exec -d sglang-omni-jaxan bash -c 'bash /data/osworld-runner/serve30.sh 1 19080 /data/osworld-runner/logs30/s19080.log 1'
docker exec -d sglang-omni-jaxan bash -c 'bash /data/osworld-runner/serve30.sh 2 19081 /data/osworld-runner/logs30/s19081.log 1'
docker exec -d sglang-omni-jaxan bash -c 'bash /data/osworld-runner/serve30.sh 3 19082 /data/osworld-runner/logs30/s19082.log 0'
CIP=$(docker inspect sglang-omni-jaxan --format '{{.NetworkSettings.IPAddress}}')
for p in 19080 19081 19082; do
  until curl -sm 3 http://$CIP:$p/health >/dev/null 2>&1; do sleep 25; done
done
echo "SERVERS_READY $CIP"

mkdir -p $BASE/output/verified30-sel $BASE/output/verified30-recent $BASE/logs30
cat > $BASE/worker30.sh <<EOF
#!/bin/bash
set -u
ARM=\$1; SHARD=\$2; EP=\$3; OUT=\$4
LOG=$BASE/logs30/worker-\$ARM-\$SHARD.log
cd $BASE/run 2>/dev/null || cd $BASE
for attempt in 1 2 3 4 5; do
  PYTHONPATH=$BASE/CausalCache/code $BASE/venv/bin/python3 \
    $BASE/CausalCache/code/scripts/run_osworld_benchmark_worker.py \
    --osworld-root $BASE/OSWorld \
    --meta-path evaluation_examples/test_nogdrive.json \
    --shard-index "\$SHARD" --shard-count 8 \
    --output-root "\$OUT" \
    --policy-endpoint "\$EP" \
    --memory-arm full --memory-budget 4 \
    --max-steps 30 \
    --cache-dir $BASE/cache >> "\$LOG" 2>&1
  code=\$?
  last=\$(grep -h '"event": "WORKER_COMPLETE"' "\$LOG" | tail -1)
  if [ \$code -eq 0 ] && echo "\$last" | grep -q '"failed": 0'; then break; fi
  sleep 30
done
EOF
chmod +x $BASE/worker30.sh
for s in 0 1 2 3 4 5 6 7; do
  [ $((s % 2)) -eq 0 ] && SEP=http://$CIP:19080/act || SEP=http://$CIP:19081/act
  nohup $BASE/worker30.sh sel $s $SEP $BASE/output/verified30-sel > /dev/null 2>&1 &
  nohup $BASE/worker30.sh recent $s http://$CIP:19082/act $BASE/output/verified30-recent > /dev/null 2>&1 &
done
echo "OSWORLD30_LAUNCHED"
