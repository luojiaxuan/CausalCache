#!/bin/bash
set -u
docker exec sglang-omni-jaxan bash -c 'cat > /bigdata/osworld/serve30_recent.sh <<"S30"
#!/usr/bin/env bash
set -uo pipefail
cd /data/CausalCache-mwhgkv
while true; do
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \
    --model-dir /bigdata/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port 19280 --visual-tokens 2560 \
    --memory-budget 4 >> /bigdata/osworld/s19280.log 2>&1
  echo REPLICA_EXIT_$? >> /bigdata/osworld/s19280.log
  sleep 10
done
S30
chmod +x /bigdata/osworld/serve30_recent.sh'
docker exec -d sglang-omni-jaxan bash /bigdata/osworld/serve30_recent.sh
CIP=$(docker inspect sglang-omni-jaxan --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
until curl -sm 3 http://$CIP:19280/health >/dev/null 2>&1; do sleep 25; done
echo "H01_SERVER_UP $CIP"
B=/data04/jaxan/osworld
docker exec sglang-omni-jaxan bash -c 'mkdir -p /bigdata/osworld/output30-recent /bigdata/osworld/logs30 /bigdata/osworld/cache30 /bigdata/osworld/run30 && chmod -R 777 /bigdata/osworld'
cat > $B/worker30.sh <<W30
#!/bin/bash
set -u
SHARD=\$1; EP=\$2
LOG=$B/logs30/worker30-recent-\$SHARD.log
cd $B/run30
for a in 1 2 3 4 5; do
  PYTHONPATH=/data02/jaxan/CausalCache-mwhgkv/code:$B/OSWorld python3 \
    /data02/jaxan/CausalCache-mwhgkv/code/scripts/run_osworld_benchmark_worker.py \
    --osworld-root $B/OSWorld \
    --meta-path evaluation_examples/test_nogdrive.json \
    --shard-index "\$SHARD" --shard-count 8 \
    --output-root $B/output30-recent \
    --policy-endpoint "\$EP" \
    --memory-arm full --memory-budget 4 \
    --max-steps 30 \
    --cache-dir $B/cache30 >> "\$LOG" 2>&1
  code=\$?
  last=\$(grep -h '"event": "WORKER_COMPLETE"' "\$LOG" | tail -1)
  if [ \$code -eq 0 ] && echo "\$last" | grep -q '"failed": 0'; then break; fi
  sleep 30
done
W30
chmod +x $B/worker30.sh
for s in 4 5 6 7; do
  nohup $B/worker30.sh $s http://$CIP:19280/act > /dev/null 2>&1 &
done
echo H01_OSW30_WORKERS_UP
