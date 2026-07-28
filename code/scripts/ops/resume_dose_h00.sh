#!/bin/bash
# note (luojiaxuan): 驱动更新后恢复 hyper00 剂量扫描(P-B8 + recent-B8)。
# 用法: bash /data02/jaxan/mw/resume_dose_h00.sh
set -u
docker start sglang-omni-jaxan >/dev/null 2>&1
sleep 5
N=$(docker ps --filter name=sglang-omni-jaxan-mwh --format '{{.Names}}' | wc -l)
if [ "$N" -lt 32 ]; then
  echo "EMULATORS_DOWN ($N/32) — 重启舰队(同种子,名字/端口确定性复现)"
  docker ps -a --filter name=sglang-omni-jaxan-mwh --format '{{.Names}}' | xargs -r docker rm -f >/dev/null
  python3 /data02/jaxan/mw/launch_mw_envs.py launch --count 16 --name-prefix sglang-omni-jaxan-mwh- \
    --port-seed h00-dose-20260729 --source-root /data02/jaxan/mw/upstreams/MobileWorld \
    --image ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240 \
    --expected-source-revision 8ae506487bf87785292d6cad101c49955d704d39 \
    --output /tmp/mw-dose8-fleet.json || exit 2
  python3 /data02/jaxan/mw/launch_mw_envs.py launch --count 16 --name-prefix sglang-omni-jaxan-mwh2- \
    --port-seed h00-dose2-20260729 --source-root /data02/jaxan/mw/upstreams/MobileWorld \
    --image ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240 \
    --expected-source-revision 8ae506487bf87785292d6cad101c49955d704d39 \
    --output /tmp/mw-dose8b-fleet.json || exit 2
fi
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=0 bash /data/mw/tools/serve_policy_pb8.sh 57141 /data/mw/runs/mw-p-b8-v1/policy-57141.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=1 bash /data/mw/tools/serve_policy_pb8.sh 57253 /data/mw/runs/mw-p-b8-v1/policy-57253.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=4 bash /data/mw/tools/serve_policy_pb8.sh 57589 /data/mw/runs/mw-p-b8-v1/policy-57589.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=5 bash /data/mw/tools/serve_policy_pb8.sh 57701 /data/mw/runs/mw-p-b8-v1/policy-57701.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=2 bash /data/mw/tools/serve_policy_recent.sh 57365 /data/mw/runs/mw-recent-b8-v1/policy-57365.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=3 bash /data/mw/tools/serve_policy_recent.sh 57477 /data/mw/runs/mw-recent-b8-v1/policy-57477.log'
for p in 57141 57253 57589 57701 57365 57477; do
  until docker exec sglang-omni-jaxan bash -c "curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1"; do sleep 20; done
done
PP=(57141 57253 57589 57701); RP=(57365 57477 57365 57477)
for i in 0 1 2 3; do
  docker exec -d sglang-omni-jaxan bash -c "bash /data/mw/tools/supervise_shard_pb8.sh 8 $i 4 ${PP[$i]} /data/mw/runs/mw-p-b8-v1 4 > /data/mw/runs/mw-p-b8-v1/supervisor-shard-$i.log 2>&1"
  docker exec -d sglang-omni-jaxan bash -c "bash /data/mw/tools/supervise_shard_rb8.sh 8 $i 4 ${RP[$i]} /data/mw/runs/mw-recent-b8-v1 4 > /data/mw/runs/mw-recent-b8-v1/supervisor-shard-$i.log 2>&1"
done
echo RESUME_H00_DONE
