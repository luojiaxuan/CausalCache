#!/bin/bash
# note (luojiaxuan): 驱动更新后恢复 hyper01 剂量扫描(P-B1 + P-B2)。
# 用法: bash /data04/jaxan/mw/resume_dose_h01.sh
set -u
docker start sglang-omni-jaxan sglang-omni-jaxan-2 >/dev/null 2>&1
sleep 5
N=$(docker ps --filter name=sglang-omni-jaxan-mws --format '{{.Names}}' | wc -l)
if [ "$N" -lt 32 ]; then
  echo "EMULATORS_DOWN ($N/32) — 重启舰队(同种子,名字/端口确定性复现)"
  docker ps -a --filter name=sglang-omni-jaxan-mws --format '{{.Names}}' | xargs -r docker rm -f >/dev/null
  python3 /data04/jaxan/mw/launch_mw_envs.py launch --count 16 --name-prefix sglang-omni-jaxan-mws- \
    --port-seed h01-dose-20260729 --source-root /data04/jaxan/mw/upstreams/MobileWorld \
    --image ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240 \
    --expected-source-revision 8ae506487bf87785292d6cad101c49955d704d39 \
    --output /tmp/mw-dose-fleet.json || exit 2
  python3 /data04/jaxan/mw/launch_mw_envs.py launch --count 16 --name-prefix sglang-omni-jaxan-mws2- \
    --port-seed h01-dose2-20260729 --source-root /data04/jaxan/mw/upstreams/MobileWorld \
    --image ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240 \
    --expected-source-revision 8ae506487bf87785292d6cad101c49955d704d39 \
    --output /tmp/mw-dose2-fleet.json || exit 2
fi
AUXIP=$(docker inspect sglang-omni-jaxan-2 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
echo "AUX_IP=$AUXIP (原 172.17.0.38,若变化需改 endpoint)"
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=0 bash /bigdata/mw/tools-hgkv-sel/serve_policy_pb.sh 56141 /bigdata/mw/runs/mw-p-b1-v1/policy-56141.log 1'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=1 bash /bigdata/mw/tools-hgkv-sel/serve_policy_pb.sh 56253 /bigdata/mw/runs/mw-p-b1-v1/policy-56253.log 1'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=2 bash /bigdata/mw/tools-hgkv-sel/serve_policy_pb.sh 56365 /bigdata/mw/runs/mw-p-b2-v1/policy-56365.log 2'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=3 bash /bigdata/mw/tools-hgkv-sel/serve_policy_pb.sh 56477 /bigdata/mw/runs/mw-p-b2-v1/policy-56477.log 2'
docker exec -d sglang-omni-jaxan-2 bash -c 'CUDA_VISIBLE_DEVICES=0 bash /bigdata/mw/tools-hgkv-sel/serve_policy_pb.sh 56589 /bigdata/mw/runs/mw-p-b1-v1/policy-56589.log 1'
docker exec -d sglang-omni-jaxan-2 bash -c 'CUDA_VISIBLE_DEVICES=2 bash /bigdata/mw/tools-hgkv-sel/serve_policy_pb.sh 56813 /bigdata/mw/runs/mw-p-b2-v1/policy-56813.log 2'
for p in 56141 56253 56365 56477; do
  until docker exec sglang-omni-jaxan bash -c "curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1"; do sleep 20; done
done
for p in 56589 56813; do
  until docker exec sglang-omni-jaxan-2 bash -c "curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1"; do sleep 20; done
done
B1E=(http://127.0.0.1:56141 http://127.0.0.1:56253 http://$AUXIP:56589 http://127.0.0.1:56253)
B2E=(http://127.0.0.1:56365 http://127.0.0.1:56477 http://$AUXIP:56813 http://127.0.0.1:56477)
for i in 0 1 2 3; do
  docker exec -d sglang-omni-jaxan bash -c "bash /bigdata/mw/tools-hgkv-sel/supervise_shard_b1e.sh 1 $i 4 ${B1E[$i]} /bigdata/mw/runs/mw-p-b1-v1 4 > /bigdata/mw/runs/mw-p-b1-v1/supervisor-shard-$i.log 2>&1"
  docker exec -d sglang-omni-jaxan bash -c "bash /bigdata/mw/tools-hgkv-sel/supervise_shard_b2e.sh 2 $i 4 ${B2E[$i]} /bigdata/mw/runs/mw-p-b2-v1 4 > /bigdata/mw/runs/mw-p-b2-v1/supervisor-shard-$i.log 2>&1"
done
echo RESUME_H01_DONE
