#!/bin/bash
# note (luojiaxuan): h01 32B 两臂:shard 4-7 × 6 envs × 2 campaigns;aux 4 服务器 + canonical 2。
set -u
docker exec sglang-omni-jaxan-2 bash -c 'pids=$(ps aux | grep -E "serve_selv5|serve_fsft" | grep -v grep | awk "{print \$2}"); [ -n "$pids" ] && kill -9 $pids; cp /bigdata/mw/tools-hgkv-sel/serve_32b.sh /tmp/ 2>/dev/null; sed -i "s|--history-render .*$||" /tmp/serve_32b.sh 2>/dev/null; true'
docker exec sglang-omni-jaxan bash -c 'for c in mw-32b-sel-v2 mw-32b-recent-v2; do mkdir -p /bigdata/mw/runs/$c; done'
python3 - <<'PY'
import json
base = json.load(open('/tmp/mw-g1.json'))
cs = []
for f in ('/tmp/mw-g1.json', '/tmp/mw-g2.json', '/tmp/mw-g3.json'):
    cs += json.load(open(f))['containers']
for c in cs:
    c['backend_url'] = c['backend_url'].replace('http://127.0.0.1:', 'http://172.17.0.1:')
camps = {'mw-32b-sel-v2': cs[0:24], 'mw-32b-recent-v2': cs[24:48]}
for camp, block in camps.items():
    for k, s in enumerate(range(4, 8)):
        m = dict(base); m['containers'] = block[6*k:6*k+6]; m['all_ready'] = True
        json.dump(m, open(f'/tmp/{camp}-fs-{s}.json', 'w'), indent=1)
print('H01_32B_SPLIT_OK')
PY
for camp in mw-32b-sel-v2 mw-32b-recent-v2; do
  for s in 4 5 6 7; do docker cp /tmp/$camp-fs-$s.json sglang-omni-jaxan:/bigdata/mw/runs/$camp/fleet-shard-$s.json; done
done
docker exec -d sglang-omni-jaxan-2 bash -c 'CUDA_VISIBLE_DEVICES=0 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58911 /bigdata/mw/runs/mw-32b-sel-v2/p1.log sel'
docker exec -d sglang-omni-jaxan-2 bash -c 'CUDA_VISIBLE_DEVICES=1 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58912 /bigdata/mw/runs/mw-32b-sel-v2/p2.log sel'
docker exec -d sglang-omni-jaxan-2 bash -c 'CUDA_VISIBLE_DEVICES=2 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58913 /bigdata/mw/runs/mw-32b-sel-v2/p4.log sel'
docker exec -d sglang-omni-jaxan-2 bash -c 'CUDA_VISIBLE_DEVICES=3 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58914 /bigdata/mw/runs/mw-32b-recent-v2/p2.log plain'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=0 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58915 /bigdata/mw/runs/mw-32b-sel-v2/p3.log sel'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=1 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58916 /bigdata/mw/runs/mw-32b-recent-v2/p3.log plain'
for p in 58911 58912 58913 58914; do
  until docker exec sglang-omni-jaxan-2 bash -c "curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1"; do sleep 30; done
done
for p in 58915 58916; do
  until docker exec sglang-omni-jaxan bash -c "curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1"; do sleep 30; done
done
SELP=(http://172.17.0.55:58911 http://172.17.0.55:58912 http://127.0.0.1:58915 http://172.17.0.55:58913)
RECP=(http://172.17.0.55:58914 http://127.0.0.1:58916 http://172.17.0.55:58914 http://127.0.0.1:58916)
k=0
for s in 4 5 6 7; do
  docker exec -d sglang-omni-jaxan bash -c "bash /bigdata/mw/tools-hgkv-sel/supervise_shard_cc4e.sh 4 $s 6 ${SELP[$k]} /bigdata/mw/runs/mw-32b-sel-v2 8 > /bigdata/mw/runs/mw-32b-sel-v2/supervisor-shard-$s.log 2>&1"
  docker exec -d sglang-omni-jaxan bash -c "bash /bigdata/mw/tools-hgkv-sel/supervise_shard_cc4e.sh 4 $s 6 ${RECP[$k]} /bigdata/mw/runs/mw-32b-recent-v2 8 > /bigdata/mw/runs/mw-32b-recent-v2/supervisor-shard-$s.log 2>&1"
  k=$((k+1))
done
echo H01_32B_UP
