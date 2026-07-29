#!/bin/bash
# note (luojiaxuan): h00 32B 两臂:shard 0-3 × 6 envs × 2 campaigns,6 卡 6 服务器。
set -u
sed -i 's|/data/runs/selector-v5a/arm-twotower|/data/runs/selector-v4/arm-twotower|' /data/mw/tools/serve_32b.sh
pids=$(ps aux | grep -E 'serve_selv5|serve_32b|port 589' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill -9 $pids; sleep 3
for c in mw-32b-sel-v2 mw-32b-recent-v2; do mkdir -p /data/mw/runs/$c; done
python3 - <<'PY'
import json
base = json.load(open('/tmp/mw-f1.json'))
cs = json.load(open('/tmp/mw-f1.json'))['containers'] + json.load(open('/tmp/mw-f2.json'))['containers'][:8]
cs += json.load(open('/tmp/mw-32bloop-f1.json'))['containers'] + json.load(open('/tmp/mw-32bloop-f2.json'))['containers']
for c in cs:
    c['backend_url'] = c['backend_url'].replace('http://127.0.0.1:', 'http://172.17.0.1:')
camps = {'mw-32b-sel-v2': cs[0:24], 'mw-32b-recent-v2': cs[24:48]}
for camp, block in camps.items():
    for s in range(4):
        m = dict(base); m['containers'] = block[6*s:6*s+6]; m['all_ready'] = True
        json.dump(m, open(f'/data/mw/runs/{camp}/fleet-shard-{s}.json', 'w'), indent=1)
print('H00_32B_SPLIT_OK')
PY
CUDA_VISIBLE_DEVICES=3 nohup bash /data/mw/tools/serve_32b.sh 58901 /data/mw/runs/mw-32b-sel-v2/p1.log sel > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup bash /data/mw/tools/serve_32b.sh 58902 /data/mw/runs/mw-32b-sel-v2/p2.log sel > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash /data/mw/tools/serve_32b.sh 58905 /data/mw/runs/mw-32b-sel-v2/p3.log sel > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup bash /data/mw/tools/serve_32b.sh 58903 /data/mw/runs/mw-32b-recent-v2/p1.log plain > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash /data/mw/tools/serve_32b.sh 58906 /data/mw/runs/mw-32b-sel-v2/p4.log sel > /dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup bash /data/mw/tools/serve_32b.sh 58907 /data/mw/runs/mw-32b-recent-v2/p3.log plain > /dev/null 2>&1 &
for p in 58901 58902 58905 58903 58906 58907; do
  until curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1; do sleep 30; done
done
SELP=(58901 58902 58905 58906); RECP=(58903 58907 58903 58907)
for s in 0 1 2 3; do
  nohup bash /data/mw/tools/supervise_shard_cc4.sh 4 $s 6 ${SELP[$s]} /data/mw/runs/mw-32b-sel-v2 8 > /data/mw/runs/mw-32b-sel-v2/supervisor-shard-$s.log 2>&1 &
  nohup bash /data/mw/tools/supervise_shard_cc4.sh 4 $s 6 ${RECP[$s]} /data/mw/runs/mw-32b-recent-v2 8 > /data/mw/runs/mw-32b-recent-v2/supervisor-shard-$s.log 2>&1 &
done
echo H00_32B_UP
