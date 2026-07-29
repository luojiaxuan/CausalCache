#!/bin/bash
set -u
# ---- h00:sel 暂停,24 emus 全归 recent,4 服务器(58903,58907,+58908@7,+58909@2) ----
cat > /tmp/h00_rmax.sh <<'INNER'
#!/bin/bash
pids=$(ps aux | grep 'supervise_shard_cc4.sh 4 [0-3] .*32b-sel' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill $pids
pids=$(ps aux | grep 'run_mobileworld.*32b-sel' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill $pids
python3 - <<'PY'
import json, glob
cs = []
for s in range(4):
    m = json.load(open(f'/data/mw/runs/mw-32b-sel-v2/fleet-shard-{s}.json'))
    cs += m['containers']
    base = m
for s in range(4):
    m2 = json.load(open(f'/data/mw/runs/mw-32b-recent-v2/fleet-shard-{s}.json'))
    cs += m2['containers']
seen, uniq = set(), []
for c in cs:
    if c['backend_url'] not in seen:
        seen.add(c['backend_url']); uniq.append(c)
for s in range(4):
    m = dict(base); m['containers'] = uniq[6*s:6*s+6]; m['all_ready'] = True
    json.dump(m, open(f'/data/mw/runs/mw-32b-recent-v2/fleet-shard-{s}.json', 'w'), indent=1)
print('H00_RMAX_SPLIT', len(uniq))
PY
INNER
scp -q /tmp/h00_rmax.sh hyper00:/tmp/ && ssh hyper00 'docker cp /tmp/h00_rmax.sh sglang-omni-jaxan:/tmp/ && docker exec sglang-omni-jaxan bash /tmp/h00_rmax.sh'
ssh hyper00 'docker exec -d sglang-omni-jaxan bash -c "CUDA_VISIBLE_DEVICES=7 bash /data/mw/tools/serve_32b.sh 58908 /data/mw/runs/mw-32b-recent-v2/p4.log plain"; docker exec -d sglang-omni-jaxan bash -c "CUDA_VISIBLE_DEVICES=2 bash /data/mw/tools/serve_32b.sh 58909 /data/mw/runs/mw-32b-recent-v2/p5.log plain"'
# ---- h01:sel 服务器/监督器全撤,4 新 plain,48 emus 12envs×4 ----
cat > /tmp/h01_rmax.sh <<'INNER'
#!/bin/bash
pids=$(ps aux | grep 'supervise_shard_cc4e.sh 4 [4-7] .*32b-sel' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill $pids
pids=$(ps aux | grep 'run_mobileworld.*32b-sel' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill $pids
python3 - <<'PY'
import json
cs = []
for s in range(4, 8):
    m = json.load(open(f'/bigdata/mw/runs/mw-32b-sel-v2/fleet-shard-{s}.json'))
    cs += m['containers']; base = m
for s in range(4, 8):
    m2 = json.load(open(f'/bigdata/mw/runs/mw-32b-recent-v2/fleet-shard-{s}.json'))
    cs += m2['containers']
seen, uniq = set(), []
for c in cs:
    if c['backend_url'] not in seen:
        seen.add(c['backend_url']); uniq.append(c)
for k, s in enumerate(range(4, 8)):
    m = dict(base); m['containers'] = uniq[12*k:12*k+12]; m['all_ready'] = True
    json.dump(m, open(f'/bigdata/mw/runs/mw-32b-recent-v2/fleet-shard-{s}.json', 'w'), indent=1)
print('H01_RMAX_SPLIT', len(uniq))
PY
INNER
scp -q /tmp/h01_rmax.sh hyper01:/tmp/ && ssh hyper01 'docker cp /tmp/h01_rmax.sh sglang-omni-jaxan:/tmp/ && docker exec sglang-omni-jaxan bash /tmp/h01_rmax.sh'
ssh hyper01 'docker exec sglang-omni-jaxan-2 bash -c "pids=\$(ps aux | grep \"port 5891[12]\" | grep -v grep | awk \"{print \\\$2}\"); [ -n \"\$pids\" ] && kill -9 \$pids; true"; docker exec sglang-omni-jaxan bash -c "pids=\$(ps aux | grep \"port 58915\" | grep -v grep | awk \"{print \\\$2}\"); [ -n \"\$pids\" ] && kill -9 \$pids; true"'
ssh hyper01 'docker exec -d sglang-omni-jaxan-2 bash -c "CUDA_VISIBLE_DEVICES=0 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58919 /bigdata/mw/runs/mw-32b-recent-v2/p6.log plain"; docker exec -d sglang-omni-jaxan-2 bash -c "CUDA_VISIBLE_DEVICES=1 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58920 /bigdata/mw/runs/mw-32b-recent-v2/p7.log plain"; docker exec -d sglang-omni-jaxan bash -c "CUDA_VISIBLE_DEVICES=0 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58921 /bigdata/mw/runs/mw-32b-recent-v2/p8.log plain"; docker exec -d sglang-omni-jaxan bash -c "CUDA_VISIBLE_DEVICES=3 bash /bigdata/mw/tools-hgkv-sel/serve_32b.sh 58922 /bigdata/mw/runs/mw-32b-recent-v2/p9.log plain"'
# 等新服务器,重启 recent 监督器(h00 6envs 1:1;h01 12envs 1:1)
ssh hyper00 'docker exec sglang-omni-jaxan bash -c "until curl -sm 3 http://127.0.0.1:58908/health >/dev/null 2>&1 && curl -sm 3 http://127.0.0.1:58909/health >/dev/null 2>&1; do sleep 30; done; echo H00_NEW_UP"'
cat > /tmp/h00_rsup.sh <<'INNER'
#!/bin/bash
pids=$(ps aux | grep 'supervise_shard_cc4.sh 4 [0-3] .*32b-recent' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill $pids
pids=$(ps aux | grep 'run_mobileworld.*32b-recent' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill $pids
sleep 2
INNER
scp -q /tmp/h00_rsup.sh hyper00:/tmp/ && ssh hyper00 'docker cp /tmp/h00_rsup.sh sglang-omni-jaxan:/tmp/ && docker exec sglang-omni-jaxan bash /tmp/h00_rsup.sh'
for s in 0 1 2 3; do
  case $s in 0) P=58903;; 1) P=58907;; 2) P=58908;; 3) P=58909;; esac
  ssh hyper00 "docker exec -d sglang-omni-jaxan bash -c 'bash /data/mw/tools/supervise_shard_cc4.sh 4 $s 6 $P /data/mw/runs/mw-32b-recent-v2 8 > /data/mw/runs/mw-32b-recent-v2/supervisor-shard-$s.log 2>&1'"
done
ssh hyper01 'docker exec sglang-omni-jaxan-2 bash -c "until curl -sm 3 http://127.0.0.1:58919/health >/dev/null 2>&1 && curl -sm 3 http://127.0.0.1:58920/health >/dev/null 2>&1; do sleep 30; done"; docker exec sglang-omni-jaxan bash -c "until curl -sm 3 http://127.0.0.1:58921/health >/dev/null 2>&1 && curl -sm 3 http://127.0.0.1:58922/health >/dev/null 2>&1; do sleep 30; done; echo H01_NEW_UP"'
cat > /tmp/h01_rsup.sh <<'INNER'
#!/bin/bash
pids=$(ps aux | grep 'supervise_shard_cc4e.sh 4 [4-7] .*32b-recent' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill $pids
pids=$(ps aux | grep 'run_mobileworld.*32b-recent' | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill $pids
sleep 2
INNER
scp -q /tmp/h01_rsup.sh hyper01:/tmp/ && ssh hyper01 'docker cp /tmp/h01_rsup.sh sglang-omni-jaxan:/tmp/ && docker exec sglang-omni-jaxan bash /tmp/h01_rsup.sh'
k=0
for s in 4 5 6 7; do
  case $k in 0) EP=http://172.17.0.55:58919;; 1) EP=http://172.17.0.55:58920;; 2) EP=http://127.0.0.1:58921;; 3) EP=http://127.0.0.1:58922;; esac
  ssh hyper01 "docker exec -d sglang-omni-jaxan bash -c 'bash /bigdata/mw/tools-hgkv-sel/supervise_shard_cc4e.sh 4 $s 12 $EP /bigdata/mw/runs/mw-32b-recent-v2 8 > /bigdata/mw/runs/mw-32b-recent-v2/supervisor-shard-$s.log 2>&1'"
  k=$((k+1))
done
echo RECENT_MAX_DONE
