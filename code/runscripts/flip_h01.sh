#!/bin/bash
# note (luojiaxuan): h01 翻臂:shard4-7 转 k1cap + frozensel;
# server 分布:canonical GPU0-1 = k1cap×2,GPU2-3 = frozensel×2(aux 停用可省)。
set -u
docker exec sglang-omni-jaxan bash -c "pkill -f '[s]upervise_shard_cc4e'; pkill -f '[r]un_mobileworld_gui_owl'; sleep 2; pkill -9 -f '[r]un_mobileworld_gui_owl' 2>/dev/null; pkill -f '[s]erve_ablate.sh'; pkill -9 -f '[h]istory-render' 2>/dev/null; sleep 1; true"
docker exec sglang-omni-jaxan-2 bash -c "pkill -f '[s]erve_ablate.sh'; pkill -9 -f '[h]istory-render' 2>/dev/null; sleep 1; true"
python3 - <<'PY'
import json
base = json.load(open('/tmp/mw-g1.json'))
cs = []
for f in ('/tmp/mw-g1.json', '/tmp/mw-g2.json', '/tmp/mw-g3.json'):
    cs += json.load(open(f))['containers']
for c in cs:
    c['backend_url'] = c['backend_url'].replace('http://127.0.0.1:', 'http://172.17.0.1:')
blocks = {'mw-cc-k1cap-v1': cs[0:24], 'mw-frozensel-v1': cs[24:48]}
for camp, block in blocks.items():
    for k, s in enumerate(range(4, 8)):
        m = dict(base); m['containers'] = block[6*k:6*k+6]; m['all_ready'] = True
        json.dump(m, open(f'/tmp/{camp}-fs-{s}.json', 'w'), indent=1)
print('H01_FLIP_SPLIT_OK')
PY
for camp in mw-cc-k1cap-v1 mw-frozensel-v1; do
  docker exec sglang-omni-jaxan mkdir -p /bigdata/mw/runs/$camp
  for s in 4 5 6 7; do docker cp /tmp/$camp-fs-$s.json sglang-omni-jaxan:/bigdata/mw/runs/$camp/fleet-shard-$s.json; done
done
docker exec sglang-omni-jaxan bash -c 'cat > /bigdata/mw/tools-hgkv-sel/serve_k1cap.sh <<"EOF"
#!/usr/bin/env bash
set -uo pipefail
PORT=$1; LOG=$2
cd /data/CausalCache-mwhgkv
while true; do
  PYTHONPATH=code python3 code/scripts/serve_mobileworld_gui_owl_policy.py \
    --model-dir /bigdata/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 \
    --adapter-checkpoint /bigdata/mw/runs/desktop-did-v4/hgkv-ckpts/lora-step300.pt \
    --adapter-checkpoint-sha256 572092c218a97d1aa88b6845086a2ad2cec77c1a6d6fcd796f2e125ed62d67a9 \
    --selector-bundle /bigdata/mw/runs/selector-v4/from-h00/arm-twotower/marginal_scorer.pt \
    --selector-arch two_tower --selection-budget 4 --selector-beam 3 \
    --max-replacements 1 >> "$LOG" 2>&1
  echo "{\"event\":\"POLICY_REPLICA_EXIT\",\"port\":$PORT,\"exit_code\":$?}" >> "$LOG"
  sleep 10
done
EOF
cat > /bigdata/mw/tools-hgkv-sel/serve_frozensel.sh <<"EOF"
#!/usr/bin/env bash
set -uo pipefail
PORT=$1; LOG=$2
cd /data/CausalCache-mwhgkv
while true; do
  PYTHONPATH=code python3 code/scripts/serve_mobileworld_gui_owl_policy.py \
    --model-dir /bigdata/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 \
    --selector-bundle /bigdata/mw/runs/selector-v4/from-h00/arm-twotower/marginal_scorer.pt \
    --selector-arch two_tower --selection-budget 4 --selector-beam 3 >> "$LOG" 2>&1
  echo "{\"event\":\"POLICY_REPLICA_EXIT\",\"port\":$PORT,\"exit_code\":$?}" >> "$LOG"
  sleep 10
done
EOF
chmod +x /bigdata/mw/tools-hgkv-sel/serve_k1cap.sh /bigdata/mw/tools-hgkv-sel/serve_frozensel.sh'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=0 bash /bigdata/mw/tools-hgkv-sel/serve_k1cap.sh 58601 /bigdata/mw/runs/mw-cc-k1cap-v1/policy-58601.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=1 bash /bigdata/mw/tools-hgkv-sel/serve_k1cap.sh 58602 /bigdata/mw/runs/mw-cc-k1cap-v1/policy-58602.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=2 bash /bigdata/mw/tools-hgkv-sel/serve_frozensel.sh 58701 /bigdata/mw/runs/mw-frozensel-v1/policy-58701.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=3 bash /bigdata/mw/tools-hgkv-sel/serve_frozensel.sh 58702 /bigdata/mw/runs/mw-frozensel-v1/policy-58702.log'
for p in 58601 58602 58701 58702; do
  until docker exec sglang-omni-jaxan bash -c "curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1"; do sleep 25; done
done
KP=(http://127.0.0.1:58601 http://127.0.0.1:58602 http://127.0.0.1:58601 http://127.0.0.1:58602)
FP=(http://127.0.0.1:58701 http://127.0.0.1:58702 http://127.0.0.1:58701 http://127.0.0.1:58702)
k=0
for s in 4 5 6 7; do
  docker exec -d sglang-omni-jaxan bash -c "bash /bigdata/mw/tools-hgkv-sel/supervise_shard_cc4e.sh 4 $s 6 ${KP[$k]} /bigdata/mw/runs/mw-cc-k1cap-v1 8 > /bigdata/mw/runs/mw-cc-k1cap-v1/supervisor-shard-$s.log 2>&1"
  docker exec -d sglang-omni-jaxan bash -c "bash /bigdata/mw/tools-hgkv-sel/supervise_shard_cc4e.sh 4 $s 6 ${FP[$k]} /bigdata/mw/runs/mw-frozensel-v1 8 > /bigdata/mw/runs/mw-frozensel-v1/supervisor-shard-$s.log 2>&1"
  k=$((k+1))
done
echo H01_FLIPPED
