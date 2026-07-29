#!/bin/bash
# note (luojiaxuan): h00 翻臂:text/OCR 收官后,shard0-3 全转 k1cap + frozensel。
set -u
docker exec sglang-omni-jaxan bash -c "pkill -f '[s]upervise_shard_cc4'; pkill -f '[r]un_mobileworld_gui_owl'; sleep 2; pkill -9 -f '[r]un_mobileworld_gui_owl' 2>/dev/null; pkill -f '[s]erve_ablate.sh'; sleep 1; pkill -9 -f '[h]istory-render' 2>/dev/null; true"
python3 - <<'PY'
import json
base = json.load(open('/tmp/mw-f1.json'))
cs = []
for f in ('/tmp/mw-f1.json', '/tmp/mw-f2.json', '/tmp/mw-f3.json'):
    cs += json.load(open(f))['containers']
for c in cs:
    c['backend_url'] = c['backend_url'].replace('http://127.0.0.1:', 'http://172.17.0.1:')
blocks = {'mw-cc-k1cap-v1': cs[0:24], 'mw-frozensel-v1': cs[24:48]}
for camp, block in blocks.items():
    for s in range(4):
        m = dict(base); m['containers'] = block[6*s:6*s+6]; m['all_ready'] = True
        json.dump(m, open(f'/tmp/{camp}-fs-{s}.json', 'w'), indent=1)
print('H00_FLIP_SPLIT_OK')
PY
for camp in mw-cc-k1cap-v1 mw-frozensel-v1; do
  docker exec sglang-omni-jaxan mkdir -p /data/mw/runs/$camp
  for s in 0 1 2 3; do docker cp /tmp/$camp-fs-$s.json sglang-omni-jaxan:/data/mw/runs/$camp/fleet-shard-$s.json; done
done
docker exec sglang-omni-jaxan bash -c 'cat > /data/mw/tools/serve_k1cap.sh <<"EOF"
#!/usr/bin/env bash
set -uo pipefail
PORT=$1; LOG=$2
cd /data/CausalCache-mwb0
while true; do
  PYTHONPATH=code python3 code/scripts/serve_mobileworld_gui_owl_policy.py \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 \
    --adapter-checkpoint /data/runs/desktop-did-v4/hgkv/lora-step300.pt \
    --adapter-checkpoint-sha256 572092c218a97d1aa88b6845086a2ad2cec77c1a6d6fcd796f2e125ed62d67a9 \
    --selector-bundle /data/runs/selector-v4/arm-twotower/marginal_scorer.pt \
    --selector-arch two_tower --selection-budget 4 --selector-beam 3 \
    --max-replacements 1 >> "$LOG" 2>&1
  echo "{\"event\":\"POLICY_REPLICA_EXIT\",\"port\":$PORT,\"exit_code\":$?}" >> "$LOG"
  sleep 10
done
EOF
cat > /data/mw/tools/serve_frozensel.sh <<"EOF"
#!/usr/bin/env bash
set -uo pipefail
PORT=$1; LOG=$2
cd /data/CausalCache-mwb0
while true; do
  PYTHONPATH=code python3 code/scripts/serve_mobileworld_gui_owl_policy.py \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 \
    --selector-bundle /data/runs/selector-v4/arm-twotower/marginal_scorer.pt \
    --selector-arch two_tower --selection-budget 4 --selector-beam 3 >> "$LOG" 2>&1
  echo "{\"event\":\"POLICY_REPLICA_EXIT\",\"port\":$PORT,\"exit_code\":$?}" >> "$LOG"
  sleep 10
done
EOF
chmod +x /data/mw/tools/serve_k1cap.sh /data/mw/tools/serve_frozensel.sh'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=3 bash /data/mw/tools/serve_k1cap.sh 58601 /data/mw/runs/mw-cc-k1cap-v1/policy-58601.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=3 bash /data/mw/tools/serve_k1cap.sh 58602 /data/mw/runs/mw-cc-k1cap-v1/policy-58602.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=3 bash /data/mw/tools/serve_frozensel.sh 58701 /data/mw/runs/mw-frozensel-v1/policy-58701.log'
docker exec -d sglang-omni-jaxan bash -c 'CUDA_VISIBLE_DEVICES=3 bash /data/mw/tools/serve_frozensel.sh 58702 /data/mw/runs/mw-frozensel-v1/policy-58702.log'
for p in 58601 58602 58701 58702; do
  until docker exec sglang-omni-jaxan bash -c "curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1"; do sleep 25; done
done
KP=(58601 58602 58601 58602); FP=(58701 58702 58701 58702)
for i in 0 1 2 3; do
  docker exec -d sglang-omni-jaxan bash -c "bash /data/mw/tools/supervise_shard_cc4.sh 4 $i 6 ${KP[$i]} /data/mw/runs/mw-cc-k1cap-v1 8 > /data/mw/runs/mw-cc-k1cap-v1/supervisor-shard-$i.log 2>&1"
  docker exec -d sglang-omni-jaxan bash -c "bash /data/mw/tools/supervise_shard_cc4.sh 4 $i 6 ${FP[$i]} /data/mw/runs/mw-frozensel-v1 8 > /data/mw/runs/mw-frozensel-v1/supervisor-shard-$i.log 2>&1"
done
echo H00_FLIPPED
