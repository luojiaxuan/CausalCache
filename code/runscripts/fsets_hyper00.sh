#!/bin/bash
# note (luojiaxuan): frozen sets 打分 h00:8 GPU × 2 proc = shard 0-15(shard_count=50,只跑 0-29 = 60% 状态子集)。
set -u
OUT=/data/runs/selector-v4-frozen/sets
mkdir -p "$OUT"
cd /data/CausalCache/code
export PYTHONPATH=/data/CausalCache/code
GPUS=(0 1 2 3 4 5 6 7)
for i in $(seq 0 15); do
  gpu=${GPUS[$((i % 8))]}
  nohup bash -c "for a in 1 2 3; do CUDA_VISIBLE_DEVICES=$gpu python3 -m scripts.score_selector_v4_sets \
    --singletons-root /data/runs/selector-v4-frozen/singletons \
    --screening-manifest /data/desktop-did-corpus-v1/agentnet_screening_manifest_ubuntu_v1.jsonl \
    --image-root /data/agentnet-frames-v3 \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest /data/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json \
    --teacher frozen --device cuda:0 --splits train dev \
    --shard-index $i --shard-count 50 \
    --output-root $OUT >> $OUT/shard$i.log 2>&1 && break; echo RETRY_\$a >> $OUT/shard$i.log; sleep 20; done" \
    > /dev/null 2>&1 &
done
echo H00_FSETS_LAUNCHED
