#!/bin/bash
# note (luojiaxuan): frozen-teacher singleton 打分,h00 侧 shard 0-6(7 卡),shard_count=11。
set -u
OUT=/data/runs/selector-v4-frozen/singletons
mkdir -p "$OUT"
cd /data/CausalCache/code
export PYTHONPATH=/data/CausalCache/code
GPUS=(0 1 3 4 5 6 7)
for k in 0 1 2 3 4 5 6; do
  gpu=${GPUS[$k]}
  nohup bash -c "for a in 1 2 3; do CUDA_VISIBLE_DEVICES=$gpu python3 -m scripts.score_selector_v4_singletons \
    --screening-manifest /data/desktop-did-corpus-v1/agentnet_screening_manifest_ubuntu_v1.jsonl \
    --image-root /data/agentnet-frames-v3 \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest /data/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json \
    --teacher frozen --device cuda:0 --splits train dev \
    --shard-index $k --shard-count 11 \
    --output-root $OUT >> $OUT/shard$k.log 2>&1 && break; echo RETRY_\$a >> $OUT/shard$k.log; sleep 20; done" \
    > /dev/null 2>&1 &
done
echo H00_FROZEN_SINGLETONS_LAUNCHED
