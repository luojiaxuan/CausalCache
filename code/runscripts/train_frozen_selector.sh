#!/bin/bash
# note (luojiaxuan): frozen-teacher selector 训练(aux 容器 GPU=host6=CUDA2)。
# 输入:frozen singletons(11 shard)+ frozen sets(30 shard,60% 状态)。
set -u
cd /data/CausalCache-mwhgkv
export PYTHONPATH=/data/CausalCache-mwhgkv/code
OUT=/bigdata/mw/runs/selector-v4-frozen/arm-twotower
mkdir -p $OUT
CUDA_VISIBLE_DEVICES=2 python3 -m scripts.train_selector_v4_marginal \
  --singletons-root /bigdata/mw/runs/selector-v4-frozen/singletons \
  --sets-root /bigdata/mw/runs/selector-v4-frozen/sets \
  --screening-manifest /data/CausalCache-v4/data/manifests/agentnet_screening_manifest_ubuntu_v1.jsonl \
  --epochs 40 --eval-every 5 --early-stop-patience 3 \
  --feature-cache /bigdata/mw/runs/selector-v4-frozen/features.pt \
  --feature-workers 64 \
  --arch two_tower --device cuda:0 \
  --output-root $OUT > $OUT/train.log 2>&1
CODE=$?
[ $CODE -eq 0 ] && touch $OUT/EXIT_0 || touch $OUT/EXIT_$CODE
echo done $CODE
