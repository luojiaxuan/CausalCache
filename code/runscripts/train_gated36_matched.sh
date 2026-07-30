#!/bin/bash
# note (luojiaxuan): 决定性对照——门控全层(last_36)训在与 full-lora 完全相同的修复语料
# 与预算上,用以把"漂移由门控控制"与"漂移由数据多样性驱动"分开。
set -u
cd /data/CausalCache
export PYTHONPATH=/data/CausalCache/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,4
OUT=/data/runs/desktop-did-v5/gated36-matched
mkdir -p $OUT
for attempt in 1 2 3; do
  echo "=== attempt $attempt $(date -u +%FT%TZ) ===" >> $OUT/train.log
  torchrun --standalone --nproc_per_node=4 code/scripts/train_success_sft_lora.py \
    --config code/configs/causalcache_desktop_did_hgkv_v5_selected_full.json \
    --dataset-root /data/desktop-did-corpus-v5-selected \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --output-root $OUT \
    --frozen-score-cache $OUT/frozen-cache \
    --encode-cache-scope group \
    --repository-root . >> $OUT/train.log 2>&1
  CODE=$?
  if [ $CODE -eq 0 ]; then touch $OUT/EXIT_0; exit 0; fi
  echo "attempt $attempt failed code $CODE" >> $OUT/train.log
  touch $OUT/EXIT_$CODE
  sleep 30
done
exit 1
