#!/usr/bin/env bash
# note (luojiaxuan): 32B 跨骨干 HGKV 支线训练(h01 aux 容器,host GPU 4/5/6)。
set -uo pipefail
OUT=/bigdata/mw/runs/desktop-did-32b/hgkv
mkdir -p "$OUT/frozen-cache"
cd /data/CausalCache-mwhgkv
export PYTHONPATH=/data/CausalCache-mwhgkv/code
export CUDA_VISIBLE_DEVICES=1,2,3
for attempt in 1 2 3; do
  echo "=== attempt $attempt $(date -u +%FT%TZ) ===" >> "$OUT/train.log"
  torchrun --standalone --nproc_per_node=3 code/scripts/train_success_sft_lora.py \
    --config code/configs/causalcache_desktop_did_hgkv_v4_32b.json \
    --dataset-root /bigdata/mw/desktop-did-corpus-v4 \
    --model-dir /bigdata/models/GUI-Owl-1.5-32B-Instruct \
    --output-root "$OUT" \
    --frozen-score-cache "$OUT/frozen-cache" \
    --encode-cache-scope group \
    --model-profile 32b \
    --repository-root . >> "$OUT/train.log" 2>&1
  CODE=$?
  if [ $CODE -eq 0 ]; then touch "$OUT/EXIT_0"; exit 0; fi
  echo "attempt $attempt failed code $CODE" >> "$OUT/train.log"
  touch "$OUT/EXIT_$CODE"
  sleep 30
done
exit 1
