#!/bin/bash
# note (luojiaxuan): v5a EXIT_0 → 双口径 devscore(GPU0=v4-b1 经典,GPU1=v5-hit 多槽)。
CK=/data/runs/desktop-did-v5/hgkv-sel
while [ ! -f $CK/EXIT_0 ]; do
  [ -f $CK/EXIT_1 ] && ps aux 2>/dev/null | true
  sleep 300
done
echo "V5A_TRAIN_DONE $(date -u +%H:%M)"
cd /data/CausalCache
export PYTHONPATH=/data/CausalCache/code
CKPTS=""
for s in 50 100 150 200 250 300; do CKPTS="$CKPTS --checkpoint lora-step$s=$CK/lora-step$s.pt"; done
mkdir -p $CK/devscore-v4b1 $CK/devscore-v5hit
CUDA_VISIBLE_DEVICES=0 nohup python3 code/scripts/score_sparse_history_arms.py \
  --repository-root . \
  --config code/configs/causalcache_desktop_did_hgkv_v5_selected.json \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --dataset-root /data/desktop-did-corpus-v4/samples-b1.jsonl \
  --image-root /data/desktop-did-corpus-v4 \
  $CKPTS --score-cache $CK/devscore-v4b1/cache.jsonl \
  --heartbeat $CK/devscore-v4b1/heartbeat.json \
  --output $CK/devscore-v4b1/gate_report.json \
  --device cuda:0 > $CK/devscore-v4b1/score.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 code/scripts/score_sparse_history_arms.py \
  --repository-root . \
  --config code/configs/causalcache_desktop_did_hgkv_v5_selected.json \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --dataset-root /data/desktop-did-corpus-v5-hit \
  $CKPTS --score-cache $CK/devscore-v5hit/cache.jsonl \
  --heartbeat $CK/devscore-v5hit/heartbeat.json \
  --output $CK/devscore-v5hit/gate_report.json \
  --device cuda:0 > $CK/devscore-v5hit/score.log 2>&1 &
wait
echo "V5A_DEVSCORE_BOTH_DONE"
