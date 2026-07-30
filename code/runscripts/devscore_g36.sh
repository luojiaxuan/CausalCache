#!/bin/bash
# note (luojiaxuan): full-layer LoRA(q/k/v)训完即打分:两套 dev 口径与 v5a 完全对齐,
# 便于"全层 LoRA vs HGKV last-8"等预算对比。
set -u
CK=/data/runs/desktop-did-v5/gated36-matched
while [ ! -f $CK/EXIT_0 ]; do
  n=$(ps aux | grep -c "[t]rain_success_sft_lora")
  [ "$n" -eq 0 ] && [ ! -f $CK/EXIT_0 ] && { echo "TRAIN_DIED_NO_GATE $(date -u +%H:%M)"; exit 1; }
  sleep 300
done
echo "G36_TRAIN_DONE $(date -u +%H:%M)"
cd /data/CausalCache
export PYTHONPATH=/data/CausalCache/code
CKPTS=""
for s in 50 100 150 200 250 300; do
  [ -f $CK/lora-step$s.pt ] && CKPTS="$CKPTS --checkpoint lora-step$s=$CK/lora-step$s.pt"
done
echo "CKPTS:$CKPTS"
mkdir -p $CK/devscore-v4b1 $CK/devscore-v5hit
CUDA_VISIBLE_DEVICES=0 nohup python3 code/scripts/score_sparse_history_arms.py \
  --repository-root . \
  --config code/configs/causalcache_desktop_did_hgkv_v5_selected_full.json \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --dataset-root /data/desktop-did-corpus-v4/samples-b1.jsonl \
  --image-root /data/desktop-did-corpus-v4 \
  $CKPTS --score-cache $CK/devscore-v4b1/cache.jsonl \
  --heartbeat $CK/devscore-v4b1/heartbeat.json \
  --output $CK/devscore-v4b1/gate_report.json \
  --device cuda:0 > $CK/devscore-v4b1/score.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 code/scripts/score_sparse_history_arms.py \
  --repository-root . \
  --config code/configs/causalcache_desktop_did_hgkv_v5_selected_full.json \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --dataset-root /data/desktop-did-corpus-v5-hit \
  $CKPTS --score-cache $CK/devscore-v5hit/cache.jsonl \
  --heartbeat $CK/devscore-v5hit/heartbeat.json \
  --output $CK/devscore-v5hit/gate_report.json \
  --device cuda:0 > $CK/devscore-v5hit/score.log 2>&1 &
wait
echo "G36_DEVSCORE_BOTH_DONE $(date -u +%H:%M)"
