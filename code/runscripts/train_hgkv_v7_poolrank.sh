#!/bin/bash
# note (luojiaxuan): did_pool_rank 训练(2026-08-02)。相对 v4 的两处改动见
# code/configs/causalcache_desktop_did_hgkv_v7_poolrank.json 的 objective 段:
# 去掉 L_gain(均匀抬升的唯一激励来源)+ 候选池上的排序。
#
# 语料是 desktop-did-corpus-v7pool(1,645 组 × 11 臂 = 18,095 条,schema v3),
# **与 v4 基线不是同一份**——因为满池要求淘汰了 25% 的组。所以这次的对照不是
# "同语料换目标",而是"新语料 + 新目标 vs 旧语料 + 旧目标"。要干净归因,
# 还需要在同一份 v7pool 语料上跑一个 did_ra_aware 臂(候选行对旧目标惰性,
# 可以直接跑)。**不要跳过那个对照**,否则又会是 2026-08-01 的语料混杂。
#
# 每组前向 15 次(旧目标 9 次),约 1.7×,300 步预计 1.5–2 小时。
set -u
VARIANT=${1:-poolrank}
GPUS=${2:-4,5,6,7}
PORT=${3:-29601}
CONFIG=${CONFIG:-code/configs/causalcache_desktop_did_hgkv_v7_poolrank.json}
REPO=${REPO:-/data/CausalCache}
CORPUS=${CORPUS:-/data/desktop-did-corpus-v7pool}
MODEL_DIR=${MODEL_DIR:-/data/artifacts/models/GUI-Owl-1.5-8B-Instruct}
OUT_ROOT=${OUT_ROOT:-/data/runs/desktop-did-v7}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
cd "$REPO"
export PYTHONPATH="$REPO/code"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPUS
OUT=$OUT_ROOT/$VARIANT
mkdir -p "$OUT"
echo "=== corpus=$CORPUS config=$CONFIG gpus=$GPUS ===" >> "$OUT/train.log"
for attempt in 1 2 3; do
  echo "=== attempt $attempt $(date -u +%FT%TZ) ===" >> "$OUT/train.log"
  torchrun --rdzv-backend=c10d --rdzv-endpoint="localhost:$PORT" \
    --nproc_per_node="$NPROC" code/scripts/train_success_sft_lora.py \
    --config "$CONFIG" \
    --dataset-root "$CORPUS" \
    --model-dir "$MODEL_DIR" \
    --output-root "$OUT" \
    --frozen-score-cache "$OUT/frozen-cache" \
    --encode-cache-scope group \
    --repository-root . >> "$OUT/train.log" 2>&1
  CODE=$?
  if [ $CODE -eq 0 ]; then touch "$OUT/EXIT_0"; echo "done 0"; exit 0; fi
  echo "attempt $attempt failed code $CODE" >> "$OUT/train.log"
  touch "$OUT/EXIT_$CODE"
  sleep 30
done
exit 1
