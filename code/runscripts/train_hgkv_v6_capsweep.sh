#!/bin/bash
# note (luojiaxuan): 漏洞 1 的修复训练——收紧 drift cap,封死"均匀抬升"捷径。
#
# 诊断依据(data/results/hgkv_k_stratified_v1):v4 适配器的效应里
# **77% 与内容无关**(k=0 即 Recent-|S| 上仍有 +0.0092),对平均远端帧的选择性
# 只有 +0.0027。原因是 eps=0.02 是 gain margin m=0.01 的两倍,
# "贴着 cap 加均匀抬升 + 一点点选择性"能同时满足 L_gain 与 L_cap,是最省力解。
#
# 收紧后均匀路线最多拿 eps,要满足 L_gain(A_c ≥ 0.01)必须从选择性挣够 (0.01 - eps)。
# 两档剂量:tightcap eps=0.002(m/5)、midcap eps=0.005(m/2)。
# 两个 config 除 eps 外与 v4 逐字节相同,效果可干净归因。
#
# 用法:train_hgkv_v6_capsweep.sh <variant> <gpu_csv> <rdzv_port>
#   variant ∈ {tightcap, midcap, nogain}
# note (luojiaxuan): rdzv_port 必须逐 job 唯一。`torchrun --standalone` 用固定端口
# 29400,同机并发两个 job 会让进程组串台,表现为 600s all-reduce 超时后崩溃
#(2026-08-01 实际踩到:tightcap 与 midcap 同时发,双双 timeout)。
set -u
VARIANT=$1
GPUS=$2
PORT=${3:-29400}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
cd /data/CausalCache
export PYTHONPATH=/data/CausalCache/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPUS
OUT=/data/runs/desktop-did-v6/$VARIANT
mkdir -p "$OUT"
for attempt in 1 2 3; do
  echo "=== attempt $attempt $(date -u +%FT%TZ) gpus=$GPUS ===" >> "$OUT/train.log"
  torchrun --rdzv-backend=c10d --rdzv-endpoint="localhost:$PORT" --nproc_per_node="$NPROC" code/scripts/train_success_sft_lora.py \
    --config "code/configs/causalcache_desktop_did_hgkv_v6_${VARIANT}.json" \
    --dataset-root /data/desktop-did-corpus-v1 \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
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
