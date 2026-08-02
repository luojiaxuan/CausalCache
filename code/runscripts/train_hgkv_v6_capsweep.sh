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
# 三档:nogain 再把 sparse_gain_weight 归零。
# 三个 config 除这些项外与 v4 逐字节相同,效果可干净归因。
#
# note (luojiaxuan): **语料必须与 v4 基线同源,否则整个对照作废。**
# 2026-08-01 第一轮三臂全部误用 desktop-did-corpus-v1(manifest sha e872828e),
# 而 v4 基线用的是 desktop-did-corpus-v4(sha 167009bc)——config 只差 eps,
# 语料却换了,单变量归因不成立,三臂全部重跑。判据:训完立刻比对
# run_manifest.json 的 dataset_manifest_sha256 与基线是否一致。
# 副作用:corpus-v1 是 schema v1,而 config 声明 v2,训练器宽容放过、
# 评分脚本拒绝——**是评分脚本的 schema 校验把这个错抓出来的**,别去放宽它。
#
# 用法:HOSTVARS... train_hgkv_v6_capsweep.sh <variant> <gpu_csv> <rdzv_port>
#   variant ∈ {tightcap, midcap, nogain}
#   环境变量(默认为 h00 布局):REPO / CORPUS / MODEL_DIR / OUT_ROOT / WARM_CACHE
# note (luojiaxuan): rdzv_port 必须逐 job 唯一。`torchrun --standalone` 用固定端口
# 29400,同机并发两个 job 会让进程组串台,表现为 600s all-reduce 超时后崩溃
#(2026-08-01 实际踩到:tightcap 与 midcap 同时发,双双 timeout)。
set -u
VARIANT=$1
GPUS=$2
PORT=${3:-29400}
REPO=${REPO:-/data/CausalCache}
CORPUS=${CORPUS:-/data/desktop-did-corpus-v4}
MODEL_DIR=${MODEL_DIR:-/data/artifacts/models/GUI-Owl-1.5-8B-Instruct}
OUT_ROOT=${OUT_ROOT:-/data/runs/desktop-did-v6}
WARM_CACHE=${WARM_CACHE:-}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
cd "$REPO"
export PYTHONPATH="$REPO/code"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPUS
OUT=$OUT_ROOT/$VARIANT
mkdir -p "$OUT/frozen-cache"
# 冻结分缓存条目自带 fingerprint,指纹不符会被忽略并重算,warm-start 是安全的。
if [ -n "$WARM_CACHE" ] && [ -d "$WARM_CACHE" ]; then
  cp -n "$WARM_CACHE"/*.jsonl "$OUT/frozen-cache/" 2>/dev/null || true
  echo "warm-start frozen-cache from $WARM_CACHE: $(ls "$OUT/frozen-cache" | wc -l) 个分片" >> "$OUT/train.log"
fi
echo "=== corpus=$CORPUS repo=$REPO ===" >> "$OUT/train.log"
for attempt in 1 2 3; do
  echo "=== attempt $attempt $(date -u +%FT%TZ) gpus=$GPUS ===" >> "$OUT/train.log"
  torchrun --rdzv-backend=c10d --rdzv-endpoint="localhost:$PORT" --nproc_per_node="$NPROC" code/scripts/train_success_sft_lora.py \
    --config "code/configs/causalcache_desktop_did_hgkv_v6_${VARIANT}.json" \
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
