#!/bin/bash
# note (luojiaxuan): v6 三臂(tightcap / midcap / nogain)的 held-out 门控评测。
# 口径必须与 v4 一致才能直接比:同一 dev 集、同一评分脚本、同一 checkpoint 网格。
# dev 集 = **desktop-did-corpus-v4 的 samples-b1.jsonl**(与 devscore_g36*.sh 一致),
# schema 为 causalcache.desktop_did_sample.v2,与 config 声明相符。
# 不要传 corpus-v1:那是 schema v1,评分脚本会直接拒绝——这条校验很有用,
# 2026-08-01 正是它抓出了"训练误用 corpus-v1"的语料混杂,别去放宽它。
#
# 要看的量:
#   adapter_on_recent(A_r) —— 均匀成分。v4 是 +0.0088,收紧 cap 后应显著变小。
#   did_select (A_s - A_r) —— 选择性。v4 是 +0.0224,**这个不能一起塌掉**,
#                              否则说明 cap 收过头,把有用的信号也压没了。
# 用法:devscore_v6.sh <variant> <gpu> <base_dir> <repo>
#   可覆盖:MODEL_DIR / DEV_SAMPLES / IMAGE_ROOT(默认为 h00 布局)
set -u
VARIANT=$1; GPU=$2; BASE=$3; REPO=$4
MODEL_DIR=${MODEL_DIR:-/data/artifacts/models/GUI-Owl-1.5-8B-Instruct}
DEV_SAMPLES=${DEV_SAMPLES:-/data/desktop-did-corpus-v4/samples-b1.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-/data/desktop-did-corpus-v4}
# note (luojiaxuan): CONFIG 可覆盖,因为 **v4 基线也必须用同一协议打一遍**。
# 论文里那两个参照数(A_r +0.0088、did_select +0.0224)不是本脚本产出的,
# /data/runs/desktop-did-v4/hgkv 下根本没有 gate_report.json ——
# 拿它们和 v6 比就是跨协议比较,与 2026-08-01 的语料混杂是同一类错误。
# 打分只用到 config 的适配器结构(rank/层/模块),eps 与 gain 权重是损失项,
# 不影响前向,所以各臂用各自 config 打分不会引入差异。
CONFIG=${CONFIG:-code/configs/causalcache_desktop_did_hgkv_v6_${VARIANT}.json}
CK=$BASE/$VARIANT
cd "$REPO"
export PYTHONPATH="$REPO/code"
CKPTS=""
for s in 50 100 150 200 250 300; do
  [ -f "$CK/lora-step$s.pt" ] && CKPTS="$CKPTS --checkpoint lora-step$s=$CK/lora-step$s.pt"
done
for e in 1 2 3; do
  [ -f "$CK/lora-epoch$e.pt" ] && CKPTS="$CKPTS --checkpoint lora-epoch$e=$CK/lora-epoch$e.pt"
done
[ -z "$CKPTS" ] && { echo "FATAL 没有 checkpoint: $CK"; exit 1; }
echo "评测 $VARIANT,checkpoints:$CKPTS"
mkdir -p "$CK/devscore"
CUDA_VISIBLE_DEVICES=$GPU python3 code/scripts/score_sparse_history_arms.py \
  --repository-root . \
  --config "$CONFIG" \
  --model-dir "$MODEL_DIR" \
  --dataset-root "$DEV_SAMPLES" \
  --image-root "$IMAGE_ROOT" \
  $CKPTS --score-cache "$CK/devscore/cache.jsonl" \
  --heartbeat "$CK/devscore/heartbeat.json" \
  --output "$CK/devscore/gate_report.json" \
  --device cuda:0 > "$CK/devscore/score.log" 2>&1
echo "V6_DEVSCORE_DONE $VARIANT $(date -u +%H:%M)"
