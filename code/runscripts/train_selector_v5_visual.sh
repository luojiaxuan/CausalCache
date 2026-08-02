#!/bin/bash
# note (luojiaxuan): 方案 3 三臂——回答"免 pass-2 后,缓存的帧 embedding 能不能
# 顶上 proposal witness 丢掉的那部分信息"。
#
# 三臂的差别只在"witness 从哪来"与"有没有视觉交互块":
#   p_cheap   真值目标 witness + 28 维 cheap 塔。= 现在的 CausalCache-P 部署塔,
#             但它需要 pass-2 的 proposal 才能在线算 witness。**上界参照**。
#   la_cheap  last_action 伪目标 witness + 同一个 cheap 塔。= CausalCache-LA,
#             单遍可算。**方案 3 要超过的基线**。
#   la_visual la_cheap + 帧 embedding 交互块。**方案 3 本体**。
#
# 关键对比:la_visual − la_cheap = 视觉交互带来的净增益;
#           la_visual − p_cheap  = 还差 pass-2 多远(≥0 则 pass-2 可以去掉)。
#
# note (luojiaxuan): **每臂必须用各自的 feature-cache**。--feature-cache 只按
# (edge_count, dim) 判命中,而 p/la 两臂的 edge_count 与 dim 完全相同、
# witness 四维的取值不同 —— 共用一个 cache 会静默复用错误特征,
# 且两臂结果会长得可疑地一致。这类"缓存键不含关键变量"的坑与 2026-08-01
# 的语料混杂同源:变量换了,标识没换。
set -u
ARM=$1
GPU=${2:-0}
# note (luojiaxuan): h01 上有四份 CausalCache 副本,修订各不相同
# (/data/CausalCache-mwhgkv 644 行、/data/CausalCache 627 行、
#  /bigdata/osworld/CausalCache 716 行)。selector 改动基于 716 行那份,
# 往别的副本上覆盖会静默跑到旧代码——2026-08-01 覆盖训练脚本已经踩过一次。
REPO=${REPO:-/bigdata/osworld/CausalCache}
CORPUS=${CORPUS:-/bigdata/mw/runs/selector-v4-frozen}
MANIFEST=${MANIFEST:-/data/CausalCache-v4/data/manifests/agentnet_screening_manifest_ubuntu_v1.jsonl}
EMB=${EMB:-/bigdata/mw/runs/selector-v5-visual/frame-emb}
OUT_ROOT=${OUT_ROOT:-/bigdata/mw/runs/selector-v5-visual}

cd "$REPO"
export PYTHONPATH="$REPO/code"
OUT=$OUT_ROOT/$ARM
mkdir -p "$OUT"

EXTRA=""
case "$ARM" in
  p_cheap)   unset CAUSALCACHE_WITNESS_PSEUDO_TARGET 2>/dev/null || true
             EXTRA="--arch two_tower" ;;
  la_cheap)  export CAUSALCACHE_WITNESS_PSEUDO_TARGET=last_action
             EXTRA="--arch two_tower" ;;
  la_visual) export CAUSALCACHE_WITNESS_PSEUDO_TARGET=last_action
             EXTRA="--arch visual --embedding-root $EMB" ;;
  *) echo "未知臂 $ARM"; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES=$GPU python3 -m scripts.train_selector_v4_marginal \
  --singletons-root "$CORPUS/singletons" \
  --sets-root "$CORPUS/sets" \
  --screening-manifest "$MANIFEST" \
  --epochs 40 --eval-every 5 --early-stop-patience 3 \
  --feature-cache "$OUT/features.pt" \
  --feature-workers 64 \
  $EXTRA --device cuda:0 \
  --output-root "$OUT" > "$OUT/train.log" 2>&1
CODE=$?
[ $CODE -eq 0 ] && touch "$OUT/EXIT_0" || touch "$OUT/EXIT_$CODE"
echo "V5VISUAL_DONE $ARM code=$CODE"
