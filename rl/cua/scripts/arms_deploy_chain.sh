#!/usr/bin/env bash
# note (luojiaxuan): 在**部署布局**(交错多轮)下重测三臂 + 底座,用 harm 线的 decode_ctx.py。规格把"附哪几帧"与"哪几轮保留原文"解耦:
#   rec0_deploy / rec2_deploy / irr2_deploy         不给图 / 最近两帧 / 无关两帧(结构=最近两轮原文)
#   pickimg:<judge>   结构固定为最近两轮原文,只把两张图换成裁判帧(纯内容效应)
#   hybrid:<judge>    最近两轮原样,裁判帧作带 PAST 标记的额外图放进首条 user(不拆结构地附老帧)
# 等平铺版 no-text 诊断跑完释放 GPU 再起,避免本账号在 hyper00 超 4 卡。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
R=/data01/jaxan/rl_v2; C=""
until grep -q OWL_NOTEXT_DONE /data01/jaxan/owl_notext_judge_chain.log 2>/dev/null; do sleep 60; done
SPECS="rec0_deploy,rec2_deploy,irr2_deploy"
for j in "judge|direct|full" "judge_glm46v|direct|full" "qwen38_27b|direct|full"; do SPECS="$SPECS,pickimg:$j,hybrid:$j"; done
PICKS="$R/picks_*direct_full.jsonl"
for tag in recency_s0 random_s0 older_s0 base; do
  M=/data01/jaxan/models/$tag; [ "$tag" = base ] && M=/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct
  G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
  PORT=$(pick_port 41261 41271 41281 41291)
  C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
    -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
    -p 172.17.0.1:$PORT:8000 vllm/vllm-omni:dev --model $M --served-model-name gui-owl --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; continue; }
  reg "$C" "$G" "sglang-omni-rl 部署布局重测($tag:rec/irr/pickimg/hybrid × 三裁判);⚠ 在用勿删;收尾:本臂结束删"
  E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
  curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL $tag"; rm_own "$C"; continue; }
  echo "$(date -u +%FT%TZ) START $tag ($C GPU$G:$PORT)"
  python3 /data01/jaxan/decode_ctx.py --base-url $E --tag ${tag}_deploy --specs "$SPECS" --picks "$PICKS" \
    --out $R/deploy2_${tag}.jsonl --workers 12 > $R/deploy2_${tag}.log 2>&1
  echo "$(date -u +%FT%TZ) DONE $tag: $(wc -l < $R/deploy2_${tag}.jsonl) specs=$(python3 -c "import json;print(list(json.loads(open(\"$R/deploy2_${tag}.jsonl\").readline())[\"decodes\"]))")"
  rm_own "$C"
done
echo ARMS_DEPLOY_DONE
