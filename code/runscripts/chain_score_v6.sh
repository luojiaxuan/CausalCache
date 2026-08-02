#!/bin/bash
# note (luojiaxuan): v6 训练→评分的串接。训练一臂约 60 分钟、评分约 30 分钟,
# 人工盯着接会白白空转,所以用轮询 EXIT_0 的方式自动接上。
#
# 判据用 **EXIT_0 文件存在**,不是"进程消失"——训练脚本自带三次重试,
# 进程消失可能只是 attempt 之间的间隙(tightcap 的 attempt 1/2 就都因
# NCCL init 阶段的 CUDA OOM 挂过,第 3 次才成功)。用进程判据会误触发。
# 反过来,EXIT_1 存在也不代表失败:同一目录下 EXIT_1 与后来的 EXIT_0 会并存。
#
# 用法:HOST=h00|h01 chain_score_v6.sh
#   h00: 等 tightcap → 在 GPU0 评分
#   h01: 等 midcap → GPU7 评分;再等 nogain → GPU7 评分(训练占 0,1,4,5)
set -u
HOST=${HOST:-h00}

wait_exit0() {  # $1=目录  $2=最多等多少个 30 秒
  local d=$1 n=$2 i
  for i in $(seq 1 "$n"); do [ -f "$d/EXIT_0" ] && return 0; sleep 30; done
  return 1
}

if [ "$HOST" = "h00" ]; then
  R=/data/runs/desktop-did-v6
  if wait_exit0 "$R/tightcap" 240; then
    bash /data/dv6.sh tightcap 0 "$R" /data/CausalCache
  else
    echo "TIGHTCAP_NEVER_FINISHED"; exit 1
  fi
else
  export MODEL_DIR=/bigdata/models/GUI-Owl-1.5-8B-Instruct
  export DEV_SAMPLES=/bigdata/mw/desktop-did-corpus-v4/samples-b1.jsonl
  export IMAGE_ROOT=/bigdata/mw/desktop-did-corpus-v4
  R=/bigdata/mw/runs/desktop-did-v6
  for arm in midcap nogain; do
    if wait_exit0 "$R/$arm" 480; then
      bash /bigdata/dv6.sh "$arm" 7 "$R" /bigdata/osworld/CausalCache
    else
      echo "ARM_NEVER_FINISHED $arm"
    fi
  done
  touch "$R/SCORE_CHAIN_DONE"
fi
