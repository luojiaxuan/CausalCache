#!/usr/bin/env bash
# note (luojiaxuan): 跑一个 MemGUI-Bench 评测臂(128 题 Pass@1,支持 resume)。
# 用法: memgui_arm.sh <arm_name> <CC_FRAME_POLICY> <CC_HISTORY_N> <n_backends> [extra mg eval args]
# executor 走本机 vLLM(gui-owl);judge 由 .env 配置(gemini-2.5-flash/pro,论文原配)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
ARM=$1; POL=$2; HN=$3; NB=$4; shift 4
cd /data01/jaxan/memgui
HOSTS=$(seq -s, -f "http://127.0.0.1:69%02g" 0 $((NB-1)))
mkdir -p /data01/jaxan/rl_v2/memgui/$ARM
CC_FRAME_POLICY=$POL CC_HISTORY_N=$HN CC_SELECTOR_URL=${CC_SELECTOR_URL:-http://172.17.0.1:41010} CC_EP_TAG=$ARM PYTHONPATH=/data01/jaxan/pyshim \
  uv run mg eval --agent-type gui_owl_1_5 --model-name gui-owl \
  --llm-base-url http://172.17.0.1:41041/v1 --api-key EMPTY \
  --task ALL --max-round 50 --aw-host "$HOSTS" --max-concurrency $NB \
  --log-file-root /data01/jaxan/rl_v2/memgui/$ARM "$@"
echo "ARM_DONE $ARM rc=$?"
