#!/usr/bin/env bash
# note (luojiaxuan): 跑一个 MemGUI-Bench 评测臂(128 题 Pass@1,支持 resume)。
# 用法: memgui_arm.sh <arm_name> <CC_FRAME_POLICY> <CC_HISTORY_N> <n_backends> [extra mg eval args]
# 环境变量: MG_HOSTS 覆盖后端列表;MG_TASKFILE 只跑指定任务 CSV;MG_PREFIX 容器名前缀(runner 据此恢复不健康后端)。
# executor 走本机 vLLM(gui-owl);judge 由 .env 配置(gemini-2.5-flash/pro,论文原配)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
ARM=$1; POL=$2; HN=$3; NB=$4; shift 4
cd /data01/jaxan/memgui
HOSTS=${MG_HOSTS:-$(seq -s, -f "http://127.0.0.1:69%02g" 0 $((NB-1)))}
mkdir -p /data01/jaxan/rl_v2/memgui/$ARM
if [ -n "${MG_TASKFILE:-}" ]; then TASK_ARGS=(--task-file "$MG_TASKFILE"); else TASK_ARGS=(--task ALL); fi
# note (luojiaxuan): MG_DISCOVER=1 时不传 --aw-host,由 runner 按容器名前缀自动发现后端并记住容器名,
# 这样不健康后端才能被它重启/重建;显式 --aw-host 会让 runner 丢失容器名而无法自愈。
if [ "${MG_DISCOVER:-0}" = "1" ]; then HOST_ARGS=(); else HOST_ARGS=(--aw-host "$HOSTS"); fi
CC_FRAME_POLICY=$POL CC_HISTORY_N=$HN CC_SELECTOR_URL=${CC_SELECTOR_URL:-http://172.17.0.1:41010} CC_EP_TAG=$ARM PYTHONPATH=/data01/jaxan/pyshim \
  uv run mg eval --agent-type gui_owl_1_5 --model-name gui-owl \
  --llm-base-url http://172.17.0.1:41041/v1 --api-key EMPTY \
  "${TASK_ARGS[@]}" --max-round 50 "${HOST_ARGS[@]}" --max-concurrency $NB \
  --env-name-prefix "${MG_PREFIX:-sglang-omni-jaxan-mg}" \
  --log-file-root /data01/jaxan/rl_v2/memgui/$ARM "$@"
echo "ARM_DONE $ARM rc=$?"
