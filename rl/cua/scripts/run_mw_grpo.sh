#!/bin/bash
# note (luojiaxuan): MobileWorld joint GRPO 发射脚本(草案,首发前逐段核验)。
# 架构:host 侧 env-server + selector service/trainer 侧车;slime 容器承载
# executor GRPO(run_grpo.sh,ASYNC=1:1 rollout GPU + 2 train GPU)。
# 依赖:cua-lite checkout($CUA)、cc_recipe($REC)、镜像 slimerl/slime:v0.3.0
# 已重打为白名单 tag(sglang-omni:dev,不覆盖既有)、mobileworld 镜像已 build。
# 共享机纪律:动态选空卡(排除 GPU6 探针)、容器名 sglang-omni-jaxan-<N>、
# pyshim 压进程名、map 登记、launcher 尾部 teardown 或显式 KEEP。
set -euo pipefail

CUA=${CUA:-/data01/jaxan/cua/cua-lite}
REC=${REC:-/data01/jaxan/cua/cc_recipe}
RUN=${RUN:-/data01/jaxan/cua/run_$(date +%m%d_%H%M)}
ARM=${ARM:-learned}            # learned | recent(对照测量批)
STEPS=${STEPS:-3}              # smoke 3;mini-run 30-50
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
mkdir -p "$RUN"/{decisions,returns,ckpt}

# ── [0] 动态选 3 张空卡(memory.used<2GB,排除 GPU6 探针;竞态教训:
#        选卡与发射同一时刻做,不留间隙)──
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F, '$1 != 6 && $2 < 2000 {print $1}' | head -3)
if [ "${#FREE[@]}" -lt 3 ]; then
  echo "ABORT: 空卡不足 3 张(现 ${FREE[*]:-none})" >&2; exit 1
fi
G_ROLLOUT=${FREE[0]}; G_TRAIN="${FREE[1]},${FREE[2]}"
echo "GPU: rollout=$G_ROLLOUT train=$G_TRAIN"

# ── [1] 任务 parquet(冻结切分)──
if [ ! -f "$RUN/train.parquet" ]; then
  (cd "$CUA" && PYTHONPATH="$REC" uv run --no-sync python \
    "$REC/scripts/export_mw_tasks.py" \
    --split-json "$REC/fixtures/mw_split_v1.json" --out-dir "$RUN")
fi

# ── [2] env-server(host 侧;mobileworld 直连本机 docker)──
export CUA_LITE_ENV_SERVER_TOKEN=${CUA_LITE_ENV_SERVER_TOKEN:-cc-$(hostname)-token}
if ! curl -s -m 3 http://127.0.0.1:30100/health > /dev/null 2>&1; then
  (cd "$CUA" && nohup uv run --no-sync python scripts/serve_env.py \
     > "$RUN/env_server.log" 2>&1 & echo $! > "$RUN/env_server.pid")
  for i in $(seq 1 30); do
    curl -s -m 3 http://127.0.0.1:30100/health > /dev/null 2>&1 && break; sleep 2
  done
fi
export CUA_LITE_ENV_SERVER_URL=http://172.17.0.1:30100   # slime 容器视角

# ── [3] selector 侧车(仅 learned 臂;CPU 即可,编码器 64x64)──
if [ "$ARM" = learned ]; then
  CC_SEL_DEVICE=cpu CC_SEL_LOG="$RUN/decisions" CC_SEL_PORT=41010 \
    nohup env PYTHONPATH="$REC" python3 -m causalcache_cua.selector.service \
    > "$RUN/selector_service.log" 2>&1 & echo $! > "$RUN/selector_service.pid"
  CC_SEL_TRAIN_DEVICE=cpu nohup env PYTHONPATH="$REC" python3 -m \
    causalcache_cua.selector.trainer \
    --decisions-dir "$RUN/decisions" \
    --returns-file "$RUN/returns/episode_returns.jsonl" \
    --ckpt-dir "$RUN/ckpt" --interval 120 \
    > "$RUN/selector_trainer.log" 2>&1 & echo $! > "$RUN/selector_trainer.pid"
fi

# ── [4] slime 容器 + run_grpo.sh(容器内)──
# 注意:scripts/train/slime/launch.sh 的镜像与容器名需按共享机纪律替换;
# 首发前人工核验其 docker run 参数(挂载 $CUA、$REC、/data04/jaxan models、
# --init、pyshim PYTHONPATH、高位端口)。
cat << EOT
[发射清单——容器内执行]
CUDA_VISIBLE_DEVICES=$G_ROLLOUT,$G_TRAIN 进容器后:
  ASYNC=1 NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=1 \\
  MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \\
  HF_CKPT=/data/models/GUI-Owl-1.5-8B-Instruct \\
  ENV_ID=mobileworld \\
  PROMPT_DATA=$RUN/train.parquet \\
  ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=8 NUM_STEPS_PER_ROLLOUT=1 \\
  NUM_ROLLOUT=$STEPS ENV_CONCURRENCY=16 \\
  ROLLOUT_MODULE=causalcache_cua.rollout_grpo \\
  CONFIG_PATH=$REC/configs/gui_owl/mobileworld_${ARM}.yaml \\
  CC_RET_DIR=$RUN/returns CC_SELECTOR_URL=http://172.17.0.1:41010 \\
  CUA_LITE_ENV_SERVER_URL=$CUA_LITE_ENV_SERVER_URL \\
  CUA_LITE_ENV_SERVER_TOKEN=$CUA_LITE_ENV_SERVER_TOKEN \\
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
EOT
echo "RUN_DIR=$RUN"
# teardown 责任:smoke 收尾删 slime 容器 + kill 三个 pid 文件进程 +
# env-server sessions 清理(docs/slime.md cleanup);KEEP 需显式说明。
