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
echo "$RUN/returns" > "$REC/CC_RET_DIR.path"   # Ray worker 经挂载读(shim 回退)
if [ "$ARM" = learned ]; then
  CC_SEL_DEVICE=cpu CC_SEL_LOG="$RUN/decisions" CC_SEL_PORT=41010 \
  CC_SEL_BIND=172.17.0.1 \
    nohup env PYTHONPATH="$REC" python3 -m causalcache_cua.selector.service \
    > "$RUN/selector_service.log" 2>&1 & echo $! > "$RUN/selector_service.pid"
  CC_SEL_TRAIN_DEVICE=cpu nohup env PYTHONPATH="$REC" python3 -m \
    causalcache_cua.selector.trainer \
    --decisions-dir "$RUN/decisions" \
    --returns-file "$RUN/returns/episode_returns.jsonl" \
    --ckpt-dir "$RUN/ckpt" --interval 120 \
    > "$RUN/selector_trainer.log" 2>&1 & echo $! > "$RUN/selector_trainer.pid"
fi

# ── [4] slime 容器(规范名复刻 launch.sh 的 docker run;镜像已重打
#        jaxanluo/sglang-omni:trainer)──
# recipe 进 PYTHONPATH 的机关:run_grpo.sh 给 Ray worker 硬编码
# PYTHONPATH=<Megatron>:<CUA_LITE_ROOT>:<slime>;故把 causalcache_cua 以
# 相对 symlink 放进 cua-lite 根(host 与容器两侧路径同构,均可解析),
# 上游零改动。pyshim 同理:sitecustomize.py 拷进 cua-lite 根即搭车。
[ -e "$CUA/causalcache_cua" ] || ln -s ../cc_recipe/causalcache_cua "$CUA/causalcache_cua"
[ -e "$CUA/sitecustomize.py" ] || cp /data01/jaxan/pyshim/sitecustomize.py "$CUA/sitecustomize.py"

CTN=""
for i in $(seq 1 99); do
  docker container inspect "sglang-omni-jaxan-$i" &>/dev/null || { CTN="sglang-omni-jaxan-$i"; break; }
done
SID="cc-mw-$(date +%m%d)"
# --ulimit nofile 必给:docker 默认 soft 1024,Ray 按 224 核自动配置
# (prestart 224 worker)把 raylet 的 FD 顶满 → NM 端口 accept 不动,
# 所有 worker 首连 "SETTINGS frame 超时"(smoke1 两连击的根因)。
docker run -d --gpus "\"device=$G_ROLLOUT,$G_TRAIN\"" --name "$CTN" --init \
  --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=524288:524288 \
  --memory=400g \
  -e CUA_LITE_ROOT=/workspaces/cua-lite \
  -e CUA_LITE_DATASETS_ROOT=/workspaces/cua-lite/.data/huggingface \
  -e CUA_LITE_ENV_SERVER_URL -e CUA_LITE_ENV_SERVER_TOKEN \
  -e SESSION_ID="$SID" \
  -v /data01/jaxan/cua/cua-lite:/workspaces/cua-lite \
  -v /data01/jaxan/cua/cc_recipe:/workspaces/cc_recipe \
  -v /data04/jaxan/models:/data/models:ro \
  -v "$RUN":"$RUN" \
  jaxanluo/sglang-omni:trainer sleep infinity
docker exec "$CTN" bash /workspaces/cua-lite/scripts/train/slime/init.sh \
  > "$RUN/slime_init.log" 2>&1
echo "$CTN gpus=$G_ROLLOUT,$G_TRAIN host=$(hostname) created=$(date -u +%FT%TZ) desc=sglang-omni-rl trainer;收尾:smoke 判读后删" >> "$HOME/jiaxuanluo-map.txt"

# ── [5] 容器内发射 run_grpo.sh ──
# 实战修正(smoke1 踩坑):
#   - 跨界文件(parquet/returns)一律走已挂载的 $REC 容器可见路径,
#     不能用宿主 $RUN 直写(除非容器带 -v $RUN:$RUN);
#   - CC_RET_DIR.path 哨兵写**容器视角**路径(worker 在容器内读);
#   - checkout 因 symlink/sitecustomize 必 dirty → 显式
#     CUA_LITE_ALLOW_DIRTY_ENV_SERVER=1(dev run 逃生阀,按其报错指引);
#   - SAVE 目录放挂载内,权重热换产物宿主可见。
mkdir -p "$REC/run_$ARM"/{returns,}
cp "$RUN/train.parquet" "$REC/run_$ARM/"
echo "/workspaces/cc_recipe/run_$ARM/returns" > "$REC/CC_RET_DIR.path"
docker exec \
  -e ASYNC=1 -e NUM_TRAIN_GPUS=2 -e NUM_ROLLOUT_GPUS=1 \
  -e MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \
  -e HF_CKPT=/data/models/GUI-Owl-1.5-8B-Instruct \
  -e ENV_ID=mobileworld \
  -e PROMPT_DATA="/workspaces/cc_recipe/run_$ARM/train.parquet" \
  -e ROLLOUT_BATCH_SIZE=4 -e N_SAMPLES_PER_PROMPT=8 -e NUM_STEPS_PER_ROLLOUT=1 \
  -e NUM_ROLLOUT="$STEPS" -e ENV_CONCURRENCY=16 \
  -e ROLLOUT_MODULE=causalcache_cua.rollout_grpo \
  -e CONFIG_PATH="/workspaces/cc_recipe/configs/gui_owl/mobileworld_${ARM}.yaml" \
  -e CUA_LITE_ENV_SERVER_URL -e CUA_LITE_ENV_SERVER_TOKEN \
  -e CUA_LITE_ALLOW_DIRTY_ENV_SERVER=1 \
  -e SAVE_DIR="/workspaces/cc_recipe/run_$ARM/ckpt_megatron" \
  -e SAVE_HF_DIR="/workspaces/cc_recipe/run_$ARM/hf/iter_{rollout_id}" \
  "$CTN" bash /workspaces/cua-lite/scripts/train/run_grpo.sh \
  2>&1 | tee "$RUN/grpo.log"

echo "RUN_DIR=$RUN"
# teardown 责任:smoke 收尾删 slime 容器($CTN)+ kill 三个 pid 文件进程
# + env-server sessions 清理(docs/slime.md cleanup)+ map 记录删除;
# KEEP 需显式打印原因。
