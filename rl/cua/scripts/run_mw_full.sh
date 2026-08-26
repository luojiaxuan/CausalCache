#!/bin/bash
# note (luojiaxuan): 全量发射脚本(2026-08-26,外审后)。与 smoke 版
# run_mw_grpo.sh 的差异全部来自 mini v5 实测与外审裁决
# (docs/reviews/fullrun_launch_review_20260826.md):
#   - selector lr 3e-4 + 探针库(门控升档 1e-3 由监控执行,判据在 recipe §6);
#   - 难度优先采样 warm-start(mini task_stats.json 预置进 returns);
#   - 容器内存 1200g(mini 900g 峰值 249G,但 train 峰随混合组数扩;批翻倍留余量)
#     + CUA_LITE_MULTIMODAL_LAZY_EXPAND=1(事故 #11);
#   - ROLLOUT_BATCH_SIZE=8 × G8 = 64 rollout/批,NUM_ROLLOUT=100,SAVE_INTERVAL=3;
#   - ENV_CONCURRENCY=32(48 实测击穿:单 rollout 引擎排队使单步 46s,
#     50 步 episode 40+ 分钟,首批 30 分钟零完成触发 stall 看门狗全批取消;
#     32 是该拓扑的 mini 实测上限)+ ROLLOUT_STALL_TIMEOUT_S=3600;
#   - GPU 现实:真空卡 1,2 + 用户授权混卡的 GPU5 占位(util 0)= 3 卡,
#     mini 同构拓扑(rollout 1 + train 2);任一整卡释放→从 ckpt 重启升 4 卡。
#     若 GPU5 持有者回归致进程死亡:从最新 HF ckpt 重启(mini 已验证恢复链)。
set -euo pipefail

CUA=${CUA:-/data01/jaxan/sglang-omni-rl/cua-lite}
REC=${REC:-/data01/jaxan/sglang-omni-rl/cc_recipe}
RUN=${RUN:-/data01/jaxan/sglang-omni-rl/run_full1}
ARM=learned
STEPS=${STEPS:-100}
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
mkdir -p "$RUN"/{decisions,returns,ckpt}

# ── [0] 选卡:<2GB 视为可用(含用户授权混卡的占位卡),排除 GPU6 ──
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F, '$1 != 6 && $2 < 2000 {print $1}' | head -3)
if [ "${#FREE[@]}" -lt 3 ]; then
  echo "ABORT: 可用卡不足 3 张(现 ${FREE[*]:-none})" >&2; exit 1
fi
G_ROLLOUT=${FREE[0]}; G_TRAIN="${FREE[1]},${FREE[2]}"
echo "GPU: rollout=$G_ROLLOUT train=$G_TRAIN"

# ── [1] 任务 parquet(冻结切分)──
if [ ! -f "$RUN/train.parquet" ]; then
  (cd "$CUA" && PYTHONPATH="$REC" uv run --no-sync python \
    "$REC/scripts/export_mw_tasks.py" \
    --split-json "$REC/fixtures/mw_split_v1.json" --out-dir "$RUN")
fi

# ── [1.5] 探针库(冻结,存在即不重建)+ 难度先验 warm-start ──
if [ ! -f "$RUN/probe_bank.jsonl" ]; then
  python3 "$REC/scripts/make_probe_bank.py" \
    --decisions-dir /data01/jaxan/sglang-omni-rl/run_mini1/decisions \
    --out "$RUN/probe_bank.jsonl" --n 200
fi
mkdir -p "$REC/run_full/returns"
if [ ! -f "$REC/run_full/returns/task_stats.json" ]; then
  cp /data01/jaxan/sglang-omni-rl/cc_recipe/run_mini1/returns/task_stats.json \
     "$REC/run_full/returns/task_stats.json" 2>/dev/null || \
  docker run --rm -v /data01/jaxan/sglang-omni-rl:/d alpine \
     cp /d/cc_recipe/run_mini1/returns/task_stats.json /d/cc_recipe/run_full/returns/task_stats.json
fi

# ── [2] env-server ──
export CUA_LITE_ENV_SERVER_TOKEN=${CUA_LITE_ENV_SERVER_TOKEN:-cc-$(hostname)-token}
if ! curl -s -m 3 http://127.0.0.1:30100/health > /dev/null 2>&1; then
  (cd "$CUA" && nohup uv run --no-sync python scripts/serve_env.py \
     > "$RUN/env_server.log" 2>&1 & echo $! > "$RUN/env_server.pid")
  for i in $(seq 1 30); do
    curl -s -m 3 http://127.0.0.1:30100/health > /dev/null 2>&1 && break; sleep 2
  done
fi
export CUA_LITE_ENV_SERVER_URL=http://172.17.0.1:30100

# ── [3] selector 侧车(lr 3e-4 + 探针库;升档 1e-3 由监控按 §6 三条执行)──
echo "/workspaces/cc_recipe/run_full/returns" > "$REC/CC_RET_DIR.path"
CC_SEL_DEVICE=cpu CC_SEL_LOG="$RUN/decisions" CC_SEL_PORT=41010 \
CC_SEL_BIND=172.17.0.1 \
  nohup env PYTHONPATH="$REC" python3 -m sglang_omni_rl.selector.service \
  > "$RUN/selector_service.log" 2>&1 & echo $! > "$RUN/selector_service.pid"
CC_SEL_TRAIN_DEVICE=cpu nohup env PYTHONPATH="$REC" python3 -m \
  sglang_omni_rl.selector.trainer \
  --decisions-dir "$RUN/decisions" \
  --returns-file "$REC/run_full/returns/episode_returns.jsonl" \
  --ckpt-dir "$RUN/ckpt" --interval 120 --lr 3e-4 \
  --probe-bank "$RUN/probe_bank.jsonl" \
  --service-url http://172.17.0.1:41010 \
  > "$RUN/selector_trainer.log" 2>&1 & echo $! > "$RUN/selector_trainer.pid"

# ── [4] slime 容器 ──
[ -e "$CUA/sglang_omni_rl" ] || ln -s ../cc_recipe/sglang_omni_rl "$CUA/sglang_omni_rl"
[ -e "$CUA/sitecustomize.py" ] || cp /data01/jaxan/pyshim/sitecustomize.py "$CUA/sitecustomize.py"
CTN=""
for i in $(seq 1 99); do
  docker container inspect "sglang-omni-jaxan-$i" &>/dev/null || { CTN="sglang-omni-jaxan-$i"; break; }
done
SID="cc-mw-full-$(date +%m%d)"
docker run -d --gpus "\"device=$G_ROLLOUT,$G_TRAIN\"" --name "$CTN" --init \
  --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=524288:524288 \
  --memory=1200g \
  -e CUA_LITE_ROOT=/workspaces/cua-lite \
  -e CUA_LITE_DATASETS_ROOT=/workspaces/cua-lite/.data/huggingface \
  -e CUA_LITE_ENV_SERVER_URL -e CUA_LITE_ENV_SERVER_TOKEN \
  -e SESSION_ID="$SID" \
  -v /data01/jaxan/sglang-omni-rl/cua-lite:/workspaces/cua-lite \
  -v /data01/jaxan/sglang-omni-rl/cc_recipe:/workspaces/cc_recipe \
  -v /data04/jaxan/models:/data/models:ro \
  -v "$RUN":"$RUN" \
  jaxanluo/sglang-omni:trainer sleep infinity
docker exec "$CTN" bash /workspaces/cua-lite/scripts/train/slime/init.sh \
  > "$RUN/slime_init.log" 2>&1
echo "$CTN gpus=$G_ROLLOUT,$G_TRAIN host=$(hostname) created=$(date -u +%FT%TZ) desc=sglang-omni-rl trainer 全量(100批,约4-5天);GPU$G_ROLLOUT/$G_TRAIN 中含用户授权混卡的占位卡;收尾:全量终判后删" >> "$HOME/jiaxuanluo-map.txt"

# ── [5] 发射(nohup 后台;监控走宿主 tail)──
# 防跨发射污染:group_index 每次发射从 0 重计,残留 returns 会与新批同组
# 混算 RLOO 基线(2026-08-26 实锤事故);重发前必须轮转 returns 与 decisions。
mkdir -p "$REC/run_full/returns"
cp "$RUN/train.parquet" "$REC/run_full/"
nohup docker exec \
  -e ASYNC=1 -e NUM_TRAIN_GPUS=2 -e NUM_ROLLOUT_GPUS=1 \
  -e TP_SIZE=2 -e OPTIM_CPU_OFFLOAD=1 \
  -e HAS_NVLINK=0 \
  -e MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \
  -e HF_CKPT=/data/models/GUI-Owl-1.5-8B-Instruct \
  -e ENV_ID=mobileworld \
  -e PROMPT_DATA="/workspaces/cc_recipe/run_full/train.parquet" \
  -e ROLLOUT_BATCH_SIZE=8 -e N_SAMPLES_PER_PROMPT=8 -e NUM_STEPS_PER_ROLLOUT=1 \
  -e NUM_ROLLOUT="$STEPS" -e ENV_CONCURRENCY=32 \
  -e ROLLOUT_STALL_TIMEOUT_S=3600 \
  -e SAVE_INTERVAL=3 \
  -e CC_TASK_PRIORITY=1 \
  -e CUA_LITE_MULTIMODAL_LAZY_EXPAND=1 \
  -e ROLLOUT_MODULE=sglang_omni_rl.rollout_grpo \
  -e CONFIG_PATH="/workspaces/cc_recipe/configs/gui_owl/mobileworld_learned.yaml" \
  -e CUA_LITE_ENV_SERVER_URL -e CUA_LITE_ENV_SERVER_TOKEN \
  -e CUA_LITE_ALLOW_DIRTY_ENV_SERVER=1 \
  -e SAVE_DIR="/workspaces/cc_recipe/run_full/ckpt_megatron" \
  -e SAVE_HF_DIR="/workspaces/cc_recipe/run_full/hf/iter_{rollout_id}" \
  "$CTN" bash /workspaces/cua-lite/scripts/train/run_grpo.sh \
  > "$RUN/grpo.log" 2>&1 & echo $! > "$RUN/grpo_exec.pid"

echo "RUN_DIR=$RUN CTN=$CTN"
echo "KEEP: 全量在跑(100 批,约 4-5 天);收尾=终判后按清单(容器+pid 三件+map)"
