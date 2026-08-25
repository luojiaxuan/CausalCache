#!/bin/bash
# note (luojiaxuan): joint GRPO smoke 主循环(v2 recipe,hyper00 GPU0/1)。
# 形态:GPU0 vLLM(--enable-lora, LoRA 热换)做 rollout;GPU1 selector 服务
# + 训练。每轮:serve -> collect(G 条 selector 臂 rollout) -> train -> 热换。
# v2 要点:G 全部 selector 臂 + RLOO;无任务硬闸门(smoke 任务手选 8 短题,
# 均匀采样);recent 对照只在第 0/5/10 轮采一批做测量。
# 验收(四条,过程量全落 metrics.jsonl):reward 非全 0 / 选帧分布在变 /
# loss·grad 有限 / 热换后 vLLM 出的 logprob 与训练侧一致。
set -u
D=/data01/jaxan/grpo
MW=/data01/jaxan/mw/MobileWorld
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
ST="$D/smoke_status.txt"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$ST"; }
mkdir -p "$D/decisions" "$D/ckpt" "$D/traj"

# note (luojiaxuan): smoke 任务 = train 池里步数短、recent 基线非零的 8 题
# (smoke 只验管线;正式训练按 v2 用均匀地板,无过滤闸门)。
TASKS="${CC_SMOKE_TASKS:?set CC_SMOKE_TASKS}"
G=4; ITERS=10; BUDGET=2

# note (luojiaxuan): 共享机竞态教训(2026-08-25):写死 device=0 在检查与发射
# 的间隙被人抢占,vLLM 以 free 31GB 起不来。改为发射时动态选两张空卡
# (排除 GPU6 —— 探针 vLLM 在用),选不满两张即中止。
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
  | awk -F', ' '$2+0<2000{print $1}' | grep -v '^6$' | head -2)
[ "${#FREE[@]}" -lt 2 ] && { log "NO_2_FREE_GPUS: ${FREE[*]:-none}"; exit 1; }
GV=${FREE[0]}; GT=${FREE[1]}
log "动态选卡: vLLM=GPU$GV train/selector=GPU$GT"

log "[-1] 独立 env 池 mw_rl(8 台,不与探针的 mw_random 池混用)"
n_rl=$(docker ps --format '{{.Names}}' | grep -c mw_rl || true)
if [ "$n_rl" -lt 8 ]; then
  timeout 1800 uv run --project "$MW" mw env run --count $((8-n_rl)) \
    --name-prefix mw_rl --launch-interval 20 >> "$ST" 2>&1 || true
  sleep 90
fi
ready=0
for c in $(docker ps --format '{{.Names}}' | grep mw_rl); do
  docker exec "$c" sh -c "adb devices 2>/dev/null" 2>/dev/null \
    | grep -qE "emulator-[0-9]+[[:space:]]+device" && ready=$((ready+1)) \
    || docker rm -f "$c" >/dev/null 2>&1
done
log "mw_rl 就绪 $ready 台(adb 真判据)"
[ "$ready" -lt 4 ] && { log "RL_POOL_TOO_SMALL"; exit 1; }

log "[0] vLLM(GPU0, enable-lora)"
docker rm -f sglang-omni-jaxan-rl0 >/dev/null 2>&1 || true
docker run -d --name sglang-omni-jaxan-rl0 --init --gpus "\"device=$GV\"" --network host \
  --ipc=host --shm-size=32g -e PYTHONPATH=/data/pyshim -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  -v /data04/jaxan:/data -v /data01/jaxan/grpo:/data/grpo --entrypoint python3 vllm/vllm-omni:dev \
  -m vllm.entrypoints.openai.api_server \
  --model /data/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --port 41003 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enable-lora --max-lora-rank 32 --limit-mm-per-prompt '{"image": 8}' >/dev/null
printf "sglang-omni-jaxan-rl0\tgpus=$GV\thost=hyper00\tcreated=%s\tdesc=sglang-omni-rl smoke;收尾:smoke 完成即删\n" \
  "$(date -u +%FT%TZ)" >> "$HOME/jiaxuanluo-map.txt"
for i in $(seq 1 60); do curl -s -m 4 http://127.0.0.1:41003/v1/models 2>/dev/null | grep -q gui-owl && break; sleep 10; done
curl -s -m 4 http://127.0.0.1:41003/v1/models | grep -q gui-owl && log "VLLM_RL0_READY" || { log "VLLM_RL0_FAIL"; exit 1; }

log "[1] selector 服务(GPU1)"
pkill -f "[s]elector_service" 2>/dev/null; sleep 2
cd "$MW"
CC_SEL_DEVICE=cuda:0 CUDA_VISIBLE_DEVICES=$GT CC_SEL_PORT=41010 CC_SEL_LOG="$D/decisions" \
  CC_SEL_CKPT="$D/ckpt/selector_latest.pt" \
  setsid nohup uv run --with pillow python "$D/selector_service.py" > "$D/sel_service.log" 2>&1 &
sleep 8
curl -s -m 5 -X POST http://127.0.0.1:41010/select -d '{"frames_b64":[],"step":0,"budget":2}' \
  -H 'Content-Type: application/json' | grep -q indices && log "SEL_SERVICE_READY" || { log "SEL_SERVICE_FAIL"; tail -5 "$D/sel_service.log"; exit 1; }

for it in $(seq 0 $((ITERS-1))); do
  log "===== iter $it ====="
  # note (luojiaxuan): it>0 后必须请求 LoRA adapter 名,否则 rollout 永远
  # 打在 base 权重上(权重同步假成功 —— 验收第 4 条的靶心)。
  MN=$([ "$it" -eq 0 ] && echo gui-owl || echo "cc_it$((it-1))")
  log "[2] rollout: G=$G selector 臂, model=$MN"
  for g in $(seq 1 $G); do
    CC_HISTORY_N=$((BUDGET+1)) CC_FRAME_POLICY=learned CC_SELECTOR_URL=http://127.0.0.1:41010 \
    CC_DUMP_DIR="$D/traj/it${it}_g${g}" timeout 7200 \
      uv run mw eval --agent_type gui_owl_1_5 --task "$TASKS" --max_round 20 \
      --model_name "$MN" --llm_base_url http://127.0.0.1:41003/v1 --api_key EMPTY \
      --step_wait_time 3 --max-concurrency 8 --env-name-prefix mw_rl \
      --log_file_root "$D/traj/it${it}_g${g}" > "$D/traj/it${it}_g${g}.log" 2>&1 || true
  done
  if [ $((it % 5)) -eq 0 ]; then
    log "[2b] recent 对照测量批(不进梯度)"
    CC_HISTORY_N=$((BUDGET+1)) CC_FRAME_POLICY=recent timeout 7200 \
      uv run mw eval --agent_type gui_owl_1_5 --task "$TASKS" --max_round 20 \
      --model_name gui-owl --llm_base_url http://127.0.0.1:41003/v1 --api_key EMPTY \
      --step_wait_time 3 --max-concurrency 8 --env-name-prefix mw_rl \
      --log_file_root "$D/traj/it${it}_recent" > /dev/null 2>&1 || true
  fi
  log "[3] train(RLOO + selector PL;executor LoRA)"
  cd "$MW" && CUDA_VISIBLE_DEVICES=$GT timeout 3600 uv run --with peft,pillow \
    python "$D/train_smoke.py" --iter "$it" --g "$G" --root "$D" 2>&1 | tail -4 | tee -a "$ST"
  log "[4] 热换 LoRA + selector 权重"
  curl -s -m 30 -X POST http://127.0.0.1:41003/v1/load_lora_adapter \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"cc_it${it}\",\"lora_path\":\"/data/grpo/ckpt/lora_it${it}\"}" | tee -a "$ST"
  curl -s -m 10 -X POST http://127.0.0.1:41010/reload \
    -d "{\"path\":\"$D/ckpt/selector_latest.pt\"}" -H 'Content-Type: application/json' >/dev/null
done
log "SMOKE_DONE"
