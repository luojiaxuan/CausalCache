#!/usr/bin/env bash
# note (luojiaxuan): selector-RL v2 全栈发射:executor vLLM(1卡)+
# selector 服务 v2(1卡,视觉塔)+ trainer 守护(CPU)+ 回合编排器。
# RL_SMOKE=1 时小规模(2 任务 × G=4 × 1 回合)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan
R=/data01/jaxan/rl_v2
mkdir -p $R/ckpt $R/decisions
PICK=""
for g in 0 1 2 3 4 6; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 3000 ] && PICK="$PICK $g"
done
set -- $PICK
[ $# -lt 2 ] && { echo "ABORT: 空卡不足 2"; exit 1; }
GE=$1; GS=$2
echo "[stack] executor GPU$GE, selector GPU$GS"

NE=sglang-omni-jaxan-rle
docker inspect $NE >/dev/null 2>&1 && docker rm -f -v $NE >/dev/null 2>&1
docker run -d --gpus "\"device=$GE\"" --name $NE --init --ipc=host --shm-size=16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim \
  -e PYTHONPATH=/pyshim -p 172.17.0.1:41041:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/sglang-omni-rl/cc_recipe/run_full/hf_p3/iter_59 \
  --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":4}' > /dev/null
echo "$NE	gpus=$GE	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl RLv2 executor推理;收尾:RL线结束删" >> "$HOME/jiaxuanluo-map.txt"

NS=sglang-omni-jaxan-rls
docker inspect $NS >/dev/null 2>&1 && docker rm -f -v $NS >/dev/null 2>&1
docker run -d --gpus "\"device=$GS\"" --name $NS --init --ipc=host --shm-size=16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim \
  -e PYTHONPATH=/pyshim -e CC_SEL_DEVICE=cuda:0 -e CC_SEL_PORT=41010 -e CC_SEL_BIND=0.0.0.0 \
  -e CC_SEL_LOG=$R/decisions -e CC_SEL_PV_FILE=$R/ckpt/selector_pv.txt \
  -p 172.17.0.1:41010:41010 \
  --entrypoint python3 vllm/vllm-omni:dev /data01/jaxan/selector_service_v2.py > /dev/null
echo "$NS	gpus=$GS	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl RLv2 selector服务;收尾:RL线结束删" >> "$HOME/jiaxuanluo-map.txt"

for i in $(seq 1 90); do
  curl -s -m 3 http://172.17.0.1:41041/v1/models 2>/dev/null | grep -q gui-owl && { echo "[stack] executor READY"; break; }
  sleep 10
done
for i in $(seq 1 90); do
  curl -s -m 3 -X POST http://172.17.0.1:41010/select -H 'Content-Type: application/json' \
    -d '{"frames_b64": []}' 2>/dev/null | grep -q indices && { echo "[stack] selector READY"; break; }
  sleep 10
done

setsid /data01/jaxan/sglang-omni-rl/selvenv/bin/python /data01/jaxan/selector_trainer_v2.py \
  --decisions-dir $R/decisions --returns-file $R/returns.jsonl --ckpt-dir $R/ckpt \
  --service-url http://172.17.0.1:41010 --interval 180 \
  > $R/trainer.log 2>&1 < /dev/null &
echo "[stack] trainer 守护已起"

if [ "${RL_SMOKE:-0}" = "1" ]; then
  RL_G=4 RL_TASKS=2 RL_ROUNDS=1 RL_CONC_TAGS=2 \
  RL_LLM_URL=http://172.17.0.1:41041/v1 RL_SEL_URL=http://172.17.0.1:41010 \
  /data01/jaxan/sglang-omni-rl/selvenv/bin/python /data01/jaxan/rl_round.py > $R/rounds.log 2>&1
  echo "SMOKE_ROUNDS_DONE"
else
  RL_LLM_URL=http://172.17.0.1:41041/v1 RL_SEL_URL=http://172.17.0.1:41010 \
  setsid /data01/jaxan/sglang-omni-rl/selvenv/bin/python /data01/jaxan/rl_round.py > $R/rounds.log 2>&1 < /dev/null &
  echo "FULL_LOOP_LAUNCHED"
fi
