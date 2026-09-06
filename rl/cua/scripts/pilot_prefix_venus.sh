#!/usr/bin/env bash
# note (luojiaxuan): B-pilot 前缀采集(Venus 版):UI-Venus-2-9B 按官方多轮协议(最近两帧)在两台独立模拟器上闭环跑全部 48 个孪生任务,
# 目的是拿到 executor 自然生成的因果前缀(含决策步),供 checkpoint 干预矩阵用;等 GUI-Owl 版前缀跑完释放模拟器后自动发射。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
EMUS=$(cat /data01/jaxan/.pilot_emus)
until grep -q PILOT_PREFIX_DONE /data01/jaxan/pilot_prefix.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 前缀采集(Venus):UI-Venus-2 闭环跑 48 孪生任务(模拟器 $EMUS);⚠ 在用勿删;收尾:采集结束删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; docker logs "$C" 2>&1 | grep -a -i error | tail -2 | cut -c1-160; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT) emulators=$EMUS"
TASKS=$(python3 - <<'PY'
fam={"QuoteRecall":8,"OrderAddressJoin":8,"PartMatch":8}
print(",".join(f"{f}Task{k:02d}{t}" for f,n in fam.items() for k in range(1,n+1) for t in "AB"))
PY
)
AW=$(echo "$EMUS" | tr ',' '\n' | sed 's#^#http://127.0.0.1:#' | paste -sd,)
NC=$(echo "$EMUS" | tr ',' '\n' | wc -l)
O=/data01/jaxan/rl_v2/pilot/prefix_venus; mkdir -p $O
cd /data01/jaxan/mw/MobileWorld
CC_VENUS_HIST=recent:2 PYTHONPATH=/data01/jaxan/pyshim timeout 14400 \
  uv run mw eval --agent_type ui_venus2 --task "$TASKS" --max_round 40 --model_name UI-Venus-2 \
  --llm_base_url $E --api_key EMPTY --step_wait_time 3 --max-concurrency $NC --aw-host "$AW" \
  --log_file_root $O > $O.log 2>&1
n=$(ls -d $O/*/ 2>/dev/null | wc -l); k=$(grep -l "^score: 1" $O/*/result.txt 2>/dev/null | wc -l)
echo "$(date -u +%FT%TZ) PREFIX_VENUS_DONE dirs=$n succ=$k | $(grep -a -o "Final: .*" $O.log | tail -1)"
rm_own "$C"; echo PILOT_PREFIX_VENUS_DONE
