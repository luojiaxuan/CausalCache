#!/usr/bin/env bash
# note (luojiaxuan): recent8 / random2 两臂首发因服务中途死亡而无效(28 / 0 题);等官方默认 ×3 对账跑完后,用同一套 32 台新池重跑。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
R=/data01/jaxan/rl_v2; C=""
TASKS=$(python3 -c "import json; s=json.load(open('/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json')); print(','.join(sorted(set(s['train'])|set(s['heldout']))))")
until grep -q "PARITY_RERUN_DONE" /data01/jaxan/parity_rerun.log 2>/dev/null; do sleep 120; done
HOSTS=$( (seq 0 11; seq 14 33) | awk '{printf "http://127.0.0.1:%d,", 6800+$1}' | sed 's/,$//'); NH=$(echo "$HOSTS" | tr "," "\n" | wc -l)
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
PORT=$(free_port); E=http://127.0.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g --network host -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan \
  -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct \
  --served-model-name gui-owl --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":12}' --host 127.0.0.1 --port $PORT || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl 底座 GUI-Owl recent8/random2 闭环重跑(32 台池,port $PORT);⚠ 在用勿删;收尾:两臂结束删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"gui-owl"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"gui-owl","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G ($C) hosts=$NH"
for spec in "recent8:9:recent" "random2:3:random"; do
  arm=${spec%%:*}; rest=${spec#*:}; N=${rest%%:*}; POL=${rest#*:}; O=$R/guiowl_base_${arm}_v2; mkdir -p $O
  echo "$(date -u +%FT%TZ) START $arm"
  cd /data01/jaxan/mw/MobileWorld && CC_HISTORY_N=$N CC_FRAME_POLICY=$POL PYTHONPATH=/data01/jaxan/pyshim timeout 14400 \
    uv run mw eval --agent_type gui_owl_1_5 --task "$TASKS" --max_round 50 --model_name gui-owl --llm_base_url $E --api_key EMPTY \
    --step_wait_time 3 --max-concurrency $NH --aw-host "$HOSTS" --log_file_root $O > $O.log 2>&1
  curl -s -m 10 $E/models | grep -q '"gui-owl"' || echo "$(date -u +%FT%TZ) WARN: server down after $arm"
  echo "$(date -u +%FT%TZ) DONE $arm succ_dirs=$(grep -l "^score: 1" $O/*/result.txt 2>/dev/null | wc -l) results=$(ls $O/*/result.txt 2>/dev/null | wc -l)"
done
rm_own "$C"; echo ARMS_RERUN_DONE
