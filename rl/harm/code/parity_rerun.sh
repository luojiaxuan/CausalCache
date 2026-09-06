#!/usr/bin/env bash
# note (luojiaxuan): 官方默认协议(history_n=1,不带历史图;CC_* 一律不设)在同一 harness 上跑 3 遍,量单次运行方差,
# 并与 8-24 对表(39/115=33.9%)、官方榜(38.2%,3 runs)对账。等四臂闭环结束后接续(共用模拟器池)。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
R=/data01/jaxan/rl_v2; C=""
TASKS=$(python3 -c "import json; s=json.load(open('/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json')); print(','.join(sorted(set(s['train'])|set(s['heldout']))))")
until grep -q "BASE_ARMS_DONE" /data01/jaxan/base_arms_closedloop2.log 2>/dev/null; do sleep 120; done
# note (luojiaxuan): 用户要求 32 台全新模拟器再对账:四臂结束后重建 p00–p11,与已重建的 p14–p33 合成 32 台;并发 32。
source /data01/jaxan/pool_lib.sh
until grep -q POOL_REBUILD_NOW_DONE /data01/jaxan/pool_rebuild_now.log 2>/dev/null; do sleep 60; done
for i in $(seq 0 11); do mk_pool $i && sleep 8; done
wait_pool "$(seq -s' ' 0 11)" || echo "WARN: p00-p11 未全部就绪"
HOSTS=$( (seq 0 11; seq 14 33) | awk '{printf "http://127.0.0.1:%d,", 6800+$1}' | sed 's/,$//')
NH=$(echo "$HOSTS" | tr "," "\n" | wc -l); echo "hosts=$NH"
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
PORT=$(free_port); E=http://127.0.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g --network host -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan \
  -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct \
  --served-model-name gui-owl --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' --host 127.0.0.1 --port $PORT || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl 官方默认协议(无历史图)×3 对账(port $PORT);⚠ 在用勿删;收尾:三轮结束删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"gui-owl"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"gui-owl","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G ($C)"
for run in 1 2 3; do
  O=$R/guiowl_official_default_run$run; mkdir -p $O
  echo "$(date -u +%FT%TZ) START official_default run$run"
  cd /data01/jaxan/mw/MobileWorld && env -u CC_HISTORY_N -u CC_FRAME_POLICY PYTHONPATH=/data01/jaxan/pyshim timeout 14400 \
    uv run mw eval --agent_type gui_owl_1_5 --task "$TASKS" --max_round 50 --model_name gui-owl --llm_base_url $E --api_key EMPTY \
    --step_wait_time 3 --max-concurrency $NH --aw-host "$HOSTS" --log_file_root $O > $O.log 2>&1
  k=$(grep -l "^score: 1" $O/*/result.txt 2>/dev/null | wc -l); n=$(ls $O/*/result.txt 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) DONE official_default run$run succ_dirs=$k results=$n"
done
rm_own "$C"; echo PARITY_RERUN_DONE
