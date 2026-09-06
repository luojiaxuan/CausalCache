#!/usr/bin/env bash
# note (luojiaxuan): H5′ 干预——指令重复到最后一条 user 消息;N=2/4/6。等 Venus N 扫描释放 GPU 后再起(账号占卡上限)。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; C=""
until grep -q "VENUS_N_DONE\|FAIL" /data01/jaxan/venus_n_chain4.log 2>/dev/null; do sleep 120; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
PORT=$(free_port); E=http://127.0.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g --network host -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan \
  -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct \
  --served-model-name gui-owl --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' --host 127.0.0.1 --port $PORT || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl history-harm:指令重复干预(port $PORT);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"gui-owl"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"gui-owl","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G ($C)"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_goal --specs "rec2_deploy_goal,rec4_deploy_goal,rec6_deploy_goal" --out $H/goal_interv_base.jsonl --workers 8 > $H/goal_interv_base.log 2>&1
echo "$(date -u +%FT%TZ) done: $(wc -l < $H/goal_interv_base.jsonl)"; rm_own "$C"; echo GOAL_INTERV_DONE
