#!/usr/bin/env bash
# note (luojiaxuan): 终止触发物剂量 3(拼图单块对照)——rec4/rec6 的早期回复换 "Noted"(noted),参考轮插到指令之后(hybridturnin);337 态,1 张卡。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; R=/data01/jaxan/rl_v2; C=""
until grep -q "DOSE2_DONE\|SERVER_FAIL\|RUN_FAIL" /data01/jaxan/dose2_chain.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
PORT=$(free_port); E=http://127.0.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g --network host -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan \
  -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct \
  --served-model-name gui-owl --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' --host 127.0.0.1 --port $PORT || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl history-harm:终止触发物剂量 3(拼图单块对照)(port $PORT);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"gui-owl"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"gui-owl","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G ($C)"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_dose3 --specs "graystackturnin2,graystackturnin3,graystackturnin4" --out $H/dose3_base.jsonl --workers 8 > $H/dose3_base.log 2>&1
echo "$(date -u +%FT%TZ) done: $(wc -l < $H/dose3_base.jsonl)"; rm_own "$C"; echo DOSE3_DONE
