#!/usr/bin/env bash
# note (luojiaxuan): 终止触发物剂量 2(同剂量放指令前 / 禁 terminate)——rec4/rec6 的早期回复换 "Noted"(noted),参考轮插到指令之后(hybridturnin);337 态,1 张卡。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; R=/data01/jaxan/rl_v2; C=""
until grep -q "DOSE_DONE\|SERVER_FAIL\|RUN_FAIL" /data01/jaxan/dose_chain.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
PORT=$(free_port); E=http://127.0.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g --network host -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan \
  -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct \
  --served-model-name gui-owl --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' --host 127.0.0.1 --port $PORT || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl history-harm:终止触发物剂量 2(同剂量放指令前 / 禁 terminate)(port $PORT);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"gui-owl"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"gui-owl","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G ($C)"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_dose2 --specs "graybefore1,graybefore2,graybefore3,graybefore4,rec4_deploy_noterm,rec6_deploy_noterm,grayturnin2_noterm,hybridturnin:judge_glm46v|direct|six_noterm" --picks "$R/picks_*.jsonl" --out $H/dose2_base.jsonl --workers 8 > $H/dose2_base.log 2>&1
echo "$(date -u +%FT%TZ) done: $(wc -l < $H/dose2_base.jsonl)"; rm_own "$C"; echo DOSE2_DONE
