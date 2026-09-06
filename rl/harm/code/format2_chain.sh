#!/usr/bin/env bash
# note (luojiaxuan): 解耦"结构"与"内容":pickimg(结构固定、只换图)与 hybrid(最近两 turn 原样 + 带标记的检索帧)。底座 GUI-Owl,337 态。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; R=/data01/jaxan/rl_v2; C=""
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
PORT=$(free_port); E=http://127.0.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g --network host -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan \
  -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct \
  --served-model-name gui-owl --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' --host 127.0.0.1 --port $PORT || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl history-harm:格式解耦 hybrid 放法 last/turn(port $PORT);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"gui-owl"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"gui-owl","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G ($C)"
# note (luojiaxuan): 首发在 109 态处服务断连(RemoteDisconnected);降并发到 8,先续跑原规格,再跑 hybrid 的另两种放法。
# note (luojiaxuan): 只跑 hybrid 的另两种放法(format1 已完成);首发 18 态崩,并发降到 4。
SPECS2="rec2_deploy,hybridlast:judge_glm46v|direct|six,hybridturn:judge_glm46v|direct|six,hybridlast:qwen38_27b|direct|six,hybridturn:qwen38_27b|direct|six"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_format2 --specs "$SPECS2" --picks "$R/picks_*.jsonl" --out $H/format2_base.jsonl --workers 4 >> $H/format2_base.log 2>&1
echo "$(date -u +%FT%TZ) format2 done: $(wc -l < $H/format2_base.jsonl)"; rm_own "$C"; echo FORMAT_DONE
