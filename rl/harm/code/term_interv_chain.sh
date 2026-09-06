#!/usr/bin/env bash
# note (luojiaxuan): 机制 A 干预:部署布局 N=2/4/6 × {原样, noresp, hint, short},看过早终止能否被压制、由什么诱发。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; C=""
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
# note (luojiaxuan): docker 端口映射在并发建容器时反复报 "failed programming external connectivity"(iptables 竞争),
# 改用宿主网络、服务只绑 127.0.0.1:<空闲端口>,不经 docker-proxy。
PORT=$(free_port); E=http://127.0.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g --network host -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan \
  -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim vllm/vllm-omni:dev \
  --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' --host 127.0.0.1 --port $PORT || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl history-harm:过早终止干预(部署布局 N×变体,port $PORT);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"gui-owl"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"gui-owl","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G ($C)"
SPECS="rec4_deploy_noresp,rec4_deploy_hint,rec4_deploy_short,rec6_deploy_noresp,rec6_deploy_hint,rec2_deploy_noresp,rec2_deploy_hint"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_term_interv --specs "$SPECS" --out $H/term_interv_base.jsonl --workers 12 > $H/term_interv_base.log 2>&1
echo "$(date -u +%FT%TZ) done: $(wc -l < $H/term_interv_base.jsonl)"; rm_own "$C"; echo TERM_INTERV_DONE
