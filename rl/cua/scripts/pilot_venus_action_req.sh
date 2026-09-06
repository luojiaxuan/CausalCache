#!/usr/bin/env bash
# note (luojiaxuan): 文本只留 action 的口径重跑一次,所有图条件带上请求帧(短信屏)——上一轮该口径没带请求帧,模型不知道要哪一项,源帧救回率被压低。
# 等 GUI-Owl 教师强制链释放卡后取卡;跑完删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_OWL_TF_DONE /data01/jaxan/pilot_owl_tf_chain.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 文本仅 action 口径 + 请求帧(Venus,Mail 30);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
rm -f $P/eval_venus_mail_action_req.jsonl
python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_mail.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag venus_mail_action_req --text-mode action --out $P/eval_venus_mail_action_req.jsonl --workers 4 2>&1 | tee $P/eval_venus_mail_action_req.txt
python3 /data01/jaxan/pilot_gates.py $P/eval_venus_mail_action_req.jsonl | tail -8
rm_own "$C"; echo PILOT_VENUS_ACTION_REQ_DONE
