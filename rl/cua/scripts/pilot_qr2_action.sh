#!/usr/bin/env bash
# note (luojiaxuan): QuoteRecall 共享请求版(孪生只差数字)在"只留 action + 请求帧"口径下的矩阵——补干净的孪生翻转端点。等 GUI-Owl 第二轮释放卡。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_OWL_TF2_DONE /data01/jaxan/pilot_owl_tf2.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot QuoteRecall 共享请求版 action+请求帧矩阵(Venus);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
rm -f $P/eval_venus_qr2_action_req.jsonl
python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_qr2.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag venus_qr2_action_req --text-mode action --out $P/eval_venus_qr2_action_req.jsonl --workers 4 2>&1 | tee $P/eval_venus_qr2_action_req.txt
python3 /data01/jaxan/pilot_gates.py $P/eval_venus_qr2_action_req.jsonl | tail -8
rm_own "$C"; echo PILOT_QR2_ACTION_DONE
