#!/usr/bin/env bash
# note (luojiaxuan): 预算曲线第二段:只留 action + 请求帧 + 证据轮,证据帧与请求帧按边长 0.5 / 0.35 / 0.25 缩放(像素 1/4、1/8、1/16),记 prompt_tokens。
# 等 QuoteRecall 共享请求版矩阵释放卡后取卡;跑完删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_QR2_ACTION_DONE /data01/jaxan/pilot_qr2_action.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 预算曲线第二段(Venus,缩放证据帧 0.5/0.35/0.25);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
rm -f $P/eval_venus_budget_scaled.jsonl
python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_mail.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag venus_budget_scaled \
  --text-mode action --scales 0.5,0.35,0.25 --conds src_at_turn,src_at_turn@0.5,src_at_turn@0.35,src_at_turn@0.25 --out $P/eval_venus_budget_scaled.jsonl --workers 4 2>&1 | tee $P/eval_venus_budget_scaled.txt
rm_own "$C"; echo PILOT_VENUS_SCALED_DONE
