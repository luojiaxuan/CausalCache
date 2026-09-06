#!/usr/bin/env bash
# note (luojiaxuan): 预算曲线(Venus,Mail 30 checkpoint):精简的视觉 token 能否在效率上胜过详细的文本摘要。
# 每个条件记录 vLLM 返回的 prompt_tokens(含图像 token)。臂:full 文本不给图(详细摘要基线);只留 action + 请求帧 + 源帧(原尺寸 / 0.5 / 0.35 / 0.25 边长);
# 只留 action + 请求帧 + 最近两帧;think 首句不给图。等"只留 action + 请求帧"链释放卡后取卡;跑完删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_VENUS_ACTION_REQ_DONE /data01/jaxan/pilot_venus_action_req.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 预算曲线(Venus,Mail 30:文本保真度 × 缩放源帧,记 prompt_tokens);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
run() { rm -f $P/eval_$1.jsonl; python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_mail.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag $1 --out $P/eval_$1.jsonl --workers 4 "${@:2}" 2>&1 | tee $P/eval_$1.txt; }
run venus_budget_full --text-mode full --conds text_only,rec2
run venus_budget_oneline --text-mode oneline --conds text_only
run venus_budget_action --text-mode action --scales 0.5,0.35,0.25 --conds text_only,rec2,src_at_turn,src_at_turn@0.5,src_at_turn@0.35,src_at_turn@0.25,ctrl_at_turn
rm_own "$C"; echo PILOT_VENUS_BUDGET_DONE
