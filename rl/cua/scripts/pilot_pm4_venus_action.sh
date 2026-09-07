#!/usr/bin/env bash
# note (luojiaxuan): PartMatch v4 的"一行摘要"制式(Venus):历史只留 action(样品的文字描述被删)vs 只在样品轮留它自己 think 的首句(描述留),
# 交叉 证据轮 / 对照轮 / 不给图 —— 视觉指代物上,文字描述是否够用、图是否只在描述缺席时有价值。等 v4 全链释放卡。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_PM4_DONE /data01/jaxan/pilot_pm4_chain.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot PartMatch v4 一行摘要制式(Venus,描述删/留 × 证据/对照);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
for mode in action action_fact; do rm -f $P/eval_pm4_venus_$mode.jsonl
  python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_prefix_venus_pm4.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag pm4_venus_$mode --text-mode $mode --conds text_only,ctrl_at_turn,src_at_turn,swap_at_turn,gold_text --out $P/eval_pm4_venus_$mode.jsonl --workers 4 2>&1 | tee $P/eval_pm4_venus_$mode.txt
  python3 /data01/jaxan/pilot_gates.py $P/eval_pm4_venus_$mode.jsonl | grep -A5 "闸门" | head -6; done
rm_own "$C"; echo PILOT_PM4_VENUS_ACTION_DONE
