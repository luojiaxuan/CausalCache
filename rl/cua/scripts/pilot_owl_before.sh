#!/usr/bin/env bash
# note (luojiaxuan): GUI-Owl 第三种放法(history-harm 线建议):检索帧作为参考轮放在指令消息之前,与保留原位(src_keep)对照;
# 在教师强制 Mail 30 与 PartMatch v4 自然前缀两批 checkpoint 上跑。等 v4 全链释放卡;跑完删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_PM4_DONE /data01/jaxan/pilot_pm4_chain.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41221 41231 41241 41271 41281)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:$PORT:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot GUI-Owl 参考轮前置放法(Mail 教师强制 + PartMatch v4);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
rm -f $P/eval_owl_tf_before.jsonl
python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/prefix_owl_tf/specs_owl_tf.jsonl --backend owl --base-url $E --model gui-owl --tag owl_tf_before \
  --conds rec2,src_keep,src_req_keep,ctrl_req_keep,src_before,ctrl_before,src_req_before,ctrl_req_before,gold_text --out $P/eval_owl_tf_before.jsonl --workers 4 2>&1 | tee $P/eval_owl_tf_before.txt
python3 /data01/jaxan/pilot_gates.py $P/eval_owl_tf_before.jsonl --source src_req_before --ctrl ctrl_req_before | grep -A5 "闸门" | head -6
if [ -f $P/specs_prefix_base_pm4.jsonl ]; then rm -f $P/eval_owl_pm4_before.jsonl
  python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_prefix_base_pm4.jsonl --backend owl --base-url $E --model gui-owl --tag owl_pm4_before \
    --conds text_only,rec2,ctrl_keep,src_keep,src_before,ctrl_before,swap_keep,gold_text --out $P/eval_owl_pm4_before.jsonl --workers 4 2>&1 | tee $P/eval_owl_pm4_before.txt
  python3 /data01/jaxan/pilot_gates.py $P/eval_owl_pm4_before.jsonl --source src_before --ctrl ctrl_before | grep -A5 "闸门" | head -6; fi
rm_own "$C"; echo PILOT_OWL_BEFORE_DONE
