#!/usr/bin/env bash
# note (luojiaxuan): GUI-Owl 教师强制前缀上的第二轮矩阵:加"请求轮 + 证据轮一起保留"(控制器该做的事)与只保留请求轮的对照。等预算曲线链释放卡。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_VENUS_BUDGET_DONE /data01/jaxan/pilot_venus_budget.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41221 41231 41241 41271 41281)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:$PORT:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot GUI-Owl 教师强制矩阵第二轮(请求轮+证据轮);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) owl up ($C GPU$G:$PORT)"
rm -f $P/eval_owl_tf2_mail.jsonl
python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/prefix_owl_tf/specs_owl_tf.jsonl --backend owl --base-url $E --model gui-owl --tag owl_tf2_mail \
  --conds text_only,rec2,ctrl_keep,src_keep,swap_keep,req_keep,src_req_keep,ctrl_req_keep,gold_text --out $P/eval_owl_tf2_mail.jsonl --workers 4 2>&1 | tee $P/eval_owl_tf2_mail.txt
python3 /data01/jaxan/pilot_gates.py $P/eval_owl_tf2_mail.jsonl --source src_req_keep --ctrl ctrl_req_keep | tail -9
rm_own "$C"; echo PILOT_OWL_TF2_DONE
