#!/usr/bin/env bash
# note (luojiaxuan): 文本保真度扫描(Venus,Mail 两族 30 个 checkpoint):同一批 checkpoint,历史文本按 oneline(think 首句+动作)与 action(只有动作)
# 两档压缩后跑同一干预矩阵——回答"一行摘要的 agent 还能靠文本记住值吗;记不住时源帧能不能救回、最近帧/对照帧能不能"。等 GUI-Owl PartMatch 矩阵释放卡后再取卡。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q "PILOT_EVAL_OWL_DONE owl_pm2" /data01/jaxan/pilot_eval_owl_pm2.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 文本保真度扫描(Venus,Mail 30 checkpoint × oneline/action);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
for mode in oneline action; do rm -f $P/eval_venus_mail_$mode.jsonl
  python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_mail.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag venus_mail_$mode --text-mode $mode --out $P/eval_venus_mail_$mode.jsonl --workers 4 2>&1 | tee $P/eval_venus_mail_$mode.txt
  python3 /data01/jaxan/pilot_gates.py $P/eval_venus_mail_$mode.jsonl | tail -8; done
rm_own "$C"; echo PILOT_VENUS_TEXTMODE_DONE
