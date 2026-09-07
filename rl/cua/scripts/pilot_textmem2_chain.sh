#!/usr/bin/env bash
# note (luojiaxuan): 文本记忆基线第二轮:补一版强制单行的简洁笔记(v1 实测约 320 token/帧,小预算上对文本不公平),
# 与 v1、OCR 转写、图像检索在同一预算表里重跑。等第一轮与延后揭示链结束;跑完删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_LATEBOUND_DONE /data01/jaxan/pilot_latebound_chain.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":16}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl 外审 §3.1 文本记忆第二轮(简洁笔记 v2 + 重跑预算表);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
for spec in $P/specs_venus_mail.jsonl $P/specs_browse.jsonl; do
  [ -f $spec ] || continue
  python3 /data01/jaxan/pilot_textmem.py --spec $spec --base-url $E --tag archive --out /dev/null --archive-only --workers 4 2>&1 | tail -2
done
rm -f $P/eval_textmem2_mail.jsonl
python3 /data01/jaxan/pilot_textmem.py --spec $P/specs_venus_mail.jsonl --base-url $E --tag textmem2_mail --out $P/eval_textmem2_mail.jsonl --workers 4 2>&1 | tee $P/eval_textmem2_mail.txt
if [ -f $P/specs_browse.jsonl ]; then rm -f $P/eval_latebound2.jsonl
  python3 /data01/jaxan/pilot_textmem.py --spec $P/specs_browse.jsonl --base-url $E --tag latebound2 --probe --out $P/eval_latebound2.jsonl --workers 4 2>&1 | tee $P/eval_latebound2.txt; fi
rm_own "$C"; echo PILOT_TEXTMEM2_DONE
