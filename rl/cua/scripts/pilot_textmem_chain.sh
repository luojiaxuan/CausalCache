#!/usr/bin/env bash
# note (luojiaxuan): 外审 §3.1 判别实验(第一段,开发集口径):Mail 两族 30 个 checkpoint 上,称职的在线文本记忆(笔记 / OCR 转写 + 检索)
# 与图像检索在 2k/4k/8k/16k 历史 token 预算下对打。先建档案(每帧两次 VLM 调用,按目录缓存),再跑矩阵。跑完删容器。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl 外审 §3.1 文本记忆基线(笔记/OCR 档案 + 预算扫描,Venus,Mail 30);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
python3 /data01/jaxan/pilot_textmem.py --spec $P/specs_venus_mail.jsonl --base-url $E --tag archive --out /dev/null --archive-only --workers 4 2>&1 | tail -3
echo "$(date -u +%FT%TZ) archives built"
rm -f $P/eval_textmem_mail.jsonl
python3 /data01/jaxan/pilot_textmem.py --spec $P/specs_venus_mail.jsonl --base-url $E --tag textmem_mail --out $P/eval_textmem_mail.jsonl --workers 4 2>&1 | tee $P/eval_textmem_mail.txt
rm_own "$C"; echo PILOT_TEXTMEM_DONE
