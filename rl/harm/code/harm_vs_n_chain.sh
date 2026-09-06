#!/usr/bin/env bash
# note (luojiaxuan): Step 1 "伤害 vs 历史图数":GUI-Owl 底座,337 自一致状态,两种协议,recN / irrN(N=0..6)+ Step 3 的两个干预
# (recN_mark 时序标记、recN_blank 同 token 数灰图)。一张卡,约 337×(11+4)≈5k 次解码。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
R=/data01/jaxan/rl_v2; H=/data01/jaxan/harm; mkdir -p $H; E=http://172.17.0.1:41211/v1; C=""
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:41211:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null
reg "$C" "$G" "sglang-omni-rl history-harm:伤害 vs N + 干预(GUI-Owl 底座);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; docker logs "$C" 2>&1 | grep -a -i "free memory\|error" | tail -2 | cut -c1-160; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G"
SPECS="rec0,rec1,rec2,rec3,rec4,rec6,irr1,irr2,irr3,irr4,irr6,rec2_mark,rec4_mark,rec2_blank,rec4_blank"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_text --specs $SPECS --out $H/harm_vs_n_base_text.jsonl --workers 12 > $H/harm_vs_n_base_text.log 2>&1
echo "$(date -u +%FT%TZ) text done: $(wc -l < $H/harm_vs_n_base_text.jsonl)"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_notext --no-text --specs $SPECS --out $H/harm_vs_n_base_notext.jsonl --workers 12 > $H/harm_vs_n_base_notext.log 2>&1
echo "$(date -u +%FT%TZ) notext done: $(wc -l < $H/harm_vs_n_base_notext.jsonl)"
rm_own "$C"; echo HARM_VS_N_DONE
