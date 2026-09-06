#!/usr/bin/env bash
# note (luojiaxuan): B-pilot 主臂:Venus 自然前缀 → checkpoint 规格(Mail 两族;PartMatch 待目录挪动后重采,另行评测)→ Venus 原生协议下的
# 干预矩阵,原生文本与无文本两种口径各跑一遍。等 Venus 全量前缀链结束后自动发射;起服务→跑→删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_PREFIX_VENUS_DONE /data01/jaxan/pilot_prefix_venus.log 2>/dev/null; do sleep 60; done
python3 /data01/jaxan/pilot_build_specs.py --prefix-dir $P/prefix_venus --seeds $P/task_seeds.txt --backend venus --out $P/specs_venus_all.jsonl --contact $P/contact_venus > $P/specs_venus.log 2>&1
grep -v '"family": "PartMatch"' $P/specs_venus_all.jsonl > $P/specs_venus_mail.jsonl
n=$(wc -l < $P/specs_venus_mail.jsonl); echo "$(date -u +%FT%TZ) venus mail specs=$n"; grep -v "^    " $P/specs_venus.log | head -6
[ "$n" -gt 0 ] || { echo "NO_VENUS_CHECKPOINTS"; exit 0; }
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 干预矩阵(Venus 原生协议,Mail 两族,原生文本+无文本);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
for proto in text notext; do
  extra=$([ "$proto" = notext ] && echo --no-text || true)
  rm -f $P/eval_venus_mail_$proto.jsonl
  python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_mail.jsonl --backend venus --base-url $E --model UI-Venus-2 \
    --tag venus_mail_$proto $extra --out $P/eval_venus_mail_$proto.jsonl --workers 4 2>&1 | tee $P/eval_venus_mail_$proto.txt
done
rm_own "$C"; echo PILOT_EVAL_VENUS_MAIL_DONE
