#!/usr/bin/env bash
# note (luojiaxuan): PartMatch v3 的 Venus 矩阵重跑:决策步改按"候选列表第一次上屏(文本点名 cand_*)"定位后重建规格,原生文本 + 无文本各一遍。等缩放档链释放卡。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
until grep -q PILOT_VENUS_SCALED_DONE /data01/jaxan/pilot_venus_scaled.log 2>/dev/null; do sleep 60; done
python3 /data01/jaxan/pilot_build_specs.py --prefix-dir $P/prefix_venus_pm3 --seeds $P/task_seeds.txt --backend venus --out $P/specs_venus_pm3b.jsonl --contact $P/contact_venus_pm3b > $P/specs_venus_pm3b.log 2>&1
grep -v "^    " $P/specs_venus_pm3b.log | head -3
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot PartMatch v3 Venus 矩阵重跑;⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
for proto in text notext; do extra=$([ "$proto" = notext ] && echo --no-text || true); rm -f $P/eval_venus_pm3b_$proto.jsonl
  python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_pm3b.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag venus_pm3b_$proto $extra --out $P/eval_venus_pm3b_$proto.jsonl --workers 4 2>&1 | tee $P/eval_venus_pm3b_$proto.txt
  python3 /data01/jaxan/pilot_gates.py $P/eval_venus_pm3b_$proto.jsonl | tail -8; done
rm_own "$C"; echo PILOT_PM3B_DONE
