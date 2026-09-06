#!/usr/bin/env bash
# note (luojiaxuan): QuoteRecall 孪生改为共享请求(目标供应商/属性由 PAIR 种子给出,只有数字随 TWIN 变)后重采 Venus 前缀 16 题,
# 重建 Mail 规格,重跑 QuoteRecall 的矩阵(原生文本 + 无文本),完成即删执行器。装任务类会重启 p12/p13 的任务服务,须在模拟器空闲时执行。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot; EMUS=$(cat /data01/jaxan/.pilot_emus)
bash /data01/jaxan/install_pilot_tasks.sh sglang-omni-jaxan-p12 sglang-omni-jaxan-p13 2>&1 | grep -c "OK:" 
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41201 41211 41231 41241 41271)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim -p 172.17.0.1:$PORT:8000 \
  vllm/vllm-omni:dev --model /data01/jaxan/models/UI-Venus-2-9b --served-model-name UI-Venus-2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot QuoteRecall 共享请求版重采 + 矩阵(Venus);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) venus up ($C GPU$G:$PORT)"
TASKS=$(python3 -c "print(','.join(f'QuoteRecallTask{k:02d}{t}' for k in range(1,9) for t in 'AB'))")
AW=$(echo "$EMUS" | tr ',' '\n' | sed 's#^#http://127.0.0.1:#' | paste -sd,); NC=$(echo "$EMUS" | tr ',' '\n' | wc -l)
rm -rf $P/prefix_venus_qr2; mkdir -p $P/prefix_venus_qr2; cd /data01/jaxan/mw/MobileWorld
CC_VENUS_HIST=recent:2 PYTHONPATH=/data01/jaxan/pyshim timeout 7200 uv run mw eval --agent_type ui_venus2 --task "$TASKS" --max_round 40 --model_name UI-Venus-2 \
  --llm_base_url $E --api_key EMPTY --step_wait_time 3 --max-concurrency $NC --aw-host "$AW" --log_file_root $P/prefix_venus_qr2 > $P/prefix_venus_qr2.log 2>&1
echo "$(date -u +%FT%TZ) DONE qr2 dirs=$(ls -d $P/prefix_venus_qr2/*/ 2>/dev/null | wc -l) succ=$(grep -l "^score: 1" $P/prefix_venus_qr2/*/result.txt 2>/dev/null | wc -l)"
for c in sglang-omni-jaxan-p12 sglang-omni-jaxan-p13; do docker logs $c 2>&1 | grep -a "pair="; done | sed "s/.*- //" | sort -u > $P/task_seeds_qr2.txt
python3 /data01/jaxan/pilot_build_specs.py --prefix-dir $P/prefix_venus_qr2 --seeds $P/task_seeds_qr2.txt --backend venus --out $P/specs_venus_qr2.jsonl --contact $P/contact_venus_qr2 > $P/specs_venus_qr2.log 2>&1
grep -v "^    " $P/specs_venus_qr2.log | head -3
for proto in text notext; do extra=$([ "$proto" = notext ] && echo --no-text || true); rm -f $P/eval_venus_qr2_$proto.jsonl
  python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $P/specs_venus_qr2.jsonl --backend venus --base-url $E --model UI-Venus-2 --tag venus_qr2_$proto $extra --out $P/eval_venus_qr2_$proto.jsonl --workers 4 2>&1 | tee $P/eval_venus_qr2_$proto.txt; done
rm_own "$C"; echo PILOT_QR2_DONE
