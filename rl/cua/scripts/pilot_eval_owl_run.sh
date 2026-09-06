#!/usr/bin/env bash
# note (luojiaxuan): B-pilot 干预矩阵(GUI-Owl 后端,异质性臂):对给定规格文件起一台 GUI-Owl 执行器跑完即删。
# 用法:SPEC=<specs.jsonl> TAG=<tag> [AFTER=<log 文件里等待出现的标记>] bash pilot_eval_owl_run.sh
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot
[ -n "${AFTER:-}" ] && until grep -q "$AFTER" /data01/jaxan/pilot_eval_owl_smoke.log 2>/dev/null; do sleep 60; done
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT=$(pick_port 41221 41231 41241 41271 41281)
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:$PORT:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C" "$G" "sglang-omni-rl B-pilot 干预矩阵(GUI-Owl 后端,$TAG);⚠ 在用勿删;收尾:出数即删"
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) owl up ($C GPU$G:$PORT) spec=$SPEC tag=$TAG"
rm -f $P/eval_$TAG.jsonl
python3 /data01/jaxan/pilot_checkpoint_eval.py --spec $SPEC --backend owl --base-url $E --model gui-owl --tag $TAG \
  --out $P/eval_$TAG.jsonl --workers 4 2>&1 | tee $P/eval_$TAG.txt
rm_own "$C"; echo "PILOT_EVAL_OWL_DONE $TAG"
