#!/usr/bin/env bash
# note (luojiaxuan): 在**部署忠实布局**下重测三裁判帧对(底座 GUI-Owl,337 态)。主线"裁判≈无关"是标注器布局下测的,必须在此复核。
# 规格:9 个裁判 pick 集 + rec2 / irr2 / rec0(已有,重复以便同批)。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; R=/data01/jaxan/rl_v2; C=""
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
PORT=$(free_port); E=http://172.17.0.1:$PORT/v1
run_labeled --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:$PORT:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' || { echo "RUN_FAIL"; exit 1; }
reg "$C" "$G" "sglang-omni-rl history-harm:裁判帧对@部署布局(GUI-Owl 底座,port $PORT);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models | grep -q '"gui-owl"' && break; sleep 20; done
curl -s -m 60 $E/chat/completions -H "Content-Type: application/json" -d '{"model":"gui-owl","messages":[{"role":"user","content":"Say OK"}],"max_tokens":4}' | grep -q '"choices"' || { echo "SERVER_FAIL"; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G ($C)"
SPECS="rec0_deploy,rec2_deploy,irr2_deploy,pick:judge_glm46v|direct|full_deploy,pick:judge_glm46v|direct|six_deploy,pick:judge_glm46v|caption|full_deploy,pick:qwen38_27b|direct|full_deploy,pick:qwen38_27b|direct|six_deploy,pick:qwen38_27b|caption|full_deploy,pick:judge|direct|full_deploy,pick:judge|direct|six_deploy,pick:judge|caption|full_deploy"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_judge_deploy --specs "$SPECS" --picks "$R/picks_*.jsonl" --out $H/judge_deploy_base.jsonl --workers 12 > $H/judge_deploy_base.log 2>&1
echo "$(date -u +%FT%TZ) done: $(wc -l < $H/judge_deploy_base.jsonl)"; rm_own "$C"; echo JUDGE_DEPLOY_DONE
