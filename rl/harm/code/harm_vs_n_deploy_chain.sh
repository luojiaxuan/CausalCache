#!/usr/bin/env bash
# note (luojiaxuan): 与 harm_vs_n_chain.sh 同样的 N 扫描,但用**部署忠实布局**(recN_deploy / irrN_deploy)。另起一张卡并行。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
H=/data01/jaxan/harm; E=http://172.17.0.1:41231/v1; C=""
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:41231:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null
reg "$C" "$G" "sglang-omni-rl history-harm:伤害 vs N(部署忠实布局);⚠ 在用勿删;收尾:出数即删"
for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; docker logs "$C" 2>&1 | grep -a -i "free memory\|error" | tail -2 | cut -c1-160; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G"
SPECS="rec0_deploy,rec1_deploy,rec2_deploy,rec3_deploy,rec4_deploy,rec6_deploy,irr1_deploy,irr2_deploy,irr4_deploy"
python3 /data01/jaxan/decode_ctx.py --base-url $E --tag base_deploy --specs $SPECS --out $H/harm_vs_n_base_deploy.jsonl --workers 12 > $H/harm_vs_n_base_deploy.log 2>&1
echo "$(date -u +%FT%TZ) deploy done: $(wc -l < $H/harm_vs_n_base_deploy.jsonl)"
rm_own "$C"; echo HARM_VS_N_DEPLOY_DONE
