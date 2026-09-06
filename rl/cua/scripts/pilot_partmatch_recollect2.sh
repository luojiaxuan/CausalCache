#!/usr/bin/env bash
# note (luojiaxuan): PartMatch(候选目录已挪到 Pictures)重采:先用已起好的 Venus 执行器(C/PORT/OWN 由调用方给出)跑 16 题并删容器,
# 再起一台 GUI-Owl 跑同 16 题。任务类须已装进 p12/p13(install_pilot_tasks.sh)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
EMUS=$(cat /data01/jaxan/.pilot_emus)
TASKS=$(python3 -c "print(','.join(f'PartMatchTask{k:02d}{t}' for k in range(1,9) for t in 'AB'))")
AW=$(echo "$EMUS" | tr ',' '\n' | sed 's#^#http://127.0.0.1:#' | paste -sd,); NC=$(echo "$EMUS" | tr ',' '\n' | wc -l)
collect() {  # $1=agent_type $2=served name $3=endpoint $4=out dir $5=extra env
  mkdir -p "$4"; cd /data01/jaxan/mw/MobileWorld
  env $5 PYTHONPATH=/data01/jaxan/pyshim timeout 7200 uv run mw eval --agent_type "$1" --task "$TASKS" --max_round 40 --model_name "$2" \
    --llm_base_url "$3" --api_key EMPTY --step_wait_time 3 --max-concurrency $NC --aw-host "$AW" --log_file_root "$4" > "$4.log" 2>&1
  echo "$(date -u +%FT%TZ) DONE $2 dirs=$(ls -d $4/*/ 2>/dev/null | wc -l) succ=$(grep -l "^score: 1" $4/*/result.txt 2>/dev/null | wc -l)"
}
E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL venus"; rm_own "$C"; exit 1; }
rm -rf /data01/jaxan/rl_v2/pilot/prefix_venus_pm2
collect ui_venus2 UI-Venus-2 $E /data01/jaxan/rl_v2/pilot/prefix_venus_pm2 CC_VENUS_HIST=recent:2
rm_own "$C"
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
PORT2=$(pick_port 41221 41231 41241 41271 41281)
C2=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C2 --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:$PORT2:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; exit 1; }
reg "$C2" "$G" "sglang-omni-rl B-pilot PartMatch 重采前缀(GUI-Owl);⚠ 在用勿删;收尾:采集结束删"
E2=http://172.17.0.1:$PORT2/v1; for t in $(seq 1 60); do curl -s -m 5 $E2/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E2/models >/dev/null || { echo "SERVER_FAIL owl"; rm_own "$C2"; exit 1; }
echo "$(date -u +%FT%TZ) owl up ($C2 GPU$G:$PORT2)"
rm -rf /data01/jaxan/rl_v2/pilot/prefix_base_pm2
collect gui_owl_1_5 gui-owl $E2 /data01/jaxan/rl_v2/pilot/prefix_base_pm2 "CC_HISTORY_N=3 CC_FRAME_POLICY=recent"
rm_own "$C2"; echo PILOT_PARTMATCH_RECOLLECT_DONE
