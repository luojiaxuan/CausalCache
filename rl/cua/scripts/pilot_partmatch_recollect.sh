#!/usr/bin/env bash
# note (luojiaxuan): PartMatch 族的候选目录挪到 Pictures 之后重采前缀(Venus 主、GUI-Owl 异质性各一遍)。等 Venus 全量前缀链结束
# (模拟器与卡都空出)再装新任务类——重启任务服务会打断正在跑的 episode。每个 executor 起服务→跑 16 题→删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
EMUS=$(cat /data01/jaxan/.pilot_emus)
until grep -q PILOT_PREFIX_VENUS_DONE /data01/jaxan/pilot_prefix_venus.log 2>/dev/null; do sleep 60; done
bash /data01/jaxan/mw/MobileWorld/src/mobile_world/tasks/definitions/work/install_pilot_tasks.sh $(echo "$EMUS" | sed 's/6812/sglang-omni-jaxan-p12/;s/6813/sglang-omni-jaxan-p13/;s/,/ /')
TASKS=$(python3 -c "print(','.join(f'PartMatchTask{k:02d}{t}' for k in range(1,9) for t in 'AB'))")
AW=$(echo "$EMUS" | tr ',' '\n' | sed 's#^#http://127.0.0.1:#' | paste -sd,); NC=$(echo "$EMUS" | tr ',' '\n' | wc -l)
run_arm() {  # $1=agent_type $2=model path $3=served name $4=out dir $5=extra env
  local G="" PORT C E
  until G=$(pick_gpu) && [ -n "$G" ]; do sleep 60; done
  PORT=$(pick_port 41201 41211 41221 41231 41241 41271)
  C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
    -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
    -p 172.17.0.1:$PORT:8000 vllm/vllm-omni:dev --model "$2" --served-model-name "$3" --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}' >/dev/null || { echo "docker run failed"; rm_created_own; return 1; }
  reg "$C" "$G" "sglang-omni-rl B-pilot PartMatch 重采前缀($3);⚠ 在用勿删;收尾:采集结束删"
  E=http://172.17.0.1:$PORT/v1; for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
  curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL $3"; rm_own "$C"; return 1; }
  echo "$(date -u +%FT%TZ) $3 up ($C GPU$G:$PORT)"
  mkdir -p "$4"; cd /data01/jaxan/mw/MobileWorld
  env $5 PYTHONPATH=/data01/jaxan/pyshim timeout 7200 uv run mw eval --agent_type "$1" --task "$TASKS" --max_round 40 --model_name "$3" \
    --llm_base_url $E --api_key EMPTY --step_wait_time 3 --max-concurrency $NC --aw-host "$AW" --log_file_root "$4" > "$4.log" 2>&1
  echo "$(date -u +%FT%TZ) DONE $3 dirs=$(ls -d $4/*/ 2>/dev/null | wc -l) succ=$(grep -l "^score: 1" $4/*/result.txt 2>/dev/null | wc -l)"
  rm_own "$C"
}
run_arm ui_venus2 /data01/jaxan/models/UI-Venus-2-9b UI-Venus-2 /data01/jaxan/rl_v2/pilot/prefix_venus_pm2 CC_VENUS_HIST=recent:2
run_arm gui_owl_1_5 /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct gui-owl /data01/jaxan/rl_v2/pilot/prefix_base_pm2 "CC_HISTORY_N=3 CC_FRAME_POLICY=recent"
echo PILOT_PARTMATCH_RECOLLECT_DONE
