#!/usr/bin/env bash
# note (luojiaxuan): PartMatch(Pictures 布局)的 GUI-Owl 前缀重采:等 Venus 版 16 题齐后先删掉本线的 Venus 采集容器(VC,按名删除,
# 保持账号占卡数不变),再起一台 GUI-Owl 跑同 16 题,跑完删。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
source /data01/jaxan/cc_container_lib.sh
P=/data01/jaxan/rl_v2/pilot; EMUS=$(cat /data01/jaxan/.pilot_emus)
until [ "$(ls $P/prefix_venus_pm2/*/result.txt 2>/dev/null | wc -l)" -ge 16 ]; do sleep 30; done
sleep 20; docker rm -f "$VC" >/dev/null 2>&1; sed -i "/^$VC\t/d" "$MAP"; echo "$(date -u +%FT%TZ) removed $VC"
TASKS=$(python3 -c "print(','.join(f'PartMatchTask{k:02d}{t}' for k in range(1,9) for t in 'AB'))")
AW=$(echo "$EMUS" | tr ',' '\n' | sed 's#^#http://127.0.0.1:#' | paste -sd,); NC=$(echo "$EMUS" | tr ',' '\n' | wc -l)
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
rm -rf $P/prefix_base_pm2; mkdir -p $P/prefix_base_pm2; cd /data01/jaxan/mw/MobileWorld
CC_HISTORY_N=3 CC_FRAME_POLICY=recent PYTHONPATH=/data01/jaxan/pyshim timeout 7200 uv run mw eval --agent_type gui_owl_1_5 --task "$TASKS" --max_round 40 \
  --model_name gui-owl --llm_base_url $E2 --api_key EMPTY --step_wait_time 3 --max-concurrency $NC --aw-host "$AW" --log_file_root $P/prefix_base_pm2 > $P/prefix_base_pm2.log 2>&1
echo "$(date -u +%FT%TZ) DONE owl dirs=$(ls -d $P/prefix_base_pm2/*/ 2>/dev/null | wc -l) succ=$(grep -l "^score: 1" $P/prefix_base_pm2/*/result.txt 2>/dev/null | wc -l)"
rm_own "$C2"; echo PILOT_PARTMATCH_OWL_DONE
