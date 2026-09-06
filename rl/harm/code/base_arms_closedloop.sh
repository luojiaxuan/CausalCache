#!/usr/bin/env bash
# note (luojiaxuan): 无偏的记忆余量口径——任务级 union-vs-best-single。冻结 GUI-Owl 底座在 117 题上按四种历史策略各跑一次闭环
# (recency-2 已有:31/117),这里补 recent0 / recent8 / random2(agent 只实现了 recent/random/learned 三种策略)。三臂共用一台服务,顺序跑(12 台模拟器是共享池,不能并行三臂)。
set -uo pipefail
source /data01/jaxan/cc_container_lib.sh
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
R=/data01/jaxan/rl_v2; E=http://172.17.0.1:41221/v1; C=""
TASKS=$(python3 -c "import json; s=json.load(open('/data01/jaxan/sglang-omni-rl/cc_recipe/fixtures/mw_split_v1.json')); print(','.join(sorted(set(s['train'])|set(s['heldout']))))")
G=""; until G=$(pick_gpu) && [ -n "$G" ]; do sleep 120; done
C=$(alloc_name); docker run -d --init --label cc.owner=$OWN --name $C --gpus "\"device=$G\"" --ipc=host --shm-size 16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:41221:8000 vllm/vllm-omni:dev --model /data04/jaxan/models/GUI-Owl-1.5-8B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":12}' >/dev/null
reg "$C" "$G" "sglang-omni-rl 底座 GUI-Owl 多臂闭环(recent0/recent8/random2,117 题,union 余量);⚠ 在用勿删;收尾:三臂结束删"
for t in $(seq 1 60); do curl -s -m 5 $E/models >/dev/null && break; sleep 20; done
curl -s -m 5 $E/models >/dev/null || { echo "SERVER_FAIL"; docker logs "$C" 2>&1 | grep -a -i "free memory\|error" | tail -2 | cut -c1-160; rm_own "$C"; exit 1; }
echo "$(date -u +%FT%TZ) gui-owl up on GPU$G"
for spec in "recent0:1:recent" "recent8:9:recent" "random2:3:random"; do
  arm=${spec%%:*}; rest=${spec#*:}; N=${rest%%:*}; POL=${rest#*:}; O=$R/guiowl_base_$arm; mkdir -p $O
  grep -q "Final:" $O.log 2>/dev/null && { echo "skip $arm"; continue; }
  echo "$(date -u +%FT%TZ) START $arm (CC_HISTORY_N=$N CC_FRAME_POLICY=$POL)"
  cd /data01/jaxan/mw/MobileWorld && CC_HISTORY_N=$N CC_FRAME_POLICY=$POL PYTHONPATH=/data01/jaxan/pyshim timeout 14400 \
    uv run mw eval --agent_type gui_owl_1_5 --task "$TASKS" --max_round 50 --model_name gui-owl --llm_base_url $E --api_key EMPTY \
    --step_wait_time 3 --max-concurrency 12 --aw-host "$(seq -s, -f 'http://127.0.0.1:68%02g' 0 11)" --log_file_root $O > $O.log 2>&1
  k=$(grep -l "^score: 1" $O/*/result.txt 2>/dev/null | wc -l); echo "$(date -u +%FT%TZ) DONE $arm succ_dirs=$k"
done
rm_own "$C"; echo BASE_ARMS_DONE
