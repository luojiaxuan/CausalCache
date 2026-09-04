#!/usr/bin/env bash
# note (luojiaxuan): 补齐"公平测试"两组:
#  (1) 原版 8B(GPU1 现有服务)在弱/无文本下 recent+gold 是否涨:rec1_gold1_notext / rec2_gold1_notext;
#  (2) iter_59(P3 用随机两帧历史训过的 executor,GPU2 新起 vLLM)复跑关键 6 条件。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan/mw/MobileWorld
PYTHONPATH=/data01/jaxan/pyshim setsid uv run python /data01/jaxan/androtmem_probe.py --workers 16 --conds rec1_gold1_notext,rec2_gold1_notext --out /data01/jaxan/androtmem/probe/results.jsonl > /data01/jaxan/androtmem/probe/probe3.log 2>&1 < /dev/null &
N=sglang-omni-jaxan-rle59
docker rm -f $N > /dev/null 2>&1
docker run -d --gpus '"device=2"' --name $N --init --ipc=host --shm-size=16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:41042:8000 vllm/vllm-omni:dev --model /data01/jaxan/sglang-omni-rl/cc_recipe/run_full/hf_p3/iter_59 \
  --served-model-name gui-owl --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":4}' > /dev/null
echo "$N	gpus=2	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl 探针 executor 推理(iter_59,公平对照);调用方=本机探针;保留至 2026-09-05 PT ⚠ 在用勿删;收尾:探针结束删" >> "$HOME/jiaxuanluo-map.txt"
for i in $(seq 1 60); do curl -s -m 3 http://172.17.0.1:41042/v1/models 2>/dev/null | grep -q gui-owl && { echo "iter_59 READY after ~$((i*10))s"; break; }; sleep 10; done
mkdir -p /data01/jaxan/androtmem/probe59
PYTHONPATH=/data01/jaxan/pyshim setsid uv run python /data01/jaxan/androtmem_probe.py --workers 16 --llm http://172.17.0.1:41042/v1 --conds none,recency2,gold,rec1_gold1,recency2_notext,rec1_gold1_notext --out /data01/jaxan/androtmem/probe59/results.jsonl > /data01/jaxan/androtmem/probe59/probe.log 2>&1 < /dev/null &
sleep 30; grep -E "jobs=" /data01/jaxan/androtmem/probe/probe3.log /data01/jaxan/androtmem/probe59/probe.log
echo FAIR_PROBES_LAUNCHED
