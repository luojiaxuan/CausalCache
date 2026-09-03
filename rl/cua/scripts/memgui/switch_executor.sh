#!/usr/bin/env bash
# note (luojiaxuan): 把 executor 推理容器 rle 切到指定权重(如原版 GUI-Owl-1.5-8B-Instruct,
# 与 MemGUI 榜单可比)。用法: switch_executor.sh <model_dir>。前提:无在途评测。
set -uo pipefail
M=$1
GE=$(docker inspect -f "{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}}{{end}}{{end}}" sglang-omni-jaxan-rle)
docker rm -f -v sglang-omni-jaxan-rle > /dev/null
docker run -d --gpus "\"device=$GE\"" --name sglang-omni-jaxan-rle --init --ipc=host --shm-size=16g \
  -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim \
  -e PYTHONPATH=/pyshim -p 172.17.0.1:41041:8000 \
  vllm/vllm-omni:dev --model "$M" --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":4}' > /dev/null
sed -i "/^sglang-omni-jaxan-rle\t/d" "$HOME/jiaxuanluo-map.txt"
echo "sglang-omni-jaxan-rle	gpus=$GE	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl executor推理($(basename $M));调用方=本机 mg eval/mw eval ⚠ 在用勿删;收尾:评测线结束删" >> "$HOME/jiaxuanluo-map.txt"
for i in $(seq 1 90); do curl -s -m 3 http://172.17.0.1:41041/v1/models 2>/dev/null | grep -q gui-owl && { echo "executor READY ($M) after ~$((i*10))s"; break; }; sleep 10; done
