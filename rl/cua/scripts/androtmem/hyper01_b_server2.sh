#!/usr/bin/env bash
# note (luojiaxuan): vllm-omni:v0.28.0rc1 镜像无默认 serve 入口,显式用 python -m vllm.entrypoints.openai.api_server。
set -uo pipefail
N=sglang-omni-jaxan-rle32
docker rm -f $N > /dev/null 2>&1
docker run -d --gpus '"device=2"' --name $N --init --ipc=host --shm-size=16g \
  -v /data04/jaxan:/data04/jaxan -v /data04/jaxan/pyshim:/pyshim -e PYTHONPATH=/pyshim \
  -p 172.17.0.1:41051:8000 --entrypoint python3 vllm-omni:dev -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 --model /data04/jaxan/models/GUI-Owl-1.5-32B-Instruct --served-model-name gui-owl \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":4}' > /dev/null
for i in $(seq 1 80); do curl -s -m 3 http://172.17.0.1:41051/v1/models 2>/dev/null | grep -q gui-owl && { echo "$(date -u +%FT%TZ) 32B executor READY after ~$((i*15))s"; break; }; docker ps -q -f name=$N -f status=running | grep -q . || { echo "CONTAINER_EXITED"; docker logs --tail 15 $N 2>&1 | cut -c1-200; break; }; sleep 15; done
echo SERVER2_DONE
