#!/usr/bin/env bash
# note (luojiaxuan): Stage B 行为标签发射。4×vLLM(iter_59,image 上限 7
# 以容纳四元组 5 图),4 shard,每 shard 重试 3 次,末尾 teardown+rebuild_map。
set -uo pipefail
cd /data01/jaxan
GPUS=(2 4)
for g in "${GPUS[@]}"; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -gt 3000 ] && { echo "ABORT: GPU$g 非空(${u}MiB)"; exit 1; }
done
for i in 0 1; do
  N=sglang-omni-jaxan-$((i+1)); P=$((41021+i)); G=${GPUS[$i]}
  docker inspect $N >/dev/null 2>&1 && docker rm -f -v $N >/dev/null 2>&1
  docker run -d --gpus "\"device=$G\"" --name $N --init --ipc=host --shm-size=16g \
    -v /data01/jaxan:/data01/jaxan -v /data04/jaxan:/data04/jaxan -v /data01/jaxan/pyshim:/pyshim \
    -e PYTHONPATH=/pyshim -p 127.0.0.1:$P:8000 \
    vllm/vllm-omni:dev --model /data01/jaxan/sglang-omni-rl/cc_recipe/run_full/hf_p3/iter_59 \
    --served-model-name probe \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":7}' > /dev/null
  echo "$N	gpus=$G	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl StageB行为标签;收尾:标完删" >> "$HOME/jiaxuanluo-map.txt"
done
for i in 0 1; do
  P=$((41021+i))
  for t in $(seq 1 120); do
    curl -s -m 3 http://127.0.0.1:$P/v1/models 2>/dev/null | grep -q probe && { echo "[$(date +%H:%M)] :$P READY"; break; }
    sleep 10
  done
  curl -s -m 3 http://127.0.0.1:$P/v1/models 2>/dev/null | grep -q probe || { echo "ABORT: :$P 未就绪"; exit 1; }
done
run_shard () {
  for try in 1 2 3; do
    python3 /data01/jaxan/behav_label_full.py --base-url http://127.0.0.1:$2/v1 \
      --out /data01/jaxan/behav_labels_shard$1.jsonl --shard $1 --n-shards 2 \
      --exclude /data01/jaxan/behav_pilot_shard0.jsonl,/data01/jaxan/behav_pilot_shard1.jsonl \
      >> /data01/jaxan/behav_labels_shard$1.log 2>&1 && break
    echo "[retry $try]" >> /data01/jaxan/behav_labels_shard$1.log
    sleep 20
  done
}
for i in 0 1; do run_shard $i $((41021+i)) & done
wait
echo "STAGEB_DONE"
for i in 1 2; do docker rm -f -v sglang-omni-jaxan-$i >/dev/null 2>&1; done
bash /data01/jaxan/sglang-omni-rl/rebuild_map.sh >/dev/null 2>&1
echo "SB_TEARDOWN_DONE"
