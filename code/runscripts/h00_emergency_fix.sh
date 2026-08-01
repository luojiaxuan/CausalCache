#!/bin/bash
# note (luojiaxuan): hyper00 内存告急抢修 + 收缩到 4 卡。
# 起因:s50 跑完后我复用了它的策略服务器接 s100,没重启;这些进程连跑十几小时后
# 宿主 RSS 涨到 150-244GB/个,八个合计约 1.25TB,把 2TB 的机器吃到只剩 17GB,
# sshd fork 不出来,别人连不上。端口 19210 只占 2GB(它中途自动重启过)——
# 直接证明这是随运行时长累积的泄漏,重启即可回收。
#
# 目标布局:GPU0/GPU1 = s100 两台服务器,GPU4/GPU5 = B 扫描(不动),共 4 卡。
set -u
echo "=== [0] 抢修前 ==="
free -g | head -2

echo "=== [1] 停 OSWorld 监督循环(否则杀了会自动重起) ==="
pkill -f "[h]00_osw_serve.sh" 2>/dev/null; echo "host supervisor: $?"
docker exec sglang-omni-jaxan bash -lc 'pkill -f "[s]erve30_recent.sh"; pkill -f "[o]sworld_serve_loop_h00.sh"; true'

echo "=== [2] 停 s100 workers(稍后按新端点重启;worker 跳过已完成任务) ==="
pkill -f "[r]un_osworld_benchmark_worker" 2>/dev/null
pkill -f "[o]sw_worker_gen.sh" 2>/dev/null
sleep 3

echo "=== [3] 杀掉全部 OSWorld 策略服务器(容器内 root,必须 docker exec) ==="
docker exec sglang-omni-jaxan bash -lc 'pkill -f "[s]erve_osworld_official_policy"; true'
sleep 15

echo "=== [4] 停掉不在 B 扫描 fleet 里的闲置模拟器 ==="
IN_USE=$(docker exec sglang-omni-jaxan bash -lc 'python3 -c "
import json
n=[]
for f in (\"fleet-a\",\"fleet-b\"):
    d=json.load(open(\"/data/mw/runs/bsweep/%s.json\"%f))
    n+=[c[\"name\"] for c in d[\"containers\"]]
print(\" \".join(n))"')
IDLE=""
for c in $(docker ps --format '{{.Names}}' | grep 'sglang-omni-jaxan-mw'); do
  echo "$IN_USE" | grep -qw "$c" || IDLE="$IDLE $c"
done
echo "在用 $(echo $IN_USE | wc -w) 个,停掉闲置 $(echo $IDLE | wc -w) 个"
[ -n "$IDLE" ] && docker stop -t 5 $IDLE >/dev/null 2>&1
sleep 5

echo "=== [5] 回收结果 ==="
free -g | head -2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

echo "=== [6] 只在 GPU0/GPU1 重起两台 s100 服务器 ==="
docker exec sglang-omni-jaxan bash -lc 'cat > /data/osworld/serve_s100.sh <<"EOS"
#!/usr/bin/env bash
# note (luojiaxuan): s100 臂服务器(HGKV-taught selector),带自动重启循环。
set -uo pipefail
DEV=$1; PORT=$2; LOG=$3
cd /data/CausalCache
while true; do
  CUDA_VISIBLE_DEVICES=$DEV PYTHONPATH=code python3 code/scripts/serve_osworld_official_policy.py \
    --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
    --snapshot-manifest code/configs/gui_owl_1_5_8b_snapshot.json \
    --device cuda:0 --port "$PORT" --visual-tokens 2560 --memory-budget 4 \
    --selector-bundle /data/runs/selector-v4/arm-twotower/marginal_scorer.pt \
    --selector-arch two_tower --selector-beam 3 >> "$LOG" 2>&1
  echo REPLICA_EXIT_$? >> "$LOG"
  sleep 10
done
EOS
chmod +x /data/osworld/serve_s100.sh; mkdir -p /data/osworld/logs30'
docker exec -d sglang-omni-jaxan bash -lc 'bash /data/osworld/serve_s100.sh 0 19200 /data/osworld/logs30/s100-19200.log'
docker exec -d sglang-omni-jaxan bash -lc 'bash /data/osworld/serve_s100.sh 1 19201 /data/osworld/logs30/s100-19201.log'

echo "=== [7] 等两台就绪 ==="
for k in $(seq 1 40); do
  u=0
  for p in 19200 19201; do curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1 && u=$((u+1)); done
  [ "$u" -eq 2 ] && break
  sleep 20
done
echo "servers_up=$u"

echo "=== [8] 重起 12 个 s100 分片,分摊到两台 ==="
if [ "$u" -eq 2 ]; then
  B=/data02/jaxan/osworld
  for s in 0 1 2 3 4 5 6 7 8 9 10 11; do
    P=$((19200 + s % 2))
    setsid nohup bash $B/osw_worker_gen.sh $s http://127.0.0.1:$P/act 100 $B/output-s100-hgkvsel 12 s100 \
      >/dev/null 2>&1 </dev/null &
  done
  sleep 10
  echo "workers=$(ps aux | grep -c '[r]un_osworld_benchmark_worker')"
else
  echo "服务器未就绪,未启动 worker"
fi

echo "=== [9] 最终状态 ==="
free -g | head -2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "占用中的 GPU 数(应为 4):"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0 > 1000) n++} END {print n+0}'
