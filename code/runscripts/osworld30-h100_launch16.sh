#!/bin/bash
pids=$(ps aux | grep 'bash /data/jaxan/osworld-runner/worker30.sh' | grep -v grep | awk '{print $2}')
[ -n "$pids" ] && kill -9 $pids 2>/dev/null
pids=$(ps aux | grep '[r]un_osworld_benchmark_worker' | awk '{print $2}')
[ -n "$pids" ] && kill -9 $pids 2>/dev/null
sleep 2
sed -i 's/--shard-count 8/--shard-count 16/' /data/jaxan/osworld-runner/worker30.sh
CIP=$(docker inspect sglang-omni-jaxan --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
for sh in $(seq 0 15); do
  nohup /data/jaxan/osworld-runner/worker30.sh sel $sh http://$CIP:19080/act /data/jaxan/osworld-runner/output/verified30-sel > /dev/null 2>&1 &
done
sleep 15
ps aux | grep -c '[r]un_osworld_benchmark_worker'
