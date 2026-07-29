#!/bin/bash
set -u
B=/data04/jaxan/osworld
pids=$(ps aux | grep 'bash /data04/jaxan/osworld/worker30.sh' | grep -v grep | awk '{print $2}')
[ -n "$pids" ] && kill -9 $pids 2>/dev/null
rm -f $B/run30/docker_vm_data/Ubuntu.qcow2.zip $B/run30/docker_vm_data/Ubuntu.qcow2
cd $B/run30/docker_vm_data
wget -q -c https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip -o Ubuntu.qcow2.zip >/dev/null && echo H01_VM_READY
CIP=$(docker inspect sglang-omni-jaxan --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
for s in 8 9 10 11 12 13 14 15; do
  nohup $B/worker30.sh $s http://$CIP:19280/act > /dev/null 2>&1 &
done
sleep 10
echo "H01_WORKERS=$(ps aux | grep -c '[r]un_osworld_benchmark_worker')"
