#!/bin/bash
set -u
R=/data/jaxan/osworld-runner
pkill -9 -f worker30.sh 2>/dev/null; pkill -9 -f run_osworld_benchmark_worker 2>/dev/null; sleep 2
docker exec sglang-omni-jaxan bash -c '
R=/data/osworld-runner
mkdir -p $R/run/docker_vm_data
rm -f $R/run/docker_vm_data/Ubuntu.qcow2.zip $R/run/docker_vm_data/Ubuntu.qcow2
ln -f $R/runtime/docker_vm_data/Ubuntu.qcow2 $R/run/docker_vm_data/Ubuntu.qcow2
ln -f $R/runtime/docker_vm_data/Ubuntu.qcow2.zip $R/run/docker_vm_data/Ubuntu.qcow2.zip
chmod -R 777 $R/run
echo RUN_LINKED'
echo "--- serve state"
ps aux | grep "[s]erve30.sh" | wc -l
ps aux | grep "[s]erve_osworld_official" | wc -l
tail -2 $R/s19080.log 2>/dev/null
tail -2 $R/s19082.log 2>/dev/null
ls $R/logs30/*.log 2>/dev/null | head -3
