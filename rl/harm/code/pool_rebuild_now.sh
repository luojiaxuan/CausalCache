#!/usr/bin/env bash
# note (luojiaxuan): 先重建不在用的 p14–p33(20 台),不碰正在被闭环臂使用的 p00–p11 和他人的 p12/p13。
set -uo pipefail
source /data01/jaxan/pool_lib.sh
for i in $(seq 14 33); do mk_pool $i && sleep 8; done
echo "launched: $(docker ps --format '{{.Names}}' | grep -c 'jaxan-p[0-9]')"
wait_pool "$(seq -s' ' 14 33)"
echo POOL_REBUILD_NOW_DONE
