#!/usr/bin/env bash
# note (luojiaxuan): 移除冒烟用的单台后端,后台起 8 台(名字 sglang-omni-jaxan-mg_0..7,
# 后端 6900-6907),完成后登记 map。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan/memgui
timeout 120 uv run mg env rm --name-prefix sglang-omni-jaxan-mg > /dev/null 2>&1 || docker rm -f sglang-omni-jaxan-mg_0 > /dev/null 2>&1
sed -i "/^sglang-omni-jaxan-mg_/d" "$HOME/jiaxuanluo-map.txt"
setsid bash /data01/jaxan/memgui_pool_up.sh 8 > /data01/jaxan/rl_v2/memgui_pool_up.log 2>&1 < /dev/null &
sleep 30
docker ps --format "{{.Names}} {{.Status}}" | grep jaxan-mg | head -3
echo POOL_RELAUNCH_SUBMITTED
