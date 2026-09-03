#!/usr/bin/env bash
# note (luojiaxuan): 起 N 台 MemGUI 后端模拟器容器(sglang-omni-jaxan-mg_<i>,后端 6900+i),
# 与 docker run 同一动作登记 map。用法: memgui_pool_up.sh <count>
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
N=${1:-8}
cd /data01/jaxan/memgui
timeout 3600 uv run mg env run --count $N --name-prefix sglang-omni-jaxan-mg --backend-start-port 6900 --viewer-start-port 7900 --adb-start-port 5700 --launch-interval 20 > /data01/jaxan/rl_v2/memgui_env_run_$N.log 2>&1
for c in $(docker ps --format "{{.Names}}" | grep "^sglang-omni-jaxan-mg"); do
  grep -q "^$c	" "$HOME/jiaxuanluo-map.txt" || echo "$c	gpus=none	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl MemGUI-Bench 后端模拟器;调用方=本机 mg eval;保留至 2026-09-06 PT ⚠ 在用勿删;收尾:评测线结束 mg env rm" >> "$HOME/jiaxuanluo-map.txt"
done
docker ps --format "{{.Names}} {{.Status}}" | grep jaxan-mg
echo "map mg lines=$(grep -c "jaxan-mg" $HOME/jiaxuanluo-map.txt)"
