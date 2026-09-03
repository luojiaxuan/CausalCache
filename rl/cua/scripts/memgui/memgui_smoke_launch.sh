#!/usr/bin/env bash
# note (luojiaxuan): MemGUI-Bench 首个后端容器(1 台模拟器)冒烟:env check → env run →
# 立即登记 map(与 docker run 同一动作)→ 等就绪。容器名走 sglang-omni-jaxan-mg_<i>。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan/memgui
echo "=== env check(无 sudo) ==="; timeout 120 uv run mg env check 2>&1 | tr -d "│╭╰─╮╯" | grep -vE "^\s*$" | tail -8 | cut -c1-120
echo "=== eval --help 里与 agent 配置相关的项 ==="; timeout 60 uv run mg eval --help 2>&1 | grep -iE "config|history|runtime|agent" | cut -c1-120 | head -8
echo "=== 启动 1 台后端 ==="
timeout 1800 uv run mg env run --count 1 --name-prefix sglang-omni-jaxan-mg --backend-start-port 6900 --viewer-start-port 7900 --adb-start-port 5700 --launch-interval 5 > /data01/jaxan/rl_v2/memgui_env_run.log 2>&1 &
sleep 20
docker ps --format "{{.Names}}\t{{.Status}}" | grep "jaxan-mg" 
for c in $(docker ps --format "{{.Names}}" | grep "^sglang-omni-jaxan-mg"); do
  grep -q "^$c" "$HOME/jiaxuanluo-map.txt" || echo "$c	gpus=none	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl MemGUI-Bench 后端模拟器(port 6900);调用方=本机 mg eval;保留至 2026-09-06 PT;收尾:评测线结束 mg env rm" >> "$HOME/jiaxuanluo-map.txt"
done
echo "map lines for mg: $(grep -c "jaxan-mg" $HOME/jiaxuanluo-map.txt)"
echo LAUNCH_SUBMITTED
