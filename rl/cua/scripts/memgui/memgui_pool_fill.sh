#!/usr/bin/env bash
# note (luojiaxuan): 补齐缺失的后端容器(逐个捕获 docker run 报错),然后按 docker ps 重建 map 中的 mg 行。
set -uo pipefail
IMG=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep memgui-bench | head -1)
launch() { docker run -d --rm --init --privileged --ulimit nofile=524288:524288 --name "$1_$2" \
    -p $3:6800 -p $4:7860 -p $5:5556 -v /data01/jaxan/memgui/.env:/app/service/.env -e EMULATOR_TIMEOUT=1200 \
    --entrypoint /usr/local/bin/memgui-runtime-entrypoint.sh "$IMG" tail -f /var/log/emulator.log /var/log/server.log /var/log/dockerd.err.log 2>&1 | tail -1 | cut -c1-160; }
for i in 0 1 2 3 4 5 6 7; do docker ps --format "{{.Names}}" | grep -q "^sglang-omni-jaxan-mga_$i$" || { echo "mga_$i: $(launch sglang-omni-jaxan-mga $i $((6900+i)) $((7900+i)) $((5700+i)))"; sleep 12; }; done
for i in 0 1 2 3 4 5 6 7; do docker ps --format "{{.Names}}" | grep -q "^sglang-omni-jaxan-mgb_$i$" || { echo "mgb_$i: $(launch sglang-omni-jaxan-mgb $i $((6908+i)) $((7908+i)) $((5708+i)))"; sleep 12; }; done
sleep 5
sed -i "/^sglang-omni-jaxan-mg[ab]_/d" "$HOME/jiaxuanluo-map.txt"
for c in $(docker ps --format "{{.Names}}" | grep "jaxan-mg[ab]_"); do p=$(docker port $c 6800 | head -1 | awk -F: "{print \$NF}"); echo "$c	gpus=none	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl MemGUI-Bench 后端模拟器(--init;端口 $p);调用方=本机 mg eval;保留至 2026-09-06 PT ⚠ 在用勿删;收尾:评测线结束删" >> "$HOME/jiaxuanluo-map.txt"; done
echo "running=$(docker ps --format '{{.Names}}' | grep -c 'jaxan-mg[ab]_') map=$(grep -c 'jaxan-mg[ab]_' $HOME/jiaxuanluo-map.txt)"
echo FILL_DONE
