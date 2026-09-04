#!/usr/bin/env bash
# note (luojiaxuan): MemGUI 后端重建。根因:mg env run 起的容器 PID 1 是 tail -f(无 init),
# 模拟器崩溃后成 <defunct> 僵尸,容器内恢复脚本永远 in_progress;runner 侧又因显式 --aw-host
# 拿不到容器名而无法自愈。改为按 dry-run 原命令自起 16 台,加 --init;两组前缀 mga_/mgb_
# 互不为前缀,让两臂各自按前缀自动发现(不传 --aw-host),runner 得以重启模拟器/重建容器。
set -uo pipefail
IMG=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep memgui-bench | head -1)
echo "image=$IMG"
echo "=== 崩溃现场:mg_0 emulator.log 尾部关键行 ==="
docker exec sglang-omni-jaxan-mg_0 sh -c 'grep -aiE "crash|killed|segmentation|abort|fatal|error" /var/log/emulator.log | tail -6; echo ---; tail -3 /var/log/dockerd.err.log' 2>/dev/null | cut -c1-200
echo "=== 拆除 16 台 ==="
docker rm -f $(docker ps -aq -f name=sglang-omni-jaxan-mg) > /dev/null 2>&1
sed -i "/^sglang-omni-jaxan-mg/d" "$HOME/jiaxuanluo-map.txt"
launch() { # prefix idx backend viewer adb
  docker run -d --rm --init --privileged --ulimit nofile=524288:524288 --name "$1_$2" \
    -p $3:6800 -p $4:7860 -p $5:5556 -v /data01/jaxan/memgui/.env:/app/service/.env -e EMULATOR_TIMEOUT=1200 \
    --entrypoint /usr/local/bin/memgui-runtime-entrypoint.sh "$IMG" tail -f /var/log/emulator.log /var/log/server.log /var/log/dockerd.err.log > /dev/null
  echo "$1_$2	gpus=none	host=$(hostname)	created=$(date -u +%FT%TZ)	desc=sglang-omni-rl MemGUI-Bench 后端模拟器(--init;端口 $3);调用方=本机 mg eval;保留至 2026-09-06 PT ⚠ 在用勿删;收尾:评测线结束删" >> "$HOME/jiaxuanluo-map.txt"
}
for i in 0 1 2 3 4 5 6 7; do launch sglang-omni-jaxan-mga $i $((6900+i)) $((7900+i)) $((5700+i)); sleep 15; done
for i in 0 1 2 3 4 5 6 7; do launch sglang-omni-jaxan-mgb $i $((6908+i)) $((7908+i)) $((5708+i)); sleep 15; done
echo "launched: $(docker ps --format '{{.Names}}' | grep -c 'jaxan-mg[ab]_') map=$(grep -c 'jaxan-mg[ab]_' $HOME/jiaxuanluo-map.txt)"
docker inspect -f "{{.Name}} init={{.HostConfig.Init}}" sglang-omni-jaxan-mga_0
echo POOL_RESET_DONE
