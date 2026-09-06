#!/usr/bin/env bash
# note (luojiaxuan): 把 B-pilot 的任务类装进模拟器容器(任务注册表在容器内的 /app/service 拷贝里,不是宿主 checkout),
# 然后重启容器内的 mobile-world server。它由 entrypoint.sh 直接拉起(supervisord 只管 dockerd),杀掉后不会自动重生,
# 要用同样的命令(cd /app/service && uv run mobile-world server --port 6800,.env 已在该目录)手动再起。
# 用法:install_pilot_tasks.sh <容器名...>   只在该容器空闲时执行——重启任务服务会打断正在跑的 episode。
set -uo pipefail
SRC=/data01/jaxan/mw/MobileWorld/src/mobile_world/tasks/definitions/work
FILES="quote_recall.py order_address_join.py part_match.py"
for c in "$@"; do
  # 宿主端口:容器 6800/tcp 映射到的 HostPort(池里 p00 是 6800,p13 是 6813),不能用容器端口去 curl 宿主
  port=$(docker inspect "$c" --format '{{(index (index .NetworkSettings.Ports "6800/tcp") 0).HostPort}}')
  for f in $FILES; do docker cp $SRC/$f "$c":/app/service/src/mobile_world/tasks/definitions/work/$f; done
  docker exec "$c" mkdir -p /app/service/src/mobile_world/tasks/definitions/work/assets/pilot_parts
  docker cp $SRC/assets/pilot_parts/. "$c":/app/service/src/mobile_world/tasks/definitions/work/assets/pilot_parts/
  docker exec "$c" sh -c 'pkill -f "mobile-world server" ; true'; sleep 2
  docker exec -d "$c" sh -c 'cd /app/service && nohup uv run mobile-world server --port 6800 >> /var/log/server.log 2>&1'
  for t in $(seq 1 40); do sleep 3; curl -s -m 5 "http://127.0.0.1:${port:-6800}/" >/dev/null 2>&1 && break; done
  for t in QuoteRecallTask01A OrderAddressJoinTask01A PartMatchTask01A; do r=$(curl -s -m 10 "http://127.0.0.1:$port/task/goal?task_name=$t" | head -c 70); echo "$c (host port $port) $t: $(echo "$r" | grep -qi "error\|detail" && echo "FAIL: $r" || echo "OK: $r")"; done
  docker exec "$c" sh -c 'grep -a -i "Traceback\|Error\|quote_recall\|order_address\|part_match" /var/log/server.log | tail -6 | cut -c1-160'
done
