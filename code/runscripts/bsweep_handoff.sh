#!/bin/bash
# note (luojiaxuan): 等 b0-recent 跑完(v1 驱动会写 DONE),然后把 v1 换成 v2。
# 不提前杀 v1,是为了不浪费当前配置已完成的任务;v1 写完 DONE 后可能已经抢跑下一个配置,
# 所以这里连 runner 一起杀干净再交接——那时下一个配置刚起步,损失可忽略。
set -u
LOG=/data02/jaxan/bsweep_handoff.log
exec >>$LOG 2>&1
echo "=== handoff watcher started $(date -Is) ==="
while true; do
  if docker exec sglang-omni-jaxan test -f /data/mw/runs/bsweep/b0-recent/DONE 2>/dev/null; then
    echo "$(date -Is) b0-recent DONE 出现,开始交接"
    pkill -f 'bsweep_driver.sh' && echo "killed v1 driver"
    sleep 2
    pkill -f 'run_mobileworld_gui_owl.py' && echo "killed in-flight runner(s)"
    sleep 5
    for p in 58900 58901; do
      pids=$(ps aux | grep -- "--port $p" | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill -9 $pids
      pids=$(ps aux | grep "bsweep_serve.sh $p" | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill -9 $pids
    done
    sleep 5
    # note (luojiaxuan): v1 若已抢跑下一个配置,其半成品目录要清掉,免得与 v2 的 a/b 布局混在一起。
    docker exec sglang-omni-jaxan bash -lc 'for d in /data/mw/runs/bsweep/b*-*/; do [ -f "$d/DONE" ] || { [ "$(basename $d)" = "b0-recent" ] || rm -rf "$d"; }; done; ls /data/mw/runs/bsweep/'
    setsid nohup bash /tmp/bsweep_driver2.sh >> /data02/jaxan/bsweep_v2.log 2>&1 < /dev/null &
    echo "$(date -Is) v2 driver launched pid=$!"
    sleep 60
    pgrep -af bsweep_driver2.sh || echo "WARN: v2 driver not found after launch"
    exit 0
  fi
  sleep 60
done
