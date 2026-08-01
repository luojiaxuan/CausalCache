#!/bin/bash
# note (luojiaxuan): 等 b0-recent 跑完(v1 驱动会写 DONE),然后把 v1 换成 v2。
#
# 【本脚本的初版在生产中失败过,三个坑都已修,改动前务必读】
# 1. 驱动与 runner 是**容器内的 root 进程**。宿主上以 sglang-omni 身份 pkill 它们
#    只会拿到 EPERM,静默失败——v1 因此存活。所有杀进程/起进程都必须走
#    `docker exec`(容器内是 root)。
# 2. 本脚本没有 set -e,而 pkill 失败不中断,于是它继续执行清理;清理那步用的是
#    docker exec(有 root),把没有 DONE 的 b1-recent 真删了,却没能换掉驱动——
#    造成"删了目录但旧驱动还在跑"的最坏中间态。
# 3. `exec >>LOG` 指向不可写路径时**不会**让脚本退出(实测打印后续行、exit=0),
#    只是重定向失败、输出继续走原 stdout。配合 `>/dev/null` 启动就成了全程静默。
#    日志路径必须先验证可写:/data02/jaxan 对 sglang-omni 不可写,容器内 /data 才行。
set -u
LOG=/data/mw/runs/bsweep/handoff.log   # 容器内路径,root 可写
exec >>"$LOG" 2>&1 || { echo "FATAL: 日志路径不可写 $LOG" >&2; exit 1; }
echo "=== handoff watcher started $(date -Is) ==="
while true; do
  if docker exec sglang-omni-jaxan test -f /data/mw/runs/bsweep/b0-recent/DONE 2>/dev/null; then
    echo "$(date -Is) b0-recent DONE 出现,开始交接"
    # note (luojiaxuan): 全部走 docker exec,否则杀不动容器内的 root 进程。
    # 模式用 [b] 字符类,避免 pkill 匹配到承载它的那条命令行本身(会自杀)。
    docker exec sglang-omni-jaxan bash -lc '
      pkill -f "[b]sweep_driver.sh"; sleep 2
      pkill -f "[r]un_mobileworld_gui_owl.py"; sleep 5
      for p in 58900 58901; do
        ps aux | grep -- "[-]-port $p"        | awk "{print \$2}" | xargs -r kill -9
        ps aux | grep "[b]sweep_serve.sh $p"  | awk "{print \$2}" | xargs -r kill -9
      done'
    sleep 5
    # note (luojiaxuan): 确认真的杀干净了再往下走,绝不能在旧驱动还活着时做清理。
    if docker exec sglang-omni-jaxan pgrep -f "[b]sweep_driver.sh" >/dev/null 2>&1; then
      echo "FATAL: v1 驱动仍存活,放弃交接(避免删目录却换不掉驱动的最坏中间态)"
      exit 1
    fi
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
