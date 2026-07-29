#!/bin/bash
for port in 58901 58902; do
  pids=$(ps aux | grep -- "--port $port" | grep -v grep | awk '{print $2}')
  [ -n "$pids" ] && kill -9 $pids
done
pids=$(ps aux | grep 'serve_32b.sh 5890[12]' | grep -v grep | awk '{print $2}')
[ -n "$pids" ] && kill $pids
sleep 2
for s in 0 1 2 3; do
  case $s in 0) P=58903;; 1) P=58907;; 2) P=58908;; 3) P=58909;; esac
  nohup bash /data/mw/tools/supervise_shard_cc4.sh 4 $s 6 $P /data/mw/runs/mw-32b-recent-v2 8 > /data/mw/runs/mw-32b-recent-v2/supervisor-shard-$s.log 2>&1 &
done
sleep 5
ps aux | grep -c '[s]upervise_shard_cc4.sh 4'
curl -sm 3 http://127.0.0.1:58901/health >/dev/null && echo 58901 STILL_UP || echo 58901 DEAD
curl -sm 3 http://127.0.0.1:58902/health >/dev/null && echo 58902 STILL_UP || echo 58902 DEAD
