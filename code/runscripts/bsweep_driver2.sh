#!/bin/bash
# note (luojiaxuan): B 扫描驱动 v2。v1 把 18 个 env 全指向 58900,58901 零流量、GPU5 空转
# (实测 p0.log 523 条请求 / p1.log 0 条,GPU4 99% / GPU5 0%)。runner 的 --policy-endpoint
# 是单值,所以改为每个配置并行跑两个 runner,各带一半任务与一半模拟器,分别打两台服务器。
# 配置级续跑靠 DONE 文件,已完成的 b0-recent 会被跳过(它的输出仍是旧的单层 trajectories/)。
set -u
BUD="0 1 2 4 6 8"
R=/data/mw/runs/bsweep
CFG=/data/CausalCache-mwb0/code/configs/causalcache_mobileworld_official_b4_hgkv_sel_v1.json
SEL=/data/runs/selector-v4/arm-twotower/marginal_scorer.pt
LOCK=/tmp/bsweep.lock

exec 9>$LOCK
flock -n 9 || { echo "另一个驱动已在运行,退出"; exit 1; }

mkdir -p $R
for b in $BUD; do
  for arm in recent sel; do
    [ "$b" = "0" ] && [ "$arm" = "sel" ] && continue
    OUT=$R/b${b}-${arm}
    mkdir -p $OUT
    [ -f $OUT/DONE ] && { echo "skip b$b-$arm"; continue; }

    for p in 58900 58901; do
      pids=$(ps aux | grep -- "--port $p" | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill -9 $pids
      pids=$(ps aux | grep "bsweep_serve.sh $p" | grep -v grep | awk '{print $2}'); [ -n "$pids" ] && kill -9 $pids
    done
    sleep 3
    for i in 0 1; do
      G=$((4 + i)); P=$((58900 + i))
      nohup bash /data/mw/tools/bsweep_serve.sh $P $G $b $arm $SEL $OUT/p$i.log > /dev/null 2>&1 &
    done
    for k in $(seq 1 60); do
      u=0; for p in 58900 58901; do curl -sm 3 http://127.0.0.1:$p/health >/dev/null 2>&1 && u=$((u+1)); done
      [ "$u" -eq 2 ] && break; sleep 20
    done
    echo "b=$b arm=$arm servers=$u (需要 2)"
    [ "$u" -ne 2 ] && { echo "b=$b arm=$arm ABORT: 服务器未就绪"; continue; }

    cd /data/CausalCache-mwb0
    # note (luojiaxuan): 两半并行。a 半 -> 58900/GPU4,b 半 -> 58901/GPU5。
    for half in a b; do
      [ "$half" = a ] && EP=58900 || EP=58901
      mkdir -p $OUT/$half
      PYTHONPATH=code uv run --project /data/mw/upstreams/MobileWorld \
        python code/scripts/run_mobileworld_gui_owl.py \
        --repository-root /data/CausalCache-mwb0 \
        --mobileworld-root /data/mw/upstreams/MobileWorld \
        --fleet-manifest $R/fleet-$half.json \
        --policy-endpoint http://127.0.0.1:$EP \
        --output-root $OUT/$half --profile full --num-envs 9 \
        --task-subset $R/subset-$half.json --config $CFG >> $OUT/$half/runner.log 2>&1 &
      eval "PID_$half=$!"
    done
    wait $PID_a; RC_a=$?
    wait $PID_b; RC_b=$?
    echo "b=$b arm=$arm exit_a=$RC_a exit_b=$RC_b"
    n=$(find $OUT -name result.txt 2>/dev/null | wc -l)
    echo "b=$b arm=$arm results=$n/36"
    # note (luojiaxuan): 只有两半都正常退出才标 DONE,否则留给下次续跑重做该配置。
    if [ "$RC_a" -eq 0 ] && [ "$RC_b" -eq 0 ]; then touch $OUT/DONE; else echo "b=$b arm=$arm NOT marked DONE"; fi
  done
done
echo BSWEEP_ALL_DONE
