#!/bin/bash
# note (luojiaxuan): 监控剩余两个在跑的作业:OSWorld s100(12 分片)与 MobileWorld B 扫描。
# s50/ft30 已跑满 361 并归约入库,从监控里摘掉。完成信号用 latch 只发一次,避免每轮重复。
# 停摆判据 = done 计数两轮不动 且 最新产出文件超过 STALE 秒没更新。
STALE=1800
POLL=600
tick=0
prev_s100=-1; prev_bs=-1
stall_s100=0; stall_bs=0
latch_s100=0; latch_bs=0

while true; do
  tick=$((tick+1))

  s100=$(ssh -o ConnectTimeout=20 hyper00 'ls -1 /data02/jaxan/osworld/output-s100-hgkvsel/*/*/result.json 2>/dev/null | wc -l' 2>/dev/null || echo -1)
  w100=$(ssh -o ConnectTimeout=20 hyper00 'ps aux | grep -c "[r]un_osworld_benchmark_worker"' 2>/dev/null || echo -1)

  bs=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "find /data/mw/runs/bsweep -name result.txt 2>/dev/null | wc -l"' 2>/dev/null || echo -1)
  bcfg=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "ls -d /data/mw/runs/bsweep/b*/DONE 2>/dev/null | wc -l"' 2>/dev/null || echo -1)
  bdrv=$(ssh -o ConnectTimeout=20 hyper00 'pgrep -f bsweep_driver >/dev/null && echo up || echo DOWN' 2>/dev/null || echo "?")
  bhand=$(ssh -o ConnectTimeout=20 hyper00 'pgrep -f bsweep_handoff >/dev/null && echo waiting || echo done' 2>/dev/null || echo "?")

  # --- 停摆:计数不动才去查文件年龄,省 ssh ---
  if [ "$s100" = "$prev_s100" ] && [ "$s100" != "-1" ]; then stall_s100=$((stall_s100+1)); else stall_s100=0; fi
  if [ "$bs" = "$prev_bs" ] && [ "$bs" != "-1" ]; then stall_bs=$((stall_bs+1)); else stall_bs=0; fi

  if [ "$stall_s100" -ge 2 ]; then
    a=$(ssh -o ConnectTimeout=20 hyper00 'now=$(date +%s); n=$(find /data02/jaxan/osworld/output-s100-hgkvsel -type f -printf "%T@\n" 2>/dev/null | sort -n | tail -1 | cut -d. -f1); [ -n "$n" ] && echo $((now-n)) || echo -1' 2>/dev/null || echo -1)
    [ "$a" -gt "$STALE" ] 2>/dev/null && echo "STALL s100 done=$s100 workers=$w100 no-new-file-for=${a}s"
  fi
  if [ "$stall_bs" -ge 2 ]; then
    a=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "now=\$(date +%s); n=\$(find /data/mw/runs/bsweep -type f -printf \"%T@\n\" 2>/dev/null | sort -n | tail -1 | cut -d. -f1); [ -n \"\$n\" ] && echo \$((now-n)) || echo -1"' 2>/dev/null || echo -1)
    [ "$a" -gt "$STALE" ] 2>/dev/null && echo "STALL bsweep results=$bs cfgs_done=$bcfg no-new-file-for=${a}s"
  fi

  [ "$bdrv" = "DOWN" ] && [ "$bhand" = "done" ] && echo "ALERT B 扫描驱动进程消失(cfgs_done=$bcfg results=$bs)"
  [ "$w100" = "0" ] && [ "$s100" -lt 361 ] 2>/dev/null && echo "ALERT s100 worker 全部退出但只完成 $s100/361"

  # --- 完成:latch 只发一次 ---
  if [ "$s100" -ge 361 ] 2>/dev/null && [ "$latch_s100" = "0" ]; then echo "DONE s100 跑满 361/361"; latch_s100=1; fi
  if [ "$bcfg" -ge 11 ] 2>/dev/null && [ "$latch_bs" = "0" ]; then echo "DONE B 扫描 11 个配置全部完成"; latch_bs=1; fi

  # --- 配置推进 / 整点进度 ---
  if [ "$bcfg" != "$prev_bcfg" ] && [ -n "${prev_bcfg:-}" ]; then
    echo "BSWEEP 配置推进 -> 已完成 $bcfg/11(results=$bs)"
  fi
  prev_bcfg=$bcfg

  if [ $((tick % 6)) -eq 1 ]; then
    echo "PROGRESS s100=$s100/361 (workers=$w100) | bsweep cfgs_done=$bcfg/11 results=$bs drv=$bdrv handoff=$bhand"
  fi

  prev_s100=$s100; prev_bs=$bs
  sleep $POLL
done
