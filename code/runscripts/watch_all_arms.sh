#!/bin/bash
# note (luojiaxuan): 统一监控四个在跑的臂。只在“状态变化 / 完成 / 停摆 / 整点”时发事件,
# 避免每轮都刷屏。停摆判据 = done 计数两轮不动 且 最新产出文件超过 STALE 秒没更新。
STALE=1200
POLL=600
tick=0
prev_sig=""
prev_s50=-1; prev_s100=-1; prev_ft30=-1; prev_bs=-1
stall_s50=0; stall_s100=0; stall_ft30=0; stall_bs=0

osw_count() { # host dir -> 完成任务数(每个任务一份 result.json,层级为 <domain>/<uuid>/)
  ssh -o ConnectTimeout=20 "$1" "ls -1 $2/*/*/result.json 2>/dev/null | wc -l" 2>/dev/null || echo -1
}
osw_age() { # host dir
  ssh -o ConnectTimeout=20 "$1" "now=\$(date +%s); n=\$(find $2 -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1); [ -n \"\$n\" ] && echo \$((now-n)) || echo -1" 2>/dev/null || echo -1
}

while true; do
  tick=$((tick+1))

  s50=$(osw_count hyper00 /data02/jaxan/osworld/output-s50-hgkvsel)
  s100=$(osw_count hyper00 /data02/jaxan/osworld/output-s100-hgkvsel)
  ft30=$(osw_count hyper01 /data04/jaxan/osworld/output-ft30)

  bs=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "find /data/mw/runs/bsweep -name result.txt 2>/dev/null | wc -l"' 2>/dev/null || echo -1)
  bcfg=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "ls -d /data/mw/runs/bsweep/b*/ 2>/dev/null | wc -l"' 2>/dev/null || echo -1)
  # note (luojiaxuan): v1 写 driver.log,v2 写 /data02/jaxan/bsweep_v2.log;pgrep 用不带
  # 后缀的 bsweep_driver 以同时匹配 v1/v2,否则交接后会误报“驱动消失”。
  bcur=$(ssh -o ConnectTimeout=20 hyper00 'tail -1 /data02/jaxan/bsweep_v2.log 2>/dev/null || tail -1 /data/mw/runs/bsweep/driver.log 2>/dev/null' 2>/dev/null)
  bdrv=$(ssh -o ConnectTimeout=20 hyper00 'pgrep -f bsweep_driver >/dev/null && echo up || echo DOWN' 2>/dev/null || echo "?")
  bhand=$(ssh -o ConnectTimeout=20 hyper00 'pgrep -f bsweep_handoff >/dev/null && echo waiting || echo done' 2>/dev/null || echo "?")

  # --- 停摆检测:计数不动才去查文件年龄,省 ssh ---
  for arm in s50 s100 ft30 bs; do
    cur=$(eval echo \$$arm); prv=$(eval echo \$prev_$arm)
    if [ "$cur" = "$prv" ] && [ "$cur" != "-1" ]; then
      eval stall_$arm=\$\(\(stall_$arm+1\)\)
    else
      eval stall_$arm=0
    fi
  done

  if [ "$stall_s50" -ge 2 ]; then
    a=$(osw_age hyper00 /data02/jaxan/osworld/output-s50-hgkvsel)
    [ "$a" -gt "$STALE" ] 2>/dev/null && echo "STALL s50 done=$s50 no-new-file-for=${a}s"
  fi
  if [ "$stall_s100" -ge 2 ]; then
    a=$(osw_age hyper00 /data02/jaxan/osworld/output-s100-hgkvsel)
    [ "$a" -gt "$STALE" ] 2>/dev/null && echo "STALL s100 done=$s100 no-new-file-for=${a}s"
  fi
  if [ "$stall_ft30" -ge 2 ]; then
    a=$(osw_age hyper01 /data04/jaxan/osworld/output-ft30)
    [ "$a" -gt "$STALE" ] 2>/dev/null && echo "STALL ft30 done=$ft30 no-new-file-for=${a}s"
  fi
  if [ "$stall_bs" -ge 2 ]; then
    a=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "now=\$(date +%s); n=\$(find /data/mw/runs/bsweep -type f -printf \"%T@\n\" 2>/dev/null | sort -n | tail -1 | cut -d. -f1); [ -n \"\$n\" ] && echo \$((now-n)) || echo -1"' 2>/dev/null || echo -1)
    [ "$a" -gt "$STALE" ] 2>/dev/null && echo "STALL bsweep results=$bs cfg=$bcfg no-new-file-for=${a}s"
  fi

  [ "$bdrv" = "DOWN" ] && echo "ALERT bsweep driver process is GONE (cfg=$bcfg results=$bs)"

  # --- 完成信号 ---
  [ "$s50" -ge 361 ] 2>/dev/null && echo "DONE s50 reached $s50/361"
  [ "$s100" -ge 361 ] 2>/dev/null && echo "DONE s100 reached $s100/361"
  [ "$ft30" -ge 361 ] 2>/dev/null && echo "DONE ft30 reached $ft30/361"
  [ "$bcfg" -ge 11 ] 2>/dev/null && [ "$bs" -ge 396 ] 2>/dev/null && echo "DONE bsweep all 11 configs"

  # --- 配置切换 / 整点进度 ---
  sig="cfg=$bcfg|$bcur"
  if [ "$sig" != "$prev_sig" ] && [ -n "$prev_sig" ]; then
    echo "BSWEEP advanced -> [$bcur] cfg_dirs=$bcfg results=$bs"
  fi
  prev_sig="$sig"

  if [ $((tick % 6)) -eq 1 ]; then
    echo "PROGRESS s50=$s50/361 s100=$s100/361 ft30=$ft30/361 | bsweep cfg=$bcfg results=$bs drv=$bdrv handoff=$bhand cur=[$bcur]"
  fi

  prev_s50=$s50; prev_s100=$s100; prev_ft30=$ft30; prev_bs=$bs
  sleep $POLL
done
