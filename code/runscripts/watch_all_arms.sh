#!/bin/bash
# note (luojiaxuan): 监控 aries 定点补跑(110 个任务,sel@30,h00 分片0-3 / h01 分片4-7)
# 与 MobileWorld B 扫描,外加两台机器的内存与容器存活。
# 已完成并归档:s50 / s100 / ft30;已废弃:四轮全量重跑(范围过大,改为定点补跑)。
POLL=600
TARGET=110
tick=0
prev_n=-1; stall=0; latch=0
prev_bs=-1; stall_bs=0; latch_bs=0

cnt() { ssh -o ConnectTimeout=20 "$1" "ls -1 $2/aries-repair/out-sel-s30/*/*/result.json 2>/dev/null | wc -l" 2>/dev/null || echo 0; }
alive() { ssh -o ConnectTimeout=20 "$1" "pgrep -f '[o]sw_aries_repair' >/dev/null && echo up || echo DOWN" 2>/dev/null || echo "?"; }
cont() { ssh -o ConnectTimeout=20 "$1" "docker ps --format '{{.Names}}' | grep -qx sglang-omni-jaxan && echo up || echo DOWN" 2>/dev/null || echo "?"; }
mem()  { ssh -o ConnectTimeout=20 "$1" "free -g | awk '/^Mem:/ {printf \"%d/%d\", \$7, \$2}'" 2>/dev/null || echo "?/?"; }

while true; do
  tick=$((tick+1))
  n0=$(cnt hyper00 /data02/jaxan/osworld); n1=$(cnt hyper01 /data04/jaxan/osworld)
  n=$((n0 + n1))
  d0=$(alive hyper00); d1=$(alive hyper01)
  c0=$(cont hyper00);  c1=$(cont hyper01)
  m0=$(mem hyper00);   m1=$(mem hyper01)

  for pair in "h00:$m0" "h01:$m1"; do
    hh=${pair%%:*}; mm=${pair#*:}; av=${mm%%/*}; tt=${mm##*/}
    [ "$tt" != "?" ] && [ "${tt:-0}" -gt 0 ] 2>/dev/null && [ $((av * 100 / tt)) -lt 15 ] && \
      echo "ALERT $hh 内存可用仅 ${av}G/${tt}G —— 逼近 sshd fork 失败"
  done
  [ "$c0" = "DOWN" ] && echo "ALERT h00 容器已停"
  [ "$c1" = "DOWN" ] && echo "ALERT h01 容器已停"

  # note (luojiaxuan): 驱动跑完会正常退出,所以"消失"只在未达标时才算异常。
  if [ "$n" -lt "$TARGET" ]; then
    [ "$d0" = "DOWN" ] && echo "ALERT h00 补跑驱动消失,进度 $n/$TARGET"
    [ "$d1" = "DOWN" ] && echo "ALERT h01 补跑驱动消失,进度 $n/$TARGET"
  fi

  if [ "$n" = "$prev_n" ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 4 ] && [ "$n" -lt "$TARGET" ] && echo "STALL 补跑停在 $n/$TARGET 已 4 轮未动"
  prev_n=$n

  if [ "$n" -ge "$TARGET" ] && [ "$latch" = "0" ]; then
    echo "DONE aries 补跑完成 $n/$TARGET (h00=$n0 h01=$n1) —— 可以合并重算 sel@30"
    latch=1
  fi

  bs=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "find /data/mw/runs/bsweep -name result.txt 2>/dev/null | wc -l"' 2>/dev/null || echo -1)
  bcfg=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "ls -d /data/mw/runs/bsweep/b*/DONE 2>/dev/null | wc -l"' 2>/dev/null || echo -1)
  if [ "$bs" = "$prev_bs" ] && [ "$bs" != "-1" ]; then stall_bs=$((stall_bs+1)); else stall_bs=0; fi
  [ "$stall_bs" -ge 4 ] && echo "STALL bsweep 停在 $bs(配置 $bcfg/11)"
  prev_bs=$bs
  [ "$bcfg" -ge 11 ] 2>/dev/null && [ "$latch_bs" = "0" ] && { echo "DONE B 扫描 11 个配置全部完成"; latch_bs=1; }

  [ $((tick % 6)) -eq 1 ] && \
    echo "PROGRESS 补跑 $n/$TARGET (h00=$n0 h01=$n1) drv=$d0/$d1 | bsweep $bcfg/11 ($bs) | RAM h00=$m0 h01=$m1"
  sleep $POLL
done
