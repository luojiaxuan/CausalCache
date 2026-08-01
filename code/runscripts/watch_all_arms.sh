#!/bin/bash
# note (luojiaxuan): 监控 OSWorld 主机匹配重跑(h00 shard0-3 / h01 shard4-11,四轮)
# 与 MobileWorld B 扫描,外加两台机器的内存看护。
# s50/s100/ft30 已全部跑满并归档,不再监控。
STALE=2400
POLL=600
tick=0
prev_sig=""; prev_bs=-1; stall_bs=0; latch_bs=0

rr() {  # host base -> "sel15:n recent15:n sel30:n recent30:n"
  ssh -o ConnectTimeout=20 "$1" "for t in sel-s15 recent-s15 sel-s30 recent-s30; do
    d=$2/rerun/out-\$t
    n=\$(ls -1 \$d/*/*/result.json 2>/dev/null | wc -l)
    printf '%s=%s ' \"\$t\" \"\$n\"
  done" 2>/dev/null || echo "?"
}

while true; do
  tick=$((tick+1))

  h00=$(rr hyper00 /data02/jaxan/osworld)
  h01=$(rr hyper01 /data04/jaxan/osworld)
  d00=$(ssh -o ConnectTimeout=20 hyper00 'pgrep -f "[o]sw_rerun_driver" >/dev/null && echo up || echo DOWN' 2>/dev/null || echo "?")
  d01=$(ssh -o ConnectTimeout=20 hyper01 'pgrep -f "[o]sw_rerun_driver" >/dev/null && echo up || echo DOWN' 2>/dev/null || echo "?")
  # note (luojiaxuan): 容器被外部 SIGKILL 过一次,四轮空转。存活性单独盯。
  c01=$(ssh -o ConnectTimeout=20 hyper01 'docker ps --format "{{.Names}}" | grep -qx sglang-omni-jaxan && echo up || echo DOWN' 2>/dev/null || echo "?")
  c00=$(ssh -o ConnectTimeout=20 hyper00 'docker ps --format "{{.Names}}" | grep -qx sglang-omni-jaxan && echo up || echo DOWN' 2>/dev/null || echo "?")

  bs=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "find /data/mw/runs/bsweep -name result.txt 2>/dev/null | wc -l"' 2>/dev/null || echo -1)
  bcfg=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "ls -d /data/mw/runs/bsweep/b*/DONE 2>/dev/null | wc -l"' 2>/dev/null || echo -1)

  m00=$(ssh -o ConnectTimeout=20 hyper00 "free -g | awk '/^Mem:/ {printf \"%d/%d\", \$7, \$2}'" 2>/dev/null || echo "?")
  m01=$(ssh -o ConnectTimeout=20 hyper01 "free -g | awk '/^Mem:/ {printf \"%d/%d\", \$7, \$2}'" 2>/dev/null || echo "?")
  for pair in "h00:$m00" "h01:$m01"; do
    hh=${pair%%:*}; mm=${pair#*:}; av=${mm%%/*}; tt=${mm##*/}
    [ "$tt" != "?" ] && [ "${tt:-0}" -gt 0 ] 2>/dev/null && [ $((av * 100 / tt)) -lt 15 ] && \
      echo "ALERT $hh 内存可用仅 ${av}G/${tt}G —— 逼近 sshd fork 失败"
  done

  [ "$d00" = "DOWN" ] && echo "ALERT h00 重跑驱动消失 [$h00]"
  [ "$d01" = "DOWN" ] && echo "ALERT h01 重跑驱动消失 [$h01]"
  [ "$c00" = "DOWN" ] && echo "ALERT h00 容器 sglang-omni-jaxan 已停"
  [ "$c01" = "DOWN" ] && echo "ALERT h01 容器 sglang-omni-jaxan 已停"

  # B 扫描停摆
  if [ "$bs" = "$prev_bs" ] && [ "$bs" != "-1" ]; then stall_bs=$((stall_bs+1)); else stall_bs=0; fi
  if [ "$stall_bs" -ge 3 ]; then
    a=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "now=\$(date +%s); n=\$(find /data/mw/runs/bsweep -type f -printf \"%T@\n\" 2>/dev/null | sort -n | tail -1 | cut -d. -f1); [ -n \"\$n\" ] && echo \$((now-n)) || echo -1"' 2>/dev/null || echo -1)
    [ "$a" -gt "$STALE" ] 2>/dev/null && echo "STALL bsweep results=$bs cfgs=$bcfg 静默 ${a}s"
  fi
  prev_bs=$bs
  if [ "$bcfg" -ge 11 ] 2>/dev/null && [ "$latch_bs" = "0" ]; then echo "DONE B 扫描 11 个配置全部完成"; latch_bs=1; fi

  sig="$h00|$h01"
  if [ "$sig" != "$prev_sig" ] && [ -n "$prev_sig" ] && [ $((tick % 3)) -eq 0 ]; then
    echo "RERUN h00[$h00] h01[$h01]"
  fi
  prev_sig="$sig"

  if [ $((tick % 6)) -eq 1 ]; then
    echo "PROGRESS 重跑 h00[$h00] h01[$h01] drv=$d00/$d01 | bsweep $bcfg/11 ($bs) | RAM h00=$m00 h01=$m01"
  fi
  sleep $POLL
done
