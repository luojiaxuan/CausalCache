#!/bin/bash
# note (luojiaxuan): 监控 MobileWorld B 扫描收尾(11 个配置,已完成 4 个)。
# OSWorld 侧全部完结:s50/s100/ft30 已归档,sel@30 的 aries 污染已定点补跑并合并。
POLL=600
tick=0; prev=-1; stall=0; latch=0

while true; do
  tick=$((tick+1))
  st=$(ssh -o ConnectTimeout=20 hyper00 'docker exec sglang-omni-jaxan bash -lc "
    d=\$(ls -d /data/mw/runs/bsweep/b*-*/DONE 2>/dev/null | wc -l)
    n=\$(find /data/mw/runs/bsweep -name result.txt 2>/dev/null | wc -l)
    cur=\$(for x in /data/mw/runs/bsweep/b*-*/; do [ -f \$x/DONE ] || echo -n \"\$(basename \$x):\$(find \$x -name result.txt | wc -l) \"; done)
    drv=\$(pgrep -fc \"[b]sweep_driver2\" || echo 0)
    run=\$(pgrep -fc \"[r]un_mobileworld_gui_owl.py\" || echo 0)
    echo \"\$d|\$n|\$drv|\$run|\$cur\""' 2>/dev/null || echo "?|?|?|?|")
  cfg=${st%%|*}; rest=${st#*|}; n=${rest%%|*}; rest=${rest#*|}
  drv=${rest%%|*}; rest=${rest#*|}; run=${rest%%|*}; cur=${rest#*|}
  mem=$(ssh -o ConnectTimeout=20 hyper00 "free -g | awk '/^Mem:/ {printf \"%d/%d\", \$7, \$2}'" 2>/dev/null || echo "?/?")

  av=${mem%%/*}; tt=${mem##*/}
  [ "$tt" != "?" ] && [ "${tt:-0}" -gt 0 ] 2>/dev/null && [ $((av * 100 / tt)) -lt 15 ] && \
    echo "ALERT h00 内存可用仅 ${av}G/${tt}G"

  # note (luojiaxuan): 驱动跑在容器内、以 root 身份。宿主 pkill 杀不动它(EPERM),
  # 本 session 曾因此"以为暂停了其实一直在跑"。这里用容器内 pgrep 计数,别用宿主的。
  [ "$drv" = "0" ] && [ "$cfg" != "11" ] && echo "ALERT B 扫描驱动消失(已完成 $cfg/11,结果 $n)"

  if [ "$n" = "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 4 ] && echo "STALL B 扫描停在 $n 个结果(配置 $cfg/11,runner=$run)"
  prev=$n

  if [ "$cfg" -ge 11 ] 2>/dev/null && [ "$latch" = "0" ]; then
    echo "DONE B 扫描 11 个配置全部完成 —— 可以归约 Δ(selector − Recent-B) 曲线"; latch=1
  fi

  [ $((tick % 5)) -eq 1 ] && echo "PROGRESS B 扫描 $cfg/11 配置 结果=$n drv=$drv runner=$run 在跑[$cur] RAM=$mem"
  sleep $POLL
done
