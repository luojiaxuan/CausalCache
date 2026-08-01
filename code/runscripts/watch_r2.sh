#!/bin/bash
# note (luojiaxuan): 第2轮同批双臂(selfix vs recent,各 361,15 步)。
# 计数按唯一 task_id;完成判据要求**两臂都**满 361,否则不能配对。
POLL=420
TARGET=361
tick=0; prev=""; stall=0; latch=0

count_arm() {
  python3 - "$1" <<'PY' 2>/dev/null || echo 0
import subprocess, sys
arm = sys.argv[1]
ids = set()
for host, base in (("hyper00", "/data02/jaxan/osworld"), ("hyper01", "/data04/jaxan/osworld")):
    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", host,
             f"ls -1 {base}/twoarm-r2/out-{arm}/*/*/result.json 2>/dev/null"],
            capture_output=True, text=True, timeout=40).stdout
    except Exception:
        continue
    for line in out.splitlines():
        # note (luojiaxuan): 必须数 result.json。任务目录在开始时就建好,
        # 数目录会把"在跑"当成"已完成"——本轮已因此误报过一次满员。
        parts = line.strip().split("/")
        if len(parts) >= 3:
            ids.add(parts[-2])
print(len(ids))
PY
}

while true; do
  tick=$((tick+1))
  a=$(count_arm selfix); b=$(count_arm recent)
  d0=$(ssh -o ConnectTimeout=15 hyper00 'pgrep -fc "[o]sw_arm" || echo 0' 2>/dev/null || echo "?")
  d1=$(ssh -o ConnectTimeout=15 hyper01 'pgrep -fc "[o]sw_arm" || echo 0' 2>/dev/null || echo "?")
  m0=$(ssh -o ConnectTimeout=15 hyper00 "free -g | awk '/^Mem:/ {printf \"%d/%d\", \$7, \$2}'" 2>/dev/null || echo "?/?")
  dk=$(ssh -o ConnectTimeout=15 hyper00 "df -h /data02 | tail -1 | awk '{print \$4}'" 2>/dev/null || echo "?")

  av=${m0%%/*}; tt=${m0##*/}
  [ "$tt" != "?" ] && [ "${tt:-0}" -gt 0 ] 2>/dev/null && [ $((av * 100 / tt)) -lt 15 ] && \
    echo "ALERT h00 内存可用仅 ${av}G/${tt}G"

  if [ "$a" -ge "$TARGET" ] 2>/dev/null && [ "$b" -ge "$TARGET" ] 2>/dev/null && [ "$latch" = "0" ]; then
    echo "DONE 双臂各满 361 —— 可以做同批配对"; latch=1
  fi
  # note (luojiaxuan): 驱动全退但未满 = 需要补差集,这是真正要人介入的失败模式。
  if [ "$d0" = "0" ] && [ "$d1" = "0" ] && { [ "$a" -lt "$TARGET" ] || [ "$b" -lt "$TARGET" ]; } 2>/dev/null; then
    echo "ALERT 驱动全部退出但 selfix=$a recent=$b 未满 361,需补差集"
  fi
  cur="$a|$b"
  if [ "$cur" = "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 5 ] && echo "STALL 双臂停在 selfix=$a recent=$b 已 5 轮未动"
  prev="$cur"

  [ $((tick % 4)) -eq 1 ] && echo "PROGRESS selfix=$a/361 recent=$b/361 | 驱动 h00=$d0 h01=$d1 | RAM $m0 | /data02 剩 $dk"
  sleep $POLL
done
