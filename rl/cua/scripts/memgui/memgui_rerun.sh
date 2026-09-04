#!/usr/bin/env bash
# note (luojiaxuan): 两臂补跑未执行的任务(轨迹为空者):等各组 8 台后端全部 healthy 后起,
# 传容器名前缀让 runner 能自愈不健康后端(此前因前缀未知而无法恢复,导致成批 0 秒失败)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
R=/data01/jaxan/rl_v2/memgui
wait_healthy() { for i in $(seq 1 60); do n=$(docker ps --format "{{.Names}} {{.Status}}" | grep -E "^sglang-omni-jaxan-$1_[0-9]+ " | grep -c healthy); [ "$n" = "8" ] && return 0; sleep 15; done; echo "WARN $1 healthy=$n/8 after 15min"; }
wait_healthy mg; wait_healthy mgb
echo "$(date -u +%FT%TZ) backends ready; launching reruns"
MG_TASKFILE=$R/armA_base_hist1_missing.csv MG_PREFIX=sglang-omni-jaxan-mg \
  setsid bash /data01/jaxan/memgui_arm.sh armA_base_hist1 recent 1 8 > $R/armA_base_hist1_rerun.log 2>&1 < /dev/null &
MG_TASKFILE=$R/armB_base_recency_h3_missing.csv MG_PREFIX=sglang-omni-jaxan-mgb MG_HOSTS=$(seq -s, -f "http://127.0.0.1:69%02g" 8 15) \
  setsid bash /data01/jaxan/memgui_arm.sh armB_base_recency_h3 recent 3 8 > $R/armB_base_recency_h3_rerun.log 2>&1 < /dev/null &
sleep 60
for a in armA_base_hist1 armB_base_recency_h3; do echo "$a: $(grep -aE "Remaining tasks|Task list" $R/${a}_rerun.log | head -1 | cut -c1-120)"; done
echo RERUN_LAUNCHED
