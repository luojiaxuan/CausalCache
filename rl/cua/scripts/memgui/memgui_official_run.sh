#!/usr/bin/env bash
# note (luojiaxuan): 官方步数预算口径(每题 golden_steps*2.5+1)下的两臂完整流程,逐臂串行:
# 多轮补跑直至该臂无缺失轨迹 → 低并发补判 429 判错项(两遍扫尾)→ 该臂终表;臂 A 先跑,
# 让与榜单可比的数字最早落地。末尾不拆后端(保留期见 map)。
set -uo pipefail
R=/data01/jaxan/rl_v2/memgui
for spec in "armA_off_hist1 recent 1" "armB_off_recency_h3 recent 3"; do set -- $spec; arm=$1
  echo "$(date -u +%FT%TZ) START $arm"
  ARM_SPECS="$spec" NBACK=5 PASSES=4 bash /data01/jaxan/memgui_pass_loop.sh > $R/pass_loop_off_$arm.log 2>&1
  for k in 1 2; do bash /data01/jaxan/rejudge_sweep.sh $arm > $R/rejudge_off_${arm}_$k.log 2>&1; done
  python3 /data01/jaxan/memgui_tally.py $arm
  echo "$(date -u +%FT%TZ) DONE $arm"
done
echo OFFICIAL_RUN_DONE
