#!/usr/bin/env bash
# note (luojiaxuan): 等闭环探针 v2 收官后,在其四臂成功轨迹上跑离线主闸门 G2(≤400 状态 × 23 上下文,并发 12),
# 再归约;全程零 API 费用。产物:rl_v2/venus/oracle_v2.jsonl 与 oracle_v2_verdict.txt。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
O=/data01/jaxan/rl_v2/venus
while ! grep -q VENUS_PROBE2_DONE $O/v2/probe2.log 2>/dev/null; do sleep 120; done
echo "$(date -u +%FT%TZ) G2 START"
cd /data01/jaxan/mw/MobileWorld
uv run python /data01/jaxan/venus_oracle.py --roots $O/v2/recent0 $O/v2/recent2 $O/v2/recent8 $O/v2/change2 \
  --out $O/oracle_v2.jsonl --per-traj 6 --limit 400 --workers 12 > $O/oracle_v2.log 2>&1
python3 /data01/jaxan/venus_oracle_verdict.py $O/oracle_v2.jsonl | tee $O/oracle_v2_verdict.txt
echo "$(date -u +%FT%TZ) G2_DONE"
