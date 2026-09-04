#!/usr/bin/env bash
# note (luojiaxuan): 辅助诊断:G2 与补跑轮结束后,用无文本历史口径(镜像 §17)在同一批状态上重跑 23 上下文解码。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
O=/data01/jaxan/rl_v2/venus
while ! grep -q MISSING_PASS_DONE $O/v2/missing_pass.log 2>/dev/null; do sleep 120; done
echo "$(date -u +%FT%TZ) G2_NOTEXT START"
cd /data01/jaxan/mw/MobileWorld
uv run python /data01/jaxan/venus_oracle.py --roots $O/v2/recent0 $O/v2/recent2 $O/v2/recent8 $O/v2/change2 \
  --out $O/oracle_v2_notext.jsonl --per-traj 6 --limit 400 --workers 12 --no-text > $O/oracle_v2_notext.log 2>&1
python3 /data01/jaxan/venus_oracle_verdict.py $O/oracle_v2_notext.jsonl | tee $O/oracle_v2_notext_verdict.txt
echo "$(date -u +%FT%TZ) G2_NOTEXT_DONE"
