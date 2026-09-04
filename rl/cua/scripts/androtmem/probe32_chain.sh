#!/usr/bin/env bash
# note (luojiaxuan): 等 32B 服务就绪 → 冒烟(12 步)→ 若冒烟出分则起全量 4 主条件探针(后台)。
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
for i in $(seq 1 80); do grep -q "READY" /data04/jaxan/androtmem/server2.log 2>/dev/null && break; grep -q "CONTAINER_EXITED" /data04/jaxan/androtmem/server2.log 2>/dev/null && { echo "SERVER_FAILED"; exit 1; }; sleep 15; done
mkdir -p /data04/jaxan/androtmem/probe32
bash /data04/jaxan/probe32_smoke.sh
n=$(wc -l < /data04/jaxan/androtmem/probe32/smoke.jsonl 2>/dev/null || echo 0)
if [ "$n" -ge 40 ]; then
  cd /data04/jaxan/mw/MobileWorld
  PYTHONPATH=/data04/jaxan/pyshim setsid uv run python /data04/jaxan/androtmem_probe_h01.py --conds none,recency2,gold,rec1_gold1 --workers 16 --out /data04/jaxan/androtmem/probe32/results.jsonl > /data04/jaxan/androtmem/probe32/probe.log 2>&1 < /dev/null &
  echo "FULL_PROBE32_LAUNCHED"
else
  echo "SMOKE_INSUFFICIENT n=$n"
fi
