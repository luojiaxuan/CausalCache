#!/usr/bin/env bash
# note (luojiaxuan): 全量探针(约 3.4k 记忆敏感步 × 4 条件),24 路并发打本机 vLLM(GPU1)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan/mw/MobileWorld
PYTHONPATH=/data01/jaxan/pyshim setsid uv run python /data01/jaxan/androtmem_probe.py --workers 24 --out /data01/jaxan/androtmem/probe/results.jsonl > /data01/jaxan/androtmem/probe/probe.log 2>&1 < /dev/null &
sleep 60
grep -E "usable|jobs=|^\[|ERR|PROBE_DONE" /data01/jaxan/androtmem/probe/probe.log | tail -4
echo "lines=$(wc -l < /data01/jaxan/androtmem/probe/results.jsonl 2>/dev/null)"
for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i 1 | tr -d "\n"; printf " "; sleep 2; done; echo
echo PROBE_LAUNCHED
