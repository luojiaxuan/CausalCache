#!/usr/bin/env bash
# note (luojiaxuan): 探针第二批条件:最近帧+gold 混合(rec1_gold1/rec2_gold1)与去文本历史消融
# (none_notext/gold_notext/recency2_notext),追加写入同一 results.jsonl(按 (task,step,cond) 续传)。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan/mw/MobileWorld
PYTHONPATH=/data01/jaxan/pyshim setsid uv run python /data01/jaxan/androtmem_probe.py --workers 24 --conds rec1_gold1,rec2_gold1,none_notext,gold_notext,recency2_notext --out /data01/jaxan/androtmem/probe/results.jsonl > /data01/jaxan/androtmem/probe/probe2.log 2>&1 < /dev/null &
sleep 30; grep -E "usable|jobs=" /data01/jaxan/androtmem/probe/probe2.log; echo PROBE2_LAUNCHED
