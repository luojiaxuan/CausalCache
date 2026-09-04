#!/usr/bin/env bash
# note (luojiaxuan): 等当前两路探针各自跑完后,分别在原版 8B(GPU1)与 iter_59(GPU2)上补 v1 式弱文本三条件。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan/mw/MobileWorld
until grep -q PROBE_DONE /data01/jaxan/androtmem/probe/probe3.log 2>/dev/null; do sleep 60; done
PYTHONPATH=/data01/jaxan/pyshim uv run python /data01/jaxan/androtmem_probe.py --workers 16 --conds none_actions,recency2_actions,rec1_gold1_actions --out /data01/jaxan/androtmem/probe/results.jsonl > /data01/jaxan/androtmem/probe/probe4.log 2>&1
echo "8B actions-only DONE $(date -u +%FT%TZ)"
until grep -q PROBE_DONE /data01/jaxan/androtmem/probe59/probe.log 2>/dev/null; do sleep 60; done
PYTHONPATH=/data01/jaxan/pyshim uv run python /data01/jaxan/androtmem_probe.py --workers 16 --llm http://172.17.0.1:41042/v1 --conds none_actions,recency2_actions,rec1_gold1_actions --out /data01/jaxan/androtmem/probe59/results.jsonl > /data01/jaxan/androtmem/probe59/probe_actions.log 2>&1
echo "iter59 actions-only DONE $(date -u +%FT%TZ)"
echo ACTIONS_CHAIN_DONE
