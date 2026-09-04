#!/usr/bin/env bash
# note (luojiaxuan): 对指定臂剩余判错项做单分片低并发扫尾(429 残留)。用法: rejudge_sweep.sh <arm>
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan/memgui
uv run python /data01/jaxan/memgui_rejudge.py /data01/jaxan/rl_v2/memgui/$1 --only-errors --no-resume --shard 0 --nshards 1 2>&1 | grep -aE "^\[|REJUDGE_DONE" | tail -4 | cut -c1-140
python3 /data01/jaxan/memgui_tally.py $1
