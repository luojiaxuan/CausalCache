#!/usr/bin/env bash
# note (luojiaxuan): 臂 A 补跑期间 judge 有 86 次 429,43 题落成 error decision;低并发(2 分片)只重判错误项。
set -uo pipefail
export PATH="/data01/jaxan/binshim:$HOME/.local/bin:$PATH"
cd /data01/jaxan/memgui
sed -i "s/^MEMGUI_LLM_MAX_CONCURRENCY=.*/MEMGUI_LLM_MAX_CONCURRENCY=2/" /data01/jaxan/memgui/.env
for s in 0 1; do setsid uv run python /data01/jaxan/memgui_rejudge.py /data01/jaxan/rl_v2/memgui/armA_base_hist1 --only-errors --no-resume --shard $s --nshards 2 >> /data01/jaxan/rl_v2/memgui/rejudge_errA_$s.log 2>&1 < /dev/null & done
sleep 5; echo "rejudge procs=$(pgrep -f "[m]emgui_rejudge.py" | wc -l)"
