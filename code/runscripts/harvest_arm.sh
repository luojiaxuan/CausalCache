#!/bin/bash
# note (luojiaxuan): 收割一个 campaign 的双机 task->score 映射。$1=campaign $2=输出json
CAMP=$1; OUT=$2
TMP=$(mktemp)
collect() {  # $1=host $2=容器 $3=runs根
  ssh -o ConnectTimeout=15 "$1" "docker exec $2 bash -c 'for f in $3/$CAMP/shard-*/trajectories/*/result.txt; do
    [ -f \"\$f\" ] || continue
    t=\$(basename \$(dirname \"\$f\"))
    s=\$(head -1 \"\$f\" | sed \"s/score: //\")
    echo \"\$t \$s\"
  done'" 2>/dev/null
}
collect hyper00 sglang-omni-jaxan /data/mw/runs > "$TMP"
collect hyper01 sglang-omni-jaxan /bigdata/mw/runs >> "$TMP"
python3 - "$TMP" "$OUT" <<'PY'
import json, sys
out = {}
conflicts = []
for line in open(sys.argv[1]):
    parts = line.split()
    if len(parts) != 2:
        continue
    t, s = parts[0], float(parts[1])
    if t in out and out[t] != s:
        conflicts.append((t, out[t], s))
    out.setdefault(t, s)
json.dump(out, open(sys.argv[2], "w"), indent=0, sort_keys=True)
print(f"tasks={len(out)} conflicts={len(conflicts)}")
for c in conflicts:
    print("CONFLICT", *c)
PY
rm -f "$TMP"
