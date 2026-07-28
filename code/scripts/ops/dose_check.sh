#!/bin/bash
# note (luojiaxuan): 剂量扫描进度检查(host 侧)。$1=容器名 $2=runs根 $3=campaign列表(空格分隔)
C=$1; ROOT=$2; shift 2
for r in "$@"; do
  for s in 0 1 2 3; do
    docker exec -i "$C" python3 - "$ROOT/$r/heartbeat-shard-$s.json" "$r" "$s" <<'PY' 2>/dev/null || echo "ALERT $r shard$s heartbeat-unreadable"
import json, sys, datetime, os
p, r, s = sys.argv[1:4]
if not os.path.exists(p):
    print(f"WAIT {r} shard{s} no-heartbeat"); sys.exit()
d = json.load(open(p))
age = (datetime.datetime.now(datetime.timezone.utc)
       - datetime.datetime.fromisoformat(d["updated_at"])).total_seconds()
tag = "ALERT-STALE" if age > 1200 else d["phase"]
print(f"{tag} {r} shard{s} done={d['completed_tasks']} pend={d['pending_at_attempt_start']} attempt={d['attempt']} age={int(age)}s")
PY
  done
done
