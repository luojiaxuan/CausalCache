#!/usr/bin/env python3
# note (luojiaxuan): 统计 v4 集合标签的唯一边数与覆盖状态数。
# 必须在 h00(权威位置)上跑:h01 只有部分镜像,按它统计会低估覆盖率。
import glob
import json
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "/data/runs/selector-v4/sets"
keys, states = set(), set()
raw = 0
per_file = {}
for f in sorted(glob.glob(root + "/*.jsonl")):
    n = 0
    with open(f, errors="ignore") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("key") == "__fingerprint__":
                continue
            raw += 1
            n += 1
            keys.add(r["key"])
            states.add(r["dp_id"])
    per_file[f.split("/")[-1]] = n
print(json.dumps({"raw_rows": raw, "unique_set_edges": len(keys),
                  "states_covered": len(states)}, indent=1))
print(json.dumps(per_file, indent=1))
