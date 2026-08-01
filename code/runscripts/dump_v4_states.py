#!/usr/bin/env python3
# note (luojiaxuan): 导出某台机器上 v4 集合标签覆盖到的 dp_id 与唯一边 key。
# 集合打分曾跨机分片,任何单机统计都是低估,必须取并集。
import glob
import json
import sys

root, out = sys.argv[1], sys.argv[2]
states, keys = set(), set()
for f in sorted(glob.glob(root + "/*.jsonl")):
    with open(f, errors="ignore") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("key") == "__fingerprint__":
                continue
            states.add(r["dp_id"])
            keys.add(r["key"])
json.dump({"states": sorted(states), "n_keys": len(keys)}, open(out, "w"))
print(json.dumps({"states": len(states), "unique_edges": len(keys)}))
