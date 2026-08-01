#!/usr/bin/env python3
# note (luojiaxuan): 统计弃权臂的 SELECT_AUDIT:提升了几个、补了几个 recent、
# 偏离 recent 多远。用来判断 -3.88pp 到底是"提升得太少"还是"提升得太差"。
import collections
import glob
import json
import sys

promoted = collections.Counter()
koff = collections.Counter()
reason = collections.Counter()
n = 0
for f in glob.glob(sys.argv[1]):
    with open(f, errors="ignore") as fh:
        for line in fh:
            i = line.find("{")
            if i < 0:
                continue
            try:
                d = json.loads(line[i:])
            except Exception:
                continue
            if d.get("event") != "SELECT_AUDIT":
                continue
            n += 1
            promoted[d.get("promoted")] += 1
            koff[d.get("k_off_recent")] += 1
            reason[d.get("reason")] += 1
print(json.dumps({
    "steps": n,
    "promoted": {str(k): v for k, v in sorted(promoted.items(), key=lambda t: (t[0] is None, t[0]))},
    "k_off_recent": {str(k): v for k, v in sorted(koff.items(), key=lambda t: (t[0] is None, t[0]))},
    "reason": {str(k): v for k, v in reason.items()},
}, ensure_ascii=False, indent=1))
