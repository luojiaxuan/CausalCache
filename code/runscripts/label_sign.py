#!/usr/bin/env python3
# note (luojiaxuan): 统计教师标签里边际的符号分布。
# 关键判别:若训练标签本来就大多为负,则部署时"97.4% 的步没有正边际候选"是模型
# 学对了,而非校准崩了——那么同域该不该提升远端帧这个前提本身就要重估。
# 若标签里正边际占比可观,则部署侧的全负就是分布偏移/校准问题。
import collections
import glob
import json
import sys

sets_root, singles_root = sys.argv[1], sys.argv[2]

# 单例边际 = U({e}) - U(∅)
b0 = {}
singles = collections.defaultdict(dict)
for f in sorted(glob.glob(singles_root + "/singletons.shard*.jsonl")):
    with open(f, errors="ignore") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r.get("kind")
            if k == "b0":
                b0[r["dp_id"]] = r
            elif k == "singleton":
                singles[r["dp_id"]][int(r["event"])] = float(r["score"])

pos = neg = zero = 0
per_state_any_pos = 0
states = 0
for dp, base in b0.items():
    have = singles.get(dp, {})
    if not have:
        continue
    states += 1
    anyp = False
    for e, s in have.items():
        m = s - float(base["score"])
        if m > 0:
            pos += 1
            anyp = True
        elif m < 0:
            neg += 1
        else:
            zero += 1
    if anyp:
        per_state_any_pos += 1

# 集合边际(已在行内)
spos = sneg = 0
for f in sorted(glob.glob(sets_root + "/sets.shard*.jsonl")):
    with open(f, errors="ignore") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") != "set":
                continue
            m = float(r["score"]) - float(r["parent_score"])
            if m > 0:
                spos += 1
            else:
                sneg += 1

print(json.dumps({
    "singleton_edges": pos + neg + zero,
    "singleton_positive_pct": round(100 * pos / max(pos + neg + zero, 1), 1),
    "states": states,
    "states_with_any_positive_singleton_pct": round(100 * per_state_any_pos / max(states, 1), 1),
    "set_edges": spos + sneg,
    "set_positive_pct": round(100 * spos / max(spos + sneg, 1), 1),
}, ensure_ascii=False, indent=1))
