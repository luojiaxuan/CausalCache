import json, glob, collections, re
rows = []
for f in glob.glob("/data01/jaxan/behav_labels_shard*.jsonl"):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
def ok(v): return 1 if (v or {}).get("ok") else 0
PAIR = re.compile(r"^(\d+)_(\d+)$")
cat = collections.Counter(); per_task = collections.defaultdict(collections.Counter); ages = []; near = 0
for r in rows:
    d = r["decodes"]; rec = ok(d.get("recency")); nul = ok(d.get("null"))
    okpairs = [tuple(map(int, PAIR.match(k).groups())) for k, v in d.items() if PAIR.match(k) and ok(v)]
    c = "A_recency_ok" if rec else ("B_controllable" if okpairs else "C_floor")
    cat[c] += 1; per_task[r["task"]][c] += 1
    if nul and not rec: cat["null_ok_recency_wrong"] += 1
    if nul: cat["null_ok(仅当前帧即可)"] += 1
    if c == "B_controllable":
        ci = r["cand_idx"]; st = r["step"]
        best = min(max(st - ci[a], st - ci[b]) for a, b in okpairs)
        ages.append(best)
        if any(st - ci[a] <= 2 and st - ci[b] <= 2 for a, b in okpairs): near += 1
n = len(rows)
print(f"=== 行为标签 {n} 个上下文(训练域成功轨迹;B=2 口径) ===")
for k, v in sorted(cat.items()): print(f"  {k:26s} {v:5d}  {v/n*100:5.1f}%")
nb = len(ages)
print(f"=== 可控上下文 {nb} 个:最优对里较老那帧的最小帧龄分布 ===")
h = collections.Counter(min(a, 20) for a in ages); print("  ", {k: h[k] for k in sorted(h)}, "(20=≥20)")
print(f"  帧龄≥5: {sum(1 for a in ages if a >= 5)/max(nb,1)*100:.1f}%   ≥10: {sum(1 for a in ages if a >= 10)/max(nb,1)*100:.1f}%   存在两帧都≤2步的正确对: {near/max(nb,1)*100:.1f}%")
print("=== 各任务可控上下文(B)前 12 / 任务数", len(per_task))
for t, c in sorted(per_task.items(), key=lambda x: -x[1]["B_controllable"])[:12]:
    tot = sum(c.values()); print(f"  {t[:36]:36s} B={c['B_controllable']:3d}/{tot:3d}  A={c['A_recency_ok']:3d} C={c['C_floor']:3d}")
print("=== 零可控上下文的任务数:", sum(1 for c in per_task.values() if c["B_controllable"] == 0), "/", len(per_task))
