# note (luojiaxuan): 条件间取并集的 oracle(逐步任一条件对即算对)与按时滞/步位的拆分。
import json, sys, collections
p = sys.argv[1] if len(sys.argv) > 1 else "/data01/jaxan/androtmem/probe/results.jsonl"
R = collections.defaultdict(dict); meta = {}
for l in open(p):
    r = json.loads(l); k = (r["task_id"], r["step_i"]); R[k][r["cond"]] = r["score"]; meta[k] = r
conds = ["none", "recency2", "gold", "random2"]
full = [k for k, v in R.items() if all(c in v for c in conds)]
def rate(ks, f): return sum(f(R[k]) for k in ks) / max(len(ks), 1) * 100
print(f"n={len(full)}  union(any cond)={rate(full, lambda v: max(v.values())):.1f}%  none|recency2={rate(full, lambda v: max(v['none'], v['recency2'])):.1f}%  recency2|gold={rate(full, lambda v: max(v['recency2'], v['gold'])):.1f}%")
print("gold-only wins (gold ok, none&recency2 wrong):", sum(1 for k in full if R[k]["gold"] >= 1 and R[k]["none"] < 1 and R[k]["recency2"] < 1), "| random2-only wins:", sum(1 for k in full if R[k]["random2"] >= 1 and R[k]["none"] < 1 and R[k]["recency2"] < 1))
print("by lag bucket (none/recency2/gold/random2, n):")
for lo, hi in [(5, 7), (8, 12), (13, 20), (21, 999)]:
    ks = [k for k in full if meta[k]["lag"] is not None and lo <= meta[k]["lag"] <= hi]
    if ks: print(f"  lag {lo:>2}-{hi:<3} " + " ".join(f"{rate(ks, lambda v, c=c: v[c]):5.1f}" for c in conds) + f"  n={len(ks)}")
print("by step position (i/n_steps):")
for lo, hi in [(0, .33), (.33, .66), (.66, 1.01)]:
    ks = [k for k in full if lo <= meta[k]["step_i"] / meta[k]["n_steps"] < hi]
    if ks: print(f"  pos {lo:.2f}-{hi:.2f} " + " ".join(f"{rate(ks, lambda v, c=c: v[c]):5.1f}" for c in conds) + f"  n={len(ks)}")
