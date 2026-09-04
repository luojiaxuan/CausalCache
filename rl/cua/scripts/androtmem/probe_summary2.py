# note (luojiaxuan): 第二批条件的汇总:与第一批同单元配对比较。
import json, sys, collections
p = sys.argv[1] if len(sys.argv) > 1 else "/data01/jaxan/androtmem/probe/results.jsonl"
R = collections.defaultdict(dict)
for l in open(p):
    r = json.loads(l); R[(r["task_id"], r["step_i"])][r["cond"]] = r["score"]
conds = ["none", "recency2", "gold", "random2", "rec1_gold1", "rec2_gold1", "none_notext", "recency2_notext", "gold_notext"]
full = [k for k, v in R.items() if all(c in v for c in conds)]
print(f"units with all 9 conds={len(full)}")
for c in conds:
    s = [R[k][c] for k in full]; print(f"  AMS {c:16s} {sum(s)/max(len(s),1)*100:5.1f}%")
def P(cond_ok, given_wrong, given_ok=None):
    ks = [k for k in full if R[k][given_wrong] < 1] if given_ok is None else [k for k in full if R[k][given_ok] >= 1]
    return sum(R[k][cond_ok] for k in ks) / max(len(ks), 1) * 100, len(ks)
r, n = P("rec1_gold1", "recency2"); print(f"  recovery P(rec1_gold1 ok | recency2 wrong) = {r:.1f}% (n={n})")
r, n = P("rec2_gold1", "recency2"); print(f"  recovery P(rec2_gold1 ok | recency2 wrong) = {r:.1f}% (n={n})")
ks = [k for k in full if R[k]["recency2"] >= 1]
print(f"  harm P(rec1_gold1 wrong | recency2 ok) = {sum(1-R[k]['rec1_gold1'] for k in ks)/max(len(ks),1)*100:.1f}%   P(rec2_gold1 wrong | recency2 ok) = {sum(1-R[k]['rec2_gold1'] for k in ks)/max(len(ks),1)*100:.1f}% (n={len(ks)})")
print(f"  union(recency2 | rec2_gold1) = {sum(max(R[k]['recency2'], R[k]['rec2_gold1']) for k in full)/max(len(full),1)*100:.1f}%")
