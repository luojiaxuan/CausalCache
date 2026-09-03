# note (luojiaxuan): 汇总探针结果:各条件 AMS、按 GT 动作类型、恢复率与伤害率。
import json, sys, collections
p = sys.argv[1] if len(sys.argv) > 1 else "/data01/jaxan/androtmem/probe/results.jsonl"
R = collections.defaultdict(dict); gt = {}
for l in open(p):
    r = json.loads(l); R[(r["task_id"], r["step_i"])][r["cond"]] = r["score"]; gt[(r["task_id"], r["step_i"])] = r["gt"]
conds = ["none", "recency2", "gold", "random2"]
full = [k for k, v in R.items() if all(c in v for c in conds)]
print(f"units with all conds={len(full)}")
for c in conds:
    s = [R[k][c] for k in full]; print(f"  AMS {c:9s} {sum(s)/max(len(s),1)*100:5.1f}%  (n={len(s)})")
print("  by GT type (none/recency2/gold/random2):")
by = collections.defaultdict(list)
for k in full: by[gt[k]].append(k)
for t, ks in sorted(by.items(), key=lambda x: -len(x[1]))[:6]:
    print(f"    {t:10s} n={len(ks):4d} " + " ".join(f"{sum(R[k][c] for k in ks)/len(ks)*100:5.1f}" for c in conds))
rec_wrong = [k for k in full if R[k]["recency2"] < 1]; none_wrong = [k for k in full if R[k]["none"] < 1]
print(f"  recovery P(gold ok | recency2 wrong) = {sum(R[k]['gold'] for k in rec_wrong)/max(len(rec_wrong),1)*100:.1f}% (n={len(rec_wrong)})")
print(f"  recovery P(gold ok | none wrong)     = {sum(R[k]['gold'] for k in none_wrong)/max(len(none_wrong),1)*100:.1f}% (n={len(none_wrong)})")
rec_ok = [k for k in full if R[k]["recency2"] >= 1]
print(f"  harm P(gold wrong | recency2 ok)     = {sum(1-R[k]['gold'] for k in rec_ok)/max(len(rec_ok),1)*100:.1f}% (n={len(rec_ok)})")
print(f"  harm P(random2 wrong | recency2 ok)  = {sum(1-R[k]['random2'] for k in rec_ok)/max(len(rec_ok),1)*100:.1f}%")
