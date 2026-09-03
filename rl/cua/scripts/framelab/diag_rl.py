import json, glob, collections
# A. 决策日志:按 pv 段看策略形态
buck = collections.defaultdict(lambda: {"n": 0, "rec": 0, "lp": 0.0, "nf": 0, "adj": 0})
for f in glob.glob("decisions/*.jsonl"):
    for l in open(f):
        try:
            r = json.loads(l)
        except Exception:
            continue
        pv = r.get("pv", 0); n = r.get("n_frames", 0); ch = sorted(r.get("chosen", []))
        if n < 3 or len(ch) != 2:
            continue
        b = "pv0-2(初始)" if pv <= 2 else ("pv10-12" if 10 <= pv <= 12 else ("pv20-22(当前)" if pv >= 20 else None))
        if b is None:
            continue
        d = buck[b]; d["n"] += 1; d["nf"] += n; d["lp"] += r["logp"]
        d["rec"] += (ch == [n - 2, n - 1]); d["adj"] += (ch[1] - ch[0] == 1)
print("=== A. 策略形态(仅 n_frames>=3 的决策) ===")
for b, d in sorted(buck.items()):
    print(f'{b:14s} n={d["n"]:6d} 选最近两帧={d["rec"]/d["n"]*100:5.1f}%  选相邻两帧={d["adj"]/d["n"]*100:5.1f}%  mean_logp={d["lp"]/d["n"]:.2f}  mean_nframes={d["nf"]/d["n"]:.1f}')
# B. RLOO 组信号
groups = collections.defaultdict(dict)
for l in open("returns.jsonl"):
    r = json.loads(l); rnd = int(r["tag"].split("t")[0][1:]); groups[(rnd, r["gmd5"])][r["tag"]] = r["score"]
mixed = tot = 0; per_rnd = collections.defaultdict(lambda: [0, 0])
for (rnd, g), t in groups.items():
    if len(t) < 2:
        continue
    tot += 1; m = 0 < sum(t.values()) < len(t); mixed += m
    per_rnd[rnd][0] += m; per_rnd[rnd][1] += 1
print(f"=== B. RLOO 组:有信号(混合结果)的组 {mixed}/{tot} = {mixed/tot*100:.1f}% ===")
print(" ".join(f"r{k}:{v[0]}/{v[1]}" for k, v in sorted(per_rnd.items())))
# C. r18 逐任务配对
def outcomes(label):
    o = {}
    for d in glob.glob(f"anchor/{label}/*/"):
        try:
            sc = float(open(d + "result.txt").read().split("score:")[1].split()[0])
        except Exception:
            continue
        o[d.rstrip("/").split("/")[-1]] = int(sc > 0)
    return o
L, R = outcomes("learned_r18"), outcomes("recency_r18")
print("=== C. r18 逐任务配对(learned/recency) ===")
both = sorted(set(L) & set(R))
print(f"配对任务数={len(both)}  learned胜={sum(L[t]>R[t] for t in both)}  recency胜={sum(L[t]<R[t] for t in both)}  同={sum(L[t]==R[t] for t in both)}")
for t in both:
    if L[t] != R[t]:
        print(f"  {t[:44]:44s} learned={L[t]} recency={R[t]}")
print("  双成功:", [t[:30] for t in both if L[t] == R[t] == 1])
