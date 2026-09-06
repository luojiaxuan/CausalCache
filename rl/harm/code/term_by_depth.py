# note (luojiaxuan): 外审的卫生项——终止塌陷是否是"轨迹后段"的混杂:按状态的步深(k-1 = 历史长度)分层,报各条件的终止率与复现率,
# 以及 rec4 vs rec2 / grayturnin2 vs rec2 的配对差(按状态配对的 bootstrap 95% CI)。
import json, random, statistics as st
files = ["/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl", "/data01/jaxan/harm/dose_base.jsonl", "/data01/jaxan/harm/reply_pattern2_base.jsonl"]
recs = {}
for f in files:
    for l in open(f):
        r = json.loads(l); k = f"{r['dir']}|{r['step']}"; recs.setdefault(k, {"step": r["step"], "m": {}, "t": {}})
        for sp, m in r["match"].items():
            if m is None: continue
            recs[k]["m"][sp] = bool(m); recs[k]["t"][sp] = "terminate" in (r["decodes"].get(sp) or "").lower()
specs = ["rec2_deploy", "rec3_deploy", "rec4_deploy", "grayturnin1", "grayturnin2", "grayturnin4", "longtextturnin4", "hybridturnin_text:judge_glm46v|direct|six"]
bins = [(2, 5), (6, 10), (11, 20), (21, 50)]
print(f"{'步深 k-1':10s} {'n':>4s} | " + " | ".join(f"{sp[:14]:>14s}" for sp in specs))
print("  (每格:终止% / 复现%)")
for lo, hi in bins:
    ks = [k for k, r in recs.items() if lo <= r["step"] - 1 <= hi and all(sp in r["m"] for sp in specs)]
    if not ks: continue
    cells = []
    for sp in specs:
        cells.append(f"{100*st.mean(recs[k]['t'][sp] for k in ks):5.1f}/{100*st.mean(recs[k]['m'][sp] for k in ks):4.0f}")
    print(f"{lo:2d}–{hi:2d}      {len(ks):4d} | " + " | ".join(f"{c:>14s}" for c in cells))
def paired(a, b, ks, B=2000):
    d = [int(recs[k]["m"][a]) - int(recs[k]["m"][b]) for k in ks]; rng = random.Random(0); n = len(d)
    ms = sorted(st.mean(rng.choices(d, k=n)) for _ in range(B)); return 100*st.mean(d), 100*ms[int(.025*B)], 100*ms[int(.975*B)]
print("\n配对差(复现率,a − b,按状态 bootstrap 95% CI):")
for a, b in (("rec4_deploy", "rec2_deploy"), ("grayturnin2", "rec2_deploy"), ("grayturnin1", "rec2_deploy"), ("longtextturnin4", "rec2_deploy"), ("grayturnin2", "grayturnin1")):
    ks = [k for k, r in recs.items() if a in r["m"] and b in r["m"]]
    mu, lo, hi = paired(a, b, ks); print(f"  {a:18s} − {b:12s}: {mu:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] n={len(ks)}")
    for blo, bhi in bins:
        kk = [k for k in ks if blo <= recs[k]["step"] - 1 <= bhi]
        if len(kk) >= 20: mu, lo, hi = paired(a, b, kk); print(f"      步深 {blo:2d}–{bhi:2d}: {mu:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] n={len(kk)}")
