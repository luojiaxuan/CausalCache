# note (luojiaxuan): 部署布局下裁判帧对的净信息量(与主线 score_judge 同定义:只在"不给图会错"的状态上,裁判救回率 − 无关帧救回率,配对 95% CI)。
import json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
rs = [r for r in rows if r["match"].get("rec0_deploy") is not None]
bad = [r for r in rs if not r["match"]["rec0_deploy"]]
print(f"n={len(rs)}  不给图会错={len(bad)} ({len(bad)/len(rs):.1%})   [部署忠实布局]")
keys = [k for k in rs[0]["match"] if k != "rec0_deploy"]
def V(k, pool): xs = [bool(r["match"][k]) for r in pool if r["match"].get(k) is not None]; return (st.mean(xs), len(xs)) if xs else (float("nan"), 0)
print(f"  {'上下文':42s} {'全集 V':>7s} {'救回率':>7s}")
for k in keys:
    v, _ = V(k, rs); b, nb = V(k, bad); print(f"  {k:42s} {v:7.3f} {b:7.3f} (n={nb})")
p_irr, _ = V("irr2_deploy", bad)
print(f"  --- 净信息量 = 救回率 − 无关帧救回率({p_irr:.3f}) ---")
for k in keys:
    if not k.startswith("pick:") and k != "rec2_deploy": continue
    d = [int(bool(r["match"][k])) - int(bool(r["match"]["irr2_deploy"])) for r in bad if r["match"].get(k) is not None and r["match"].get("irr2_deploy") is not None]
    if len(d) < 5: continue
    m = st.mean(d); se = (st.pvariance(d) / len(d)) ** 0.5
    print(f"    {k:40s} {m:+.3f} (±{1.96*se:.3f}, n={len(d)})")
