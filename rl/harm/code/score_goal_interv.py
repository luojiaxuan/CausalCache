# note (luojiaxuan): H5′ 干预打分——指令重复到最后一条消息(goal)vs 原样,N=2/4/6:复现率、错误里 terminate 占比、配对差 CI。
import json, statistics as st, sys
sys.path.insert(0, "/data01/jaxan")
from guiowl_oracle import parse_action
def load(p):
    d = {}
    for l in open(p): r = json.loads(l); d[f"{r['dir']}|{r['step']}"] = r
    return d
base = load("/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl"); g = load("/data01/jaxan/harm/goal_interv_base.jsonl")
keys = sorted(set(base) & set(g)); print(f"共同状态 {len(keys)}")
def term_share(getter, sp):
    bad = [k for k in keys if getter(k)["match"].get(sp) is not None and not getter(k)["match"][sp]]
    t = sum(1 for k in bad if (parse_action(getter(k)["decodes"][sp]) or {}).get("action") in ("terminate", "finished")); return len(bad), (t / len(bad) if bad else float("nan"))
print(f"  {'规格':20s} {'复现率':>7s} {'错误中 terminate':>16s} {'vs 原样 (±95%CI)':>20s}")
for n in (2, 4, 6):
    b = f"rec{n}_deploy"; s = f"rec{n}_deploy_goal"
    m0 = st.mean(bool(base[k]["match"][b]) for k in keys); nb, t0 = term_share(lambda k: base[k], b)
    m1 = st.mean(bool(g[k]["match"][s]) for k in keys if g[k]["match"].get(s) is not None); nb1, t1 = term_share(lambda k: g[k], s)
    d = [int(bool(g[k]["match"][s])) - int(bool(base[k]["match"][b])) for k in keys if g[k]["match"].get(s) is not None]
    mu = st.mean(d); se = (st.pvariance(d) / len(d)) ** 0.5
    print(f"  {b:20s} {100*m0:7.1f} {100*t0:16.0f}%")
    print(f"  {s:20s} {100*m1:7.1f} {100*t1:16.0f}% {100*mu:+8.1f} ±{100*1.96*se:4.1f}")
