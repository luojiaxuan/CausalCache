# note (luojiaxuan): 机制 A 干预打分:部署布局 N=2/4/6 × {原样, noresp(回复只留 conclusion), hint(加"未完成"提示), short(回复置空)}
# 报动作复现率、过早终止率(错误里 terminate 的占比)、以及相对原样的配对差 95% CI。原样数据来自 harm_vs_n_base_deploy.jsonl。
import json, statistics as st, sys
sys.path.insert(0, "/data01/jaxan")
from guiowl_oracle import parse_action
def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l); d[f"{r['dir']}|{r['step']}"] = r
    return d
base = load("/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl"); iv = load("/data01/jaxan/harm/term_interv_base.jsonl")
keys = sorted(set(base) & set(iv)); print(f"共同状态 {len(keys)}")
def stats(getter):
    m = [bool(getter(k)["match"][sp]) for k in keys if getter(k)["match"].get(sp) is not None]
    term = [ (parse_action(getter(k)["decodes"][sp]) or {}).get("action") in ("terminate", "finished") for k in keys
             if getter(k)["match"].get(sp) is not None and not getter(k)["match"][sp]]
    return st.mean(m), (sum(term) / len(term) if term else float("nan")), len(m)
print(f"  {'规格':22s} {'复现率':>7s} {'错误中 terminate':>16s} {'vs 原样 (配对差 ±95%CI)':>26s}")
for n in (2, 4, 6):
    sp = f"rec{n}_deploy"; m0, t0, _ = stats(lambda k: base[k]); print(f"  {sp:22s} {100*m0:7.1f} {100*t0:16.0f}%")
    for v in ("noresp", "hint", "short"):
        sp = f"rec{n}_deploy_{v}"
        if not any(sp in iv[k]["match"] for k in keys): continue
        m1, t1, nn = stats(lambda k: iv[k])
        d = [int(bool(iv[k]["match"][sp])) - int(bool(base[k]["match"][f"rec{n}_deploy"])) for k in keys if iv[k]["match"].get(sp) is not None]
        mu = st.mean(d); se = (st.pvariance(d) / len(d)) ** 0.5
        print(f"  {sp:22s} {100*m1:7.1f} {100*t1:16.0f}% {100*mu:+8.1f} ±{100*1.96*se:4.1f} (n={nn})")
