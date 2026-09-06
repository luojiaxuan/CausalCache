# note (luojiaxuan): 格式解耦打分——rec2(原生)vs pickimg(结构固定、图换裁判老帧)vs hybrid(原生 + 带标记的检索帧)。
# 以 rec2_deploy 为基线报配对差;"不给图会错"子集用 harm_vs_n_base_deploy 的 rec0_deploy 判定。
import json, statistics as st, sys
fmt = {}
for l in open(sys.argv[1] if len(sys.argv) > 1 else "/data01/jaxan/harm/format_base.jsonl"):
    r = json.loads(l); fmt[f"{r['dir']}|{r['step']}"] = r
base = {}
for l in open("/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl"):
    r = json.loads(l); base[f"{r['dir']}|{r['step']}"] = r
keys = sorted(set(fmt) & set(base)); bad = [k for k in keys if not base[k]["match"]["rec0_deploy"]]
print(f"状态 {len(keys)},其中不给图会错 {len(bad)}")
specs = [s for s in fmt[keys[0]]["match"]]
print(f"  {'规格':36s} {'全集V':>6s} {'救回率':>7s} {'vs rec2 (配对 ±95%CI)':>24s}")
for sp in specs:
    v = [bool(fmt[k]["match"][sp]) for k in keys if fmt[k]["match"].get(sp) is not None]
    b = [bool(fmt[k]["match"][sp]) for k in bad if fmt[k]["match"].get(sp) is not None]
    d = [int(bool(fmt[k]["match"][sp])) - int(bool(fmt[k]["match"]["rec2_deploy"])) for k in keys if fmt[k]["match"].get(sp) is not None and fmt[k]["match"].get("rec2_deploy") is not None]
    mu = st.mean(d) if d else float("nan"); se = (st.pvariance(d) / len(d)) ** 0.5 if len(d) > 1 else 0
    print(f"  {sp:36s} {100*st.mean(v):6.1f} {100*st.mean(b) if b else float('nan'):7.1f} {100*mu:+8.1f} ±{100*1.96*se:4.1f} (n={len(d)})")
