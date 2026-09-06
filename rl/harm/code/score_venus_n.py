# note (luojiaxuan): Venus N 扫描汇总——V(recN)/V(irrN) 带 bootstrap CI、错误里 Finished(早停)占比、HTTP 错误计数。跨模型看 H5 是否 GUI-Owl 特有。
import json, random, statistics as st, sys, re, types, math
_src = open("/data01/jaxan/venus_oracle_verdict.py").read().split("def summarize", 1)[0]
vv = types.ModuleType("vv"); vv.__dict__.update({"re": re, "math": math, "ast": __import__("ast")}); exec(_src, vv.__dict__)
rows = [json.loads(l) for l in open("/data01/jaxan/harm/harm_vs_n_venus_text.jsonl")]
specs = ["rec0", "rec1", "rec2", "rec3", "rec4", "rec6", "rec8", "irr2", "irr4", "irr8"]
def ci(v, B=2000):
    r = random.Random(0); n = len(v); ms = sorted(st.mean(r.choices(v, k=n)) for _ in range(B)); return ms[int(.025*B)], ms[int(.975*B)]
print(f"UI-Venus-2-9B 冻结,保留文本轨迹,n={len(rows)}")
print(f"  {'规格':6s} {'复现率':>7s} {'95%CI':>16s} {'错误数':>6s} {'错误中 Finished':>14s} {'HTTP错误':>8s}")
for sp in specs:
    v = [bool(r["match"][sp]) for r in rows if r["match"].get(sp) is not None]
    if not v: continue
    lo, hi = ci(v)
    bad = [r for r in rows if r["match"].get(sp) is not None and not r["match"][sp]]
    fin = sum(1 for r in bad if (vv.parse(r["decodes"][sp]) or ("", {}))[0] == "Finished")
    http = sum(1 for r in rows if str(r["decodes"].get(sp, "")).startswith("__HTTP"))
    print(f"  {sp:6s} {100*st.mean(v):7.1f} [{100*lo:5.1f}, {100*hi:5.1f}] {len(bad):6d} {100*fin/max(len(bad),1):13.0f}% {http:8d}")
