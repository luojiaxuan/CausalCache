# note (luojiaxuan): 检验"过早终止随带原文的 assistant 轮数上升,而非随图数上升":同一状态集上按规格列出
# (轮数, 图数) → 复现率、错误数、错误中 terminate 占比、总体 terminate 率。
import json, re, statistics as st
files = {"deploy": "/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl", "format": "/data01/jaxan/harm/format_base.jsonl",
         "format2": "/data01/jaxan/harm/format2_base.jsonl", "term": "/data01/jaxan/harm/term_interv_base.jsonl", "goal": "/data01/jaxan/harm/goal_interv_base.jsonl"}
recs = {}
for tag, p in files.items():
    try:
        for l in open(p):
            r = json.loads(l); k = f"{r['dir']}|{r['step']}"; recs.setdefault(k, {"match": {}, "dec": {}})
            for sp in r["match"]: recs[k]["match"][sp] = r["match"][sp]; recs[k]["dec"][sp] = r["decodes"].get(sp, "")
    except FileNotFoundError: print("missing", p)
# 轮数 = 带原文回复的 assistant 轮(不含 hybridturn 的 "Noted" 轮);图数 = 历史图 + 当前图
def shape(sp):
    m = re.match(r"rec(\d+)_deploy(_\w+)?$", sp)
    if m: n = int(m.group(1)); return (n, n + 1, "recN" + (m.group(2) or ""))
    m = re.match(r"irr(\d+)_deploy$", sp)
    if m: n = int(m.group(1)); return (n, n + 1, "irrN")
    kind = sp.split(":")[0]
    return {"pickimg": (2, 3, "pickimg 2轮3图(图换老帧)"), "hybrid": (2, 5, "hybrid 2轮+2标记图放首条"), "hybridlast": (2, 5, "hybridlast 2轮+2标记图放末条"),
            "hybridturn": (2, 5, "hybridturn 2轮+2参考轮(Noted 回复)")}.get(kind, (None, None, sp))
rows = {}
for k, r in recs.items():
    for sp, m in r["match"].items():
        if m is None: continue
        t = "terminate" in r["dec"][sp].lower()
        rows.setdefault(sp, []).append((bool(m), t))
print(f"{'规格':46s} {'轮':>3s} {'图':>3s} {'n':>4s} {'复现':>6s} {'错误':>5s} {'错误中term%':>10s} {'总term%':>8s}")
out = []
for sp, v in rows.items():
    a, im, lab = shape(sp); n = len(v); ok = sum(m for m, _ in v); err = n - ok; term_err = sum(t for m, t in v if not m); term_all = sum(t for _, t in v)
    out.append((a if a is not None else 99, im if im is not None else 99, sp, lab, n, ok, err, term_err, term_all))
for a, im, sp, lab, n, ok, err, term_err, term_all in sorted(out):
    print(f"{sp[:46]:46s} {a:3d} {im:3d} {n:4d} {100*ok/n:6.1f} {err:5d} {100*term_err/max(err,1):10.1f} {100*term_all/n:8.1f}   {lab}")
