# note (luojiaxuan): 检验"过早终止随带原文的 assistant 轮数上升,而非随图数上升":同一状态集上按规格列出
# (轮数, 图数) → 复现率、错误数、错误中 terminate 占比、总体 terminate 率。
import json, re, statistics as st
files = {"deploy": "/data01/jaxan/harm/harm_vs_n_base_deploy.jsonl", "format": "/data01/jaxan/harm/format_base.jsonl",
         "format2": "/data01/jaxan/harm/format2_base.jsonl", "term": "/data01/jaxan/harm/term_interv_base.jsonl", "goal": "/data01/jaxan/harm/goal_interv_base.jsonl", "replypat": "/data01/jaxan/harm/reply_pattern_base.jsonl", "replypat2": "/data01/jaxan/harm/reply_pattern2_base.jsonl", "dose": "/data01/jaxan/harm/dose_base.jsonl", "dose2": "/data01/jaxan/harm/dose2_base.jsonl", "dose3": "/data01/jaxan/harm/dose3_base.jsonl", "dose4": "/data01/jaxan/harm/dose4_base.jsonl", "dose5": "/data01/jaxan/harm/dose5_base.jsonl"}
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
    m = re.match(r"(grayturnin|longtextturnin|graybefore|graystackturnin|grayfracturnin)(\d+)(_noterm)?$", sp)
    if m and m.group(1) == "graystackturnin": d = int(m.group(2)); return (2, 4, f"{sp}: 1 个参考轮,{d} 张灰图拼成一张(token≈{d} 张,图块 1)插在指令之后")
    m2 = re.match(r"grayfracturnin(\d+)$", sp)
    if m2: return (2, 4, f"{sp}: 1 个参考轮,{int(m2.group(1))/10:.1f} 张截图高的灰图插在指令之后")
    if m: d = int(m.group(2)); return (2, 3 + (0 if m.group(1) == "longtextturnin" else d), f"{sp}: {d} 个{'长文本' if m.group(1) == 'longtextturnin' else '灰图'}参考轮插在指令{'之前' if m.group(1) == 'graybefore' else '之后'}{' 禁terminate' if m.group(3) else ''}")
    if sp.endswith("_noterm") and ":" in sp: sp2 = sp[:-7]; a, im, lab = shape(sp2); return (a, im, lab + " 禁terminate")
    kind = sp.split(":")[0]
    return {"pickimg": (2, 3, "pickimg 2轮3图(图换老帧)"), "hybrid": (2, 5, "hybrid 2轮+2标记图放首条"), "hybridlast": (2, 5, "hybridlast 2轮+2标记图放末条"),
            "hybridturn": (2, 5, "hybridturn 2轮+2参考轮(Noted 回复,插在指令之前)"), "hybridturnin": (2, 5, "hybridturnin 2轮+2参考轮(Noted 回复,插在指令之后)"), "hybridturnin_gray": (2, 5, "hybridturnin_gray 参考轮换灰图,插在指令之后"), "hybridturnin_text": (2, 3, "hybridturnin_text 参考轮只有文本,插在指令之后")}.get(kind, (None, None, sp))
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
