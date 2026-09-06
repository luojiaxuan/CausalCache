# note (luojiaxuan): 从前缀轨迹建 B-pilot 的 checkpoint 规格(供 pilot_checkpoint_eval.py)。每个任务:
#   决策步 k = executor 第一次输出携带 expected 的动作(Type/Answer;PartMatch 为提到目标文件名)的那一步 → checkpoint 状态 = 第 k 步之前的全部历史;
#   源帧 = 打开证据后、离开该视图(Back/Home)那一步观察到的帧:第 j 轮(0-based)的文本提到证据关键词(供应商名或其邮箱域 / 订单号 / approved_sample)且动作是点击,
#         之后第一次离开动作在第 b 步 → 帧 b−1(没有离开动作则帧 j+1);
#   孪生 = 同族同对另一版(源帧取它自己检出的源帧,expected_swap = 它的答案);无关 = 同族另一对 A 版的源帧(同一工作流阶段)。
# 源帧落在最近两帧窗口内(idx ≥ k−3)的 checkpoint 标 in_recency,不进主终点。可选 --contact 输出每个 checkpoint 的拼图供人工核对。
import argparse, ast, glob, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--prefix-dir", required=True); ap.add_argument("--seeds", required=True)
ap.add_argument("--backend", choices=["owl", "venus"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--contact", default="")
args = ap.parse_args()

def norm(s): return re.sub(r"[\s,]+", " ", str(s)).strip().lower()

def seeds():
    out = {}
    for line in open(args.seeds):
        m = re.match(r"(\w+) pair=(\d+) twin=(\d): (.*)$", line.strip())
        if not m: continue
        fam, pair, twin, rest = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        exp = rest.split("expected=", 1)[1].strip()
        t = {"family": fam, "pair": pair, "twin": twin, "task": f"{fam}Task{pair:02d}{'AB'[twin]}", "expected": exp}
        if fam == "QuoteRecall":
            sup = re.search(r"target=(.*?)/", rest).group(1)
            t["kind"] = "text"; t["keys"] = [[sup, f"@{sup.split()[0].lower()}."]]
        elif fam == "OrderAddressJoin":
            sup = re.search(r"supplier=(.*?) expected=", rest).group(1)
            t["kind"] = "text"; t["keys"] = [[re.search(r"order=(\S+)", rest).group(1)], [sup, f"@{sup.split()[0].lower()}."]]
        else:
            t["kind"] = "file"; t["keys"] = [["approved_sample"]]
        out[t["task"]] = t
    return out

def shots_of(d):
    return sorted(glob.glob(os.path.join(d, "screenshots", "*.png")), key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))

def action_of(txt):
    if args.backend == "owl":
        m = re.search(r'"arguments":\s*(\{.*?\})', txt, re.S)
        try: return json.loads(m.group(1)) if m else {}
        except json.JSONDecodeError: return {}
    m = re.search(r"<action>(.*?)</action>", txt, re.S)
    if not m: return {}
    mm = re.match(r"(\w+)\((.*)\)\s*$", m.group(1).strip(), re.S)
    if not mm: return {}
    try:
        tree = ast.parse("_(" + mm.group(2).replace("\n", "\\n") + ")", mode="eval")
        p = {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
    except (SyntaxError, ValueError):
        p = {}
    p["action"] = {"Click": "click", "Type": "type", "Answer": "answer", "CallUser": "answer", "Finished": "answer",
                   "LongPress": "click", "DoubleClick": "click"}.get(mm.group(1), mm.group(1).lower()); return p

def is_click(a): return a.get("action") in ("click", "long_press", "double_click")

def leaves_view(a, txt):
    return a.get("action") in ("system_button", "pressback", "presshome") or bool(re.search(r"\b(back|exit|home screen)\b", txt.split("<tool_call>")[0], re.I))

def carries(txt, t):
    a = action_of(txt); exp = norm(t["expected"])
    if t["kind"] == "file": return exp in norm(txt)
    if a.get("action") not in ("type", "answer"): return False
    body = norm(a.get("text") or a.get("content") or "")
    try:
        e = float(exp); return any(abs(float(n) - e) < 0.005 for n in re.findall(r"\d+(?:\.\d+)?", body.replace(",", "")))
    except ValueError:
        return exp in body or exp.split(" ")[0] in body and exp.split(",")[0] in body

def build(d, t):
    data = json.load(open(os.path.join(d, "traj.json")))   # note (luojiaxuan): 系统边界:采集中的 episode 目录 traj.json 可能尚为空
    if not data: return None
    traj = list(data.values())[0].get("traj") or []
    preds = {s["step"]: s.get("prediction") or "" for s in traj}; shots = shots_of(d); n = len(preds)
    k = next((s for s in range(1, n + 1) if carries(preds[s], t)), None)
    src = []
    for aliases in t["keys"]:
        j = next((j for j in range(0, n) if any(norm(a) in norm(preds[j + 1]) for a in aliases) and is_click(action_of(preds[j + 1]))), None)
        if j is None or j + 1 >= len(shots):
            src.append(None); continue
        # note (luojiaxuan): 证据屏可能要经过选择器(Files 的"用哪个应用打开")才真正显示;取离开该视图(Back/Home/"back"字样)那一步
        # 观察到的帧 = 证据显示得最完整的一帧;决策步之前没有离开动作则取点击后的下一帧。
        b = next((b for b in range(j + 2, (k or n) + 1) if leaves_view(action_of(preds[b]), preds[b])), None)
        src.append(b - 1 if b and b - 1 < len(shots) else j + 1)
    return {"dir": d, "task": t["task"], "family": t["family"], "pair": t["pair"], "twin": t["twin"], "expected": t["expected"],
            "expected_kind": t["kind"], "step": k, "n_steps": n, "source_frames": src,
            "score": open(os.path.join(d, "result.txt")).read().split("score:")[1].split()[0] if os.path.exists(os.path.join(d, "result.txt")) else None}

S = seeds(); rows = {}
for d in sorted(glob.glob(os.path.join(args.prefix_dir, "*/"))):
    name = os.path.basename(d.rstrip("/"))
    if name in S and os.path.exists(os.path.join(d, "traj.json")):
        r = build(d.rstrip("/"), S[name])
        if r: rows[name] = r
specs = []
for name, r in rows.items():
    ok_src = all(i is not None for i in r["source_frames"])
    r["valid"] = bool(r["step"] and ok_src and all(i < r["step"] - 1 for i in r["source_frames"]))
    r["in_recency"] = bool(r["valid"] and any(i >= r["step"] - 3 for i in r["source_frames"]))
    twin = rows.get(f"{r['family']}Task{r['pair']:02d}{'BA'[r['twin']]}")
    if twin and all(i is not None for i in twin["source_frames"]):
        r["swap_frames_dir"] = twin["dir"]; r["swap_source_frames"] = twin["source_frames"]; r["expected_swap"] = twin["expected"]
    others = [o for o in rows.values() if o["family"] == r["family"] and o["pair"] != r["pair"] and o["twin"] == 0 and all(i is not None for i in o["source_frames"])]
    if others:
        o = others[(r["pair"] - 1) % len(others)]; r["irrelevant_dir"] = o["dir"]; r["irrelevant_frames"] = o["source_frames"]
    if r["valid"]: specs.append(r)
with open(args.out, "w") as f:
    for r in specs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
fams = sorted({r["family"] for r in rows.values()})
print(f"tasks={len(rows)} valid_checkpoints={len(specs)} (in_recency={sum(r['in_recency'] for r in specs)}) -> {args.out}")
for fam in fams:
    rs = [r for r in rows.values() if r["family"] == fam]
    print(f"  {fam:18s} n={len(rs):2d} succ={sum(r['score']=='1.0' for r in rs):2d} decision_step={sum(bool(r['step']) for r in rs):2d} "
          f"all_sources={sum(all(i is not None for i in r['source_frames']) for r in rs):2d} valid={sum(r['valid'] for r in rs):2d} "
          f"with_twin={sum('swap_frames_dir' in r for r in rs if r['valid']):2d}")
for r in sorted(rows.values(), key=lambda r: r["task"]):
    print(f"    {r['task']:24s} steps={r['n_steps']:2d} score={r['score']} k={r['step']} src={r['source_frames']} valid={r['valid']}")
if args.contact and specs:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL unavailable; no contact sheets"); sys.exit(0)
    os.makedirs(args.contact, exist_ok=True)
    for r in specs:
        shots = shots_of(r["dir"]); frames = [(f"src {i}", shots[i]) for i in r["source_frames"]] + [(f"rec {r['step']-3}", shots[r['step']-3]), (f"rec {r['step']-2}", shots[r['step']-2]), (f"now {r['step']-1}", shots[r['step']-1])]
        ims = [Image.open(p).convert("RGB").resize((270, 600)) for _, p in frames]
        sheet = Image.new("RGB", (280 * len(ims), 630), "white"); dr = ImageDraw.Draw(sheet)
        for i, ((lab, _), im) in enumerate(zip(frames, ims)):
            sheet.paste(im, (i * 280 + 5, 25)); dr.text((i * 280 + 8, 6), lab, fill="black")
        sheet.save(os.path.join(args.contact, f"{r['task']}.png"))
    print(f"contact sheets -> {args.contact}")
