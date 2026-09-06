# note (luojiaxuan): 从前缀轨迹建 B-pilot 的 checkpoint 规格(供 pilot_checkpoint_eval.py)。每个任务:
#   决策步 k:PartMatch 按结构定义 = 看过样品之后、候选列表(Candidates)第一次出现在屏幕上的那一步(不看 agent 当时选没选对,避免按结果选态);
#            Mail 两族暂按结果定义 = executor 第一次输出携带 expected 的动作(Type/Answer)的那一步;k_outcome 一律另记。checkpoint 状态 = 第 k 步之前的全部历史;
#   源帧 = 证据正文显示在屏幕上的那一帧:观察帧 f 时写下的文本(preds[f+1])同时提到该证据的身份关键词(供应商名或其邮箱域 / 订单号 /
#         approved_sample)与只在正文里出现的内容关键词(报价项 / "placed with" / "shipping address" / 样品形状描述;收件箱主题行里
#         也出现的词不能用)的最早 f;
#   对照帧 = 与源帧同龄的无证据帧(源帧前一帧,通常是收件箱或文件夹列表;若它本身是源帧则取后一帧)——"保留老图 = 连该轮原始回复一起保留"
#         的控制变量要求对照也是同一轨迹里的一个真实轮次;
#   请求帧(Mail 两族)= 打开短信线程后的帧:无文本口径下"要哪一项"只存在于这一屏,所有图条件都要带上它,否则模型无从知道该答什么;
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
            t["kind"] = "text"; t["keys"] = [[sup, f"@{sup.split()[0].lower()}."]]; t["content"] = [["unit price", "delivery charge", "lead time", "valid for 14"]]
        elif fam == "OrderAddressJoin":
            sup = re.search(r"supplier=(.*?) expected=", rest).group(1)
            t["kind"] = "text"; t["keys"] = [[re.search(r"order=(\S+)", rest).group(1)], [sup]]
            t["content"] = [["placed with", "pallets", "net 30"], ["shipping address", "vendor management"]]
        else:
            t["kind"] = "file"; t["keys"] = [["approved_sample", "approved sample", "sample image"]]
            t["content"] = [[r"\b(shape|triangle|hexagon|gear|circle|square|star|ring|colou?r|gr[ae]y|red|blue|green|orange|purple|yellow|hole|image shows)\b"]]
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
    k_out = next((s for s in range(1, n + 1) if carries(preds[s], t)), None)
    k = k_out
    src = []
    for aliases, words in zip(t["keys"], t["content"]):
        if t["family"] == "PartMatch":
            # note (luojiaxuan): Files 要经过"用哪个应用打开"选择器,证据真正显示的帧 = 写下形状/颜色描述时观察的那一帧;
            # 描述词按整词匹配(否则 start/whole/during 之类会误命中 star/hole/ring)。GUI-Owl 的文本只写动作不写所见,
            # 对它改用"点击 approved_sample 后第一次离开视图(Back)那一步观察到的帧"。
            f = next((f for f in range(0, min(n, len(shots)) - 1) if any(norm(a) in norm(preds[f + 1]) for a in aliases)
                      and any(re.search(w, norm(preds[f + 1])) for w in words)), None)
            if f is None and args.backend == "owl":
                j = next((j for j in range(0, n) if "approved_sample" in norm(preds[j + 1]) and is_click(action_of(preds[j + 1]))), None)
                if j is not None:
                    b = next((b for b in range(j + 2, n + 1) if leaves_view(action_of(preds[b]), preds[b])), None)
                    f = b - 1 if b and j + 1 <= b - 1 < len(shots) else None
        else:
            # note (luojiaxuan): Mail:点击主题行含身份关键词的那封邮件(第 j+1 步的文本提到它且动作是点击)→ 打开后的帧 j+1;
            # 要求观察该帧时写下的文本含只在正文里出现的内容词,或下一步是离开动作,以确认邮件确实打开了。
            f = None
            for j in range(0, min(n, len(shots)) - 1):
                if not (any(norm(a) in norm(preds[j + 1]) for a in aliases) and is_click(action_of(preds[j + 1]))): continue
                seen = preds.get(j + 2, ""); opened = any(w in norm(seen) for w in words) or leaves_view(action_of(seen), seen)
                if opened: f = j + 1; break
        src.append(f)
    req = None
    if t["family"] != "PartMatch":
        for j in range(0, min(n, len(shots)) - 1):
            txt = norm(preds[j + 1])
            if is_click(action_of(preds[j + 1])) and any(a in txt for a in ("987-6543", "987 6543", "sms", "message", "conversation", "text from")):
                seen = norm(preds.get(j + 2, ""))
                if any(w in seen for w in ("logistics needs", "send accounting", "quoted by", "please send", "shipping address for order", "the quote emails")):
                    req = j + 1; break
    ctrl = []
    for f in src:
        if f is None: ctrl.append(None); continue
        if t["family"] == "PartMatch":
            # note (luojiaxuan):Approved 文件夹列表与其上的"用哪个应用打开"选择器都露出样品缩略图,同龄无证据帧只能取 Files 打开时的 Download 列表(帧 1)。
            c = 1
        else:
            c = f - 1 if f - 1 >= 0 and (f - 1) not in src else f + 1
        ctrl.append(c if 0 <= c < len(shots) else None)
    if t["family"] == "PartMatch" and src and src[0] is not None:
        # note (luojiaxuan): 决策屏 = 候选列表第一次在屏幕上:观察该屏时写下的文本会点名候选文件(cand_*);按"点击 Candidates 文件夹"定位
        # 会在导航文字提到 Documents/Candidates 时提前触发(Documents 列表被当成决策屏)。
        k = next((s_ for s_ in range(src[0] + 2, n + 1) if "cand_" in norm(preds[s_])), None)
        # note (luojiaxuan): 候选列表上屏之前经过的帧里若有 Approved 文件夹列表(含样品缩略图),对照帧取它之后的无缩略图列表帧。
    return {"dir": d, "task": t["task"], "family": t["family"], "pair": t["pair"], "twin": t["twin"], "expected": t["expected"],
            "expected_kind": t["kind"], "step": k, "k_outcome": k_out, "n_steps": n, "source_frames": src, "control_frames": ctrl, "request_frames": [] if req is None else [req],
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
    r["valid"] = bool(r["step"] and ok_src and all(i < r["step"] - 1 for i in r["source_frames"]) and all(c is not None and c < r["step"] - 1 for c in r["control_frames"])
                      and (r["family"] == "PartMatch" or r["request_frames"]))
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
    print(f"    {r['task']:24s} steps={r['n_steps']:2d} score={r['score']} k={r['step']} src={r['source_frames']} ctrl={r['control_frames']} req={r['request_frames']} valid={r['valid']}")
if args.contact and specs:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL unavailable; no contact sheets"); sys.exit(0)
    os.makedirs(args.contact, exist_ok=True)
    for r in specs:
        shots = shots_of(r["dir"]); frames = [(f"req {i}", shots[i]) for i in r["request_frames"]] + [(f"src {i}", shots[i]) for i in r["source_frames"]] + [(f"ctrl {i}", shots[i]) for i in r["control_frames"]] + [(f"rec {r['step']-3}", shots[r['step']-3]), (f"rec {r['step']-2}", shots[r['step']-2]), (f"now {r['step']-1}", shots[r['step']-1])]
        ims = [Image.open(p).convert("RGB").resize((270, 600)) for _, p in frames]
        sheet = Image.new("RGB", (280 * len(ims), 630), "white"); dr = ImageDraw.Draw(sheet)
        for i, ((lab, _), im) in enumerate(zip(frames, ims)):
            sheet.paste(im, (i * 280 + 5, 25)); dr.text((i * 280 + 8, 6), lab, fill="black")
        sheet.save(os.path.join(args.contact, f"{r['task']}.png"))
    print(f"contact sheets -> {args.contact}")
