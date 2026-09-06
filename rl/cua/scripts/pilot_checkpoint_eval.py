# note (luojiaxuan): B-pilot 的 checkpoint 干预矩阵。两种 executor 后端,各用自己的**部署协议**构造上下文:
#   owl   GUI-Owl 交错多轮布局(复用 harm 线 decode_ctx.py 的 messages_deploy / messages_hybrid);
#   venus UI-Venus-2 官方多轮协议(复用 venus_oracle.messages;--no-text 为无文本诊断口径)。
# 主条件 = "保留哪几轮"这一个控制变量,其余全按部署协议(保留一轮 = 该轮截图 + 该轮原始回复一起保留、并从折叠文本里抽走):
#   text_only   B=0                      rec2        保留最近两轮(部署默认)
#   src_keep    保留源帧所在轮             ctrl_keep   保留同龄的无证据轮(同一轨迹里源帧前一帧,通常是列表屏)
#   swap_keep   源帧所在轮整轮换成孪生另一版的对应轮(截图 + 原始回复;判定按另一版答案)
#   gold_text   文本里直接给出事实(checkpoint 有效性前提)
# Venus 的对应物:src_at_turn / ctrl_at_turn / swap_at_turn(Venus 协议文本恒保留,只有图按轮窗口化)。
# 无文本口径(--no-text)下"要哪一项"只存在于请求帧(短信屏),所有图条件一律带上 request_frames;gold_text 的事实以文本行附在当前屏前。
# 机制诊断(不作闸门):src_pickimg / swap_pickimg / irr2(最近两帧槽位只换图)、src_hybrid / swap_hybrid / judge_hybrid:<judge>(附带 PAST 标记的额外图)。
# 每行另记 leak 标志:expected 是否已出现在 goal 或决策步之前的自写文本里(竞争通道,不是泄漏;用于分层)。
import argparse, ast, glob, importlib.util, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True, help="jsonl: dir, step, source_frames, expected, expected_kind, irrelevant_dir, irrelevant_frames, swap_frames_dir, swap_source_frames, expected_swap, judge_picks(可选)")
ap.add_argument("--backend", choices=["owl", "venus"], required=True)
ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--no-text", action="store_true", help="venus: 去掉全部文本推理历史(诊断口径)")
ap.add_argument("--text-mode", choices=["full", "oneline", "action"], default="full",
                help="venus: 历史文本保真度——full=官方协议(整段 think+action);oneline=每轮只留 think 首句+action(模拟一行摘要);action=只留 action")
ap.add_argument("--rescore", action="store_true", help="不解码:读 --out 里已保存的解码文本重新判定并汇总")
ap.add_argument("--scales", default="", help="venus: 额外跑源帧缩放版 src_at_turn@<s>(s 为边长比例,逗号分隔),用于'精简视觉 token vs 详细文本'的预算曲线")
ap.add_argument("--conds", default="", help="只跑这些条件(逗号分隔);空 = 全部")
args = ap.parse_args()

go = load("/data01/jaxan/guiowl_oracle.py", "go")
if args.rescore:
    pass
elif args.backend == "owl":
    sys.argv = ["decode_ctx.py", "--base-url", args.base_url, "--tag", "x", "--out", "/dev/null", "--specs", "rec0", "--roots", "/nonexistent"]
    dc = load("/data01/jaxan/decode_ctx.py", "dc")
else:
    vo = load("/data01/jaxan/venus_oracle.py", "vo")

def shots_of(d):
    return sorted(glob.glob(os.path.join(d, "screenshots", "*.png")), key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))

def squeeze(p):
    # note (luojiaxuan): 把 Venus 一轮的原文压成"一行摘要 + 动作"或"只有动作",模拟常见 GUI agent 的文本历史保真度。
    if args.text_mode == "full": return p
    act = re.search(r"<action>.*?</action>", p, re.S); act = act.group(0) if act else ""
    if args.text_mode == "action": return act
    th = re.search(r"<think>(.*?)</think>", p, re.S); first = re.split(r"(?<=[.!?。])\s+", th.group(1).strip(), maxsplit=1)[0] if th else ""
    return f"<think>{first}</think>\n{act}"

def state_of(d, k):
    traj = list(json.load(open(os.path.join(d, "traj.json"))).values())[0].get("traj") or []
    preds = {s["step"]: (squeeze(s.get("prediction") or "") if args.backend == "venus" else (s.get("prediction") or "")) for s in traj}
    hist = [preds[j] for j in range(1, k)]
    return {"dir": d, "task": os.path.basename(d.rstrip("/")), "goal": traj[0].get("task_goal", ""), "step": k, "shots": shots_of(d),
            "concls": [go.extract_conclusion(h) for h in hist], "hist": hist, "preds": preds, "target": preds.get(k, "")}

def norm(s):
    return re.sub(r"[\s,]+", " ", str(s)).strip().lower()

# note (luojiaxuan): PartMatch 的决策屏是 Files 的列表视图(Documents/Candidates,cand_a..cand_f 按名排,六行全可见,
# 行心约在 0–1000 归一化纵坐标 317/397/477/557/637/717,行高 80);模型点击行而非报文件名,所以按点击纵坐标落到哪一行判定。
def grid_file(x, y):
    row = int((y - 277) // 80)
    return f"cand_{'abcdef'[row]}.png" if 0 <= row < 6 else ""

def click_point(txt):
    if args.backend == "owl":
        act = go.parse_action(txt)
        if act and act.get("action") in ("click", "long_press", "double_click") and act.get("coordinate"): return tuple(act["coordinate"][:2])
        return None
    m = re.search(r"<action>\s*(Click|LongPress|DoubleClick)\(point=\((\d+),\s*(\d+)\)", txt)
    return (int(m.group(2)), int(m.group(3))) if m else None

def carries_text(body, exp, kind, whole):
    exp = norm(exp)
    if kind == "file":
        if exp in norm(whole): return 1
        pt = click_point(whole); return int(pt is not None and grid_file(*pt) == exp)
    try:
        e = float(exp); return int(any(abs(float(n) - e) < 0.005 for n in re.findall(r"\d+(?:\.\d+)?", body.replace(",", ""))))
    except ValueError:
        return int(exp in norm(body))

def carries(txt, exp, kind):
    # note (luojiaxuan): file 类(PartMatch)的决策是点击候选文件,判定看整段输出是否指名目标文件;文本类只看 type/answer 的内容。
    if kind == "file": return carries_text("", exp, kind, txt)
    if args.backend == "owl":
        act = go.parse_action(txt)
        if act is None or act["action"] not in ("type", "answer"): return 0
        return carries_text(act.get("text") or "", exp, kind, txt)
    m = re.search(r"<action>(.*?)</action>", txt, re.S)
    if not m: return 0
    mm = re.match(r"(\w+)\((.*)\)\s*$", m.group(1).strip(), re.S)
    if not mm or mm.group(1) not in ("Type", "Answer", "CallUser", "Finished"): return 0
    try:
        tree = ast.parse("_(" + mm.group(2).replace("\n", "\\n") + ")", mode="eval")
        body = str({kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}.get("content", ""))
    except (SyntaxError, ValueError):
        body = mm.group(2)
    return carries_text(body, exp, kind, txt)

def decode(msgs):
    if args.backend == "owl":
        out = go.post(args.base_url, {"model": args.model, "temperature": 0.0, "max_tokens": 256, "messages": msgs})
    else:
        out = vo.post(args.base_url, {"model": args.model, "temperature": 0.0, "max_tokens": 2048, "messages": msgs,
                                      "repetition_penalty": 1.05, "frequency_penalty": 0.3})
    # note (luojiaxuan): vLLM 的 usage.prompt_tokens 含图像 token,直接作为该条件的上下文成本。
    return out["choices"][0]["message"]["content"] or "", out.get("usage", {}).get("prompt_tokens")

def scaled(path, scale):
    # note (luojiaxuan): 缩放截图(边长比例 scale)后缓存到盘,给"精简视觉 token"条件用;像素数按 scale² 降,图像 token 随之降。
    from PIL import Image
    d = os.path.join("/data01/jaxan/rl_v2/pilot/scaled", f"{scale:g}"); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, os.path.basename(os.path.dirname(os.path.dirname(path))) + "_" + os.path.basename(path))
    if not os.path.exists(out):
        im = Image.open(path); im.resize((max(28, int(im.width * scale)), max(28, int(im.height * scale)))).save(out)
    return out

def slots(st):
    k = st["step"]; return [k - 3, k - 2] if k >= 3 else [0, 1]

# ---- owl 构造器 ----
def owl_hybrid(st, frames):
    msgs = dc.messages_deploy(st, 2); k = st["step"]; extra = []
    for age, path in frames:
        extra += [{"type": "text", "text": f"[PAST screenshot from {age} steps ago, for reference only]"},
                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(path)}"}}]
    msgs[1]["content"] = [msgs[1]["content"][0]] + extra + msgs[1]["content"][1:]
    return msgs

def owl_gold(st, exp):
    g = dict(st); g["concls"] = st["concls"][:-1] + [(st["concls"][-1] if st["concls"] else "") + f" Note: the requested item is {exp}."]
    return dc.messages_deploy(g, 0)

# ---- venus 构造器 ----
def venus_slots(st, frames):
    fake = dict(st); fake["shots"] = list(st["shots"]); idx = slots(st)
    for i, p in zip(idx, frames): fake["shots"][i] = p
    return vo.messages(fake, set(idx[:len(frames)]), args.no_text)

def venus_at_turn(st, turn_idx, frames):
    fake = dict(st); fake["shots"] = list(st["shots"])
    for i, p in zip(turn_idx, frames): fake["shots"][i] = p
    return vo.messages(fake, set(turn_idx[:len(frames)]), args.no_text)

def venus_gold(st, exp, req):
    if args.no_text:
        msgs = vo.messages(st, set(req), True)
        msgs[-1]["content"].insert(0, {"type": "text", "text": f"Note: the requested item is {exp}.\n"}); return msgs
    g = dict(st); g["hist"] = list(st["hist"])
    note = f"<think>Note: the requested item is {exp}.</think>"
    g["hist"] = g["hist"][:-1] + [(g["hist"][-1] if g["hist"] else "") + note]
    return vo.messages(g, set(), False)

def swapped_state(st, src, twin_dir, twin_idx):
    # note (luojiaxuan): 整轮反事实:源帧所在轮的截图与原始回复都换成孪生另一版对应轮的,其余轮不动。
    tw = state_of(twin_dir, max(twin_idx) + 2); g = dict(st); g["shots"] = list(st["shots"]); g["hist"] = list(st["hist"]); g["concls"] = list(st["concls"]); g["preds"] = dict(st["preds"])
    for i, j in zip(src, twin_idx):
        g["shots"][i] = tw["shots"][j]; g["hist"][i] = tw["hist"][j]; g["concls"][i] = tw["concls"][j]; g["preds"][i + 1] = tw["preds"][j + 1]
    return g

def conditions(sp, st):
    k = st["step"]
    src = [i for i in sp["source_frames"] if 0 <= i < k - 1][:2]
    ctrl = [i for i in sp.get("control_frames", []) if i is not None and 0 <= i < k - 1][:2]
    src_paths = [st["shots"][i] for i in src]
    swap_paths, sw_idx = [], []
    if sp.get("swap_frames_dir") and sp.get("expected_swap") is not None:
        s3 = shots_of(sp["swap_frames_dir"]); sw_idx = [i for i in sp.get("swap_source_frames", src) if i < len(s3)][:2]
        swap_paths = [s3[i] for i in sw_idx]
    irr_paths = []
    if sp.get("irrelevant_dir"):
        s2 = shots_of(sp["irrelevant_dir"]); ii = [i for i in sp.get("irrelevant_frames", src) if i < len(s2)][:2]
        irr_paths = [s2[i] for i in ii]
    c = {}
    if args.backend == "owl":
        c["text_only"] = dc.messages_deploy(st, 0); c["rec2"] = dc.messages_deploy(st, 2)
        if src_paths:
            c["src_keep"] = dc.messages_deploy(st, 0, keep_set=set(src))
            c["src_pickimg"] = dc.messages_deploy(st, 2, irr=src_paths)
            c["src_hybrid"] = owl_hybrid(st, [(k - 1 - i, p) for i, p in zip(src, src_paths)])
        if ctrl: c["ctrl_keep"] = dc.messages_deploy(st, 0, keep_set=set(ctrl))
        if swap_paths:
            if len(sw_idx) == len(src): c["swap_keep"] = dc.messages_deploy(swapped_state(st, src, sp["swap_frames_dir"], sw_idx), 0, keep_set=set(src))
            c["swap_pickimg"] = dc.messages_deploy(st, 2, irr=swap_paths)
            c["swap_hybrid"] = owl_hybrid(st, [(k - 1 - i, p) for i, p in zip(sw_idx, swap_paths)])
        if irr_paths: c["irr2"] = dc.messages_deploy(st, 2, irr=irr_paths)
        for jname, pk in (sp.get("judge_picks") or {}).items():
            pk = [i for i in pk if 0 <= i < k - 1][:2]
            if pk: c[f"judge_hybrid:{jname}"] = owl_hybrid(st, [(k - 1 - i, st["shots"][i]) for i in pk])
        c["gold_text"] = owl_gold(st, sp["expected"])
    else:
        # note (luojiaxuan): 文本里没有请求内容的口径(无文本、只留 action)下,"要哪一项"只在短信屏上,所有图条件都带请求帧。
        req = set(i for i in sp.get("request_frames", []) if 0 <= i < k - 1) if (args.no_text or args.text_mode == "action") else set()
        c["text_only"] = vo.messages(st, req, args.no_text); c["rec2"] = vo.messages(st, req | set(slots(st)), args.no_text)
        if src_paths:
            c["src_at_turn"] = vo.messages(st, req | set(src), args.no_text)
        if ctrl: c["ctrl_at_turn"] = vo.messages(st, req | set(ctrl), args.no_text)
        if swap_paths and len(sw_idx) == len(src):
            c["swap_at_turn"] = vo.messages(swapped_state(st, src, sp["swap_frames_dir"], sw_idx), req | set(src), args.no_text)
        if irr_paths and len(irr_paths) == len(src):
            g = dict(st); g["shots"] = list(st["shots"])
            for i, p in zip(src, irr_paths): g["shots"][i] = p
            c["irr_at_turn"] = vo.messages(g, req | set(src), args.no_text)
        c["gold_text"] = venus_gold(st, sp["expected"], req)
    return c

lock = threading.Lock()
def run(sp):
    st = state_of(sp["dir"], sp["step"]); kind = sp.get("expected_kind", "text")
    exp = sp["expected"]
    leak = {"goal": int(norm(exp) in norm(st["goal"])), "hist": int(any(norm(exp) in norm(h) for h in st["hist"]))}
    rec = {"tag": args.tag, "backend": args.backend, "no_text": args.no_text, "text_mode": args.text_mode, "dir": st["dir"], "task": st["task"], "family": sp.get("family", ""),
           "pair": sp.get("pair"), "step": st["step"], "expected": exp, "expected_swap": sp.get("expected_swap"), "leak": leak, "decodes": {}, "hit": {}}
    rec["ptoks"] = {}; only = set(x for x in args.conds.split(",") if x)
    for name, msgs in conditions(sp, st).items():
        if only and name not in only: continue
        txt, pt = decode(msgs); rec["decodes"][name] = txt; rec["ptoks"][name] = pt
        rec["hit"][name] = carries(txt, sp["expected_swap"] if name.startswith("swap_") else exp, kind)
    with lock:
        with open(args.out, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

specs = [json.loads(l) for l in open(args.spec)]
if args.rescore:
    kinds = {sp["dir"]: sp.get("expected_kind", "text") for sp in specs}; swaps = {sp["dir"]: sp.get("expected_swap") for sp in specs}
    rows = [json.loads(l) for l in open(args.out) if json.loads(l)["tag"] == args.tag]
    for r in rows:
        r["hit"] = {c: carries(t, swaps[r["dir"]] if c.startswith("swap_") else r["expected"], kinds[r["dir"]]) for c, t in r["decodes"].items()}
    with open(args.out, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
else:
    with ThreadPoolExecutor(args.workers) as ex: list(ex.map(run, specs))
    rows = [json.loads(l) for l in open(args.out) if json.loads(l)["tag"] == args.tag]
conds = sorted({c for r in rows for c in r["hit"]})
print(f"[{args.tag}] backend={args.backend} no_text={args.no_text} n={len(rows)}  leak goal={sum(r['leak']['goal'] for r in rows)} hist={sum(r['leak']['hist'] for r in rows)}")
print("  命中率(动作携带正确答案;swap_* 按另一版答案判):")
for c in conds:
    xs = [r["hit"][c] for r in rows if c in r["hit"]]; pts = [r["ptoks"][c] for r in rows if r.get("ptoks", {}).get(c)]
    print(f"  {c:26s} {sum(xs)/max(len(xs),1):.3f} (n={len(xs)})" + (f"  prompt_tokens median={sorted(pts)[len(pts)//2]}" if pts else ""))
