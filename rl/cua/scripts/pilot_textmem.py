# note (luojiaxuan): 外审 §3.1 判别实验的核心:给"称职的在线文本记忆"一个公平的位置,与图像检索在同一历史 token 预算下对打。
# 浏览阶段(不知道之后要问什么)为每一帧写两种档案:notes = 一句结构化笔记;ocr = 稠密文本转写。决策步上按预算填充上下文:
#   none      不给历史图、不给档案(地板)          img_rec   最近帧(原尺寸)
#   img_src   请求帧 + 证据帧(oracle 上界)        img_src35 同上,边长 0.35
#   notes     全部笔记,新到旧填满预算              notes_q   按与任务目标的词面相似度排序后填满
#   ocr       全部转写,新到旧填满预算              ocr_q     同上按相似度
#   gold      事实直接给出(有效性对照)
# agent 自己的文本历史一律压成"只留动作",即业界常态的一行摘要制式——图在这个制式下才有余量(台账 2026-09-06 17:15 PT)。
import argparse, glob, importlib.util, json, os, re, threading
from concurrent.futures import ThreadPoolExecutor

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True); ap.add_argument("--base-url", required=True)
ap.add_argument("--model", default="UI-Venus-2"); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--budgets", default="2000,4000,8000,16000"); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--archive-only", action="store_true")
args = ap.parse_args()

vo = load("/data01/jaxan/venus_oracle.py", "vo")
IMG_TOK = 2500          # 全分辨率手机截图的 prompt token 数(实测 rec2 与 rec0 之差)
IMG_TOK35 = 350         # 边长 0.35
CHARS_PER_TOK = 3.5

NOTE_PROMPT = ("You are keeping a running memory while operating a phone. In at most 40 words, note this screen: which app, "
               "what it shows, and every concrete value visible (numbers, prices, addresses, names, ids, file names). "
               "You do not know what you will be asked later. Output the note only.")
OCR_PROMPT = "Transcribe every readable line of text on this screen, in reading order. No commentary."

def shots_of(d):
    return sorted(glob.glob(os.path.join(d, "screenshots", "*.png")), key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))

def action_of(raw):
    m = re.search(r"<action>.*?</action>", raw, re.S); return m.group(0) if m else ""

def state_of(d, k):
    traj = list(json.load(open(os.path.join(d, "traj.json"))).values())[0]["traj"]
    preds = {s["step"]: s.get("prediction") or "" for s in traj}
    return {"dir": d, "goal": traj[0].get("task_goal", ""), "step": k, "shots": shots_of(d),
            "hist": [action_of(preds[j]) for j in range(1, k)]}

def ask(img_path, prompt):
    body = {"model": args.model, "temperature": 0.0, "max_tokens": 512,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{vo.b64(img_path)}"}}]}]}
    return (vo.post(args.base_url, body)["choices"][0]["message"]["content"] or "").strip()

def archive(d, k):
    # note (luojiaxuan): 档案在"浏览阶段"写成,只看当帧、不知道后续请求;按目录缓存,重跑免费。
    path = os.path.join(d, "archive_v1.json")
    arc = json.load(open(path)) if os.path.exists(path) else {"notes": {}, "ocr": {}}
    shots = shots_of(d); todo = [j for j in range(0, min(k - 1, len(shots))) if str(j) not in arc["notes"]]
    for j in todo:
        arc["notes"][str(j)] = ask(shots[j], NOTE_PROMPT)
        arc["ocr"][str(j)] = ask(shots[j], OCR_PROMPT)
    if todo: json.dump(arc, open(path, "w"), ensure_ascii=False)
    return arc

WORD = re.compile(r"[a-z0-9]+")
def sim(query, text):
    q = set(WORD.findall(query.lower())); t = WORD.findall(text.lower())
    return sum(w in q for w in t) / (len(t) ** 0.5 + 1e-6)

def pack(items, budget):
    # items: [(idx, text)] 按已定顺序;按估算 token 填到预算为止
    out, used = [], 0
    for j, t in items:
        n = len(t) / CHARS_PER_TOK
        if used + n > budget: break
        out.append((j, t)); used += n
    return sorted(out), used

def messages(st, image_turns, archive_block, scale_paths=None):
    msgs = [{"role": "system", "content": vo.SYSTEM_PROMPT.format(user_task=st["goal"])}]
    for j, raw in enumerate(st["hist"]):
        content = ""
        if j in image_turns:
            p = (scale_paths or {}).get(j, st["shots"][j])
            content = [{"type": "text", "text": "History Screenshot:"},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{vo.b64(p)}"}}]
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": raw})
    if archive_block:
        msgs.append({"role": "user", "content": [{"type": "text", "text": archive_block}]})
        msgs.append({"role": "assistant", "content": "Noted."})
    msgs.append({"role": "user", "content": [{"type": "text", "text": "Current Screenshot:\n"},
                                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{vo.b64(st['shots'][st['step'] - 1])}"}}]})
    return msgs

def block(kind, items):
    head = {"notes": "[Memory archive: notes you wrote while browsing]", "ocr": "[Memory archive: text transcribed from screens you visited]"}[kind]
    return head + "\n" + "\n".join(f"- step {j + 1}: {t}" for j, t in items)

def scaled(path, s):
    from PIL import Image
    d = f"/data01/jaxan/rl_v2/pilot/scaled/{s:g}"; os.makedirs(d, exist_ok=True)
    out = os.path.join(d, os.path.basename(os.path.dirname(os.path.dirname(path))) + "_" + os.path.basename(path))
    if not os.path.exists(out):
        im = Image.open(path); im.resize((max(28, int(im.width * s)), max(28, int(im.height * s)))).save(out)
    return out

def carries(txt, exp):
    m = re.search(r"<action>(.*?)</action>", txt, re.S)
    if not m: return 0
    mm = re.match(r"(\w+)\((.*)\)\s*$", m.group(1).strip(), re.S)
    if not mm or mm.group(1) not in ("Type", "Answer", "CallUser", "Finished"): return 0
    body = mm.group(2).lower().replace(",", ""); exp = str(exp).strip().lower()
    try:
        e = float(exp.replace(",", "")); return int(any(abs(float(n) - e) < 0.005 for n in re.findall(r"\d+(?:\.\d+)?", body)))
    except ValueError:
        return int(exp.replace(",", "") in body)

lock = threading.Lock()
def run(sp):
    d, k = sp["dir"], sp["step"]
    st = state_of(d, k); arc = archive(d, k)
    if args.archive_only:
        print(f"{sp['task']} archived {len(arc['notes'])} frames", flush=True); return
    src = [i for i in sp["source_frames"] + sp.get("request_frames", []) if 0 <= i < k - 1]
    n_hist = k - 1
    notes = [(j, arc["notes"][str(j)]) for j in range(n_hist) if str(j) in arc["notes"]]
    ocr = [(j, arc["ocr"][str(j)]) for j in range(n_hist) if str(j) in arc["ocr"]]
    rec = {"tag": args.tag, "task": sp["task"], "family": sp["family"], "pair": sp["pair"], "dir": d, "step": k,
           "expected": sp["expected"], "n_hist": n_hist, "hit": {}, "ptoks": {}, "fill": {}}
    conds = {}
    for B in [int(x) for x in args.budgets.split(",")]:
        n_img = int(B // IMG_TOK); n_img35 = int(B // IMG_TOK35)
        conds[f"img_rec@{B}"] = ("img", sorted(range(max(0, n_hist - n_img), n_hist))[:n_img], None, None)
        conds[f"img_src@{B}"] = ("img", sorted(src)[:n_img], None, None)
        conds[f"img_src35@{B}"] = ("img35", sorted(src)[:n_img35], None, None)
        for kind, items in (("notes", notes), ("ocr", ocr)):
            newest, used = pack(list(reversed(items)), B)
            conds[f"{kind}@{B}"] = ("txt", [], block(kind, newest), used)
            ranked, used_q = pack(sorted(items, key=lambda it: -sim(st["goal"], it[1])), B)
            conds[f"{kind}_q@{B}"] = ("txt", [], block(kind, ranked), used_q)
    conds["none"] = ("img", [], None, None)
    conds["gold"] = ("txt", [], f"[Memory archive]\n- the requested item is {sp['expected']}", None)
    for name, (mode, turns, blk, used) in conds.items():
        if mode == "img35":
            paths = {j: scaled(st["shots"][j], 0.35) for j in turns}
            msgs = messages(st, set(turns), blk, paths)
        else:
            msgs = messages(st, set(turns), blk)
        out = vo.post(args.base_url, {"model": args.model, "temperature": 0.0, "max_tokens": 2048, "messages": msgs,
                                      "repetition_penalty": 1.05, "frequency_penalty": 0.3})
        txt = out["choices"][0]["message"]["content"] or ""
        rec["hit"][name] = carries(txt, sp["expected"]); rec["ptoks"][name] = out.get("usage", {}).get("prompt_tokens")
        rec["fill"][name] = round(used) if used is not None else len(turns)
    with lock:
        with open(args.out, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{sp['task']} done", flush=True)

specs = [json.loads(l) for l in open(args.spec)]
with ThreadPoolExecutor(args.workers) as ex: list(ex.map(run, specs))
if args.archive_only: raise SystemExit
rows = [json.loads(l) for l in open(args.out) if json.loads(l)["tag"] == args.tag]
names = sorted({c for r in rows for c in r["hit"]}, key=lambda n: (n.split("@")[-1].rjust(6), n))
print(f"[{args.tag}] n={len(rows)}  历史 token 预算 × 记忆策略(命中率 / 实测 prompt_tokens 中位 / 填充量)")
for c in names:
    xs = [r["hit"][c] for r in rows if c in r["hit"]]; pts = sorted(r["ptoks"][c] for r in rows if r["ptoks"].get(c))
    fl = sorted(r["fill"][c] for r in rows if c in r["fill"])
    print(f"  {c:18s} {sum(xs)/max(1,len(xs)):.3f} (n={len(xs)})  tok={pts[len(pts)//2] if pts else '-'}  fill={fl[len(fl)//2] if fl else '-'}")
