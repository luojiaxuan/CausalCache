# note (luojiaxuan): history-harm Step 1 —— 按上下文规格对状态集补解(GUI-Owl 布局)。规格:
#   recN   最近 N 帧(N=0..6);irrN   来自别的轨迹的 N 帧(同图数对照);
#   recN_mark  最近 N 帧但每张前加 "[PAST screenshot, t-j]" 文本标记(Step 3 干预:时序消歧);
#   recN_blank 最近 N 帧换成同尺寸纯灰图(Step 3 干预:纯 token 数效应)。
import argparse, glob, hashlib, importlib.util, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("--oracle-py", default="/data01/jaxan/guiowl_oracle.py")
ap.add_argument("--prompts", default="/data01/jaxan/sglang-omni-rl/cc_recipe/sglang_omni_rl/gui_owl/prompts.py")
ap.add_argument("--roots", nargs="+", default=["/data01/jaxan/rl_v2/guiowl_base"])
ap.add_argument("--per-traj", type=int, default=12)
ap.add_argument("--specs", default="rec0,rec1,rec2,rec3,rec4,rec6,irr1,irr2,irr3,irr4,irr6")
ap.add_argument("--base-url", required=True); ap.add_argument("--model", default="gui-owl")
ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--no-text", action="store_true"); ap.add_argument("--workers", type=int, default=12)
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("go", args.oracle_py); go = importlib.util.module_from_spec(spec); spec.loader.exec_module(go)
ns = {}; exec(open(args.prompts).read(), ns)
sysprompt, tp, th = ns["SYSTEM_PROMPT"], ns["USER_PROMPT_TEMPLATE"], ns["USER_PROMPT_WITH_HISTSTEPS_TEMPLATE"]

# note (luojiaxuan): 官方布局的消息构造(与 guiowl_oracle.messages 相同,另支持 no_text=去掉全部文本 conclusion 与 irr=外部帧列表)。
# hyper00 上的 guiowl_oracle.py 没有 no_text 参数,这里自带一份以免依赖版本差异。
def messages(st, keep, irr=None, no_text=False):
    hist = irr if irr is not None else [st["shots"][i] for i in sorted(keep)]
    imgs = hist + [st["shots"][st["step"] - 1]]
    if st["concls"] and not no_text:
        prev = "\n".join(f"Step{j + 1}: {c}" for j, c in enumerate(st["concls"]))
        user = th.format(instruction=st["goal"], previous_steps=prev)
    else:
        user = tp.format(instruction=st["goal"])
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(p)}"}} for p in imgs]
    content.append({"type": "text", "text": user})
    return [{"role": "system", "content": sysprompt}, {"role": "user", "content": content}]
states = list(go.iter_states(args.roots, args.per_traj)); pool = {s["dir"]: s["shots"] for s in states}
BLANK = "/data01/jaxan/rl_v2/blank_gray.png"
if not os.path.exists(BLANK):
    from PIL import Image; Image.new("RGB", (1080, 2400), (128, 128, 128)).save(BLANK)

def irr_frames(st, n):
    others = [d for d in pool if d != st["dir"]]
    rng = random.Random(int(hashlib.md5(f"{st['dir']}|{st['step']}|irr{n}".encode()).hexdigest()[:8], 16))
    shots = pool[rng.choice(others)]
    return [shots[i] for i in sorted(rng.sample(range(len(shots)), min(n, len(shots))))]

# note (luojiaxuan): 部署忠实布局(与 gui_owl_1_5.py 逐行对应):第一条 user = [文本模板(只含**未保留为图**的步的 conclusion)+ 第一张保留帧],
# 之后每相邻两张保留帧之间插入该 turn 的 assistant 原始回复,最后一张 user 图是当前屏。标注器的"全部图在前 + 全部 conclusion"布局
# 与之不同(同协议 T=0 仅复现 55.8%),伤害必须在这个布局下也成立才算数。
_resp_cache = {}
def responses(st):
    d = st["dir"]
    if d not in _resp_cache:
        traj = list(json.load(open(os.path.join(d, "traj.json"))).values())[0]["traj"]
        _resp_cache[d] = {int(t["step"]): (t.get("prediction") or "").strip() for t in traj}
    return _resp_cache[d]

def add_period(c):
    c = (c or "").strip(); return c if not c or c[-1] in ".!?。" else c + "."

def messages_deploy(st, n, irr=None):
    k = st["step"]; total = k - 1                      # 0-based 当前 turn 下标 = total
    keep = min(n, total); text_cnt = total - keep
    Sidx = list(range(text_cnt, total + 1))            # 保留为图的 turn(含当前)
    frames = {i: st["shots"][i] for i in Sidx}
    if irr is not None:
        for i, p in zip(Sidx[:-1], irr): frames[i] = p
    resp = responses(st)
    if text_cnt:
        prev = "\n".join(f"Step{i + 1}: {add_period(st['concls'][i])}" for i in range(text_cnt))
        first_text = th.format(instruction=st["goal"], previous_steps=prev)
    else:
        first_text = tp.format(instruction=st["goal"])
    msgs = [{"role": "system", "content": sysprompt},
            {"role": "user", "content": [{"type": "text", "text": first_text},
                                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(frames[Sidx[0]])}"}}]}]
    for a in range(len(Sidx) - 1):
        turn = Sidx[a]
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": resp.get(turn + 1, "")}]})
        msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(frames[Sidx[a + 1]])}"}}]})
    return msgs

def build(st, spec_name):
    if spec_name.endswith("_deploy"):
        core = spec_name[:-7]; kind = core[:3]; n = int(core[3:])
        return messages_deploy(st, n, irr=(irr_frames(st, n) if kind == "irr" else None))
    k = st["step"]; kind = spec_name[:3]; rest = spec_name[3:]; n = int(rest.split("_")[0]); variant = rest.split("_")[1] if "_" in rest else ""
    if kind == "rec":
        keep = {k - 1 - j for j in range(1, n + 1) if k - 1 - j >= 0}
        if variant == "blank":
            fake = dict(st); fake["shots"] = [BLANK if i in keep else p for i, p in enumerate(st["shots"])]
            return messages(fake, keep, no_text=args.no_text)
        msgs = messages(st, keep, no_text=args.no_text)
        if variant == "mark":
            content = msgs[-1]["content"]; ages = sorted(keep); out = []; ii = 0
            for c in content:
                if c["type"] == "image_url" and ii < len(ages):
                    out.append({"type": "text", "text": f"[PAST screenshot from {k - 1 - ages[ii]} steps ago]"}); ii += 1
                elif c["type"] == "image_url":
                    out.append({"type": "text", "text": "[CURRENT screenshot]"})
                out.append(c)
            msgs[-1]["content"] = out
        return msgs
    if kind == "irr":
        return messages(st, set(), irr=irr_frames(st, n), no_text=args.no_text)
    raise ValueError(spec_name)

done = set()
if os.path.exists(args.out):
    for l in open(args.out):
        r = json.loads(l); done.add(f"{r['dir']}|{r['step']}")
todo = [s for s in states if f"{s['dir']}|{s['step']}" not in done]
specs = args.specs.split(","); print(f"states {len(states)} todo {len(todo)} specs {specs}", flush=True); lock = threading.Lock()
def run(st):
    ref = go.parse_action(st["target"]); rec = {"tag": args.tag, "dir": st["dir"], "task": st["task"], "step": st["step"], "target": st["target"], "decodes": {}, "match": {}}
    for sp in specs:
        out = go.post(args.base_url, {"model": args.model, "temperature": 0.0, "max_tokens": 512, "messages": build(st, sp)})
        txt = out["choices"][0]["message"]["content"] or ""
        rec["decodes"][sp] = txt; rec["match"][sp] = go.match(go.parse_action(txt), ref) if ref is not None else None
    with lock:
        with open(args.out, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
with ThreadPoolExecutor(args.workers) as ex: list(ex.map(run, todo))
print("DECODE_CTX_DONE", args.tag, flush=True)
