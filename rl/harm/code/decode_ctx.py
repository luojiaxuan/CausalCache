# note (luojiaxuan): history-harm Step 1 —— 按上下文规格对状态集补解(GUI-Owl 布局)。规格:
#   recN   最近 N 帧(N=0..6);irrN   来自别的轨迹的 N 帧(同图数对照);
#   recN_mark  最近 N 帧但每张前加 "[PAST screenshot, t-j]" 文本标记(Step 3 干预:时序消歧);
#   recN_blank 最近 N 帧换成同尺寸纯灰图(Step 3 干预:纯 token 数效应)。
import argparse, glob, hashlib, importlib.util, json, os, random, re, sys, threading
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
ap.add_argument("--picks", default="", help="glob;给出时 specs 里可用 pick:<judge>|<mode>|<cands>_deploy")
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

# note (luojiaxuan): 机制 A(N≥4 过早终止)的干预变体:
#   variant="noresp"  交错轮次里 assistant 回复只留 "Action: <conclusion>" 一行(去掉 <tool_call> 与长文本)——测"冗长的回复历史"是否是诱因;
#   variant="hint"    首条 user 文本末尾加一句 "The task is NOT finished yet; do not terminate unless the goal is verifiably complete."——测提示能否压制;
#   variant="short"   保留 N 张图但 assistant 回复换成空串——极端版 noresp(只剩图与轮次结构)。
#   variant="noted"   保留 N 张图与轮次,但最近两轮之前的 assistant 回复换成 "Noted the screenshot."——测"已完成动作的序列"是否是终止的触发物。
def messages_deploy(st, n, irr=None, keep_set=None, variant=""):
    k = st["step"]; total = k - 1                      # 0-based 当前 turn 下标 = total
    if keep_set is None:
        keep = min(n, total); text_cnt = total - keep
        Sidx = list(range(text_cnt, total + 1))        # 保留为图的 turn(含当前)
        text_idx = list(range(text_cnt))
    else:
        # note (luojiaxuan): 任意保留集合(裁判帧对):与 agent 的 _cc_text_idx 同法,非保留步压成 conclusion 文本
        Sidx = sorted({i for i in keep_set if 0 <= i < total}) + [total]
        text_idx = [i for i in range(total) if i not in Sidx]
    frames = {i: st["shots"][i] for i in Sidx}
    if irr is not None:
        for i, p in zip(Sidx[:-1], irr): frames[i] = p
    resp = responses(st)
    if text_idx:
        prev = "\n".join(f"Step{i + 1}: {add_period(st['concls'][i])}" for i in text_idx)
        first_text = th.format(instruction=st["goal"], previous_steps=prev)
    else:
        first_text = tp.format(instruction=st["goal"])
    if variant == "hint":
        first_text += "\nNote: the task is NOT finished yet. Do not terminate unless the goal is verifiably complete on the current screen."
    # note (luojiaxuan): H5′ 目标稀释的干预:把任务指令**再放一遍**到最后一条 user 消息(当前帧旁边),让它靠近决策点。
    goal_tail = f"\nReminder of the task: {st['goal']}" if variant == "goal" else ""
    msgs = [{"role": "system", "content": sysprompt},
            {"role": "user", "content": [{"type": "text", "text": first_text},
                                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(frames[Sidx[0]])}"}}]}]
    for a in range(len(Sidx) - 1):
        turn = Sidx[a]
        rtxt = resp.get(turn + 1, "")
        if variant == "noresp": rtxt = "Action: " + add_period(st["concls"][turn]) if turn < len(st["concls"]) else rtxt
        elif variant == "short": rtxt = ""
        elif variant == "noted" and a < len(Sidx) - 3: rtxt = "Noted the screenshot."
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": rtxt}]})
        msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(frames[Sidx[a + 1]])}"}}]})
    if goal_tail:
        if len(msgs) > 2: msgs[-1]["content"].append({"type": "text", "text": goal_tail})
        else: msgs[1]["content"].append({"type": "text", "text": goal_tail})
    return msgs

PICKS = {}
def load_picks(pattern):
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            r = json.loads(l); PICKS.setdefault(f"{r['dir']}|{r['step']}", {})[f"{r['judge']}|{r['mode']}|{r['cands']}"] = r["pick"]

# note (luojiaxuan): 部署布局里"选哪几帧"与"哪几个 turn 的 assistant 回复保留原文"绑在一起——pick:* 把裁判选的老帧变成保留 turn,
# 同时把最近两步压成 conclusion,结构和内容一起变了。两个解耦的规格:
#   pickimg:<judge>   结构固定为最近两 turn(回复原文),只把这两个 turn 的**图**换成裁判帧(与 irr2_deploy 同法,图不同)→ 纯内容效应;
#   hybrid:<judge>    最近两 turn 原样保留,裁判帧作为带 "[PAST screenshot from j steps ago]" 标记的额外图放进第一条 user 消息 → 候选的检索友好格式。
# note (luojiaxuan): hybrid 的三种放法(GUI-Owl 原生格式是"每个 user turn 恰好一张图",额外图放哪里可能决定成败):
#   first  放在第一条 user 消息(文本之后、首帧之前)——首发版,109 态上 −27pp;
#   last   放在最后一条 user 消息、当前帧之前(离决策最近);
#   turn   作为额外的 user/assistant 轮次插在最近两轮之前:user=[标记文本+图],assistant="(reference only)"——不破坏"一 turn 一图"。
#   turnin 同 turn,但插在首条 user 消息(指令)及其回复之后——测"参考轮在指令之后"是否触发终止。
def messages_hybrid(st, picks, where="first"):
    msgs = messages_deploy(st, 2); k = st["step"]
    items = [(j, {"type": "text", "text": f"[PAST screenshot from {k - 1 - j} steps ago, for reference only]"},
              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(st['shots'][j])}"}})
             for j in sorted(i for i in picks if 0 <= i < k - 1)]
    extra = [c for _, t, im in items for c in (t, im)]
    if where == "first":
        msgs[1]["content"] = [msgs[1]["content"][0]] + extra + msgs[1]["content"][1:]
    elif where == "last":
        msgs[-1]["content"] = extra + msgs[-1]["content"]
    elif where == "turn":
        ins = []
        for _, t, im in items:
            ins += [{"role": "user", "content": [t, im]}, {"role": "assistant", "content": [{"type": "text", "text": "Noted the reference screenshot."}]}]
        msgs = [msgs[0]] + ins + msgs[1:]  # system | 参考轮次(user/assistant 交替) | 原生首条 user(文本+首帧)及其后交错轮次——保持角色交替
    elif where == "turnin":
        ins = []
        for _, t, im in items:
            ins += [{"role": "user", "content": [t, im]}, {"role": "assistant", "content": [{"type": "text", "text": "Noted the reference screenshot."}]}]
        msgs = msgs[:3] + ins + msgs[3:]  # system | 首条 user + 其 assistant 回复 | 参考轮次 | 其余交错轮次——参考轮落在指令消息之后、最近两轮之前
    elif where in ("turnin_gray", "turnin_text"):
        # note (luojiaxuan): 同 turnin 的位置,但参考轮的内容换成灰图(同 token、无内容)或纯文本(无图、少 token),分辨"按 token 算"还是"按轮算"。
        ins = []
        for j, t, im in items:
            if where == "turnin_gray":
                content = [t, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(BLANK)}"}}]
            else:
                content = [{"type": "text", "text": f"[PAST step {j + 1}, {k - 1 - j} steps ago, for reference only: {add_period(st['concls'][j])}]"}]
            ins += [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": "Noted the reference."}]}]
        msgs = msgs[:3] + ins + msgs[3:]
    return msgs

# note (luojiaxuan): 触发物的剂量曲线——M 个"内容为空"的参考轮插在指令消息之后、最近两轮之前:
#   grayturninM      每轮一张灰图(与真实截图同 token 数)+ "Noted" 回复;
#   longtextturninM  每轮一段与一张图 token 数相当(约 4,400 字符)的纯文本(早期 conclusion 循环拼接)+ "Noted" 回复。
def messages_dose(st, m, kind, where="after"):
    msgs = messages_deploy(st, 2); k = st["step"]
    concls = [add_period(c) for c in st["concls"][:max(k - 1, 1)]] or ["No earlier step."]
    ins = []
    # note (luojiaxuan): kind="stack":只有 1 个参考轮,里面是 m 张灰图竖向拼成的一张图(token ≈ m 张,图块 = 1)——分辨"图块数"与"图 token 数"。
    if kind == "stack":
        stack = BLANK.replace(".png", f"_x{m}.png")
        content = [{"type": "text", "text": "[PAST screenshots, for reference only]"},
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(stack)}"}}]
        ins = [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": "Noted the reference."}]}]
        return msgs[:3] + ins + msgs[3:]
    for i in range(m):
        if kind == "gray":
            content = [{"type": "text", "text": "[PAST screenshot, for reference only]"},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(BLANK)}"}}]
        else:
            txt = ""; j = i
            while len(txt) < 4400: txt += f"[PAST step note {j % len(concls) + 1}: {concls[j % len(concls)]}] "; j += 1
            content = [{"type": "text", "text": txt}]
        ins += [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": "Noted the reference."}]}]
    if where == "before": return [msgs[0]] + ins + msgs[1:]      # system | 参考轮 | 指令消息及其后
    return msgs[:3] + ins + msgs[3:]

# note (luojiaxuan): 规格名尾缀 _noterm = 解码时用 logit_bias 禁掉 "terminate" 的首 token(外审建议:若复现率回到 ~90%,说明长视觉历史
# 只是把策略推进"假完成"模式,而非普遍破坏能力)。token id 在启动时通过 vLLM 的 /tokenize 取得。
NOTERM_BIAS = {}
def noterm_ids(base_url):
    import urllib.request
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    # note (luojiaxuan): 只禁"terminate"这个词本身的首 token;带前引号的写法会把 `"`(id 1)也禁掉,破坏 JSON 输出——首轮 dose2 犯过这个错,
    # 那一轮的 *_noterm 结果作废。每个候选先 detokenize 回来核对,首 token 必须以 "term"/"Term" 开头才收。
    ids = {}
    for txt in ["terminate", " terminate", "Terminate", " Terminate"]:
        req = urllib.request.Request(root + "/tokenize", data=json.dumps({"model": args.model, "prompt": txt, "add_special_tokens": False}).encode(), headers={"Content-Type": "application/json"})
        toks = json.loads(urllib.request.urlopen(req, timeout=60).read())["tokens"]
        req2 = urllib.request.Request(root + "/detokenize", data=json.dumps({"model": args.model, "tokens": [toks[0]]}).encode(), headers={"Content-Type": "application/json"})
        piece = json.loads(urllib.request.urlopen(req2, timeout=60).read())["prompt"]
        if piece.strip().lower().startswith("term"): ids[toks[0]] = piece
    print("noterm tokens:", ids, flush=True)
    return {str(i): -100 for i in ids}

def build(st, spec_name):
    m = re.match(r"(grayturnin|longtextturnin|graybefore|graystackturnin)(\d+)$", spec_name)
    if m: return messages_dose(st, int(m.group(2)), {"longtextturnin": "text", "graystackturnin": "stack"}.get(m.group(1), "gray"), where="before" if m.group(1) == "graybefore" else "after")
    if spec_name.split(":")[0] in ("pickimg", "hybrid", "hybridlast", "hybridturn", "hybridturnin", "hybridturnin_gray", "hybridturnin_text"):
        kind, key = spec_name.split(":", 1); key = key[:-7] if key.endswith("_deploy") else key
        pk = PICKS.get(f"{st['dir']}|{st['step']}", {}).get(key)
        if pk is None: return None
        if kind == "hybrid": return messages_hybrid(st, pk, where="first")
        if kind == "hybridlast": return messages_hybrid(st, pk, where="last")
        if kind == "hybridturn": return messages_hybrid(st, pk, where="turn")
        if kind == "hybridturnin": return messages_hybrid(st, pk, where="turnin")
        if kind == "hybridturnin_gray": return messages_hybrid(st, pk, where="turnin_gray")
        if kind == "hybridturnin_text": return messages_hybrid(st, pk, where="turnin_text")
        return messages_deploy(st, 2, irr=[st["shots"][i] for i in sorted(pk)[:2]])
    if spec_name.startswith("pick:"):                  # pick:<judge>|<mode>|<cands>_deploy
        key = spec_name[5:-7]; pk = PICKS.get(f"{st['dir']}|{st['step']}", {}).get(key)
        if pk is None: return None
        return messages_deploy(st, 0, keep_set=set(pk))
    if "_deploy" in spec_name:                          # recN_deploy[_noresp|_hint|_short]
        core, _, variant = spec_name.partition("_deploy"); variant = variant.lstrip("_")
        kind = core[:3]; n = int(core[3:])
        return messages_deploy(st, n, irr=(irr_frames(st, n) if kind == "irr" else None), variant=variant)
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

if args.picks: load_picks(args.picks)
done = set()
if os.path.exists(args.out):
    for l in open(args.out):
        r = json.loads(l); done.add(f"{r['dir']}|{r['step']}")
todo = [s for s in states if f"{s['dir']}|{s['step']}" not in done]
specs = args.specs.split(","); print(f"states {len(states)} todo {len(todo)} specs {specs}", flush=True); lock = threading.Lock()
if any(sp.endswith("_noterm") for sp in specs): NOTERM_BIAS = noterm_ids(args.base_url); print("noterm logit_bias ids:", NOTERM_BIAS, flush=True)
def run(st):
    ref = go.parse_action(st["target"]); rec = {"tag": args.tag, "dir": st["dir"], "task": st["task"], "step": st["step"], "target": st["target"], "decodes": {}, "match": {}}
    rec["ptoks"] = {}
    for sp in specs:
        noterm = sp.endswith("_noterm"); msgs = build(st, sp[:-7] if noterm else sp)
        if msgs is None: continue
        payload = {"model": args.model, "temperature": 0.0, "max_tokens": 512, "messages": msgs}
        if noterm: payload["logit_bias"] = NOTERM_BIAS
        out = go.post(args.base_url, payload)
        txt = out["choices"][0]["message"]["content"] or ""
        rec["decodes"][sp] = txt; rec["match"][sp] = go.match(go.parse_action(txt), ref) if ref is not None else None
        rec["ptoks"][sp] = (out.get("usage") or {}).get("prompt_tokens")
    with lock:
        with open(args.out, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
with ThreadPoolExecutor(args.workers) as ex: list(ex.map(run, todo))
print("DECODE_CTX_DONE", args.tag, flush=True)
