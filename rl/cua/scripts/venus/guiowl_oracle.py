# note (luojiaxuan): GUI-Owl 侧的同口径标注器(与 venus_oracle.py 对应)。部署条件:**文本历史恒定保留**
# (官方 WITH_HISTSTEPS 布局里所有已完成步的 conclusion 都在),只改历史截图集合;每状态枚举 23 个上下文
# = B0(空)+ recency-2 + 6 单帧 + 15 帧对,贪心解码后与该轨迹自身的参考动作做等价类匹配。
# 参考动作与匹配口径沿用 audit_analysis(mobile_use tool_call:点击类坐标容差、文本类子串、其余参数须一致)。
# 按 (task, step) 断点续跑;--shard/--nshards 便于多卡并行。
import argparse, base64, glob, hashlib, itertools, json, math, os, random, re, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

K_CAND = 6
TOL_CLICK = 60
TOL_SWIPE = 120
ALIAS = {"tap": "click", "input": "type", "press": "long_press"}


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def post(base, body, timeout=600):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_action(text):
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.S)
    if not m:
        return None
    try:
        args = json.loads(m.group(1)).get("arguments") or {}
    except json.JSONDecodeError:
        return None
    a = args.get("action")
    if not a:
        return None
    return {"action": ALIAS.get(a, a), "coord": args.get("coordinate"), "coord2": args.get("coordinate2"),
            "text": (args.get("text") or "").strip().lower(),
            "extra": {k: v for k, v in args.items() if k not in ("action", "coordinate", "coordinate2", "text")}}


def _pt(v):
    return tuple(v) if isinstance(v, (tuple, list)) and len(v) == 2 and all(
        isinstance(x, (int, float)) for x in v) else None


def match(dec, ref):
    if dec is None or ref is None or dec["action"] != ref["action"]:
        return 0 if (dec and ref) else None
    a = dec["action"]
    if a in ("click", "long_press"):
        p, q = _pt(dec["coord"]), _pt(ref["coord"])
        return int(bool(p and q) and math.dist(p, q) <= TOL_CLICK)
    if a == "swipe":
        p, q, p2, q2 = _pt(dec["coord"]), _pt(ref["coord"]), _pt(dec["coord2"]), _pt(ref["coord2"])
        if p and q and p2 and q2:
            return int(math.dist(p, q) <= TOL_SWIPE and math.dist(p2, q2) <= TOL_SWIPE)
        return int(bool(dec["coord"]) == bool(ref["coord"]))
    if a in ("type", "key", "open", "answer"):
        return int(dec["text"] == ref["text"] or (len(ref["text"]) > 3 and ref["text"] in dec["text"]))
    if a == "wait":
        return 1
    return int(dec.get("extra") == ref.get("extra"))


def extract_conclusion(pred):
    m = re.search(r"Action:\s*(.*?)\s*<tool_call>", pred or "", re.S)
    return m.group(1).strip() if m else ""


def shots_for(d):
    return sorted(glob.glob(os.path.join(d, "screenshots", "*.png")),
                  key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))


def iter_states(roots, per_traj):
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            rp, tp = os.path.join(d, "result.txt"), os.path.join(d, "traj.json")
            if not (os.path.exists(rp) and os.path.exists(tp)):
                continue
            try:
                if float(open(rp).read().split("score:")[1].split()[0]) <= 0:
                    continue
                traj = list(json.load(open(tp)).values())[0].get("traj") or []
            except (IndexError, ValueError, KeyError, json.JSONDecodeError):
                continue
            shots = shots_for(d)
            cand = [s["step"] for s in traj
                    if s.get("step", 0) >= K_CAND + 1 and s["step"] <= len(shots) and (s.get("prediction") or "")]
            if not cand:
                continue
            idx = (list(range(len(cand))) if len(cand) <= per_traj else
                   [round(i * (len(cand) - 1) / (per_traj - 1)) for i in range(per_traj)])
            preds = {s["step"]: s.get("prediction") or "" for s in traj}
            for k in sorted({cand[i] for i in idx}):
                rng = random.Random(int(hashlib.md5(f"{d}|{k}".encode()).hexdigest()[:8], 16))
                yield {"dir": d.rstrip("/"), "task": os.path.basename(d.rstrip("/")),
                       "root": os.path.basename(root.rstrip("/")), "goal": traj[0].get("task_goal", ""),
                       "step": k, "target": preds[k], "shots": shots,
                       "concls": [extract_conclusion(preds.get(j, "")) for j in range(1, k)],
                       "cand_idx": sorted(rng.sample(range(0, k - 1), K_CAND))}


def messages(st, keep, sysprompt, tmpl_plain, tmpl_hist, irr=None):
    # note (luojiaxuan): 官方布局——images = [选中的历史帧(时间序)] + [当前帧];文本里列出全部已完成步的
    # conclusion(与选帧无关,这就是"文本历史恒定保留"的部署条件)。irr 给定时,历史帧换成外部轨迹的帧。
    hist = irr if irr is not None else [st["shots"][i] for i in sorted(keep)]
    imgs = hist + [st["shots"][st["step"] - 1]]
    if st["concls"]:
        prev = "\n".join(f"Step{j + 1}: {c}" for j, c in enumerate(st["concls"]))
        user = tmpl_hist.format(instruction=st["goal"], previous_steps=prev)
    else:
        user = tmpl_plain.format(instruction=st["goal"])
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(p)}"}} for p in imgs]
    content.append({"type": "text", "text": user})
    return [{"role": "system", "content": sysprompt}, {"role": "user", "content": content}]


def contexts(st):
    k, c = st["step"], st["cand_idx"]
    yield "null", set()
    yield "recency", {k - 2, k - 3}
    for i in range(K_CAND):
        yield f"s{i}", {c[i]}
    for a, b in itertools.combinations(range(K_CAND), 2):
        yield f"{a}_{b}", {c[a], c[b]}


def irrelevant_shots(st, pool):
    """外审要求的干扰易感性对照:喂来自**别的轨迹**的两张历史帧,张数与 recency 相同。
    读法是与 null 比——若无关历史比不给历史更差,说明该臂被无关内容带偏。"""
    others = [d for d in pool if d != st["dir"]]
    if not others:
        return None
    rng = random.Random(int(hashlib.md5(f"{st['dir']}|{st['step']}|irr".encode()).hexdigest()[:8], 16))
    shots = pool[rng.choice(others)]
    if len(shots) < 2:
        return None
    return [shots[i] for i in sorted(rng.sample(range(len(shots)), 2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gui-owl")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--prompts", required=True, help="sglang_omni_rl/gui_owl/prompts.py 的路径")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-traj", type=int, default=6)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ns = {}
    exec(compile(open(args.prompts).read(), args.prompts, "exec"), ns)
    sysprompt, tmpl_plain, tmpl_hist = ns["SYSTEM_PROMPT"], ns["USER_PROMPT_TEMPLATE"], ns["USER_PROMPT_WITH_HISTSTEPS_TEMPLATE"]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add(f"{r['dir']}|{r['step']}")
    states = [s for s in iter_states(args.roots, args.per_traj) if f"{s['dir']}|{s['step']}" not in done]
    random.Random(20260904).shuffle(states)
    states = [s for i, s in enumerate(states) if i % args.nshards == args.shard][:args.limit]
    print(f"shard {args.shard}/{args.nshards}: {len(states)} states (done {len(done)})", flush=True)

    if args.dry_run:
        st = states[0]
        for name, keep in list(contexts(st))[:3]:
            m = messages(st, keep, sysprompt, tmpl_plain, tmpl_hist)
            imgs = sum(1 for c in m[1]["content"] if c["type"] == "image_url")
            txt = next(c["text"] for c in m[1]["content"] if c["type"] == "text")
            print(f"--- {name}: keep={sorted(keep)} images={imgs}")
            print("   user text:", txt[:300].replace("\n", " | "))
        print("ref action:", parse_action(st["target"]))
        return

    pool = {}
    for root in args.roots:
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            sh = shots_for(d.rstrip("/"))
            if sh:
                pool[d.rstrip("/")] = sh
    lock = threading.Lock(); n = [0]; t0 = time.time()

    def work(st):
        rec = {k: st[k] for k in ("dir", "task", "root", "goal", "step", "cand_idx")}
        rec["target"] = st["target"]; rec["decodes"] = {}
        todo = list(contexts(st))
        irr = irrelevant_shots(st, pool)
        if irr is not None:
            todo.append(("irrelevant", None))
        for name, keep in todo:
            msgs = messages(st, keep or set(), sysprompt, tmpl_plain, tmpl_hist,
                            irr if name == "irrelevant" else None)
            out = post(args.base_url, {"model": args.model, "temperature": 0.0, "max_tokens": 256,
                                       "messages": msgs})
            rec["decodes"][name] = out["choices"][0]["message"]["content"] or ""
        with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[0] += 1
            if n[0] % 10 == 0:
                print(f"{n[0]}/{len(states)} states {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, states))
    print("GUIOWL_ORACLE_DONE", flush=True)


if __name__ == "__main__":
    main()
