# note (luojiaxuan): Venus 版 N 扫描(其标注布局即部署布局:多轮 + 各步 prediction 原文)。规格 recN(最近 N 帧)/ irrN(无关帧放在最近 N 个 turn 的位置)。
# 回答"N≥4 的终止病理是否 GUI-Owl 特有":Venus 上若无骤降、无 terminate 激增,则 H5 是模型特异的。
import argparse, glob, hashlib, importlib.util, json, os, random, re, sys, threading, types, math
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="/data01/jaxan/rl_v2/venus/oracle_v2.jsonl")
ap.add_argument("--base-url", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--specs", default="rec0,rec1,rec2,rec3,rec4,rec6,rec8,irr2,irr4,irr8"); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--no-text", action="store_true")
args = ap.parse_args()
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
vo = load("/data01/jaxan/venus_oracle.py", "vo")
_src = open("/data01/jaxan/venus_oracle_verdict.py").read().split("def summarize", 1)[0]
vv = types.ModuleType("vv"); vv.__dict__.update({"re": re, "math": math, "ast": __import__("ast")}); exec(_src, vv.__dict__)
def build_state(r):
    d = r["dir"]; k = r["step"]
    traj = list(json.load(open(os.path.join(d, "traj.json"))).values())[0].get("traj") or []
    shots = sorted(glob.glob(os.path.join(d, "screenshots", "*.png")), key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))
    preds = {s["step"]: s.get("prediction") or "" for s in traj}
    return {"dir": d, "task": r["task"], "goal": r["goal"], "step": k, "target": r["target"], "shots": shots, "hist": [preds[j] for j in range(1, k)]}
rows = [json.loads(l) for l in open(args.labels)]; states = [build_state(r) for r in rows]
pool = {}
for s in states: pool.setdefault(s["dir"], s["shots"])
def irr_frames(st, n):
    others = [d for d in pool if d != st["dir"]]
    rng = random.Random(int(hashlib.md5(f"{st['dir']}|{st['step']}|irr{n}".encode()).hexdigest()[:8], 16))
    shots = pool[rng.choice(others)]; return [shots[i] for i in sorted(rng.sample(range(len(shots)), min(n, len(shots))))]
def msgs_for(st, sp):
    k = st["step"]; kind, n = sp[:3], int(sp[3:]); turns = {k - 1 - j for j in range(1, n + 1) if k - 1 - j >= 0}
    if kind == "rec": return vo.messages(st, turns, args.no_text)
    fake = dict(st); fake["shots"] = list(st["shots"]); irr = irr_frames(st, len(turns))
    for i, p in zip(sorted(turns), irr): fake["shots"][i] = p
    return vo.messages(fake, turns, args.no_text)
done = set()
if os.path.exists(args.out):
    for l in open(args.out): r = json.loads(l); done.add(f"{r['dir']}|{r['step']}")
todo = [s for s in states if f"{s['dir']}|{s['step']}" not in done]; specs = args.specs.split(","); lock = threading.Lock()
print(f"states {len(states)} todo {len(todo)} specs {specs}", flush=True)
def decode(m):
    out = vo.post(args.base_url, {"model": "UI-Venus-2", "temperature": 0.0, "max_tokens": 4096, "repetition_penalty": 1.05, "frequency_penalty": 0.3, "messages": m})
    txt = out["choices"][0]["message"]["content"] or ""; mm = re.search(r"<action>(.*?)</action>", txt, re.S); return mm.group(1).strip() if mm else ""
def run(st):
    ref = vv.parse(st["target"]); rec = {"tag": args.tag, "dir": st["dir"], "task": st["task"], "step": st["step"], "target": st["target"], "decodes": {}, "match": {}}
    for sp in specs:
        act = decode(msgs_for(st, sp)); rec["decodes"][sp] = act; rec["match"][sp] = vv.match(vv.parse(act), ref) if ref is not None else None
    with lock:
        with open(args.out, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
with ThreadPoolExecutor(args.workers) as ex: list(ex.map(run, todo))
print("DECODE_CTX_VENUS_DONE", flush=True)
