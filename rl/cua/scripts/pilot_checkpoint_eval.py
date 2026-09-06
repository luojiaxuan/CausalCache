# note (luojiaxuan): B-pilot 的 checkpoint 干预矩阵(外审 09-06:原生文本 + 真实源帧是两条路之间最高价值的判别器)。
# 上下文一律按 GUI-Owl **部署布局**(交错多轮)构造,复用 harm 线 decode_ctx.py 的 messages_deploy / messages_hybrid,
# 把"附哪几帧"与"哪几轮保留原文"解耦。条件(全部在预指定 checkpoint 上跑,不筛 B=0 失败):
#   text_only     B=0(当前屏 + 保留的文本轨迹)          rec2          最近两帧(部署默认)
#   src_keep      源帧作为保留 turn(部署语义:同时保留其原文,最近轮折叠)   src_hybrid    最近两轮原样 + 源帧作带 PAST 标记的额外图
#   src_pickimg   结构固定最近两轮,只把图换成源帧          irr2          匹配无关两帧(结构=最近两轮)
#   judge_hybrid:<judge>  裁判帧按 hybrid 附入            gold_text     文本里直接给出事实(排障对照)
#   swap_hybrid   反事实交换:附入**孪生另一版**的源帧 → 决策应跟随另一版答案(测决策是否随证据走)
# 判定:动作是否携带 expected(type/answer 的文本或 file 名);swap 条件判定是否携带 expected_swap。
import argparse, glob, importlib.util, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True, help="jsonl: dir, step, source_frames, expected, expected_kind, irrelevant_dir, swap_frames_dir(可选), expected_swap(可选), judge_picks(可选 dict)")
ap.add_argument("--base-url", required=True); ap.add_argument("--model", default="gui-owl")
ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

# 复用 decode_ctx 的构造器:它在模块顶层解析 argparse,这里用最小参数导入
sys.argv = ["decode_ctx.py", "--base-url", args.base_url, "--tag", "x", "--out", "/dev/null", "--specs", "rec0", "--roots", "/nonexistent"]
dc = load("/data01/jaxan/decode_ctx.py", "dc")
go = dc.go

def shots_of(d):
    return sorted(glob.glob(os.path.join(d, "screenshots", "*.png")), key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))

def state_of(d, k):
    traj = list(json.load(open(os.path.join(d, "traj.json"))).values())[0].get("traj") or []
    preds = {s["step"]: s.get("prediction") or "" for s in traj}
    return {"dir": d, "task": os.path.basename(d.rstrip("/")), "goal": traj[0].get("task_goal", ""), "step": k, "shots": shots_of(d),
            "concls": [go.extract_conclusion(preds[j]) for j in range(1, k)], "preds": preds, "target": preds.get(k, "")}

def carries(txt, exp, kind):
    act = go.parse_action(txt)
    if act is None: return 0
    exp = str(exp).strip().lower()
    if kind == "file": return int(exp in (txt or "").lower())
    body = (act.get("text") or "").replace(",", "").lower()
    if act["action"] not in ("type", "answer"): return 0
    try:
        e = float(exp.replace(",", "")); return int(any(abs(float(n) - e) < 0.005 for n in re.findall(r"\d+(?:\.\d+)?", body)))
    except ValueError:
        return int(exp in body)

def decode(msgs):
    out = go.post(args.base_url, {"model": args.model, "temperature": 0.0, "max_tokens": 256, "messages": msgs})
    return out["choices"][0]["message"]["content"] or ""

def hybrid_with(st, frames):
    msgs = dc.messages_deploy(st, 2); k = st["step"]; extra = []
    for age, path in frames:
        extra += [{"type": "text", "text": f"[PAST screenshot from {age} steps ago, for reference only]"},
                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{go.b64(path)}"}}]
    msgs[1]["content"] = [msgs[1]["content"][0]] + extra + msgs[1]["content"][1:]
    return msgs

lock = threading.Lock()
def run(sp):
    st = state_of(sp["dir"], sp["step"]); k = st["step"]; kind = sp.get("expected_kind", "text")
    src = [i for i in sp["source_frames"] if 0 <= i < k - 1][:2]
    conds = {"text_only": dc.messages_deploy(st, 0), "rec2": dc.messages_deploy(st, 2)}
    if src:
        conds["src_keep"] = dc.messages_deploy(st, 0, keep_set=set(src))
        conds["src_hybrid"] = hybrid_with(st, [(k - 1 - i, st["shots"][i]) for i in src])
        conds["src_pickimg"] = dc.messages_deploy(st, 2, irr=[st["shots"][i] for i in src])
    if sp.get("irrelevant_dir"):
        s2 = shots_of(sp["irrelevant_dir"])
        if len(s2) >= 2: conds["irr2"] = dc.messages_deploy(st, 2, irr=s2[:2])
    for jname, pk in (sp.get("judge_picks") or {}).items():
        pk = [i for i in pk if 0 <= i < k - 1][:2]
        if pk: conds[f"judge_hybrid:{jname}"] = hybrid_with(st, [(k - 1 - i, st["shots"][i]) for i in pk])
    g = dict(st); g["concls"] = st["concls"][:-1] + [(st["concls"][-1] if st["concls"] else "") + f" Note: the requested item is {sp['expected']}."]
    conds["gold_text"] = dc.messages_deploy(g, 0)
    if sp.get("swap_frames_dir") and sp.get("expected_swap") is not None:
        s3 = shots_of(sp["swap_frames_dir"]); sw = [i for i in sp.get("swap_source_frames", src) if i < len(s3)][:2]
        if sw: conds["swap_hybrid"] = hybrid_with(st, [(k - 1 - i, s3[i]) for i in sw])
    rec = {"tag": args.tag, "dir": st["dir"], "task": st["task"], "step": k, "expected": sp["expected"], "decodes": {}, "hit": {}}
    for name, msgs in conds.items():
        txt = decode(msgs); rec["decodes"][name] = txt
        exp = sp["expected_swap"] if name == "swap_hybrid" else sp["expected"]
        rec["hit"][name] = carries(txt, exp, kind)
    with lock:
        with open(args.out, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

specs = [json.loads(l) for l in open(args.spec)]
with ThreadPoolExecutor(args.workers) as ex: list(ex.map(run, specs))
rows = [json.loads(l) for l in open(args.out)]
conds = sorted({c for r in rows for c in r["hit"]})
print(f"[{args.tag}] n={len(rows)}  命中率(动作携带正确答案;swap_hybrid 按另一版答案判):")
for c in conds:
    xs = [r["hit"][c] for r in rows if c in r["hit"]]; print(f"  {c:26s} {sum(xs)/max(len(xs),1):.3f} (n={len(xs)})")
