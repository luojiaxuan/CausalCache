# note (luojiaxuan): 离线主闸门 G2(reviews/executor_swap_review_20260904.md §3):在 Venus-2 自己的成功轨迹上,
# 逐状态用官方多轮协议做贪心解码,只换历史截图集合:空 / recency-2 / 6 个候选单帧 / 15 个候选对(共 23 个上下文),
# 文本推理历史三者一律保留(全部过往 assistant 原文)。参考动作 = 该轨迹在该步实际输出的 <action>;等价类匹配见
# venus_oracle_verdict.py。状态构造同 frame_label_mass.iter_states(成功轨迹、step≥7、每轨迹均匀取 ≤per_traj 个、
# 候选 6 帧由 (dir,step) 派生的种子可复现);按 (dir,step) 断点续跑。
import argparse, base64, glob, hashlib, itertools, json, os, random, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data01/jaxan/mw/MobileWorld/src")
from mobile_world.agents.implementations.ui_venus2_agent import SYSTEM_PROMPT  # noqa: E402

K_CAND = 6


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def post(base, body, timeout=600):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def iter_states(roots, per_traj):
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            r, t = os.path.join(d, "result.txt"), os.path.join(d, "traj.json")
            if not (os.path.exists(r) and os.path.exists(t)):
                continue
            if float(open(r).read().split("score:")[1].split()[0]) <= 0:
                continue
            traj = list(json.load(open(t)).values())[0].get("traj") or []
            shots = sorted(glob.glob(os.path.join(d, "screenshots", "*.png")),
                           key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))
            preds = {s["step"]: s.get("prediction") or "" for s in traj}
            cand = []
            for s in traj:
                k = s["step"]
                if k < K_CAND + 1 or k > len(shots) or "<action>" not in preds.get(k, ""):
                    continue
                if any("<action>" not in preds.get(j, "") for j in range(1, k)):
                    continue
                cand.append(k)
            if not cand:
                continue
            idx = (list(range(len(cand))) if len(cand) <= per_traj else
                   [len(cand) // 2] if per_traj == 1 else
                   [round(i * (len(cand) - 1) / (per_traj - 1)) for i in range(per_traj)])
            for k in sorted({cand[i] for i in idx}):
                rng = random.Random(int(hashlib.md5(f"{d}|{k}".encode()).hexdigest()[:8], 16))
                picked = sorted(rng.sample(range(0, k - 1), K_CAND))
                yield {"dir": d.rstrip("/"), "task": os.path.basename(d.rstrip("/")),
                       "root": os.path.basename(root.rstrip("/")), "goal": traj[0].get("task_goal", ""),
                       "step": k, "target": preds[k], "shots": shots, "hist": [preds[j] for j in range(1, k)],
                       "cand_idx": picked}


def messages(st, image_turns):
    # note (luojiaxuan): 过往 turn j(0-based)的截图 = shots[j];当前截图 = shots[k-1]。
    msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(user_task=st["goal"])}]
    for j, raw in enumerate(st["hist"]):
        content = ""
        if j in image_turns:
            content = [{"type": "text", "text": "History Screenshot:"},
                       {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(st['shots'][j])}"}}]
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": raw})
    msgs.append({"role": "user", "content": [
        {"type": "text", "text": "Current Screenshot:\n"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(st['shots'][st['step'] - 1])}"}}]})
    return msgs


def contexts(st):
    k = st["step"]; c = st["cand_idx"]
    yield "null", set()
    yield "recency", {k - 2, k - 3}
    for i in range(K_CAND):
        yield f"s{i}", {c[i]}
    for a, b in itertools.combinations(range(K_CAND), 2):
        yield f"{a}_{b}", {c[a], c[b]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://172.17.0.1:41041/v1")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-traj", type=int, default=6)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line); done.add(f"{r['dir']}|{r['step']}")
    states = [s for s in iter_states(args.roots, args.per_traj) if f"{s['dir']}|{s['step']}" not in done]
    random.Random(20260904).shuffle(states)
    states = states[:args.limit]
    print(f"states to do: {len(states)} (done {len(done)})", flush=True)
    lock = threading.Lock(); n = [0]; t0 = time.time()

    def decode(st, name, turns):
        out = post(args.base_url, {"model": "UI-Venus-2", "temperature": 0.0, "max_tokens": args.max_tokens,
                                   "repetition_penalty": 1.05, "frequency_penalty": 0.3,
                                   "messages": messages(st, turns)})
        txt = out["choices"][0]["message"]["content"] or ""
        m = re.search(r"<action>(.*?)</action>", txt, re.S)
        return name, (m.group(1).strip() if m else ""), out.get("usage", {}).get("prompt_tokens")

    def work(st):
        rec = {k: st[k] for k in ("dir", "task", "root", "goal", "step", "cand_idx")}
        m = re.search(r"<action>(.*?)</action>", st["target"], re.S)
        rec["target"] = m.group(1).strip() if m else ""
        rec["decodes"] = {}; rec["ptoks"] = {}
        for name, turns in contexts(st):
            nm, act, pt = decode(st, name, turns)
            rec["decodes"][nm] = act; rec["ptoks"][nm] = pt
        with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[0] += 1
            if n[0] % 5 == 0:
                print(f"{n[0]}/{len(states)} states {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, states))
    print("ORACLE_DONE", flush=True)


if __name__ == "__main__":
    main()
