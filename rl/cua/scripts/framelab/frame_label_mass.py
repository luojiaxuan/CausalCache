# note (luojiaxuan): 帧效用标签生产。协议 = frame_utility_probe2 时间对齐
# (file N = 第 N 步前置观察;当前帧 shots[k-1],候选池 shots[0..k-2])。
# 每状态的离线动作空间 {空} ∪ {单帧} ∪ {双帧}:空 1 + 单 6 + 对 15 + recency
# = 23 次 teacher-forcing。单帧分用于协同项 γ_ij = u_ij−u_i−u_j+u_∅ 与可加
# 基线;空上下文分是状态级归一化零点。target 段记逐 token (id, logprob),
# 供事后拆分动作类型/坐标/参数 token 的增益归属。
# --audit 附加:15 对全部逆序重打(时序表征是否必要)+ 空/recency/最优对/
# 最差对四个上下文的贪心解码(离线核对 Δlogp 是否预测解码动作正确性——
# 外审 20260901 的量产前置闸门)。候选采样种子由 (dir,step) 派生,可复现;
# 按状态断点续跑。
import argparse, base64, glob, hashlib, itertools, json, os, random, re, sys, threading, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data01/jaxan/sglang-omni-rl")
from frame_utility_probe import _target_tokens, b64  # noqa: E402

ROOTS = sorted(glob.glob("/data01/jaxan/mw/traj_*"))
K_CAND = 6
PAIRS = list(itertools.combinations(range(K_CAND), 2))


def _post(base, path, body, timeout=240):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _content(imgs, goal):
    c = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(p)}"}}
         for p in imgs]
    c.append({"type": "text", "text":
              f"Please generate the next move according to the UI screenshot "
              f"and instruction.\n\nInstruction: {goal}"})
    return c


def score2(base, model, imgs, goal, target):
    """teacher forcing;返回 (target 段均值, [(token_id, logprob), ...])。"""
    out = _post(base, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": _content(imgs, goal)},
                     {"role": "assistant", "content": target}],
        "max_tokens": 1, "temperature": 0.0,
        "prompt_logprobs": 0, "add_generation_prompt": False,
        "continue_final_message": True})
    toks = []
    for entry in out.get("prompt_logprobs") or []:
        if not entry:
            continue
        for tid, info in entry.items():
            lp = info.get("logprob")
            if lp is not None:
                toks.append((int(tid), lp))
            break
    if not toks:
        return None, []
    n_t = _target_tokens(base, model, target)
    tail = toks[-n_t:] if n_t and n_t <= len(toks) else toks[-max(8, len(toks) // 8):]
    return sum(lp for _, lp in tail) / len(tail), [[t, round(lp, 4)] for t, lp in tail]


def greedy(base, model, imgs, goal):
    out = _post(base, "/chat/completions", {
        "model": model, "max_tokens": 96, "temperature": 0.0,
        "messages": [{"role": "user", "content": _content(imgs, goal)}]})
    return out["choices"][0]["message"]["content"]


def iter_states(min_step):
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "*/"))):
            r, t = os.path.join(d, "result.txt"), os.path.join(d, "traj.json")
            if not (os.path.exists(r) and os.path.exists(t)):
                continue
            try:
                if float(open(r).read().split("score:")[1].split()[0]) <= 0:
                    continue
                traj = list(json.load(open(t)).values())[0].get("traj") or []
            except Exception:  # noqa: BLE001
                continue
            shots = sorted(glob.glob(os.path.join(d, "screenshots", "*.png")),
                           key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1))
                           if re.search(r"-(\d+)\.png$", p) else 0)
            if len(shots) < min_step + 1:
                continue
            for step in traj:
                k = step.get("step", 0)
                pred = step.get("prediction") or ""
                if k < max(min_step, K_CAND + 1) or k > len(shots) or not pred:
                    continue
                rng = random.Random(int(hashlib.md5(f"{d}|{k}".encode()).hexdigest()[:8], 16))
                picked = sorted(rng.sample(range(0, k - 1), K_CAND))
                yield {
                    "dir": d.rstrip("/"), "task": os.path.basename(d.rstrip("/")),
                    "root": os.path.basename(root), "traj_len": len(shots),
                    "goal": step.get("task_goal", ""), "step": k, "target": pred,
                    "action": step.get("action"),
                    "cur_shot": shots[k - 1],
                    "cands": [shots[i] for i in picked], "cand_idx": picked,
                    "recency": [shots[k - 2], shots[k - 3]],
                }


def audit_subset(states, per_traj, limit):
    """轨迹等权 + 隔开取点:每轨迹至多 per_traj 个均匀分布的状态。"""
    by_dir = {}
    for s in states:
        by_dir.setdefault(s["dir"], []).append(s)
    picked = []
    for d in sorted(by_dir):
        ss = sorted(by_dir[d], key=lambda s: s["step"])
        idx = ([0] if len(ss) <= per_traj else
               [round(i * (len(ss) - 1) / (per_traj - 1)) for i in range(per_traj)])
        picked.extend(ss[i] for i in sorted(set(idx)))
    random.Random(20260901).shuffle(picked)
    return picked[:limit]


def label_state(base, model, st, audit):
    rec = {k: st[k] for k in ("dir", "task", "root", "traj_len", "goal", "step",
                              "target", "action", "cand_idx")}
    rec["cand_files"] = [int(re.search(r"-(\d+)\.png$", c).group(1)) for c in st["cands"]]
    rec["recency"], rec["recency_tok"] = score2(base, model, st["recency"] + [st["cur_shot"]],
                                               st["goal"], st["target"])
    rec["cur_only"], rec["cur_only_tok"] = score2(base, model, [st["cur_shot"]],
                                                 st["goal"], st["target"])
    rec["singles"], rec["singles_tok"] = {}, {}
    for i in range(K_CAND):
        m, tk = score2(base, model, [st["cands"][i], st["cur_shot"]], st["goal"], st["target"])
        rec["singles"][str(i)] = m
        rec["singles_tok"][str(i)] = tk
    rec["pairs"], rec["pairs_tok"] = {}, {}
    for a, bx in PAIRS:
        m, tk = score2(base, model, [st["cands"][a], st["cands"][bx], st["cur_shot"]],
                       st["goal"], st["target"])
        rec["pairs"][f"{a}_{bx}"] = m
        rec["pairs_tok"][f"{a}_{bx}"] = tk
    if audit:
        rec["pairs_rev"] = {}
        for a, bx in PAIRS:
            m, _ = score2(base, model, [st["cands"][bx], st["cands"][a], st["cur_shot"]],
                          st["goal"], st["target"])
            rec["pairs_rev"][f"{a}_{bx}"] = m
        valid = {k: v for k, v in rec["pairs"].items() if v is not None}
        best = max(valid, key=valid.get)
        worst = min(valid, key=valid.get)
        def pimgs(key):
            a, bx = (int(x) for x in key.split("_"))
            return [st["cands"][a], st["cands"][bx], st["cur_shot"]]
        rec["decode"] = {
            "null": greedy(base, model, [st["cur_shot"]], st["goal"]),
            "recency": greedy(base, model, st["recency"] + [st["cur_shot"]], st["goal"]),
            "best_pair": {"key": best, "text": greedy(base, model, pimgs(best), st["goal"])},
            "worst_pair": {"key": worst, "text": greedy(base, model, pimgs(worst), st["goal"])},
        }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--min-step", type=int, default=7)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--per-traj", type=int, default=3)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add(f"{r['dir']}|{r['step']}")
            except Exception:  # noqa: BLE001
                continue
    allst = list(iter_states(args.min_step))
    if args.audit:
        allst = audit_subset(allst, args.per_traj, args.limit)
    states = [s for i, s in enumerate(allst)
              if i % args.n_shards == args.shard and f"{s['dir']}|{s['step']}" not in done]
    print(f"shard {args.shard}/{args.n_shards}: 待标 {len(states)}(已完成 {len(done)},"
          f"总池 {len(allst)})", flush=True)

    lock = threading.Lock()
    n_done, t0 = 0, time.time()

    def work(st):
        nonlocal n_done
        try:
            rec = label_state(args.base_url, args.model, st, args.audit)
        except Exception as e:  # noqa: BLE001
            print(f"ERR {st['dir']}|{st['step']}: {str(e)[:100]}", flush=True)
            return
        with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_done += 1
            if n_done % 10 == 0:
                rate = n_done / (time.time() - t0)
                eta_h = (len(states) - n_done) / max(rate, 1e-9) / 3600
                print(f"  {n_done}/{len(states)}  {rate*3600:.0f} 状态/时  ETA {eta_h:.1f}h",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, states))
    print(f"SHARD_DONE {args.shard} 共 {n_done}", flush=True)


if __name__ == "__main__":
    main()
