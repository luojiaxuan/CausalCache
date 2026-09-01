# note (luojiaxuan): Stage B 行为标签量产(路线 A,luojiaxuan 20260901 批准)。
# 标签 = 每上下文贪心解码正确性(0/1,等价类匹配),不再有任何
# teacher-forcing 代理。上下文:常规槽(6 候选)= 空 + 6 单 + 15 对 +
# 15 四元组(B=4 监督);宽槽(约 1/5 且 step≥14,12 候选)= 空 + 12 单 +
# 66 对。呈现一律按时间序。解码文本与匹配结果都落盘,供事后审计。
# 断点续跑按状态;行为试点已标状态由 --exclude 排除,训练期合并。
import argparse, glob, hashlib, itertools, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data01/jaxan")
from frame_label_mass import iter_states, audit_subset, _post, _content  # noqa: E402
from audit_analysis import parse_action, match  # noqa: E402

SYSTEM = json.loads(open("/data01/jaxan/sglang-omni-rl/sft_random_s_v1.jsonl")
                    .readline())["system"]


def greedy_sys(base, imgs, goal):
    out = _post(base, "/chat/completions", {
        "model": "probe", "max_tokens": 96, "temperature": 0.0,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": _content(imgs, goal)}]})
    return out["choices"][0]["message"]["content"]


def wide_slate(st):
    h = int(hashlib.md5(f"{st['dir']}|{st['step']}|w".encode()).hexdigest()[:8], 16)
    if h % 5 != 0 or st["step"] < 14:
        return None
    rng = random.Random(h)
    picked = sorted(rng.sample(range(0, st["step"] - 1), 12))
    shots = sorted(glob.glob(os.path.join(st["dir"], "screenshots", "*.png")),
                   key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))
    return picked, [shots[i] for i in picked]


def contexts_for(st, wide):
    cur = st["cur_shot"]
    if wide:
        _, cands = wide
        yield "null", [cur]
        for i in range(12):
            yield f"s{i}", [cands[i], cur]
        for a, b in itertools.combinations(range(12), 2):
            yield f"{a}_{b}", [cands[a], cands[b], cur]
    else:
        cands = st["cands"]
        yield "null", [cur]
        for i in range(6):
            yield f"s{i}", [cands[i], cur]
        for a, b in itertools.combinations(range(6), 2):
            yield f"{a}_{b}", [cands[a], cands[b], cur]
        for S in itertools.combinations(range(6), 4):
            yield "q" + "_".join(map(str, S)), [cands[i] for i in S] + [cur]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--per-traj", type=int, default=8)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add(f"{r['dir']}|{r['step']}")
            except Exception:  # noqa: BLE001
                continue
    excl = set()
    for f in filter(None, args.exclude.split(",")):
        for line in open(f):
            try:
                r = json.loads(line)
                excl.add(f"{r['dir']}|{r['step']}")
            except Exception:  # noqa: BLE001
                continue
    pool = [s for s in audit_subset(list(iter_states(7)), args.per_traj, args.limit)
            if f"{s['dir']}|{s['step']}" not in excl]
    states = [s for i, s in enumerate(pool)
              if i % args.n_shards == args.shard and f"{s['dir']}|{s['step']}" not in done]
    print(f"shard {args.shard}/{args.n_shards}: 待标 {len(states)}"
          f"(排除 {len(excl)},已完成 {len(done)},池 {len(pool)})", flush=True)

    lock = threading.Lock()
    n_done, t0 = 0, time.time()

    def work(st):
        nonlocal n_done
        try:
            ref = parse_action(st["target"])
            if ref is None:
                return
            wide = wide_slate(st)
            rec = {"dir": st["dir"], "task": st["task"], "root": st["root"],
                   "traj_len": st["traj_len"], "goal": st["goal"], "step": st["step"],
                   "target": st["target"], "slate": 12 if wide else 6,
                   "cand_idx": wide[0] if wide else st["cand_idx"], "decodes": {}}
            for name, imgs in contexts_for(st, wide):
                t = greedy_sys(args.base_url, imgs, st["goal"])
                rec["decodes"][name] = {"t": t, "ok": match(parse_action(t), ref)}
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
