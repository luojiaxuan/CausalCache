# note (luojiaxuan): Stage B 全子集格标注。对 6 候选的全部 2^6=64 个子集
# (空集即仅当前帧)+ recency 各打一次 teacher-forcing 分,一次拿全
# B=1/2/3/4 监督、协同/高阶交互检验与 budget 条件化的精确目标。
# 服务端需 --limit-mm-per-prompt '{"image":7}'(B=6 + 当前帧)。
# 逐 token 明细只存 |S|<=2 的上下文(体积控制);>=3 只存均值。
# 轨迹等权隔开采样与断点续跑同审计版。
import argparse, glob, hashlib, itertools, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/data01/jaxan")
from frame_label_mass import iter_states, audit_subset, score2  # noqa: E402

K = 6


def subset_key(S):
    return "_".join(str(i) for i in S)


def wide_slate(st):
    """约 1/5 且历史足够长的状态换 12 候选宽槽(luojiaxuan 20260901:
    训练要见过更大的池)。宽槽只标单帧+双帧(12+66),高阶监督由 6 槽
    全格组承担;判定用 (dir,step) 哈希,确定性可复现。"""
    h = int(hashlib.md5(f"{st['dir']}|{st['step']}|w".encode()).hexdigest()[:8], 16)
    if h % 5 != 0 or st["step"] < 14:
        return None
    rng = random.Random(h)
    picked = sorted(rng.sample(range(0, st["step"] - 1), 12))
    d = os.path.join(st["dir"], "screenshots")
    shots = sorted(glob.glob(os.path.join(d, "*.png")),
                   key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1)))
    return picked, [shots[i] for i in picked]


def label_state(base, model, st):
    rec = {k: st[k] for k in ("dir", "task", "root", "traj_len", "goal", "step",
                              "target", "action")}
    rec["recency"], rec["recency_tok"] = score2(base, model, st["recency"] + [st["cur_shot"]],
                                               st["goal"], st["target"])
    wide = wide_slate(st)
    if wide:
        idx, cands = wide
        rec["slate"] = 12
        rec["cand_idx"] = idx
        rec["cand_files"] = [int(re.search(r"-(\d+)\.png$", c).group(1)) for c in cands]
        rec["subsets"], rec["subsets_tok"] = {}, {}
        for size in (0, 1, 2):
            for S in itertools.combinations(range(12), size):
                imgs = [cands[i] for i in S] + [st["cur_shot"]]
                m, tk = score2(base, model, imgs, st["goal"], st["target"])
                rec["subsets"][subset_key(S)] = m
                if size <= 1:
                    rec["subsets_tok"][subset_key(S)] = tk
        return rec
    rec["slate"] = 6
    rec["cand_idx"] = st["cand_idx"]
    rec["cand_files"] = [int(re.search(r"-(\d+)\.png$", c).group(1)) for c in st["cands"]]
    rec["subsets"], rec["subsets_tok"] = {}, {}
    for size in range(0, K + 1):
        for S in itertools.combinations(range(K), size):
            imgs = [st["cands"][i] for i in S] + [st["cur_shot"]]
            m, tk = score2(base, model, imgs, st["goal"], st["target"])
            rec["subsets"][subset_key(S)] = m
            if size <= 2:
                rec["subsets_tok"][subset_key(S)] = tk
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
    ap.add_argument("--per-traj", type=int, default=8)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--exclude", default="",
                    help="逗号分隔的 jsonl,含 dir|step 键的状态跳过(审计集独立成评测)")
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
    allst = [s for s in audit_subset(list(iter_states(args.min_step)),
                                     args.per_traj, args.limit)
             if f"{s['dir']}|{s['step']}" not in excl]
    states = [s for i, s in enumerate(allst)
              if i % args.n_shards == args.shard and f"{s['dir']}|{s['step']}" not in done]
    print(f"shard {args.shard}/{args.n_shards}: 待标 {len(states)}"
          f"(已完成 {len(done)},排除 {len(excl)},总池 {len(allst)})", flush=True)

    lock = threading.Lock()
    n_done, t0 = 0, time.time()

    def work(st):
        nonlocal n_done
        try:
            rec = label_state(args.base_url, args.model, st)
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
