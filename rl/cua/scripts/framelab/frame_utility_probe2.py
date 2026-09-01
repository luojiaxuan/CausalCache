# note (luojiaxuan): 帧效用探针 v2。与 v1 的差别只在时间对齐:
# 截图 file N 是第 N 步的**前置观察**(已由 TakeSelfieTask file1=未操作的桌面
# 而 step1=上滑打开抽屉 实证),因此决策第 k 步时的当前屏幕是 shots[k-1],
# 历史候选池是 shots[0..k-2],recency 基线是其中最近的两帧。
# v1 把 shots[k] 当作当前屏幕,那是目标动作执行完之后的画面。
#
# 状态选取与 v1 逐条对齐:cand_idx 长度不变,rng.sample 消耗的随机量相同,
# 故 (task, step) 序列与被选中的候选位置都与 v1 一致,可直接做前后对比。
import argparse, glob, itertools, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_utility_probe import score  # noqa: E402


def load_states(roots, n_states, min_step, k_cand, rng):
    out, dirs = [], []
    for root in roots:
        dirs.extend(sorted(glob.glob(os.path.join(root, "*/"))))
    rng.shuffle(dirs)
    for d in dirs:
        if len(out) >= n_states:
            break
        r, t = os.path.join(d, "result.txt"), os.path.join(d, "traj.json")
        if not (os.path.exists(r) and os.path.exists(t)):
            continue
        try:
            if float(open(r).read().split("score:")[1].split()[0]) <= 0:
                continue
            traj = list(json.load(open(t)).values())[0].get("traj") or []
        except Exception:
            continue
        shots = sorted(glob.glob(os.path.join(d, "screenshots", "*.png")),
                       key=lambda p: int(re.search(r"-(\d+)\.png$", p).group(1))
                       if re.search(r"-(\d+)\.png$", p) else 0)
        if len(shots) < min_step + 2:
            continue
        for step in traj:
            k = step.get("step", 0)
            if k < min_step or k >= len(shots) - 1:
                continue
            pred = step.get("prediction") or ""
            if not pred:
                continue
            cand_idx = list(range(0, k - 1))
            if len(cand_idx) < k_cand:
                continue
            picked = sorted(rng.sample(cand_idx, k_cand))
            out.append({
                "task": os.path.basename(d.rstrip("/")),
                "goal": step.get("task_goal", ""),
                "step": k,
                "target": pred,
                "cur_shot": shots[k - 1],
                "cands": [shots[i] for i in picked],
                "recency": [shots[k - 2], shots[k - 3]],
                "dir": d.rstrip("/"),
            })
            break
    return out[:n_states]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states-out", default="")
    ap.add_argument("--n-states", type=int, default=80)
    ap.add_argument("--k-cand", type=int, default=6)
    ap.add_argument("--min-step", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260831)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    roots = ["/data01/jaxan/mw/traj_recentB11", "/data01/jaxan/mw/traj_full",
             "/data01/jaxan/mw/traj_recentB2", "/data01/jaxan/mw/traj_recentB2_r2",
             "/data01/jaxan/mw/traj_randomB2", "/data01/jaxan/mw/traj_smoke3"]
    states = load_states(roots, args.n_states, args.min_step, args.k_cand, rng)
    print(f"状态数 {len(states)}", flush=True)
    if args.states_out:
        json.dump(states, open(args.states_out, "w"))

    results = []
    for si, st in enumerate(states):
        rec = {"task": st["task"], "step": st["step"], "pairs": {}, "recency": None}
        rec["recency"] = score(args.base_url, args.model,
                               st["recency"] + [st["cur_shot"]], st["goal"], st["target"])
        for a, b in itertools.combinations(range(args.k_cand), 2):
            u = score(args.base_url, args.model,
                      [st["cands"][a], st["cands"][b], st["cur_shot"]],
                      st["goal"], st["target"])
            if u is not None:
                rec["pairs"][f"{a}_{b}"] = u
        results.append(rec)
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(states)}", flush=True)
            json.dump(results, open(args.out, "w"))
    json.dump(results, open(args.out, "w"))
    print(f"写出 {args.out}", flush=True)


if __name__ == "__main__":
    main()
