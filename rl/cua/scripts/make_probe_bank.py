#!/usr/bin/env python3
# note (luojiaxuan): 从 mini decisions 采样构建冻结探针库(外审 20260826
# 采纳的固定态行为位移判据)。按任务分层均匀抽,偏好 n_frames>=3 的状态
# (才有非平凡选择);同一 episode 至多 2 条防轨迹垄断。输出 JSONL:
# feats/n_frames/step,供 trainer --probe-bank 消费。库一经生成即冻结,
# 全量期间不得再生成(否则 KL 基线漂移)。
import argparse
import glob
import json
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_task = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.decisions_dir, "*.jsonl"))):
        per_ep = defaultdict(int)
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("n_frames", 0) < 3 or not r.get("feats"):
                continue
            ep = str(r.get("episode"))
            if per_ep[ep] >= 2:
                continue
            per_ep[ep] += 1
            by_task[os.path.basename(f)].append(
                {"feats": r["feats"], "n_frames": r["n_frames"],
                 "step": r["step"]})

    tasks = sorted(by_task)
    picked = []
    quota = max(1, args.n // max(1, len(tasks)))
    for t in tasks:
        pool = by_task[t]
        rng.shuffle(pool)
        picked.extend(pool[:quota])
    rest = [r for t in tasks for r in by_task[t][quota:]]
    rng.shuffle(rest)
    picked.extend(rest[: max(0, args.n - len(picked))])
    picked = picked[: args.n]

    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(f"probe bank: {len(picked)} states from {len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
