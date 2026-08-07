#!/usr/bin/env python3
"""噪声源任务判据:同组内赢家与输家的选帧行为是否可区分?

# note (luojiaxuan): GRPO 的隐含前提是"组内成败差异来自 rollout 自己的选择
# 差异"。若某任务的成败其实由环境掷骰子决定(VM 时序、setup 下载超时、
# 评测器竞态),该前提失效,推高赢家选帧 = 推向随机方向(抬方差不引偏差)。
#
# 判据:对每个 (task, iter) 组,取赢家与输家的选帧统计量之差 delta。
#   纯噪声任务 → delta 围绕 0 且符号随机(赢输不可区分);
#   真边缘任务 → delta 系统性偏离 0(选帧确实驱动成败)。
# 用 NOISY 集与非 NOISY 集对照,并做组内标签置换检验给出零分布。
#
# 两个统计量(都只用审计已存的字段,零环境重建):
#   age_norm     每轮被选帧的归一化年龄均值(0=最新,1=最老)
#   off_recent   最终展示帧中"非最近 B 帧"的比例(= 偏离 recent-B 基线的程度)
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

NOISY_DEFAULT = ("4c26e3f3,51b11269,6a33f9b9,6e99a1ad,716a6079,"
                 "7f52cab9,9f3bb592,aa3a8974,edb61b14")


def episode_stats(ep: dict, max_step: int = 0) -> tuple[float, float, float] | None:
    """返回 (age_norm, off_recent, n_steps);无可用选帧步则 None。

    # note (luojiaxuan): n_steps 是混淆对照——输家常跑满 30 步 cap,池子更大,
    # 纯机械地抬高 off_recent(候选多则挤出 recent-B 的概率大)。不控这一项
    # 就会把"赢家 episode 短"误读成"赢家选帧策略不同"。
    """
    ages: list[float] = []
    off: list[float] = []
    for step in ep.get("steps") or []:
        si = step.get("step_index")
        rounds = step.get("rounds") or []
        if not rounds or not si or si < 2:
            continue
        if max_step and si > max_step:
            continue    # 只看结局分化之前的早期步(控住"赢家更短"的混淆)
        span = max(si - 1, 1)
        for rnd in rounds:
            ch = rnd.get("chosen")
            if ch is None:
                continue
            ages.append((si - ch) / span)
        shown = step.get("shown_events") or []
        cands = rounds[0].get("candidates") or []
        if shown and cands:
            recent = set(sorted(cands)[-len(shown):])
            off.append(len(set(shown) - recent) / len(shown))
    if not ages:
        return None
    return (sum(ages) / len(ages), sum(off) / len(off) if off else 0.0,
            float(ep.get("n_steps") or len(ep.get("steps") or [])))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rl-root", default="/data04/jaxan/rl-probe")
    p.add_argument("--noisy", default=NOISY_DEFAULT,
                   help="逗号分隔的 8 位任务前缀(rl_task_probe.py 的 NOISY 标记)")
    p.add_argument("--max-step", type=int, default=0,
                   help="只统计 step_index<=K 的早期步(0=全程)。全程比较被"
                        "'赢家 episode 更短→池子更小'机械混淆,K=10 可控住")
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260807)
    args = p.parse_args()

    noisy = set(args.noisy.split(","))
    # groups[(task, iter)] = [(reward, age_norm, off_recent, n_steps), ...]
    groups: dict[tuple[str, int], list[tuple[int, float, float, float]]] = defaultdict(list)
    for f in sorted(glob.glob(f"{args.rl_root}/iter-*/groups.jsonl")):
        it = int(f.split("iter-")[1].split("/")[0])
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            st = episode_stats(ep, args.max_step)
            if st is None:
                continue
            groups[(ep["task_id"][:8], it)].append(
                (int(float(ep["reward"])), st[0], st[1], st[2]))

    rng = random.Random(args.seed)

    def analyse(label: str, keys: list[tuple[str, int]]) -> None:
        rows = []
        for k in keys:
            eps = groups[k]
            win = [e for e in eps if e[0] == 1]
            lose = [e for e in eps if e[0] == 0]
            if not win or not lose:
                continue          # 退化组无对比意义(GRPO 本就跳过)
            for idx, name in ((1, "age"), (2, "off"), (3, "len")):
                dw = sum(e[idx] for e in win) / len(win)
                dl = sum(e[idx] for e in lose) / len(lose)
                rows.append((k, name, dw - dl, len(win), len(lose)))
        for name in ("age", "off", "len"):
            ds = [r[2] for r in rows if r[1] == name]
            if not ds:
                continue
            mean_d = sum(ds) / len(ds)
            mean_abs = sum(abs(d) for d in ds) / len(ds)
            pos = sum(1 for d in ds if d > 0)
            # 组内标签置换零分布:|均值差| 的经验分位
            null = []
            for _ in range(args.permutations):
                sd = []
                for k in keys:
                    eps = groups[k]
                    if not any(e[0] == 1 for e in eps) or not any(e[0] == 0 for e in eps):
                        continue
                    idx = {"age": 1, "off": 2, "len": 3}[name]
                    vals = [e[idx] for e in eps]
                    labs = [e[0] for e in eps]
                    rng.shuffle(labs)
                    w = [v for v, l in zip(vals, labs) if l == 1]
                    lo = [v for v, l in zip(vals, labs) if l == 0]
                    if w and lo:
                        sd.append(sum(w) / len(w) - sum(lo) / len(lo))
                if sd:
                    null.append(abs(sum(sd) / len(sd)))
            pval = (sum(1 for n in null if n >= abs(mean_d)) + 1) / (len(null) + 1) \
                if null else float("nan")
            print(f"{label:>10} {name}: 组数={len(ds):3d} "
                  f"均值Δ={mean_d:+.4f} 平均|Δ|={mean_abs:.4f} "
                  f"符号 {pos}+/{len(ds)-pos}- 置换p={pval:.3f}")

    noisy_keys = [k for k in groups if k[0] in noisy]
    other_keys = [k for k in groups if k[0] not in noisy]
    print(f"总组数 {len(groups)}(NOISY {len(noisy_keys)} / 其他 {len(other_keys)})")
    print("Δ = 赢家统计量均值 − 输家统计量均值;置换 p 小 = 赢输可区分")
    analyse("NOISY", noisy_keys)
    analyse("其他", other_keys)


if __name__ == "__main__":
    main()
