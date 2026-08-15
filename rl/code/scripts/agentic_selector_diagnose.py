#!/usr/bin/env python3
"""Phase 3 局限诊断:selector 是"找不到目标帧"还是"找到了也没用上"?

# note (luojiaxuan): Phase 3 判决(n=300)后剩下四条局限,其中两条指向同一个
# 未知:平均选中帧龄仅 1.89、two_frame_complementary 零捕获。二者可能来自
# 两种完全不同的病因,处方相反:
#   (A) **检索失败**:决策步根本没把 required_steps 选进来 → 该攻 selector
#       的远帧表达/先验/训练信号;
#   (B) **利用失败**:选进来了但 executor 仍做错 → 该攻 Phase 2 的 executor
#       (它在 oracle 供帧时是 92-98%,但那是 teacher-forced 前缀;闭环里
#       前缀由策略自己走出来,可能已经偏离)。
# 本脚本用 rollout 落盘的 chosen_subset + 任务的 memory_probe 直接分离二者。
# **只读诊断,不产出任何训练信号**(路线 §9)。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=Path, required=True)
    ap.add_argument("--split", default="syn_ood")
    ap.add_argument("--n-tasks", type=int, default=300)
    ap.add_argument("--task-seed", type=int, default=2101)
    ap.add_argument("--label", default="learned")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    from causalcache_agentic import tasks as tasks_mod

    probes = {}
    for t in tasks_mod.make_dataset(args.split, args.n_tasks, args.task_seed):
        probes[t.task_id] = {"probe": t.memory_probe or {}, "regime": t.regime}

    rows = []
    for p in args.rows:
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    hard = {"one_old_frame", "two_frame_complementary", "distractor_heavy"}
    agg: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "hit_all": 0, "hit_any": 0, "succ": 0,
                 "hit_and_succ": 0, "miss_and_succ": 0,
                 "req_age": [], "sel_age": [], "n_cand": []})

    for r in rows:
        meta = probes.get(r["task_id"])
        if meta is None:
            continue
        pr = meta["probe"]
        dec = pr.get("decision_step")
        req = [int(x) for x in (pr.get("required_steps") or [])]
        if dec is None or not req:
            continue
        dec = int(dec)
        step = next((s for s in r.get("steps", []) if int(s["step"]) == dec), None)
        if step is None:
            continue                       # 轨迹没走到决策步(闭环里常见)
        chosen = set(int(x) for x in (step.get("chosen_subset") or []))
        cands = [int(x) for x in (step.get("candidates") or [])]
        req_in_pool = [x for x in req if x in cands]
        if not req_in_pool:
            continue
        ok = int(bool(r.get("success")))
        for key in ("ALL", meta["regime"],
                    "hard" if meta["regime"] in hard else "easy"):
            a = agg[key]
            a["n"] += 1
            hit_all = int(set(req_in_pool) <= chosen)
            hit_any = int(bool(set(req_in_pool) & chosen))
            a["hit_all"] += hit_all
            a["hit_any"] += hit_any
            a["succ"] += ok
            if hit_all:
                a["hit_and_succ"] += ok
            else:
                a["miss_and_succ"] += ok
            a["req_age"] += [dec - x for x in req_in_pool]
            a["sel_age"] += [dec - x for x in chosen]
            a["n_cand"].append(len(cands))

    out = {}
    print(f"[{args.label}] 到达决策步且 required 在候选池内的 rollout 数 "
          f"= {agg['ALL']['n']}")
    print(f"{'分层':24s} {'n':>5s} {'全命中':>7s} {'任一命中':>8s} "
          f"{'命中时成功':>10s} {'未命中时成功':>12s} {'req帧龄':>8s} {'选中帧龄':>8s}")
    for key in ("ALL", "hard", "easy", "one_old_frame",
                "two_frame_complementary", "distractor_heavy",
                "recent_sufficient", "history_irrelevant"):
        a = agg.get(key)
        if not a or not a["n"]:
            continue
        n = a["n"]
        ha = a["hit_all"]
        rec = {
            "n": n,
            "hit_all_rate": round(ha / n, 4),
            "hit_any_rate": round(a["hit_any"] / n, 4),
            "succ_rate": round(a["succ"] / n, 4),
            "succ_given_hit": round(a["hit_and_succ"] / ha, 4) if ha else None,
            "succ_given_miss": round(a["miss_and_succ"] / (n - ha), 4)
            if n - ha else None,
            "mean_req_age": round(sum(a["req_age"]) / len(a["req_age"]), 2),
            "mean_sel_age": round(sum(a["sel_age"]) / len(a["sel_age"]), 2),
            "mean_candidates": round(sum(a["n_cand"]) / len(a["n_cand"]), 2),
        }
        out[key] = rec
        print(f"{key:24s} {n:5d} {100*rec['hit_all_rate']:6.1f}% "
              f"{100*rec['hit_any_rate']:7.1f}% "
              f"{('%.1f%%' % (100*rec['succ_given_hit'])) if rec['succ_given_hit'] is not None else '   n/a':>10s} "
              f"{('%.1f%%' % (100*rec['succ_given_miss'])) if rec['succ_given_miss'] is not None else '   n/a':>12s} "
              f"{rec['mean_req_age']:8.2f} {rec['mean_sel_age']:8.2f}")
    if args.json_out:
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print("写出", args.json_out)


if __name__ == "__main__":
    main()
