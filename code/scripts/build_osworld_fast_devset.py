#!/usr/bin/env python3
"""构造 OSWorld 快速开发集(仅供迭代,不得用于最终成绩)。

设计意图:修 HGKV 需要一个能高频迭代、且同时能回答"变好了没"和"弄坏了没"的集合。

- **A 组(还没吃下的记忆任务)**:B0 / Recent-4 / selector **三臂全失败**,且域为
  multi_apps —— 跨应用任务的证据最可能落在 recent 窗口之外,是记忆调度该起作用
  却还没起作用的地方。这里成功率地板为 0,没有天花板,灵敏度最高。
- **B 组(防回归护栏)**:B0 失败但 Recent-4 成功 —— **记忆确实起作用**的实锤任务。
  修 HGKV 不能把这些弄砸。

**严禁用于报告最终数字。** 本集合是从已有 arm 的结果里挑出来的,在它上面报成绩
等于在测试集上选样本。最终口径必须回全量 361。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B15 = ROOT / "data/results/osworld_verified_closed_loop"


def load(name: str) -> dict[str, dict]:
    raw = json.loads((B15 / name).read_text())
    out = {}
    for t, v in raw.items():
        dom, uuid = t.split("/")[0], t.split("/")[-1]
        out[uuid] = {"s": float(v.get("s", v.get("score", 0.0))),
                     "dom": dom, "steps": int(v.get("steps", -1))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-domains", default="multi_apps",
                    help="A 组限定的域,逗号分隔")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    b0, rec, sel = load("per_task_b0.json"), load("per_task_recent.json"), load("per_task_sel1p.json")
    K = sorted(set(b0) & set(rec) & set(sel))
    doms = set(args.a_domains.split(","))
    fail = lambda d, t: d[t]["s"] < 1.0  # noqa: E731

    a = sorted(t for t in K
               if fail(b0, t) and fail(rec, t) and fail(sel, t) and sel[t]["dom"] in doms)
    # note (luojiaxuan): B 组取全部,不抽样 —— 抽样会让"有没有弄坏"的判据本身带偏。
    b = sorted(t for t in K if fail(b0, t) and not fail(rec, t))

    tasks = sorted(set(a) | set(b))
    dom_of = {t: sel[t]["dom"] for t in tasks}
    meta: dict[str, list[str]] = collections.defaultdict(list)
    for t in tasks:
        meta[dom_of[t]].append(t)
    for d in meta:
        meta[d].sort()

    payload = {
        "schema_version": "causalcache.osworld.fast_devset.v1",
        "purpose": "开发迭代专用;不得用于报告最终成绩(样本由已有 arm 结果挑选)",
        "group_a_hard_memory": {"n": len(a), "rule": f"B0/Recent-4/selector 三臂全失败 且 域∈{sorted(doms)}"},
        "group_b_regression_guard": {"n": len(b), "rule": "B0 失败但 Recent-4 成功(记忆确实起作用)"},
        "overlap_a_b": len(set(a) & set(b)),
        "task_count": len(tasks),
        "tasks_sha256": hashlib.sha256(
            json.dumps(tasks, separators=(",", ":")).encode()).hexdigest(),
        "by_domain": {d: len(v) for d, v in sorted(meta.items())},
        "meta": dict(sorted(meta.items())),
        "group_a": a,
        "group_b": b,
    }
    Path(args.output).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in
                      ("group_a_hard_memory", "group_b_regression_guard",
                       "overlap_a_b", "task_count", "by_domain")},
                     indent=1, ensure_ascii=False))
    print("sha256:", payload["tasks_sha256"][:16])


if __name__ == "__main__":
    main()
