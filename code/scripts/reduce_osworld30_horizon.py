#!/usr/bin/env python3
"""OSWorld-30 扩展步长诊断归约:Δ15/Δ30/Δhorizon + success-by-step。

输入:
  --arm30 sel=<dir_or_json> recent=<dir_or_json>(可多目录逗号分隔,按 task_id 去重,后者不覆盖前者)
  15-step 基线固定读 repo:data/results/osworld_verified_closed_loop/per_task_{sel1p,recent}.json
输出:JSON(Δ15、Δ30、每臂 Δhorizon、paired bootstrap、success-by-step 曲线)。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE15 = {
    "sel": ROOT / "data/results/osworld_verified_closed_loop/per_task_sel1p.json",
    "recent": ROOT / "data/results/osworld_verified_closed_loop/per_task_recent.json",
}


def load30(spec: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for part in spec.split(","):
        p = Path(part)
        if p.is_file():
            for t, rec in json.loads(p.read_text()).items():
                out.setdefault(t, rec if isinstance(rec, dict) else {"score": float(rec)})
            continue
        for rj in sorted(p.glob("*/*/result.json")):
            rec = json.loads(rj.read_text())
            tid = rec.get("task_id") or rj.parent.name
            out.setdefault(tid, {"score": float(rec.get("score", 0.0)),
                                 "steps": int(rec.get("steps", -1))})
    return out


def rate(d: dict[str, dict]) -> float:
    return 100.0 * sum(1 for v in d.values() if v["score"] >= 1.0) / len(d) if d else 0.0


def paired_boot(a: dict, b: dict, n_boot: int = 10000, seed: int = 7) -> dict:
    common = sorted(set(a) & set(b))
    da = [1 if a[t]["score"] >= 1.0 else 0 for t in common]
    db = [1 if b[t]["score"] >= 1.0 else 0 for t in common]
    diffs = [x - y for x, y in zip(da, db)]
    rng = random.Random(seed)
    n = len(common)
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(100.0 * s / n)
    boots.sort()
    mean = 100.0 * sum(diffs) / n
    return {"n": n, "delta_pp": round(mean, 2),
            "ci95": [round(boots[int(0.025 * n_boot)], 2), round(boots[int(0.975 * n_boot)], 2)],
            "boot_p_leq0": round(sum(1 for x in boots if x <= 0) / n_boot, 4)}


def success_by_step(d: dict, max_steps: int = 30) -> list[float]:
    # note (luojiaxuan): 成功任务按其终止步数计入首达曲线;steps 缺失(-1)的按 max_steps 计。
    n = len(d)
    curve = []
    for s in range(1, max_steps + 1):
        k = sum(1 for v in d.values()
                if v["score"] >= 1.0 and (v.get("steps", max_steps) if v.get("steps", -1) > 0 else max_steps) <= s)
        curve.append(round(100.0 * k / n, 2) if n else 0.0)
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel30", required=True)
    ap.add_argument("--recent30", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    arms30 = {"sel": load30(args.sel30), "recent": load30(args.recent30)}
    arms15 = {}
    for arm, path in BASE15.items():
        raw = json.loads(path.read_text())
        arms15[arm] = {t: (v if isinstance(v, dict) else {"score": float(v)}) for t, v in raw.items()}

    out = {"n_tasks": {a: len(m) for a, m in arms30.items()},
           "rates": {f"{a}@{h}": round(rate(m), 2)
                     for h, arms in (("15", arms15), ("30", arms30)) for a, m in arms.items()},
           "delta15_sel_vs_recent": paired_boot(arms15["sel"], arms15["recent"]),
           "delta30_sel_vs_recent": paired_boot(arms30["sel"], arms30["recent"]),
           "horizon_gain": {a: paired_boot(arms30[a], arms15[a]) for a in ("sel", "recent")},
           "success_by_step_30": {a: success_by_step(arms30[a]) for a in ("sel", "recent")}}
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_tasks", "rates", "delta30_sel_vs_recent", "horizon_gain")}, indent=1))


if __name__ == "__main__":
    main()
