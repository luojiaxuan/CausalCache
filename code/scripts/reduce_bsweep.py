#!/usr/bin/env python3
"""B 预算扫描归约:Δ(selector − Recent-B) 随 B 的曲线。

回答"B 等于几时 selector 相对 Recent-B 的增益最大"。固定 36 任务子集,每个 B 跑
recent 与 sel 两臂;B=0 时 selector 结构上退化为 Recent-0,故 Δ 恒为 0,不单独跑。

输入:主机上 /data/mw/runs/bsweep/b{B}-{recent,sel}/{a,b}/trajectories/<Task>/result.txt
      (v2 驱动把 36 个任务拆成 a/b 两半并行,归约时合并)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

BUDGETS = [0, 1, 2, 4, 6, 8]
SCORE_RE = re.compile(r"^score:\s*([0-9.]+)", re.M)


def load_arm(root: Path) -> dict[str, float]:
    """逐任务分数。兼容两种布局:v2 驱动的 <cfg>/{a,b}/trajectories/<Task>/,
    以及 v1 驱动的 <cfg>/trajectories/<Task>/(b0-recent 是 v1 跑的,没有 a/b 两半,
    只用带半区的 glob 会静默漏掉整个配置)。"""
    out: dict[str, float] = {}
    for pattern in ("*/trajectories/*/result.txt", "trajectories/*/result.txt"):
        for f in root.glob(pattern):
            m = SCORE_RE.search(f.read_text(errors="ignore"))
            if m:
                out[f.parent.name] = float(m.group(1))
    return out


def paired_boot(a: dict, b: dict, keys: list[str], n_boot: int = 4000, seed: int = 11) -> dict:
    diffs = [(1 if a[t] >= 1.0 else 0) - (1 if b[t] >= 1.0 else 0) for t in keys]
    if not diffs:
        return {"n": 0}
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(100.0 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return {"n": n, "delta_pp": round(100.0 * sum(diffs) / n, 2),
            "ci95": [round(boots[int(0.025 * n_boot)], 2), round(boots[int(0.975 * n_boot)], 2)]}


def rate(d: dict, keys: list[str]) -> float:
    return round(100.0 * sum(1 for t in keys if d[t] >= 1.0) / len(keys), 2) if keys else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--memory-split", required=True,
                    help="mobileworld_memory_split_v1.json,用于分出 memory-critical / control")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.sweep_root)
    # note (luojiaxuan): manifest 是 records 列表,按 memory_split 字段区分;
    # 'cross_app_memory_candidate' 即论文里的 memory-critical 组。
    split = json.loads(Path(args.memory_split).read_text())
    memcrit = {r["task_name"] for r in split["records"]
               if r.get("memory_split") == "cross_app_memory_candidate"}
    if not memcrit:
        raise SystemExit("memory-critical 集合为空,manifest 字段可能变了")

    arms: dict[str, dict[str, float]] = {}
    for b in BUDGETS:
        for arm in ("recent", "sel"):
            d = root / f"b{b}-{arm}"
            if d.is_dir():
                arms[f"b{b}-{arm}"] = load_arm(d)

    out: dict = {"coverage": {k: len(v) for k, v in arms.items()}, "curve": {}}
    for b in BUDGETS:
        rec = arms.get(f"b{b}-recent")
        sel = arms.get(f"b{b}-sel")
        if rec is None:
            continue
        if b == 0:
            # note (luojiaxuan): B=0 时 selector 无槽位可分配,与 Recent-0 结构等同。
            out["curve"]["B=0"] = {"recent": rate(rec, sorted(rec)),
                                   "selector": rate(rec, sorted(rec)),
                                   "delta": {"n": len(rec), "delta_pp": 0.0,
                                             "ci95": [0.0, 0.0], "note": "结构等同,未单独跑"}}
            continue
        if sel is None:
            continue
        keys = sorted(set(rec) & set(sel))
        mk = [t for t in keys if t in memcrit]
        ck = [t for t in keys if t not in memcrit]
        out["curve"][f"B={b}"] = {
            "recent": rate(rec, keys), "selector": rate(sel, keys),
            "delta": paired_boot(sel, rec, keys),
            "memory_critical": {"n": len(mk), "recent": rate(rec, mk), "selector": rate(sel, mk),
                                "delta": paired_boot(sel, rec, mk)} if mk else None,
            "control": {"n": len(ck), "recent": rate(rec, ck), "selector": rate(sel, ck),
                        "delta": paired_boot(sel, rec, ck)} if ck else None,
        }

    done = [k for k, v in out["curve"].items() if v.get("delta", {}).get("n")]
    peaks = [(v["delta"]["delta_pp"], k) for k, v in out["curve"].items() if v.get("delta", {}).get("n")]
    out["summary"] = {
        "budgets_reduced": done,
        "best_B_full": max(peaks)[1] if peaks else None,
        "best_delta_pp": max(peaks)[0] if peaks else None,
        # note (luojiaxuan): 36 任务子集统计力弱,峰值位置只作趋势参考;全量口径见
        # supplement 的 Table S12(117 任务,B=0/1/2/4/8)。
        "caveat": "36 任务固定子集,单轮;峰值位置仅供趋势参考",
    }
    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["coverage"], ensure_ascii=False))
    print(json.dumps(out["summary"], ensure_ascii=False))
    for k, v in out["curve"].items():
        d = v.get("delta", {})
        print(f"  {k}: recent={v['recent']:5.2f} sel={v['selector']:5.2f} "
              f"Δ={d.get('delta_pp')} CI={d.get('ci95')}")


if __name__ == "__main__":
    main()
