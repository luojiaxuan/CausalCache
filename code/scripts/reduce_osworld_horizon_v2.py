#!/usr/bin/env python3
"""OSWorld 步长曲线 v2 + 教师来源对照。

两个问题:
  1. HGKV-taught selector 的成功率随步长预算怎么变(15/30/50/100)?
  2. 用冻结策略自身打标签训出的 selector(frozen-taught),在 30 步下与 HGKV-taught 差多少?

复用 reduce_osworld30_horizon.py 的 load30 / paired_boot,口径与既有产物一致。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
from reduce_osworld30_horizon import load30, paired_boot, rate  # noqa: E402

H = ROOT / "data/results/osworld30_horizon_v1"
BASE = ROOT / "data/results/osworld_verified_closed_loop"


def cap_pct(d: dict, cap: int) -> float:
    n = len(d)
    return round(100.0 * sum(1 for v in d.values() if v.get("steps", -1) >= cap) / n, 1) if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s50", required=True)
    ap.add_argument("--ft30", required=True)
    ap.add_argument("--s100", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # note (luojiaxuan): 15-step 基线键为 domain/uuid、值为 {'s':0/1,'steps':n},
    # 归一到 uuid + score(与 reduce_osworld30_horizon.py 同口径)。
    def load15(path: Path) -> dict[str, dict]:
        raw = json.loads(path.read_text())
        return {t.split("/")[-1]: {"score": float(v.get("s", v.get("score", 0.0))),
                                   "steps": int(v.get("steps", -1))}
                for t, v in raw.items()}

    sel15 = load15(BASE / "per_task_sel1p.json")
    rec15 = load15(BASE / "per_task_recent.json")
    # note (luojiaxuan): 顺序必须与 v1 产物一致(h100 优先),否则 10 个跨机重复任务里
    # 那 7 个结果不一致的会把 sel@30 在 32.96/33.24 之间来回改写。
    sel30 = load30(f"{H / 'osw30_sel_h100.json'},{H / 'osw30_sel_aries.json'}")
    rec30 = load30(f"{H / 'osw30_recent_h00.json'},{H / 'osw30_recent_h01.json'}")

    # note (luojiaxuan): 跨机重复任务的复现率——这是判断多大的差值才有意义的标尺。
    a_raw = json.loads((H / "osw30_sel_aries.json").read_text())
    h_raw = json.loads((H / "osw30_sel_h100.json").read_text())
    dup = sorted(set(a_raw) & set(h_raw))
    disagree = [t for t in dup
                if (a_raw[t]["score"] >= 1.0) != (h_raw[t]["score"] >= 1.0)]
    sel50 = load30(args.s50)
    ft30 = load30(args.ft30)
    s100 = load30(args.s100) if args.s100 else {}

    out: dict = {
        "n_tasks": {k: len(v) for k, v in
                    {"sel15": sel15, "sel30": sel30, "sel50": sel50,
                     "sel100": s100, "ft30": ft30, "recent15": rec15, "recent30": rec30}.items()},
        "rates": {
            "sel@15": round(rate(sel15), 2), "sel@30": round(rate(sel30), 2),
            "sel@50": round(rate(sel50), 2), "sel@100": round(rate(s100), 2) if s100 else None,
            "recent@15": round(rate(rec15), 2), "recent@30": round(rate(rec30), 2),
            "frozen_taught@30": round(rate(ft30), 2),
        },
        "hit_cap_pct": {"sel@30": cap_pct(sel30, 30), "sel@50": cap_pct(sel50, 50),
                        "frozen_taught@30": cap_pct(ft30, 30),
                        "sel@100": cap_pct(s100, 100) if s100 else None},
        # note (luojiaxuan): 步长曲线——每一段都相对 15 步基线配对,避免链式比较累积噪声。
        "horizon_vs_15": {
            "sel@30": paired_boot(sel30, sel15),
            "sel@50": paired_boot(sel50, sel15),
        },
        "horizon_50_vs_30": paired_boot(sel50, sel30),
        # note (luojiaxuan): 教师来源对照,两臂同为 30 步、同 361 任务,唯一变量是标签教师。
        "teacher_source_30": paired_boot(ft30, sel30),
        # note (luojiaxuan): 同一配置在两台机器上重复跑到的任务的复现情况。这是本基准
        # 单轮噪声的直接测量,任何小于它的差值都不应解读为效应。
        "replicate_disagreement": {
            "n_duplicated_tasks": len(dup),
            "n_flipped": len(disagree),
            "flip_rate_pct": round(100.0 * len(disagree) / len(dup), 1) if dup else None,
        },
    }
    if s100:
        out["horizon_vs_15"]["sel@100_partial"] = paired_boot(s100, sel15)

    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
