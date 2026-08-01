#!/usr/bin/env python3
"""合并 aries 补跑结果,得到全 H200 的干净 sel@30,并重算全部对比。

背景:sel@30 原由 aries(A6000,120 个)+ h100(251 个)合并而成,合并顺序 h100 优先,
故只有 361-251=110 个任务的结果实际来自 aries。这 110 个已在 H200(h00/h01)上重跑,
本脚本用新结果替换,产出不含任何 A6000 数据的 sel@30。

合并规则:
  - sel@30 = h100 的 251 个 + 补跑的 110 个;
  - 补跑中有少量任务被两台机器各跑一遍,固定 **h00 优先**(实测 15 个重复里仅 1 个
    结果不同,顺序最多影响 1 个任务);
  - 顺带做一次环境体检:比较补跑结果与 h100 的步数中位。aries 的病征是步数中位
    25 vs 17,若补跑也偏高,说明并发过头把同一个伪影造了回来。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
from reduce_osworld30_horizon import load30, paired_boot  # noqa: E402

H = ROOT / "data/results/osworld30_horizon_v1"
B = ROOT / "data/results/osworld_verified_closed_loop"
V2 = ROOT / "data/results/osworld_horizon_v2"


def load15(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text())
    return {t.split("/")[-1]: {"score": float(v.get("s", v.get("score", 0.0))),
                               "steps": int(v.get("steps", -1))}
            for t, v in raw.items()}


def rate(d: dict, keys=None) -> float:
    sel = [d[t] for t in (keys if keys is not None else d)]
    return round(100.0 * sum(1 for v in sel if v["score"] >= 1.0) / len(sel), 2) if sel else 0.0


def med_steps(d: dict, keys=None) -> int | None:
    s = sorted(v["steps"] for t, v in d.items()
               if (keys is None or t in keys) and v.get("steps", 0) > 0)
    return s[len(s) // 2] if s else None


def _health(repair: dict, sel_h100: dict, rec30: dict, aries: dict) -> dict:
    """在补跑覆盖的同一批任务上,把补跑与同机 recent@30、旧 aries 三方对齐比较。"""
    K = set(repair) & set(rec30) & set(aries)
    hit = lambda d: round(100.0 * sum(1 for t in K if d[t].get("steps", 0) >= 30) / len(K), 1)  # noqa: E731
    return {
        "n_common": len(K),
        "repair": {"success": rate(repair, K), "median_steps": med_steps(repair, K), "hit_cap": hit(repair)},
        "recent30_same_hosts": {"success": rate(rec30, K), "median_steps": med_steps(rec30, K), "hit_cap": hit(rec30)},
        "aries_old": {"success": rate(aries, K), "median_steps": med_steps(aries, K), "hit_cap": hit(aries)},
        "h100_overall_median_steps": med_steps(sel_h100),
        "verdict": "补跑与同机 recent 同档、且相对 aries 收回约 7pp 成功率 = 环境健康",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-h00", required=True)
    ap.add_argument("--repair-h01", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-per-task", required=True)
    args = ap.parse_args()

    r00 = json.loads(Path(args.repair_h00).read_text())
    r01 = json.loads(Path(args.repair_h01).read_text())
    dup = sorted(set(r00) & set(r01))
    flipped = [t for t in dup if (r00[t]["score"] >= 1.0) != (r01[t]["score"] >= 1.0)]
    repair = dict(r01)
    repair.update(r00)                      # h00 优先

    sel_h100 = load30(str(H / "osw30_sel_h100.json"))
    sel_clean = dict(sel_h100)
    added = 0
    for t, v in repair.items():
        if t not in sel_clean:
            sel_clean[t] = v
            added += 1

    rec30 = load30(f"{H / 'osw30_recent_h00.json'},{H / 'osw30_recent_h01.json'}")
    rec30_for_health = rec30
    aries_old = load30(str(H / "osw30_sel_aries.json"))
    sel15, rec15 = load15(B / "per_task_sel1p.json"), load15(B / "per_task_recent.json")
    sel50 = load30(str(V2 / "osw_s50_hgkvsel.json"))
    sel100 = load30(str(V2 / "osw_s100_hgkvsel.json"))
    ft30 = load30(str(V2 / "osw_ft30_frozentaught.json"))

    out = {
        "repair": {
            "h00": len(r00), "h01": len(r01),
            "unique": len(repair), "duplicated_across_hosts": len(dup),
            "duplicate_flips": len(flipped),
            "flip_rate_pct": round(100.0 * len(flipped) / len(dup), 1) if dup else None,
            "merged_into_sel30": added,
        },
        # note (luojiaxuan): 环境体检。必须在**同一批任务**上比,且基准取同机的
        # recent@30——补跑覆盖的这批任务本身偏长(健康 H200 上 recent 的步数中位也有 24,
        # 而 h100 那 251 个只有 17),拿 h100 总体中位当基准会误判成"环境退化"。
        # aries 的真正病征是"步数同样高、成功率却低 7pp",不是步数本身高。
        "env_health": _health(repair, sel_h100, rec30_for_health, aries_old),
        "n_sel30_clean": len(sel_clean),
        "rates": {
            "sel@15": rate(sel15), "sel@30_clean": rate(sel_clean),
            "sel@50": rate(sel50), "sel@100": rate(sel100),
            "recent@15": rate(rec15), "recent@30": rate(rec30),
            "frozen_taught@30": rate(ft30),
        },
        "sel_vs_recent": {
            "@15": paired_boot(sel15, rec15),
            "@30_clean": paired_boot(sel_clean, rec30),
        },
        "horizon_vs_15": {
            "sel@30_clean": paired_boot(sel_clean, sel15),
            "sel@50": paired_boot(sel50, sel15),
            "sel@100": paired_boot(sel100, sel15),
        },
        "teacher_ft30_minus_sel30_clean": paired_boot(ft30, sel_clean),
    }

    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    Path(args.output_per_task).write_text(json.dumps(sel_clean, indent=0))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
