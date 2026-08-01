#!/usr/bin/env python3
"""Reduce review-ablation MobileWorld arms (text-only / OCR / k1cap / HGKV+selector).

输入是每臂一个 task->score 映射(双机收割去重后),输出 full/memory-critical/control
三列成功率,并对照 Frozen+selector B=4(3 轮均值)与 HGKV+Recent-4(3 轮均值)
给出任务配对差与 bootstrap CI。历史 ``frozensel`` 输入名实际对应论文主臂
HGKV+selector；旧 ``mobileworld_hgkv_selected_b4`` 目录实际保存
Frozen+selector repeated comparator。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_MANIFEST = ROOT / "data/manifests/mobileworld_memory_split_v1.json"
# note (luojiaxuan): 目录名来自已确认的历史 arm-label 对调；不得据此把
# 33.9/28.0/40.6 重新标成论文的 HGKV+selector 主臂。
LEGACY_FROZEN_SELECTOR_DIR = ROOT / "data/results/mobileworld_hgkv_selected_b4"


def load_split() -> tuple[set[str], set[str]]:
    manifest = json.loads(SPLIT_MANIFEST.read_text())
    mem, ctl = set(), set()
    for rec in manifest["records"]:
        if rec["interface"] != "gui_only":
            continue
        if rec["memory_split"] == "cross_app_memory_candidate":
            mem.add(rec["task_name"])
        elif rec["memory_split"] == "single_app_control":
            ctl.add(rec["task_name"])
    return mem, ctl


def round_mean_maps(paths: list[Path]) -> dict[str, float]:
    rounds = [json.loads(p.read_text()) for p in paths]
    tasks = set(rounds[0])
    for r in rounds[1:]:
        if set(r) != tasks:
            raise SystemExit("baseline rounds disagree on roster")
    return {t: sum(float(r[t]) for r in rounds) / len(rounds) for t in tasks}


def rate(scores: dict[str, float], subset: set[str] | None = None) -> tuple[float, int]:
    keys = [t for t in scores if subset is None or t in subset]
    return (100.0 * sum(scores[t] for t in keys) / len(keys), len(keys)) if keys else (float("nan"), 0)


def paired_boot(arm: dict[str, float], base: dict[str, float], subset: set[str] | None,
                n_boot: int = 20000, seed: int = 20260728) -> dict:
    tasks = sorted(t for t in arm if t in base and (subset is None or t in subset))
    diffs = [arm[t] - base[t] for t in tasks]
    mean = 100.0 * sum(diffs) / len(diffs)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        stats.append(100.0 * sum(s) / len(s))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[int(0.975 * n_boot) - 1]
    p = 2.0 * min(sum(1 for x in stats if x <= 0), sum(1 for x in stats if x >= 0)) / n_boot
    return {"n": len(tasks), "delta_pp": round(mean, 2), "ci95": [round(lo, 2), round(hi, 2)],
            "boot_p": round(min(p, 1.0), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-map", action="append", required=True,
                    help="name=path/to/task_map.json,可多次")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mem, ctl = load_split()
    frozen_selector_b4 = round_mean_maps(
        [LEGACY_FROZEN_SELECTOR_DIR / f"per_task_success_r{i}.json" for i in (1, 2, 3)]
    )
    recent_b4 = round_mean_maps(
        [LEGACY_FROZEN_SELECTOR_DIR / f"recent_per_task_r{i}.json" for i in (1, 2, 3)]
    )

    out = {"split_counts": {"memory_critical": len(mem), "control": len(ctl)}, "arms": {}}
    for spec in args.arm_map:
        name, path = spec.split("=", 1)
        scores = {t: float(v) for t, v in json.loads(Path(path).read_text()).items()}
        missing = sorted(set(frozen_selector_b4) - set(scores))
        row = {
            "n_tasks": len(scores),
            "missing_vs_roster": missing,
            "full_pct": round(rate(scores)[0], 1),
            "memory_critical_pct": round(rate(scores, mem)[0], 1),
            "control_pct": round(rate(scores, ctl)[0], 1),
            "vs_frozen_selector_b4": {
                "full": paired_boot(scores, frozen_selector_b4, None),
                "memory_critical": paired_boot(scores, frozen_selector_b4, mem),
                "control": paired_boot(scores, frozen_selector_b4, ctl),
            },
            "vs_recent_b4": {
                "full": paired_boot(scores, recent_b4, None),
                "memory_critical": paired_boot(scores, recent_b4, mem),
                "control": paired_boot(scores, recent_b4, ctl),
            },
        }
        out["arms"][name] = row
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    for name, row in out["arms"].items():
        print(f"{name}: full={row['full_pct']} mem={row['memory_critical_pct']} "
              f"ctl={row['control_pct']} missing={len(row['missing_vs_roster'])}")


if __name__ == "__main__":
    main()
