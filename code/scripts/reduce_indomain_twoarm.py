#!/usr/bin/env python3
"""同批双臂同域对照归约:修复后的 selector vs 同批 Recent-4。

为什么必须同批:实测两个 98% 行为相同的臂,任务级仍翻转 14.4%、净差 −3.88pp
(见 ../data/results/indomain_gap_v1/ROOT_CAUSE.md §5)。跨轮比较在这个噪声地板下
没有分辨力,所以对照臂必须在同一时间窗、同一批机器上跑。

输出:总体配对差、按域拆分、以及与历史臂(跨轮,仅供参考)的对比。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
from reduce_osworld30_horizon import paired_boot  # noqa: E402

B15 = ROOT / "data/results/osworld_verified_closed_loop"


def load_harvest(paths: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        out.update(json.loads(Path(p).read_text()))
    return out


def load_hist(name: str) -> dict[str, dict]:
    raw = json.loads((B15 / name).read_text())
    return {t.split("/")[-1]: {"score": float(raw[t].get("s", raw[t].get("score", 0.0))),
                               "domain": t.split("/")[0]}
            for t in raw}


def rate(d: dict, keys=None) -> float:
    k = list(keys) if keys is not None else list(d)
    return round(100.0 * sum(1 for t in k if d[t]["score"] >= 1.0) / len(k), 2) if k else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfix", nargs="+", required=True)
    ap.add_argument("--recent", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sel = load_harvest(args.selfix)
    rec = load_harvest(args.recent)
    K = sorted(set(sel) & set(rec))
    hist_sel = load_hist("per_task_sel1p.json")
    hist_rec = load_hist("per_task_recent.json")
    dom = {t: v.get("domain") for t, v in hist_sel.items()}

    win = [t for t in K if sel[t]["score"] >= 1 and rec[t]["score"] < 1]
    loss = [t for t in K if sel[t]["score"] < 1 and rec[t]["score"] >= 1]

    by_dom: dict[str, dict] = {}
    for t in K:
        d = dom.get(t, "?")
        e = by_dom.setdefault(d, {"n": 0, "win": 0, "loss": 0})
        e["n"] += 1
        if t in win:
            e["win"] += 1
        elif t in loss:
            e["loss"] += 1
    for d in by_dom:
        by_dom[d]["net"] = by_dom[d]["win"] - by_dom[d]["loss"]

    out = {
        "n_paired": len(K),
        "coverage": {"selfix": len(sel), "recent": len(rec)},
        "rates": {"selfix": rate(sel, K), "recent_same_batch": rate(rec, K)},
        "paired_same_batch": paired_boot(sel, rec),
        "discordance": {"selfix_win": len(win), "recent_win": len(loss),
                        "flip_pct": round(100.0 * (len(win) + len(loss)) / len(K), 1)},
        "by_domain": dict(sorted(by_dom.items())),
        # note (luojiaxuan): 跨轮对照仅供参考——单轮噪声地板 14.4%,不能据此下结论。
        "cross_round_reference": {
            "hist_sel1p": rate(hist_sel), "hist_recent": rate(hist_rec),
            "selfix_vs_hist_recent": paired_boot(sel, hist_rec),
            "caveat": "跨轮,受 14.4% 噪声地板影响,仅作参考",
        },
    }
    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in
                      ("n_paired", "rates", "paired_same_batch", "discordance")},
                     indent=1, ensure_ascii=False))
    print("按域:")
    for d, v in out["by_domain"].items():
        print(f"  {d:22} n={v['n']:3} 赢 {v['win']:2} 输 {v['loss']:2} 净 {v['net']:+2}")


if __name__ == "__main__":
    main()
