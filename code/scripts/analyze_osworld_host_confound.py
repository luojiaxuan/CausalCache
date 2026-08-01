#!/usr/bin/env python3
"""OSWorld 跨主机混杂诊断。

动机:各臂是按可用机器分片跑的,不同臂的主机构成并不一致。若主机之间成功率有系统性
差异,臂间比较就被主机效应污染。本脚本量化主机效应,并给出剔除离群主机后的重算。

用法:python3 code/scripts/analyze_osworld_host_confound.py --output <json>
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


def host_profile(d: dict) -> dict:
    steps = sorted(v["steps"] for v in d.values() if v.get("steps", 0) > 0)
    return {
        "n": len(d),
        "success_pct": rate(d),
        "hit_cap_pct": round(100.0 * sum(1 for v in d.values() if v.get("steps", 0) >= 30) / len(d), 1),
        "median_steps": steps[len(steps) // 2] if steps else None,
    }


def standardize(src: dict, ref: dict) -> float:
    """把 src 的分域成功率套到 ref 的域构成上,剥离任务构成差异。"""
    doms = {v.get("domain") for v in ref.values()}
    num = den = 0.0
    for dom in doms:
        s = [v for v in src.values() if v.get("domain") == dom]
        r = [v for v in ref.values() if v.get("domain") == dom]
        if not s or not r:
            continue
        num += (sum(1 for v in s if v["score"] >= 1.0) / len(s)) * len(r)
        den += len(r)
    return round(100.0 * num / den, 2) if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    per_host = {
        "sel30_aries": load30(str(H / "osw30_sel_aries.json")),
        "sel30_h100": load30(str(H / "osw30_sel_h100.json")),
        "recent30_h00": load30(str(H / "osw30_recent_h00.json")),
        "recent30_h01": load30(str(H / "osw30_recent_h01.json")),
    }
    out: dict = {"host_profiles": {k: host_profile(v) for k, v in per_host.items()}}

    # note (luojiaxuan): aries 是唯一的离群主机,且只出现在 selector 臂里。先确认这不是
    # 任务构成造成的——把 aries 的分域率套到 h100 的构成上重算。
    out["aries_domain_standardized_to_h100"] = standardize(
        per_host["sel30_aries"], per_host["sel30_h100"])

    sel30_all = load30(f"{H / 'osw30_sel_h100.json'},{H / 'osw30_sel_aries.json'}")
    sel30_clean = per_host["sel30_h100"]
    rec30 = load30(f"{H / 'osw30_recent_h00.json'},{H / 'osw30_recent_h01.json'}")
    sel15, rec15 = load15(B / "per_task_sel1p.json"), load15(B / "per_task_recent.json")
    sel50 = load30(str(V2 / "osw_s50_hgkvsel.json"))
    ft30 = load30(str(V2 / "osw_ft30_frozentaught.json"))

    out["contaminated"] = {
        "sel30": rate(sel30_all), "recent30": rate(rec30),
        "delta": paired_boot(sel30_all, rec30),
        "teacher_ft30_minus_sel30": paired_boot(ft30, sel30_all),
    }

    # note (luojiaxuan): 干净口径 = 全部臂限制在 h100 跑过的那 251 个任务上,任务匹配。
    K = set(sel30_clean) & set(sel15) & set(sel50) & set(rec30) & set(rec15)
    sub = lambda d: {t: d[t] for t in K}  # noqa: E731
    out["aries_removed"] = {
        "n": len(K),
        "rates": {"sel@15": rate(sel15, K), "sel@30": rate(sel30_clean, K),
                  "sel@50": rate(sel50, K), "recent@15": rate(rec15, K),
                  "recent@30": rate(rec30, K)},
        "sel_30_vs_15": paired_boot(sub(sel30_clean), sub(sel15)),
        "sel_50_vs_15": paired_boot(sub(sel50), sub(sel15)),
        "recent_30_vs_15": paired_boot(sub(rec30), sub(rec15)),
        "sel_vs_recent_15": paired_boot(sub(sel15), sub(rec15)),
        "sel_vs_recent_30": paired_boot(sub(sel30_clean), sub(rec30)),
    }

    Kt = set(sel30_clean) & set(ft30)
    out["teacher_after_removal"] = {
        "n": len(Kt),
        "ft30": rate(ft30, Kt), "sel30": rate(sel30_clean, Kt),
        "delta": paired_boot({t: ft30[t] for t in Kt}, {t: sel30_clean[t] for t in Kt}),
        # note (luojiaxuan): 即便剔除 aries,这两臂仍分别只跑在 h01 与 h100 上,
        # 从未共享主机,因此该对比仍不可解释,数字仅供记录。
        "caveat": "ft30 全部来自 h01,sel30 全部来自 h100;两臂无共同主机,不可解释",
    }

    Path(args.output).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
