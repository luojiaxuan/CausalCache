#!/usr/bin/env python3
"""pass-1 探针规则在全部 winnable 态上的部署口径验证(阶梯 P0)。

# note (luojiaxuan): 侦察版(rl_draft_probe_rules.py)在 bcurve 的 292 个
# winnable 上给出 disagree-draft +4.58pp(p=0.0146)。本脚本把同三条规则
# (零参数,无拟合 → 无需划分,全量评估合法)扩到与 selector 部署评测
# 完全同一总体:labels_all 的 1263 个 winnable(有 token 缓存)态。
#   * b1 特征:probe1_sh*.jsonl(新枚举 1148 态)∪ bcurve_b1_sh*(115 态);
#   * 帧对正确性:labels_all 的 B=2 枚举表查表;
#   * 选中对不在表内 → 落盘 missing_pairs.jsonl,交 rl_score_chosen_subsets
#     冻结策略补测,--supplement 合并后 n_unmeasured=0 才算完整口径;
#   * recent-2 对账断言与部署评测器同款(错位即硬失败)。
用法:python3 rl_draft_probe_full.py <oracle目录> [--supplement x.jsonl]
      [--domain-json /data/task_domain.json --manifest ...]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mcnemar_p(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(diffs: list[int], reps: int = 1999, seed: int = 12345):
    import random
    r = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(r.choice(diffs) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def disagree(a: dict | None, b: dict | None, tol: float = 25.0) -> bool:
    if not a or not b:
        return a != b
    if a.get("action") != b.get("action"):
        return True
    ca, cb = a.get("coordinate"), b.get("coordinate")
    if ca and cb:
        return math.dist(ca, cb) > tol
    return any(a.get(k) != b.get(k) for k in ("text", "keys", "key"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--token-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--domain-json", type=Path, required=True)
    ap.add_argument("--supplement", type=Path, default=None)
    ap.add_argument("--missing-out", type=Path, default=None)
    args = ap.parse_args()

    # B=2 表(与部署评测同一总体口径:winnable 且有 token 缓存)
    table: dict[str, dict] = {}
    recent_ref: dict[str, bool] = {}
    for line in (args.dir / "labels_all.jsonl").open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        subs = {frozenset(a["s"]): bool(a["c"]) for a in d.get("all", [])
                if len(a["s"]) == 2}
        vals = list(subs.values())
        if not (vals and any(vals) and not all(vals)):
            continue
        if not (args.token_dir / f"{d['dp_id']}.pt").exists():
            continue
        table[d["dp_id"]] = subs
        recent_ref[d["dp_id"]] = bool(d["recent_correct"])

    # b1 特征合并(新探针优先,bcurve 兜底)
    b1: dict[str, dict] = {}
    for pat in ("bcurve_b1_sh*.jsonl", "probe1_sh*.jsonl"):
        for f in sorted(args.dir.glob(pat)):
            for line in f.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("all") and r["dp_id"] in table:
                    b1[r["dp_id"]] = r

    supp: dict[tuple[str, frozenset], bool] = {}
    if args.supplement and args.supplement.exists():
        for line in args.supplement.open():
            line = line.strip()
            if line:
                d = json.loads(line)
                supp[(d["dp_id"], frozenset(d["s"]))] = bool(d["c"])

    dom = json.loads(args.domain_json.read_text())
    t2d: dict[str, str] = {}
    for line in args.manifest.open():
        line = line.strip()
        if line:
            d = json.loads(line)
            t2d[d["dp_id"]] = dom.get(d.get("task_id"), "?")

    ids = sorted(set(table) & set(b1))
    print(json.dumps({"winnable_total": len(table), "with_b1": len(ids),
                      "missing_b1": len(table) - len(ids),
                      "supplement": len(supp)}, ensure_ascii=False))

    rules = {
        "UB-b1(不可部署)": lambda dp, j, e: (int(e["c"]),),
        "disagree-draft": lambda dp, j, e: (int(disagree(e.get("p"),
                                                         b1[dp]["b0_pred"])),),
        "agree-draft": lambda dp, j, e: (int(not disagree(e.get("p"),
                                                          b1[dp]["b0_pred"])),),
    }
    missing_rows = []
    for name, key in rules.items():
        rows, miss = [], 0
        for dp in ids:
            entries = [(a["s"][0], a) for a in b1[dp]["all"]]
            if len(entries) < 2:
                continue
            entries.sort(key=lambda x: (key(dp, x[0], x[1]), x[0]), reverse=True)
            pick = frozenset(x[0] for x in entries[:2])
            recent = frozenset(sorted(x[0] for x in entries)[-2:])
            subs = table[dp]
            if recent not in subs or subs[recent] != recent_ref[dp]:
                raise SystemExit(f"候选池错位:{dp}")
            c_sel = subs.get(pick)
            if c_sel is None:
                c_sel = supp.get((dp, pick))
            if c_sel is None:
                miss += 1
                missing_rows.append({"dp_id": dp, "subset": sorted(pick)})
                continue
            rows.append({"dp": dp, "domain": t2d.get(dp, "?"),
                         "c_sel": int(c_sel), "c_rec": int(recent_ref[dp]),
                         "moved": int(pick != recent)})
        for tag, sel in (("总体", rows),
                         ("MultiApp", [r for r in rows if r["domain"] == "MultiApp"]),
                         ("单应用", [r for r in rows if r["domain"] != "MultiApp"])):
            n = len(sel)
            if not n:
                continue
            hit = sum(r["c_sel"] for r in sel)
            rec = sum(r["c_rec"] for r in sel)
            n01 = sum(1 for r in sel if r["c_sel"] and not r["c_rec"])
            n10 = sum(1 for r in sel if r["c_rec"] and not r["c_sel"])
            lo, hi = boot_ci([r["c_sel"] - r["c_rec"] for r in sel])
            print(f"{name} {tag}: n={n}(未测 {miss if tag == '总体' else '-'}) "
                  f"sel={100*hit/n:.1f}% rec={100*rec/n:.1f}% "
                  f"diff={100*(hit-rec)/n:+.2f}pp CI[{100*lo:+.2f},{100*hi:+.2f}] "
                  f"赢/输={n01}/{n10} p={mcnemar_p(n01, n10):.4f} "
                  f"moved={sum(r['moved'] for r in sel)}/{n}")

    if args.missing_out and missing_rows:
        seen = set()
        with args.missing_out.open("w") as w:
            for r in missing_rows:
                k = (r["dp_id"], tuple(r["subset"]))
                if k not in seen:
                    seen.add(k)
                    w.write(json.dumps(r) + "\n")
        print(json.dumps({"missing_pairs_out": len(seen)}))


if __name__ == "__main__":
    main()
