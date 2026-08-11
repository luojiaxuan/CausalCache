#!/usr/bin/env python3
"""selector 2.6 部署口径归约:逐(种子,折)选检查点 → out-of-fold 合并 + 分层。

# note (luojiaxuan): 检查点选择协议沿用预注册读法(池化线起就固定的
# "1/4/16 步/态取 holdout 最好"),这里按 proxy holdout_pair 选,再报同一
# scorer 的部署口径数字。已知缺陷如实报:检查点是在同折留出集上按代理指标
# 选的,部署数字与选择共享样本,有轻微乐观倾向 —— 因此同时报固定 s=1 的
# 敏感性行。选中对未被枚举覆盖的态由 rl_score_chosen_subsets.py 冻结策略
# 真跑补齐,经 --supplement 并回;归约后 n_unmeasured 必须为 0 才是完整口径。
用法:python3 rl_selector_deploy_reduce.py <deploy2_json 目录> [--supplement 补测.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_selector_deploy_eval import summarize


def pick(reports: list[dict], config: str, steps: int | None) -> dict:
    """steps=None → 按 proxy holdout_pair 取最好(预注册读法)。"""
    cand = [r for r in reports if r["config"] == config]
    if steps is not None:
        cand = [r for r in cand if r["steps_per_state"] == steps]
    return max(cand, key=lambda r: r["proxy_holdout_pair"])


def fmt(tag: str, agg: dict) -> str:
    if not agg.get("n"):
        return f"{tag}: n=0 unmeasured={agg.get('n_unmeasured', 0)}"
    s = (f"{tag}: n={agg['n']}(未测 {agg['n_unmeasured']}) "
         f"sel={100*agg['sel_acc']:.1f}% rec={100*agg['rec_acc']:.1f}% "
         f"diff={agg['diff_pp']:+.2f}pp "
         f"CI[{agg['ci95_pp'][0]:+.2f},{agg['ci95_pp'][1]:+.2f}] "
         f"p={agg['mcnemar_p']} 赢/输={agg['n01_sel_win']}/{agg['n10_rec_win']} "
         f"moved={agg['moved']}/{agg['n']}")
    if "pool_acc" in agg:
        s += f" | 池内argmax={100*agg['pool_acc']:.1f}%({agg['pool_diff_pp']:+.2f}pp)"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--supplement", type=Path, default=None)
    args = ap.parse_args()

    supp: dict[tuple[str, frozenset], bool] = {}
    if args.supplement and args.supplement.exists():
        for line in args.supplement.open(encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                supp[(d["dp_id"], frozenset(d["s"]))] = bool(d["c"])
    print(f"补测表:{len(supp)} 对")

    files = {}
    for f in sorted(args.dir.glob("deploy2_s*_f*.json")):
        s, k = f.stem.split("_")[1:]
        doc = json.loads(f.read_text())
        for r in doc["reports"]:
            filled = 0
            for row in r["rows"]:
                key = (row["dp"], frozenset(row["chosen"]))
                if row["c_sel"] is None and key in supp:
                    row["c_sel"] = int(supp[key])
                    filled += 1
            r["_filled"] = filled
        files[(int(s[1:]), int(k[1:]))] = doc

    print("== 逐(种子,折):xattn+recency,检查点按 proxy 最好 ==")
    for (s, k), doc in sorted(files.items()):
        r = pick(doc["reports"], "xattn+recency", None)
        print(f"s{s} f{k} ckpt=s{r['steps_per_state']} "
              f"proxy={r['proxy_holdout_pair']:.4f} 补测并入={r['_filled']} | "
              + fmt("deploy", summarize(r["rows"])))

    def merged(config: str, steps: int | None, seed: int = 0):
        rows = []
        for (s, k), doc in sorted(files.items()):
            if s != seed:
                continue
            rows += pick(doc["reports"], config, steps)["rows"]
        return rows

    for config in ("xattn+recency", "xattn"):
        for steps, tag in ((None, "best-ckpt"), (1, "s=1 固定")):
            rows = merged(config, steps)
            print(f"\n== out-of-fold 合并 seed0 folds0-4  {config} / {tag} ==")
            print(fmt("总体", summarize(rows)))
            for dom in ("MultiApp", "单应用"):
                sel = [r for r in rows
                       if (r["domain"] == "MultiApp") == (dom == "MultiApp")]
                print(fmt(f"  {dom}", summarize(sel)))

    print("\n== fold0 跨种子(xattn+recency,best-ckpt)==")
    for s in (0, 1, 2):
        doc = files.get((s, 0))
        if doc:
            r = pick(doc["reports"], "xattn+recency", None)
            print(f"seed{s} ckpt=s{r['steps_per_state']} "
                  f"proxy={r['proxy_holdout_pair']:.4f} | "
                  + fmt("deploy", summarize(r["rows"])))


if __name__ == "__main__":
    main()
