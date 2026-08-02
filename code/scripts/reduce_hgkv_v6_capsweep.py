#!/usr/bin/env python3
"""v6 收紧 drift cap 三臂 vs v4 基线的 held-out 门控归约。

# note (luojiaxuan): 判据是**两个量必须同时看**,只看一个会得出相反结论:
#   adapter_on_recent(A_r)—— 均匀成分,收紧 cap 后**应当变小**。
#   did_select (A_s − A_r) —— 选择性,**不能跟着一起塌**。
# 只看 A_r 变小就宣布成功,等于奖励"把适配器学成恒等映射"——A_r=0 且
# did_select=0 是最平凡的解,门控上好看,闭环上什么都没有。
#
# v4 基线必须来自**同一脚本、同一 dev 集**的重打(v4baseline 臂),不能引用
# 论文里的 +0.0088 / +0.0224 —— 那两个数不是本协议产出的,
# /data/runs/desktop-did-v4/hgkv 下没有 gate_report.json。
#
# 选点口径:各臂都取该臂 gate_report 自己选中的 checkpoint(selection 字段);
# 若没选出可选点(所有 checkpoint 都没过硬门控),回退到 did_select 最大的那个,
# 并在输出里标 `fallback_no_selectable`,**不要静默当成选中**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (
    "adapter_on_recent",
    "adapter_on_recent_abs",
    "adapter_on_sparse",
    "did_select",
    "wrong_drift_abs",
    "SA_minus_RA",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True,
                   metavar="LABEL=PATH",
                   help="臂标签=gate_report.json 路径,可重复。第一个应是 v4 基线")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--title", default="held-out 门控对比")
    return p.parse_args()


def _q(ck: dict, key: str) -> dict | None:
    v = (ck.get("derived_quantities") or {}).get(key)
    if not isinstance(v, dict) or v.get("value") is None:
        return None
    return v


def _fmt(v: dict | None) -> str:
    if v is None:
        return "—"
    lo, hi = v.get("ci_low"), v.get("ci_high")
    if lo is None or hi is None:
        return f"{v['value']:+.5f}"
    return f"{v['value']:+.5f} [{lo:+.5f}, {hi:+.5f}]"


def _excludes_zero(v: dict | None) -> bool:
    if v is None:
        return False
    lo, hi = v.get("ci_low"), v.get("ci_high")
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


def pick(report: dict) -> tuple[dict, str]:
    """取该臂 gate_report 自己选中的 checkpoint。

    # note (luojiaxuan): 2026-08-02 修 bug —— 字段名是 ``selection.selected``,
    # 原来只找 ``selected_label``/``label``,取不到就**静默**退到"第一个 selectable"
    # (= lora-step50,训练最早、几乎什么都没学到),还把 pick_mode 标成 selectable
    # 而不是 fallback。后果:didbase 被读成 A_r=+0.0036/did_select=+0.0002,
    # 而它真正选中的 lora-step300 是 +0.0009/+0.0152 —— 判定完全反过来。
    # 现在三个键都认,且一个都取不到时必须显式落到 fallback 分支并被标出来。
    """
    cks = report.get("checkpoints") or []
    sel = report.get("selection") or {}
    label = sel.get("selected") or sel.get("selected_label") or sel.get("label")
    if label:
        for ck in cks:
            if ck.get("label") == label:
                return ck, "selected"
        raise ValueError(
            f"gate_report 选中了 {label!r},但 checkpoints 里没有这个 label —— "
            "报告自相矛盾,不做静默回退"
        )
    for ck in cks:
        if ck.get("selectable"):
            return ck, "fallback_first_selectable"
    best, bv = None, None
    for ck in cks:
        v = _q(ck, "did_select")
        if v is not None and (bv is None or v["value"] > bv):
            best, bv = ck, v["value"]
    return (best or (cks[0] if cks else {})), "fallback_no_selectable"


def main() -> None:
    args = parse_args()
    arms: list[tuple[str, dict]] = []
    for spec in args.arm:
        label, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            print(f"跳过缺失的臂 {label}: {p}")
            continue
        arms.append((label, json.loads(p.read_text(encoding="utf-8"))))
    if not arms:
        raise SystemExit("没有可读的 gate_report")

    rows = []
    for label, rep in arms:
        ck, how = pick(rep)
        rows.append({
            "arm": label,
            "checkpoint": ck.get("label"),
            "pick_mode": how,
            "selectable": bool(ck.get("selectable")),
            "n_groups": ((_q(ck, "did_select") or {}).get("support_groups")),
            **{k: _q(ck, k) for k in KEYS},
        })

    base = rows[0]
    lines = [
        f"# {args.title}",
        "",
        f"dev 集:`{arms[0][1].get('dataset_root')}`,"
        f"{base.get('n_groups')} 组 / "
        f"{arms[0][1].get('heldout', {}).get('episodes', '?')} episode。"
        "所有臂同脚本同 dev 集。",
        "",
        "| 臂 | checkpoint | 选点 | A_r(均匀,应变小) | did_select(选择性,不能塌) | A_c | \\|wrong drift\\| |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['arm']}** | {r['checkpoint']} | {r['pick_mode']} | "
            f"{_fmt(r['adapter_on_recent'])} | {_fmt(r['did_select'])} | "
            f"{_fmt(r['adapter_on_sparse'])} | {_fmt(r['wrong_drift_abs'])} |"
        )

    lines += ["", "## 判定", ""]
    b_ar = base["adapter_on_recent"]
    b_ds = base["did_select"]
    verdicts = []
    for r in rows[1:]:
        ar, ds = r["adapter_on_recent"], r["did_select"]
        if ar is None or ds is None or b_ar is None or b_ds is None:
            verdicts.append((r["arm"], "数据不全", "—"))
            continue
        shrank = abs(ar["value"]) < abs(b_ar["value"])
        # 选择性保住 = CI 仍排除 0,且点估计不低于基线的一半
        held = _excludes_zero(ds) and ds["value"] >= 0.5 * b_ds["value"]
        if shrank and held:
            v = "**通过**:均匀成分下降且选择性保住"
        elif shrank and not held:
            v = "**cap 收过头**:均匀成分下降但选择性一起塌了"
        elif not shrank and held:
            v = "**无效**:均匀成分没降下来"
        else:
            v = "**更差**:均匀成分没降、选择性还塌了"
        ratio_ar = abs(ar["value"]) / abs(b_ar["value"]) if b_ar["value"] else float("nan")
        ratio_ds = ds["value"] / b_ds["value"] if b_ds["value"] else float("nan")
        verdicts.append((r["arm"], v, f"A_r 为基线的 {ratio_ar:.0%},did_select 为 {ratio_ds:.0%}"))
        lines.append(f"- `{r['arm']}` — {v}({ratio_ar:.0%} / {ratio_ds:.0%})")

    lines += [
        "",
        "判据说明:只看 A_r 变小会奖励平凡解(适配器退化成恒等映射时 "
        "A_r=0 且 did_select=0,门控好看、闭环无物),所以两个量必须同时满足。"
        "选择性保住的定义是 CI 仍排除 0 **且**点估计不低于基线的一半。",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    (args.output_dir / "capsweep.json").write_text(
        json.dumps({"rows": rows,
                    "verdicts": [{"arm": a, "verdict": v, "detail": d}
                                 for a, v, d in verdicts]},
                   indent=1, ensure_ascii=False),
        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
