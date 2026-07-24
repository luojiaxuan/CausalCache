#!/usr/bin/env python3
"""Reduce Frozen/Recent main-table row: template-macro success per memory budget.

# note (luojiaxuan): 口径(用户 2026-07-24 冻结)——每 template 内先平均 3 instance
# (task_index 0/1/2),再对 116 template 等权 macro;bootstrap CI 以 template 为
# cluster(不把 348 局当独立单位)。所有 headline 方法同 denominator。
# Success = (1/116) Σ_t (y_t0 + y_t1 + y_t2)/3。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

BUDGET_ARMS = [
    ("B0", "summary_B0"),
    ("B1", "recent_B1"),
    ("B2", "recent_B2"),
    ("B4", "recent_B4"),
    ("B8", "recent_B8"),
]


def load(results_root: Path) -> dict[tuple[str, str], dict[int, bool]]:
    # (arm, task_type) -> {task_index: success}
    table: dict[tuple[str, str], dict[int, bool]] = defaultdict(dict)
    for path in results_root.glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        arm = d.get("arm")
        tt = d.get("task_type")
        ti = d.get("task_index")
        if arm is None or tt is None or ti is None:
            continue
        table[(arm, tt)][int(ti)] = bool(d.get("official_terminal_success"))
    return table


def template_macro(
    table: dict[tuple[str, str], dict[int, bool]], arm: str, templates: list[str]
) -> tuple[float, int, list[float]]:
    """Return (macro_success, n_templates_full, per_template_rates)."""
    rates = []
    full = 0
    for tt in templates:
        inst = table.get((arm, tt))
        if not inst:
            continue
        vals = [int(inst[i]) for i in sorted(inst)]
        if len(vals) == 3:
            full += 1
        rates.append(sum(vals) / len(vals))
    macro = sum(rates) / len(rates) if rates else 0.0
    return macro, full, rates


def bootstrap_ci(rates: list[float], seed: int = 7) -> tuple[float, float]:
    if not rates:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(rates)
    boots = sorted(
        sum(rates[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000)
    )
    return (boots[100], boots[3899])  # 95% percentile CI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    templates = sorted({i["task_type"] for i in roster["instances"]})
    table = load(args.results_root)

    report = {"n_templates": len(templates), "budgets": {}}
    pos_rates = None
    lines = ["budget  macro%   templates(3/3)   95%CI(template-cluster)"]
    for label, arm in BUDGET_ARMS:
        macro, full, rates = template_macro(table, arm, templates)
        lo, hi = bootstrap_ci(rates)
        report["budgets"][label] = {
            "arm": arm,
            "macro_success": macro,
            "templates_scored": len(rates),
            "templates_full_3_instances": full,
            "ci95": [lo, hi],
        }
        lines.append(
            "%-6s  %5.1f    %3d/%3d          [%.1f, %.1f]"
            % (label, 100 * macro, full, len(templates), 100 * lo, 100 * hi)
        )
    # Avg B>0 (macro over budgets 1/2/4/8 of the per-template mean)
    bgt0 = [report["budgets"][l]["macro_success"] for l in ("B1", "B2", "B4", "B8")]
    report["avg_b_gt_0"] = sum(bgt0) / len(bgt0)
    lines.append("Avg B>0  %5.1f" % (100 * report["avg_b_gt_0"]))

    text = "\n".join(lines)
    print(text)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(
            text + "\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
