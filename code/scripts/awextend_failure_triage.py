"""Triage every AW-Extend failure before attributing anything to memory.

# note (luojiaxuan): 在把任何一局失败算到"记忆窗口不够"头上之前必须先分类,否则会把
# 与 CausalCache 无关的失败当成支持证据。四类:
#   A 所需证据从未出现在任何一帧截图里          -> 历史缓存救不了
#   B 证据就在当前屏幕上却被忽略                -> 感知/推理问题
#   C 证据曾出现过,决策时已滑出 Recent-4       -> **唯一核心可解类型**
#   D 操作/解析/过早终止失败                    -> 通常也救不了
#
# 这一版做的是**自动初筛**,给出可判定的信号(终止方式、步数、是否出现过 terminate、
# 动作类型分布、是否重复同一动作),并把每局的动作序列摘要打出来供人工定类。
# 自动信号不足以区分 A/B/C —— 那需要看截图内容,所以最终定类必须人工过一遍。
"""

import argparse
import collections
import json
from pathlib import Path


def summarize(d: dict) -> dict:
    steps = d.get("steps") or []
    actions = []
    for s in steps:
        raw = (s.get("raw_output") or "")
        line = raw.split("<tool_call>")[0].strip().replace("\n", " ")
        canon = (s.get("canonical_action") or {}).get("action")
        actions.append((canon, line[:70]))
    kinds = collections.Counter(a for a, _ in actions)
    repeats = 0
    for i in range(1, len(actions)):
        if actions[i][1] and actions[i][1] == actions[i - 1][1]:
            repeats += 1
    return {
        "task_type": d["task_type"],
        "task_index": d["task_index"],
        "success": bool(d.get("official_terminal_success")),
        "steps": int(d.get("model_step_count") or 0),
        "termination": d.get("termination_reason"),
        "unknown": d.get("unknown_action_steps"),
        "score_before": d.get("score_before"),
        "action_kinds": dict(kinds),
        "consecutive_repeats": repeats,
        "first_actions": [line for _, line in actions[:4]],
        "last_actions": [line for _, line in actions[-3:]],
        # note (luojiaxuan): 自动线索,**不是**最终分类。
        # terminate 且步数远低于预算 -> 疑似 D(过早终止);
        # 大量重复动作 -> 疑似 B/D(卡在同一屏);
        # 步数接近预算 -> 疑似 C 或任务本身太长。
        "auto_hint": (
            "D?premature_terminate"
            if d.get("termination_reason") == "policy_terminated"
            and int(d.get("model_step_count") or 0) < 150
            else "B/D?loop" if repeats >= 5 else "needs_manual_review"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(args.results_root.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("arm") != "official_B4" or "task_type" not in d:
            continue
        if d.get("official_terminal_success"):
            continue
        rows.append(summarize(d))

    hints = collections.Counter(r["auto_hint"] for r in rows)
    payload = {
        "schema_version": "causalcache.awextend_failure_triage.v1",
        "failures": len(rows),
        "auto_hint_distribution": dict(hints),
        "classification_scheme": {
            "A": "所需证据从未出现在任何一帧截图中 —— 历史缓存救不了",
            "B": "证据仍在当前截图上但被忽略 —— 感知/推理问题",
            "C": "证据曾出现,决策时已超出 Recent-4 —— 唯一核心可解类型",
            "D": "操作、解析或过早终止失败 —— 通常救不了",
        },
        "warning": (
            "auto_hint 只是初筛线索,不能替代人工定类:区分 A/B/C 必须看截图内容。"
            "只有被人工定为 C 的局才允许进入 oracle restoration 验证,"
            "也只有 oracle restoration 真能修复的,才可列为 memory-solvable。"
        ),
        "episodes": rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"failures": len(rows), "auto_hints": dict(hints)}, ensure_ascii=False))
    for r in rows:
        print("  %-46s steps=%-4d %-22s %s" % (
            r["task_type"] + "/" + str(r["task_index"]), r["steps"],
            r["auto_hint"], r["first_actions"][0][:44] if r["first_actions"] else ""))


if __name__ == "__main__":
    main()
