"""Mine structural C-cases in GUI-Odyssey: past-only evidence the current action needs.

# note (luojiaxuan): C 类的定义必须**完全不依赖模型分数**。如果先用冻结模型挑出
# "S0 明显赢 R0" 的组,再在同分布上报告 sparse 优势,那结论就是构造出来的 ——
# 先用模型找到"稀疏会赢"的数据,再证明稀疏会赢。所以这里只用文本结构判定,
# 模型分数留到 Gate 3 的 oracle headroom,且只用于分层与查错,不用于定义 test split。
#
# 可判定的结构信号:**目标动作里键入了一个 goal 文本中不存在的字面值**。
# 那个值不可能凭空而来,只能是 agent 从某一帧屏幕上读到的。于是:
#   - 它是候选证据 d;
#   - 首次可得的步 = 最早提到它的那一步(或它出现在某个动作描述里的位置);
#   - 若 first_seen 与 decision_step 的距离 > K,则决策时它已滑出 Recent-K。
#
# 这只是**候选**。条件 (1)(3)(7)(证据确实在早帧可见、当前帧不可见、不可从当前状态恢复)
# 需要看截图,必须人工复核。脚本输出候选与全部可判定的字段,precision 由人工估计。
"""

import argparse
import collections
import json
import re
from pathlib import Path

# 键入类动作里被输入的字面值
TYPE_PAT = re.compile(r'(?:type|enter|input|fill)[^"\']*["\']([^"\']{3,60})["\']', re.I)
# 形如验证码 / 号码 / 地址 / 文件名 的高信息量值
VALUE_PAT = re.compile(r"\b(\d{4,8}|\+?\d{10,13}|[A-Za-z0-9_]+\.(?:md|txt|png|jpg))\b")


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).lower()


def extract_typed_values(action: str) -> list[str]:
    vals = [m.group(1).strip() for m in TYPE_PAT.finditer(action)]
    vals += [m.group(1) for m in VALUE_PAT.finditer(action)]
    return [v for v in {v.strip() for v in vals} if len(v.strip()) >= 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--recent-k", type=int, default=4)
    ap.add_argument("--max-trajectories", type=int, default=0)
    args = ap.parse_args()

    selected = json.loads(args.selection.read_text(encoding="utf-8"))["selected"]
    if args.max_trajectories:
        selected = selected[: args.max_trajectories]

    stats = collections.Counter()
    candidates = []
    age_hist = collections.Counter()

    for item in selected:
        sid = item["source_id"]
        path = args.annotations / f"{sid}.json"
        if not path.exists():
            stats["annotation_missing"] += 1
            continue
        ann = json.loads(path.read_text(encoding="utf-8"))
        steps = ann.get("steps") or []
        goal = str(ann.get("task_info") or ann.get("instruction") or ann.get("goal") or "")
        goal_norm = normalize(goal)
        actions = [str(s.get("description") or s.get("low_level_instruction") or "")
                   for s in steps]
        apps = [str(s.get("info") or "") for s in steps]

        for d, action in enumerate(actions):
            stats["decision_points"] += 1
            values = extract_typed_values(action)
            if not values:
                continue
            stats["typed_value_steps"] += 1
            for value in values:
                vnorm = normalize(value)
                if len(vnorm) < 3:
                    continue
                # 条件 (2):不出现在 goal 里 —— 否则信息随指令给全,与记忆无关
                if vnorm and vnorm in goal_norm:
                    stats["value_in_goal"] += 1
                    continue
                # 找它最早在哪一步被提及(候选证据来源)
                first = None
                for j in range(d):
                    if vnorm and vnorm in normalize(actions[j]):
                        first = j
                        break
                if first is None:
                    stats["no_earlier_mention"] += 1
                    continue
                age = d - first
                # 条件 (4):决策时已滑出 Recent-K
                if age <= args.recent_k:
                    stats["inside_recent_k"] += 1
                    continue
                stats["structural_candidate"] += 1
                age_hist[min(age, 30)] += 1
                candidates.append({
                    "episode": sid,
                    "decision_step": d,
                    "first_seen_step": first,
                    "age": age,
                    "value": value[:40],
                    "action": action[:120],
                    "source_action": actions[first][:120],
                    "cross_app": apps[first] != apps[d] if apps[first] and apps[d] else None,
                    "goal_excerpt": goal[:100],
                    # note (luojiaxuan): 以下三条必须人工看截图才能定,脚本给不了。
                    "manual_needed": [
                        "evidence_visible_in_source_frame",
                        "evidence_absent_from_current_frame",
                        "not_recoverable_without_returning_to_source_app",
                    ],
                })

    cross = sum(1 for c in candidates if c["cross_app"])
    payload = {
        "schema_version": "causalcache.c_case_structural_candidates.v1",
        "recent_k": args.recent_k,
        "trajectories_scanned": len(selected),
        "counters": dict(stats),
        "candidates": len(candidates),
        "candidate_rate_per_decision": (
            round(stats["structural_candidate"] / max(stats["decision_points"], 1), 5)
        ),
        "cross_app_candidates": cross,
        "age_histogram": dict(sorted(age_hist.items())),
        "selection_bias_guard": (
            "候选完全由文本结构判定,**未使用任何模型分数**。"
            "严禁按 S0-R0 的大小决定样本是否进入 test split —— 那会让结论由构造成立。"
        ),
        "manual_review_required": (
            "条件(1)证据确在早帧可见、(3)当前帧不可见、(7)不回源 app 无法恢复,"
            "都需要看截图人工复核;本文件只给候选与可判定字段,precision 由人工抽样估计。"
        ),
        "examples": candidates[:40],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "trajectories_scanned", "counters", "candidates",
        "candidate_rate_per_decision", "cross_app_candidates",
    )}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
