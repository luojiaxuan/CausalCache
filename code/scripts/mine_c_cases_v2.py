"""Mine structural C-cases in GUI-Odyssey using the annotation's own screen descriptions.

# note (luojiaxuan): v1 测错了字段 —— 它把 `description`(**截图内容描述**)当成动作,
# 于是从 "This is a screenshot of a flight booking app..." 里正则出 "2024" 这种年份,
# 42 个候选基本是噪声。
#
# 但那个错误暴露了一个更好的路子:`description` 恰恰是每一帧屏幕的**文字描述**,
# 所以判定"值 d 在第 j 帧是否可见"**不需要 VLM**,在描述文本里搜即可。于是用户列的
# 七个条件里有五个可以纯文本判定:
#   (1) d 曾在较早截图中可见      -> d 出现在 description[j],j < decision_step
#   (2) d 不出现在 goal text      -> d 不在 goal 里
#   (3) 当前截图中不再可见        -> d **不**出现在 description[decision_step]
#   (4) Recent-K 中不含该证据     -> decision_step - j > K,且 d 不在最近 K 帧的描述里
#   (5) 当前动作依赖 d            -> d 出现在 low_level_instruction[decision_step]
# 剩下的 (6) 唯一定位、(7) 不回源 app 无法恢复,仍需人工看图复核。
#
# 依旧**完全不使用模型分数** —— 用 S0-R0 的大小挑样本会让结论由构造成立。
"""

import argparse
import collections
import json
import re
from pathlib import Path

# 只接受高信息量的字面值,排除年份 / 小整数这类到处都是的噪声
CODE = re.compile(r"\b\d{4,8}\b")
PHONE = re.compile(r"\+?\d{10,13}\b")
FILE = re.compile(r"\b[A-Za-z0-9_\-]{3,40}\.(?:md|txt|png|jpg|pdf|docx?)\b")
QUOTED = re.compile(r"['\"]([^'\"]{4,50})['\"]")
YEARISH = re.compile(r"^(19|20)\d{2}$")
STOP = {"screenshot", "displaying", "showing", "interface", "application", "mobile"}


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


def harvest(action_text: str) -> set[str]:
    out: set[str] = set()
    for m in PHONE.finditer(action_text):
        out.add(m.group(0))
    for m in FILE.finditer(action_text):
        out.add(m.group(0))
    for m in CODE.finditer(action_text):
        v = m.group(0)
        if not YEARISH.match(v):          # 年份到处都是,排除
            out.add(v)
    for m in QUOTED.finditer(action_text):
        v = m.group(1).strip()
        if len(v) >= 4 and norm(v) and not any(s in v.lower() for s in STOP):
            out.add(v)
    return {v for v in out if len(norm(v)) >= 4}


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
    cands = []
    ages = collections.Counter()

    for item in selected:
        sid = item["source_id"]
        path = args.annotations / f"{sid}.json"
        if not path.exists():
            stats["annotation_missing"] += 1
            continue
        ann = json.loads(path.read_text(encoding="utf-8"))
        steps = ann.get("steps") or []
        if len(steps) < 8:
            continue
        goal_norm = norm(json.dumps(ann.get("task_info") or ann.get("instruction") or "",
                                    ensure_ascii=False))
        acts = [str(s.get("low_level_instruction") or "") for s in steps]
        screens = [norm(s.get("description") or "") for s in steps]
        apps = [str((s.get("info") or "")) for s in steps]

        for d in range(args.recent_k + 1, len(steps)):
            stats["decision_points"] += 1
            values = harvest(acts[d])
            if not values:
                continue
            stats["action_carries_value"] += 1
            for value in values:
                v = norm(value)
                if v in goal_norm:                       # (2)
                    stats["value_in_goal"] += 1
                    continue
                if v in screens[d]:                      # (3) 当前帧还看得见
                    stats["visible_in_current_frame"] += 1
                    continue
                recent = range(max(0, d - args.recent_k), d)
                if any(v in screens[j] for j in recent): # (4) Recent-K 里有
                    stats["inside_recent_k"] += 1
                    continue
                source = [j for j in range(0, d - args.recent_k) if v in screens[j]]
                if not source:                           # (1) 早帧从未可见
                    stats["never_visible_earlier"] += 1
                    continue
                j = source[-1]
                age = d - j
                stats["structural_candidate"] += 1
                ages[min(age, 40)] += 1
                cands.append({
                    "episode": sid, "decision_step": d, "evidence_step": j, "age": age,
                    "value": value[:40],
                    "action": acts[d][:130],
                    "evidence_frame_excerpt": (steps[j].get("description") or "")[:130],
                    "cross_app": (apps[j] != apps[d]) if apps[j] and apps[d] else None,
                    "n_source_frames": len(source),
                    "manual_needed": ["uniquely_locatable", "not_recoverable_in_place"],
                })

    cross = sum(1 for c in cands if c["cross_app"])
    uniq_ep = len({c["episode"] for c in cands})
    payload = {
        "schema_version": "causalcache.c_case_structural_candidates.v2",
        "recent_k": args.recent_k,
        "trajectories_scanned": len(selected),
        "counters": dict(stats),
        "candidates": len(cands),
        "candidate_rate_per_decision": round(
            stats["structural_candidate"] / max(stats["decision_points"], 1), 5),
        "episodes_with_candidates": uniq_ep,
        "cross_app_candidates": cross,
        "age_histogram": dict(sorted(ages.items())),
        "operationalization": {
            "(1) visible earlier": "value appears in description[j], j < decision_step - K",
            "(2) not in goal": "value absent from task_info",
            "(3) gone from current frame": "value absent from description[decision_step]",
            "(4) outside Recent-K": "value absent from the last K frame descriptions",
            "(5) action depends on it": "value appears in low_level_instruction[decision_step]",
            "(6)(7)": "MANUAL — uniquely locatable / not recoverable in place",
        },
        "selection_bias_guard": (
            "候选完全由文本结构判定,未使用任何模型分数。严禁按 S0-R0 大小决定样本是否进入 "
            "test split —— 那会让结论由构造成立。"
        ),
        "examples": cands[:40],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "trajectories_scanned", "counters", "candidates",
        "candidate_rate_per_decision", "episodes_with_candidates",
        "cross_app_candidates")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
