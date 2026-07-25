"""Mine structural C-cases in GUI-Odyssey, counted as evidence-dependency EVENTS.

# note (luojiaxuan): v2 数对了字段但**计数被重复膨胀**:同一个 (轨迹, 证据帧, 值) 会在
# 连续多个决策点各计一次(agent 重复键入同一串),实测 40 条候选去重后只有 5 个事件,
# 平均每事件覆盖 8 个决策点、最高 18 个。所以 prevalence 必须按**事件**报,
# 决策点覆盖数另报 —— 前者回答"这种依赖有多常见",后者回答"能造出多少训练样本"。
#
# 判定依旧**完全不使用模型分数**。用 S0-R0 的大小挑样本会让结论由构造成立。
#
# 可自动判定(用 annotation 自带的每帧 description,不需要 VLM):
#   (1) 早帧可见  (2) 不在 goal  (3) 当前帧不可见  (4) 不在 Recent-K  (5) 动作依赖它
# 仍需人工看图:
#   (6) 唯一定位  (7) 不回源 app 无法原地恢复
"""

import argparse
import collections
import json
import re
from pathlib import Path

CODE = re.compile(r"\b\d{4,8}\b")
PHONE = re.compile(r"\+?\d{10,13}\b")
FILE = re.compile(r"\b[A-Za-z0-9_\-]{3,40}\.(?:md|txt|png|jpg|pdf|docx?)\b")
QUOTED = re.compile(r"['\"]([^'\"]{4,60})['\"]")
YEARISH = re.compile(r"^(19|20)\d{2}$")
STOP = {"screenshot", "displaying", "showing", "interface", "application", "mobile",
        "search or type web addres", "search"}


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


def harvest(action_text: str) -> set[str]:
    out: set[str] = set()
    for m in PHONE.finditer(action_text):
        out.add(m.group(0))
    for m in FILE.finditer(action_text):
        out.add(m.group(0))
    for m in CODE.finditer(action_text):
        if not YEARISH.match(m.group(0)):
            out.add(m.group(0))
    for m in QUOTED.finditer(action_text):
        v = m.group(1).strip()
        if len(v) >= 4 and v.lower() not in STOP and not any(s in v.lower() for s in STOP):
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
    events: dict[tuple, dict] = {}
    scanned = 0

    for item in selected:
        sid = item["source_id"]
        path = args.annotations / f"{sid}.json"
        if not path.exists():
            stats["annotation_missing"] += 1
            continue
        ann = json.loads(path.read_text(encoding="utf-8"))
        steps = ann.get("steps") or []
        if len(steps) < 8:
            stats["too_short"] += 1
            continue
        scanned += 1
        goal_norm = norm(json.dumps(ann.get("task_info") or ann.get("instruction") or "",
                                    ensure_ascii=False))
        acts = [str(s.get("low_level_instruction") or "") for s in steps]
        screens = [norm(s.get("description") or "") for s in steps]
        apps = [str(s.get("info") or "") for s in steps]

        for d in range(args.recent_k + 1, len(steps)):
            stats["decision_points"] += 1
            values = harvest(acts[d])
            if not values:
                continue
            for value in values:
                v = norm(value)
                if v in goal_norm:
                    stats["reject_value_in_goal"] += 1
                    continue
                if v in screens[d]:
                    stats["reject_visible_now"] += 1
                    continue
                if any(v in screens[j] for j in range(max(0, d - args.recent_k), d)):
                    stats["reject_inside_recent_k"] += 1
                    continue
                source = [j for j in range(0, d - args.recent_k) if v in screens[j]]
                if not source:
                    stats["reject_never_visible"] += 1
                    continue
                j = source[-1]
                # note (luojiaxuan): 事件 = (轨迹, 证据帧, 值),同一事件在多个决策点
                # 复用时只记一次,但把覆盖到的决策点累计起来。
                key = (sid, j, v[:32])
                stats["candidate_decision_points"] += 1
                ev = events.get(key)
                if ev is None:
                    events[key] = {
                        "episode": sid, "evidence_step": j, "value": value[:48],
                        "first_use_step": d, "last_use_step": d, "uses": 1,
                        "age_at_first_use": d - j,
                        "action": acts[d][:120],
                        "evidence_frame": (steps[j].get("description") or "")[:120],
                        "cross_app": (apps[j] != apps[d]) if apps[j] and apps[d] else None,
                        "source_frames": len(source),
                    }
                else:
                    ev["last_use_step"] = d
                    ev["uses"] += 1

    evs = list(events.values())
    ages = collections.Counter(min(e["age_at_first_use"], 40) for e in evs)
    uses = collections.Counter(min(e["uses"], 10) for e in evs)
    cross = sum(1 for e in evs if e["cross_app"])
    eps = len({e["episode"] for e in evs})
    unique_src = sum(1 for e in evs if e["source_frames"] == 1)

    payload = {
        "schema_version": "causalcache.c_case_events.v3",
        "recent_k": args.recent_k,
        "trajectories_scanned": scanned,
        "counters": dict(stats),
        "events": len(evs),
        "event_rate_per_trajectory": round(len(evs) / max(scanned, 1), 4),
        "episodes_with_events": eps,
        "episode_coverage": round(eps / max(scanned, 1), 4),
        "candidate_decision_points": stats["candidate_decision_points"],
        "decision_points_per_event": round(
            stats["candidate_decision_points"] / max(len(evs), 1), 2),
        "cross_app_events": cross,
        "single_source_frame_events": unique_src,
        "age_at_first_use_histogram": dict(sorted(ages.items())),
        "uses_per_event_histogram": dict(sorted(uses.items())),
        "what_still_needs_manual_review": [
            "(6) 证据在源帧中是否唯一定位",
            "(7) 是否真的不回源 app 就无法恢复",
            "文本匹配的假阳性:值在早帧描述里出现可能是巧合",
        ],
        "selection_bias_guard": (
            "候选完全由文本结构判定,未使用任何模型分数。"
            "严禁按 S0-R0 大小决定样本是否进入 test split。"
        ),
        "examples": sorted(evs, key=lambda e: -e["uses"])[:30],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "trajectories_scanned", "events", "event_rate_per_trajectory",
        "episodes_with_events", "episode_coverage", "candidate_decision_points",
        "decision_points_per_event", "cross_app_events",
        "single_source_frame_events")}, ensure_ascii=False, indent=1))
    print("counters:", json.dumps(dict(stats), ensure_ascii=False))


if __name__ == "__main__":
    main()
