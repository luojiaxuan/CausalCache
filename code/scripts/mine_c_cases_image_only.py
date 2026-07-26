"""Mine **image-only** C-cases: v3 的四条拒绝之上再加"证据值已在动作文本里"。

# note (luojiaxuan): v3 漏掉了一条与冻结 renderer 直接相关的拒绝。
# ``gui_owl_sparse_multiturn`` 把被折叠掉的步骤以**文本**保留:
#
#     Intervening actions from Step7 through Step12:
#     Step7: Tap the search bar and type "a travel guide to Disney World in Orlando"
#
# 也就是说,对任何决策点、任何预算 B,**每一个已完成步骤做了什么,模型都以文本形式
# 看得到** —— 落在 Recent-B 里的步以 assistant 完整响应出现,Recent-B 之外的步以
# ``Intervening actions`` 块出现,两者都含该步的 action 文本(渲染器的冻结不变量 2:
# 1..current_step-1 每一步恰好出现一次)。所以只要证据值出现在**任何**已完成步骤的
# action 文本里,把那张截图以高保真恢复回来就是冗余的:模型不需要看图也知道这个值。
#
# 这一条不是吹毛求疵,它决定了实测量的含义。v3 口径下测到的 selection 效应是
# "图片相对于**已有文本**的增量",而方法宣称要测的是"旧证据相对于**没有**的增量"。
# 两者只有在证据确实只存在于像素里时才重合。加上第五条之后剩下的那部分,才是
# image-only C-side 的真实密度。
#
# 判据(前四条与 v3 逐字一致,顺序不变,故 ``candidate_decision_points`` 可与 v3
# 直接对账;第五条追加在最后,被它拒掉的数量就是 v3 口径的净虚高):
#   (1) reject_value_in_goal        值在指令里
#   (2) reject_visible_now          当前帧仍可见
#   (3) reject_inside_recent_k      在 Recent-B 窗口内可见
#   (4) reject_never_visible        更早从未可见
#   (5) reject_value_in_action_text 值出现在任一已完成步骤的 action 文本里  <-- 新增
#
# 归一化与取值抽取直接复用 v3 的 ``norm`` / ``harvest``,不另写一套 —— 两份口径一旦
# 分叉,"加了一条拒绝"的差分就同时混进了抽取器差异,数就没法归因了。
#
# 结构性推论(先写下来,免得看到数再解释):同一个 (轨迹, 证据帧, 值) 事件在 v3 里
# 平均覆盖 2.87 个决策点,但第五条会把**除第一次使用外**的全部后续使用拒掉 ——
# 第一次使用本身就把这个值写进了动作文本,之后它一直是文本可得的。所以
# image-only 决策点数的上界就是 v3 的事件数,而不是 v3 的候选决策点数。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from scripts.mine_c_cases_v3 import harvest, norm


def _event_record(
    *, sid: str, evidence_step: int, value: str, decision: int, acts, screens, apps, source
) -> dict:
    return {
        "episode": sid,
        "evidence_step": evidence_step,
        "value": value[:48],
        "first_use_step": decision,
        "last_use_step": decision,
        "uses": 1,
        "age_at_first_use": decision - evidence_step,
        "action": acts[decision][:120],
        "evidence_frame": screens[evidence_step][:120],
        # note (luojiaxuan): ``info`` 在 GUI-Odyssey 里是**动作参数**(CLICK 时是点击
        # 坐标),不是 app 名。沿用 v3 的字段是为了对账,但这个 cross_app 基本等价于
        # "两步的动作参数不同",不能当作跨 app 证据读 —— 见报告里的口径质疑。
        "cross_app": (apps[evidence_step] != apps[decision])
        if apps[evidence_step] and apps[decision]
        else None,
        "source_frames": len(source),
    }


def mine(selected: list[dict], annotations: Path, recent_k: int) -> dict:
    """Scan every selected trajectory at one budget and split v3 candidates in two."""
    stats: collections.Counter = collections.Counter()
    image_only: dict[tuple, dict] = {}
    text_available: dict[tuple, dict] = {}
    # note (luojiaxuan): v3 的事件是 (轨迹, 证据帧, 值) 去重后的键。同一个键可能
    # **同时**落进两个桶(首次使用是 image-only,后续使用变成文本可得),所以 v3 事件数
    # 必须用键集合数,不能用两桶长度相加 —— 相加会把这种事件数成两个,与 v3 对不上账。
    v3_keys: set[tuple] = set()
    scanned = 0

    for item in selected:
        sid = item["source_id"]
        path = annotations / f"{sid}.json"
        if not path.exists():
            stats["annotation_missing"] += 1
            continue
        ann = json.loads(path.read_text(encoding="utf-8"))
        steps = ann.get("steps") or []
        if len(steps) < 8:
            stats["too_short"] += 1
            continue
        scanned += 1
        goal_norm = norm(
            json.dumps(ann.get("task_info") or ann.get("instruction") or "", ensure_ascii=False)
        )
        acts = [str(s.get("low_level_instruction") or "") for s in steps]
        screens_raw = [str(s.get("description") or "") for s in steps]
        screens = [norm(text) for text in screens_raw]
        apps = [str(s.get("info") or "") for s in steps]
        # note (luojiaxuan): 逐步归一化,**不拼接**。norm 会把空白与标点全删掉,
        # 拼接两步的文本会凭空造出跨步边界的子串(``...abc`` + ``def...`` -> ``abcdef``),
        # 那是纯假阳性。逐步 membership 检查代价可忽略。
        acts_norm = [norm(text) for text in acts]

        for d in range(recent_k + 1, len(steps)):
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
                if any(v in screens[j] for j in range(max(0, d - recent_k), d)):
                    stats["reject_inside_recent_k"] += 1
                    continue
                source = [j for j in range(0, d - recent_k) if v in screens[j]]
                if not source:
                    stats["reject_never_visible"] += 1
                    continue
                stats["candidate_decision_points"] += 1
                j = source[-1]
                key = (sid, j, v[:32])
                v3_keys.add(key)
                # (5) 第五条:值是否已经在**任一已完成步骤**的 action 文本里。
                # 已完成步骤 = 0..d-1(不含 d 自己,d 的动作正是要预测的目标)。
                # 覆盖 Recent-B 内外的全部步:renderer 对两者都会输出该步的动作文本。
                in_text = [j2 for j2 in range(0, d) if v in acts_norm[j2]]
                bucket = text_available if in_text else image_only
                if in_text:
                    stats["reject_value_in_action_text"] += 1
                else:
                    stats["image_only_decision_points"] += 1
                event = bucket.get(key)
                if event is None:
                    event = _event_record(
                        sid=sid,
                        evidence_step=j,
                        value=value,
                        decision=d,
                        acts=acts,
                        screens=screens_raw,
                        apps=apps,
                        source=source,
                    )
                    if in_text:
                        event["action_text_steps"] = in_text[:5]
                    bucket[key] = event
                else:
                    event["last_use_step"] = d
                    event["uses"] += 1

    v3_events = len(v3_keys)
    dps = stats["decision_points"]
    v3_dps = stats["candidate_decision_points"]
    io_dps = stats["image_only_decision_points"]
    killed = stats["reject_value_in_action_text"]
    io_events = sorted(image_only.values(), key=lambda e: (e["episode"], e["evidence_step"]))
    ta_events = sorted(text_available.values(), key=lambda e: (e["episode"], e["evidence_step"]))
    return {
        "recent_k": recent_k,
        "trajectories_scanned": scanned,
        "counters": dict(stats),
        "decision_points": dps,
        # v3 口径(前四条)
        "v3_candidate_decision_points": v3_dps,
        "v3_candidate_share": round(v3_dps / max(dps, 1), 5),
        "v3_events": v3_events,
        # image-only 口径(五条)
        "image_only_decision_points": io_dps,
        "image_only_share": round(io_dps / max(dps, 1), 5),
        "image_only_events": len(io_events),
        "image_only_episodes": len({e["episode"] for e in io_events}),
        "image_only_episode_coverage": round(
            len({e["episode"] for e in io_events}) / max(scanned, 1), 5
        ),
        # 第五条的杀伤力
        "rejected_by_action_text_decision_points": killed,
        "rejected_by_action_text_share_of_v3": round(killed / max(v3_dps, 1), 5),
        "text_available_events": len(ta_events),
        "image_only_age_at_first_use_histogram": dict(
            sorted(collections.Counter(min(e["age_at_first_use"], 40) for e in io_events).items())
        ),
        "image_only_single_source_frame_events": sum(
            1 for e in io_events if e["source_frames"] == 1
        ),
        "density_shrink_factor": round(v3_dps / max(io_dps, 1), 2) if io_dps else None,
        # 下游富集采样要的完整清单
        "all_events": io_events,
        "all_text_available_events": ta_events,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--recent-k",
        default="4",
        help="逗号分隔的预算列表;每个预算独立扫描一遍(决策点集合本身随 B 变化)",
    )
    ap.add_argument("--max-trajectories", type=int, default=0)
    args = ap.parse_args()

    selected = json.loads(args.selection.read_text(encoding="utf-8"))["selected"]
    if args.max_trajectories:
        selected = selected[: args.max_trajectories]

    budgets = [int(part) for part in str(args.recent_k).split(",") if part.strip()]
    per_budget = {str(k): mine(selected, args.annotations, k) for k in budgets}

    payload = {
        "schema_version": "causalcache.c_case_events.image_only.v1",
        "selection": str(args.selection),
        "annotations": str(args.annotations),
        "budgets": budgets,
        "new_rejection": (
            "reject_value_in_action_text:证据值(norm 后)出现在任一已完成步骤的 "
            "low_level_instruction 里。冻结 renderer 会把每个已完成步骤的动作文本都放进 "
            "prompt(Recent-B 内是 assistant 完整响应,Recent-B 外是 Intervening actions "
            "块),所以这类证据不需要恢复截图。"
        ),
        "selection_bias_guard": (
            "候选完全由文本结构判定,未使用任何模型分数。"
            "严禁按 S0-R0 大小决定样本是否进入 test split。"
        ),
        "what_still_needs_manual_review": [
            "(6) 证据在源帧中是否唯一定位",
            "(7) 是否真的不回源 app 就无法恢复",
            "第五条是**精确子串**判据:同义改写('搜了那家餐厅' vs 具体店名)、部分匹配、"
            "以及 description 与 low_level_instruction 用词不同,都会让它偏松,"
            "即真实 image-only 密度还要更低",
        ],
        "per_budget": per_budget,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    header = (
        f"{'B':>3} {'decision_pts':>12} {'v3_cand':>8} {'v3_share':>9} "
        f"{'img_only':>9} {'io_share':>9} {'killed_by_5':>12} {'kill_rate':>10} "
        f"{'v3_ev':>6} {'io_ev':>6}"
    )
    print(header)
    for k in budgets:
        b = per_budget[str(k)]
        print(
            f"{k:>3} {b['decision_points']:>12} {b['v3_candidate_decision_points']:>8} "
            f"{100 * b['v3_candidate_share']:>8.3f}% {b['image_only_decision_points']:>9} "
            f"{100 * b['image_only_share']:>8.3f}% "
            f"{b['rejected_by_action_text_decision_points']:>12} "
            f"{100 * b['rejected_by_action_text_share_of_v3']:>9.2f}% "
            f"{b['v3_events']:>6} {b['image_only_events']:>6}"
        )


if __name__ == "__main__":
    main()
