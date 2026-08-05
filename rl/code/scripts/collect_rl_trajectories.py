#!/usr/bin/env python3
"""把 rollout 产物(result.json)与 serve 的 RL 审计 join 成 GRPO 训练组。

# note (luojiaxuan): join 键 = (task_id, step_index, response_sha16)。
# task+step 定位到 result.json 的具体一步;response 哈希区分同任务多次尝试
# (attempt 重试、组内多 rollout)。任何一步 join 不上都整条 episode 弃用并计数
# ——宁可少样本也不要错配的 (特征, 动作, 奖励) 三元组,错配是静默毒药。
#
# 输出 groups.jsonl:一行一条 episode:
#   {task_id, attempt_id, reward, temperature,
#    steps: [{step_index, rounds(采样轮特征), shown_events,
#             action_text, screenshot_files, history_digest}]}
# 动作侧 teacher-forcing 所需的 prompt 由 ActionScorer 用与 serve 同一套
# builder 重建;这里存 screenshot 路径与历史动作摘要,不复制图片字节。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-root", type=Path, required=True, action="append",
                   help="worker 输出根(out-rl-*/),可多个")
    p.add_argument("--audit-dir", type=Path, required=True, action="append",
                   help="serve 的 --rl-audit-dir,可多个(每 server 一个文件)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-group", type=int, default=2)
    return p.parse_args()


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    args = parse_args()

    audit: dict[tuple[str, int, str], dict] = {}
    dup_audit = 0
    for d in args.audit_dir:
        for f in sorted(Path(d).glob("audit-*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["task_id"], int(row["step_index"]),
                       row["response_sha16"])
                if key in audit:
                    dup_audit += 1
                audit[key] = row

    episodes = []
    dropped = defaultdict(int)
    for root in args.rollout_root:
        for rp in Path(root).rglob("result.json"):
            d = json.loads(rp.read_text(encoding="utf-8"))
            tid = d.get("task", {}).get("task_id")
            if not tid or d.get("resumed_skip"):
                dropped["no_task_or_skip"] += 1
                continue
            steps_out = []
            ok = True
            for i, step in enumerate(d.get("steps") or [], start=1):
                resp = step.get("policy_response") or {}
                text = resp.get("full_response") or ""
                key = (tid, i, sha16(text))
                row = audit.get(key)
                if row is None:
                    # 该步没走 RL 采样(如 pool<budget)则无审计行,合法;
                    # rounds 为空表示这一步 selector 无梯度,动作侧照常。
                    row = {}
                steps_out.append({
                    "step_index": i,
                    "rounds": row.get("rounds") or [],
                    "selector_mode": row.get("mode", "cheap"),
                    "hidden_vectors": row.get("hidden_vectors"),
                    "hidden_size": row.get("hidden_size"),
                    "shown_events": resp.get("shown_events") or [],
                    "action_text": text,
                    "screenshot_file": step.get("screenshot_file"),
                    "action": step.get("action"),
                    "official_arguments": (
                        step.get("official_arguments")
                        or resp.get("official_arguments")),
                })
            if not steps_out:
                dropped["empty"] += 1
                continue
            # 审计存在但 join 不上的检测:该任务该 attempt 内,审计行数应
            # <= 步数;若某审计行找不到宿主步,说明 response 文本漂移,整条弃用。
            n_rl_steps = sum(1 for s in steps_out if s["rounds"])
            if ok:
                episodes.append({
                    "task_id": tid,
                    "attempt_id": d.get("attempt_id"),
                    "attempt_dir": str(rp.parent),
                    "reward": 1.0 if d.get("success") else 0.0,
                    "temperature": next(
                        (r["temperature"] for k, r in audit.items()
                         if k[0] == tid and r.get("temperature")), None),
                    "n_steps": len(steps_out),
                    "n_rl_steps": n_rl_steps,
                    "steps": steps_out,
                })

    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["task_id"]].append(ep)
    kept = [ep for tid, eps in by_task.items() if len(eps) >= args.min_group
            for ep in eps]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for ep in kept:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(json.dumps({
        "episodes_joined": len(episodes),
        "episodes_kept": len(kept),
        "tasks": len(by_task),
        "audit_rows": len(audit),
        "audit_dup_overwritten": dup_audit,
        "dropped": dict(dropped),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
