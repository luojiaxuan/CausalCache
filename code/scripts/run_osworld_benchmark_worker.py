#!/usr/bin/env python3
"""OSWorld benchmark worker:清单分片 → 逐任务 run_osworld_episode(断点续跑)。

# note (luojiaxuan): 每个 worker 持有一台 docker-provider VM,顺序跑自己分片
# (roster[shard::count],与 MobileWorld 同一固定归属口径);result.json 已存在的
# 任务自动跳过,因此崩溃后重启 worker 即续跑。episode 失败(env 崩/超时)记入
# failure.json 后重建 VM 继续下一任务;整片跑完打印 WORKER_COMPLETE 行。
# 记忆语义:B0 → --memory-arm summary(请求不带图);recent/selected →
# --memory-arm full(全池截图随请求,server 端按 budget/selector 决定 shown)。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from causalcache.osworld import (
    HTTPOSWorldPolicy,
    create_osworld_environment,
    load_osworld_inventory,
    load_osworld_task,
    osworld_git_revision,
    run_osworld_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--meta-path", default="evaluation_examples/test_small.json")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-endpoint", required=True)
    parser.add_argument("--memory-arm", choices=("summary", "full"), required=True)
    parser.add_argument("--memory-budget", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--policy-timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screen = (args.screen_width, args.screen_height)
    inventory = load_osworld_inventory(args.osworld_root, meta_path=args.meta_path)
    shard = list(inventory)[args.shard_index::args.shard_count]
    provenance = {
        "osworld_revision": osworld_git_revision(args.osworld_root),
        "meta_path": str(args.meta_path),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "policy_endpoint": args.policy_endpoint,
        "memory_arm": args.memory_arm,
        "memory_budget": args.memory_budget,
        "prompt_protocol": "desktop_official_multiturn_gapfold",
    }
    policy = HTTPOSWorldPolicy(
        args.policy_endpoint, timeout_seconds=args.policy_timeout_seconds)

    def fresh_env():
        return create_osworld_environment(
            args.osworld_root,
            provider_name="docker",
            path_to_vm=None,
            cache_dir=args.cache_dir,
            screen_size=screen,
            headless=True,
        )

    environment = fresh_env()
    completed = failed = skipped = 0
    try:
        for domain, task_id in shard:
            task = load_osworld_task(
                args.osworld_root, domain=domain, task_id=task_id)
            result_path = (args.output_root / domain / task_id / "result.json")
            if result_path.exists():
                skipped += 1
                continue
            started = time.time()
            try:
                result = run_osworld_episode(
                    environment=environment,
                    policy=policy,
                    task=task,
                    output_root=args.output_root,
                    memory_arm=args.memory_arm,
                    memory_budget=args.memory_budget,
                    max_steps=args.max_steps,
                    pause_seconds=args.pause_seconds,
                    screen_size=screen,
                    provenance=provenance,
                )
                completed += 1
                print(json.dumps({
                    "event": "TASK_DONE",
                    "domain": domain,
                    "task_id": task_id,
                    "score": result.get("score"),
                    "steps": result.get("completed_steps"),
                    "seconds": round(time.time() - started, 1),
                }), flush=True)
            except BaseException as error:  # noqa: BLE001
                failed += 1
                print(json.dumps({
                    "event": "TASK_FAILED",
                    "domain": domain,
                    "task_id": task_id,
                    "error_type": error.__class__.__name__,
                    "error": str(error)[:300],
                }), flush=True)
                if isinstance(error, KeyboardInterrupt):
                    raise
                # VM 可能已坏:重建再继续。
                try:
                    environment.close()
                except Exception:  # noqa: BLE001
                    pass
                environment = fresh_env()
    finally:
        try:
            environment.close()
        except Exception:  # noqa: BLE001
            pass
    print(json.dumps({
        "event": "WORKER_COMPLETE",
        "shard_index": args.shard_index,
        "assigned": len(shard),
        "completed": completed,
        "failed": failed,
        "skipped_existing": skipped,
    }), flush=True)


if __name__ == "__main__":
    main()
