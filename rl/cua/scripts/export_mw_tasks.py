#!/usr/bin/env python3
# note (luojiaxuan): 按冻结切分(fixtures/mw_split_v1.json)导出 MobileWorld
# 任务 parquet(CUA-Lite rollout 输入)。在 cua-lite checkout 内用 uv run 执行:
#   uv run python <此文件> --split-json <fixtures>/mw_split_v1.json --out-dir <dir>
# 生成 train.parquet(78)/ heldout.parquet(39)。schema 同
# lite.train.export.export_tasks:problem + metadata{env_key, split}。
# heldout 全程不碰(过拟合对照);strict_alt 备选切分用 --variant strict。
import argparse
import json

from lite.utils.parquet import write_records_to_parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", choices=["primary", "strict"], default="primary")
    args = ap.parse_args()

    spec = json.load(open(args.split_json))
    if args.variant == "strict":
        train, held = spec["strict_alt"]["train"], spec["strict_alt"]["heldout"]
    else:
        train, held = spec["train"], spec["heldout"]

    import lite.gym as gym

    registered = set(gym.registry.task_ids("mobileworld")["eval"])
    missing = [t for t in train + held if t not in registered]
    if missing:
        raise SystemExit(f"切分中的任务未注册(上游 pin 变了?): {missing}")

    for name, tasks in (("train", train), ("heldout", held)):
        rows = [
            {
                "problem": t,
                "metadata": {"env_key": f"mobileworld@{t}", "split": name},
            }
            for t in sorted(tasks)
        ]
        out = f"{args.out_dir}/{name}.parquet"
        write_records_to_parquet(rows, out)
        print(f"{out}: {len(rows)} tasks")


if __name__ == "__main__":
    main()
