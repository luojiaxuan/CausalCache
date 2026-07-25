"""Freeze an episode-level confirmation split inside the existing heldout set.

# note (luojiaxuan): 现有 heldout 的 176 个 episode 已经被用来评了 identity/s25/s50/s75/
# epoch1 五个状态,再拿同一批做新 pilot 的 PASS/FAIL 判定就是重复检验。所以按 **episode**
# 把 heldout 一分为二:
#   dev          —— 训练期随便看,用于监控与调试
#   confirmation —— 只在 s25/s50 最终决策时碰一次
# 两半都不参与训练(它们本来就在 heldout 里),所以不存在泄漏;区别只是"被看过几次"。
#
# 切分必须按 episode 而不是按 group,否则同一条轨迹的不同决策点会跨越两半,
# 而组间本来就不独立 —— 那样 confirmation 的 CI 会被 dev 的信息污染。
"""

import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path

SALT = "confirmation_v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--confirmation-fraction", type=float, default=0.5)
    args = ap.parse_args()

    groups_by_ep: dict[str, set[str]] = collections.defaultdict(set)
    path = args.dataset_root / "samples.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    heldout_groups = 0
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("split") != "heldout":
            continue
        groups_by_ep[row["episode"]].add(row["pair_group"])
    heldout_groups = len({g for gs in groups_by_ep.values() for g in gs})

    episodes = sorted(groups_by_ep)
    confirmation = []
    dev = []
    for ep in episodes:
        digest = hashlib.sha256(f"{SALT}:{ep}".encode()).hexdigest()
        frac = int(digest[:8], 16) / 0xFFFFFFFF
        (confirmation if frac < args.confirmation_fraction else dev).append(ep)

    payload = {
        "schema_version": "causalcache.confirmation_split.v1",
        "salt": SALT,
        "confirmation_fraction": args.confirmation_fraction,
        "source_dataset_root": str(args.dataset_root),
        "heldout_episodes": len(episodes),
        "heldout_groups": heldout_groups,
        "confirmation_episodes": len(confirmation),
        "confirmation_groups": sum(len(groups_by_ep[e]) for e in confirmation),
        "dev_episodes": len(dev),
        "dev_groups": sum(len(groups_by_ep[e]) for e in dev),
        "confirmation": confirmation,
        "dev": dev,
        "usage_contract": (
            "confirmation 只允许在 RA-aware pilot 的 s25/s50 最终决策时读取一次;"
            "训练期监控、调参、调试一律只能用 dev。两半都不参与训练。"
            "切分按 episode 而非 group,因为组间不独立。"
        ),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    overlap = set(confirmation) & set(dev)
    print(json.dumps({
        "heldout_episodes": len(episodes),
        "heldout_groups": heldout_groups,
        "confirmation_episodes": len(confirmation),
        "confirmation_groups": payload["confirmation_groups"],
        "dev_episodes": len(dev),
        "dev_groups": payload["dev_groups"],
        "overlap": len(overlap),
        "covers_all": len(confirmation) + len(dev) == len(episodes),
    }, ensure_ascii=False))
    if overlap or len(confirmation) + len(dev) != len(episodes):
        raise SystemExit("split is not a partition")


if __name__ == "__main__":
    main()
