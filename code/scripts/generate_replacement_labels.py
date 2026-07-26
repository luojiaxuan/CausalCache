"""Generate Recent-anchored replacement labels on the train split (frozen policy).

# note (luojiaxuan): 与审计脚本的分工:审计在 60 组 dev 上**全穷举**,因为它要判定
# "互补性存不存在",按单帧效用预筛会预判答案。本脚本在 2758 组 train 上生成标签,
# 那里互补性是否存在已由审计定论,唯一的问题是"高效找到组合、并知道漏掉多少" ——
# 审计实测 top-8 召回 79.8%、top-12 召回 90.4%,所以默认取 12。
# 全穷举 k=2 是 721,618 次前向(两台 16 卡 17.2 小时),top-12 降到约 182,000。
#
# 两遍式且在同一进程内完成:k=2 的候选名单依赖 k=1 的打分结果,分成两个作业会多
# 载一次模型、多扫一次语料。
#
# 本脚本**只产出量**,不产出标签类别。delta_recent / epsilon / 两折稳定性 / 最小充分 k
# 全部留给下游决策步骤,这样调阈值不必重跑 GPU。
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

from scripts.probe_oracle_headroom import (
    BEAM_WIDTH,
    BUDGETS,
    FORMATS,
    DecisionPoint,
    FrozenSetScorer,
    SetScoreCache,
    build_parser,
    group_dev_rows,
    load_config,
    load_sparse_samples,
    resolve_sample_files,
    sha256_of,
)

FROZEN_FORMAT = "official_style_sparse_multiturn"


def fold_mean(entry: dict[str, Any], fold: str) -> float:
    return entry[f"sum_{fold}"] / entry[f"n_{fold}"]


def crossfit_gain(pool: dict[tuple[int, ...], dict], anchor: dict) -> dict[str, Any] | None:
    """Select on one token fold, evaluate on the other; both directions."""
    if not pool:
        return None
    directions: dict[str, Any] = {}
    for select, evaluate in (("even", "odd"), ("odd", "even")):
        chosen = max(pool, key=lambda steps: fold_mean(pool[steps], select))
        directions[select] = {
            "steps": list(chosen),
            "gain": fold_mean(pool[chosen], evaluate) - fold_mean(anchor, evaluate),
        }
    gains = [block["gain"] for block in directions.values()]
    in_sample = max(pool, key=lambda steps: pool[steps]["mean"])
    return {
        "crossfit_gain": sum(gains) / len(gains),
        "both_folds_positive": all(gain > 0 for gain in gains),
        "directions": directions,
        "in_sample_best_steps": list(in_sample),
        "in_sample_gain": pool[in_sample]["mean"] - anchor["mean"],
        "pool_size": len(pool),
    }


def main() -> None:
    parser = build_parser()
    parser.add_argument("--budgets", default="1,2,3,4")
    parser.add_argument(
        "--k2-top-n",
        type=int,
        default=12,
        help="rank old frames by their k=1 utility and exhaust C(N,2) pairs; 0 skips k=2",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=0,
        help="subsample this many EPISODES (0 = all); clusters stay intact",
    )
    parser.add_argument("--episode-seed", type=int, default=20260727)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    if args.score_cache is None:
        raise SystemExit("--score-cache is required")
    budgets = [int(part) for part in args.budgets.split(",") if part.strip()]
    if not budgets or any(b < 1 for b in budgets):
        raise SystemExit(f"--budgets must be positive integers, got {args.budgets!r}")

    config = load_config(args.config)
    samples = load_sparse_samples(resolve_sample_files(args.dataset_root))
    payload = json.loads(args.episode_filter.read_text(encoding="utf-8"))
    if args.episode_filter_key not in payload:
        raise SystemExit(f"--episode-filter-key {args.episode_filter_key!r} absent")
    allowed = set(payload[args.episode_filter_key])

    # note (luojiaxuan): 子采样按 **episode** 而不是 group。同一 episode 内的决策点
    # 高度相关,按 group 抽会把同一条轨迹劈开,既污染下游 episode-cluster bootstrap,
    # 也让"训练/验收 episode-disjoint"这个契约变得难以保证。
    if args.episode_limit:
        ordered = sorted(allowed)
        random.Random(args.episode_seed).shuffle(ordered)
        allowed = set(ordered[: args.episode_limit])

    rows = group_dev_rows(samples, allowed)
    if not rows:
        raise SystemExit("episode filter removed every group")
    groups = sorted(rows)
    all_groups = list(groups)
    if args.shard_count > 1:
        groups = groups[args.shard_index :: args.shard_count]
        if not groups:
            raise SystemExit(f"shard {args.shard_index}/{args.shard_count} covers nothing")

    import torch

    image_root = args.image_root or args.dataset_root

    def per_shard(path: Path) -> Path:
        if args.shard_count <= 1:
            return path
        return path.with_name(
            f"{path.stem}.shard{args.shard_index:03d}"
            f"-of-{args.shard_count:03d}{path.suffix}"
        )

    cache = SetScoreCache(
        args.score_cache,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fingerprint={
            "config_sha256": sha256_of(args.config),
            "model_dir": str(args.model_dir),
            "dataset_root": str(args.dataset_root),
            "image_root": str(image_root),
            "annotations": str(args.annotations),
            "formats": list(FORMATS),
            "budgets": list(BUDGETS),
            "beam_width": BEAM_WIDTH,
            "recent_seeded_beam": True,
            "adapter": "none",
        },
    )
    scorer = FrozenSetScorer(
        args=args, config=config, image_root=image_root, cache=cache, torch=torch
    )

    labels_path = per_shard(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path = None if args.heartbeat is None else per_shard(args.heartbeat)
    started = time.time()
    annotation_cache: dict[str, list[dict]] = {}
    written = 0

    with labels_path.open("w", encoding="utf-8") as handle:
        for position, pair_group in enumerate(groups, start=1):
            point = DecisionPoint(
                rows[pair_group],
                annotations=args.annotations,
                annotation_cache=annotation_cache,
                image_root=image_root,
            )
            current = point.current_step
            record: dict[str, Any] = {
                "pair_group": pair_group,
                "episode": point.episode,
                "current_step": current,
                "n_candidates": point.n_candidates,
                "format": FROZEN_FORMAT,
                "budgets": {},
            }
            for budget in budgets:
                if budget > point.n_candidates:
                    continue
                recent = tuple(range(current - budget, current))
                olds = [step for step in range(1, current - budget)]
                anchor = scorer.score(point, recent, FROZEN_FORMAT)
                block: dict[str, Any] = {
                    "recent_steps": list(recent),
                    "anchor_mean": anchor["mean"],
                    "n_old": len(olds),
                }
                if not olds:
                    block["k1"] = None
                    record["budgets"][str(budget)] = block
                    continue

                kept = list(recent[1:])  # 丢掉最老那张 recent
                k1_pool: dict[tuple[int, ...], dict] = {}
                for old in olds:
                    steps = tuple(sorted(kept + [old]))
                    k1_pool[steps] = scorer.score(point, steps, FROZEN_FORMAT)
                block["k1"] = crossfit_gain(k1_pool, anchor)

                if args.k2_top_n and budget >= 2 and len(olds) >= 2:
                    ranked = sorted(
                        olds,
                        key=lambda old: -k1_pool[tuple(sorted(kept + [old]))]["mean"],
                    )
                    shortlist = ranked[: args.k2_top_n]
                    tail = list(recent[2:])
                    k2_pool: dict[tuple[int, ...], dict] = {}
                    for first, second in itertools.combinations(sorted(shortlist), 2):
                        steps = tuple(sorted(tail + [first, second]))
                        if len(steps) != budget:
                            continue
                        k2_pool[steps] = scorer.score(point, steps, FROZEN_FORMAT)
                    block["k2"] = crossfit_gain(k2_pool, anchor)
                    block["k2_shortlist"] = shortlist
                record["budgets"][str(budget)] = block

            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            written += 1

            state = {
                "stage": "generating_labels",
                "groups_done": position,
                "groups_total": len(groups),
                "labels_written": written,
                "forwards": scorer.forwards,
                "cache_hits": scorer.cache_hits,
                "elapsed_seconds": round(time.time() - started, 1),
            }
            print(json.dumps({"label_progress": state}), flush=True)
            if heartbeat_path is not None:
                heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                heartbeat_path.write_text(
                    json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
                )

    cache.close()
    manifest = per_shard(args.output)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "causalcache.replacement_labels.v1",
                "format": FROZEN_FORMAT,
                "budgets": budgets,
                "k2_top_n": args.k2_top_n,
                "episode_filter_key": args.episode_filter_key,
                "episode_limit": args.episode_limit,
                "episode_seed": args.episode_seed,
                "episodes_used": len(allowed),
                "groups_in_full_sample": len(all_groups),
                "groups_in_this_shard": len(groups),
                "labels_written": written,
                "forwards": scorer.forwards,
                "cache_hits": scorer.cache_hits,
                "labels_path": str(labels_path),
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest), "labels": str(labels_path)}), flush=True)


if __name__ == "__main__":
    main()
