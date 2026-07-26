"""Score explicit Recent-anchored replacement families on the frozen policy.

# note (luojiaxuan): Gate 3 的 beam 只把"丢最老那张 recent"这一支 k=1 穷举完了
# (实测 688/688),而 k=2 的覆盖只有 6%、"丢中间/最新 recent"只有 14%。决定 selector
# 复杂度的量是**第二张旧图的边际**
#     marginal = max Q_cf(Recent-2 + 两张旧) - max Q_cf(Recent-3 + 一张旧)
# 它不依赖 pool mean,所以不必把 --exact-max-candidates 堆到全穷举那一档(约 6 小时)。
#
# 关键设计:k=2 **穷举全部旧图对**,不走"先按 singleton utility 取 top-8"的捷径。
# 互补性的定义就是"两张单独都不突出、合起来才有用",top-8 预筛会结构性地把这种组合
# 排除掉,于是只能找到"两张各自都不错的图" —— 那等于用一个偏向零假设的搜索去检验
# 备择假设。全穷举只多花约 11 分钟。
#
# DecisionPoint / FrozenSetScorer / SetScoreCache 全部复用 probe_oracle_headroom,
# 所以 cache key、渲染、编码路径与 Gate 3 逐位一致,分数可直接并入既有缓存。
"""

from __future__ import annotations

import itertools
import json
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
    set_key,
    sha256_of,
    stratified_sample,
)

# 主预算。方法定义允许 B in {1,2,3,4},但需要补搜的只有部署预算 B=4:
# B=1 的 k=1 就是全部单元素集合(Gate 3 depth-1 已穷举),B=2/3 的 k=1 同理由
# Recent-(B-1) 前缀扩展覆盖。
MAIN_BUDGET = 4


def families_for(point: DecisionPoint, budget: int) -> dict[str, list[tuple[int, ...]]]:
    """Enumerate the three families this probe adds on top of Gate 3's beam cache.

    ``olds`` 必须是 ``[1, current_step-budget-1]``:落在 R_B 里的 step 一旦被当作
    "旧图"塞回去,得到的要么是 R_B 自己(k=0),要么是一个只有 B-1 个元素的集合。
    """
    current = point.current_step
    recent = list(range(current - budget, current))
    olds = [step for step in range(1, current - budget)]
    out: dict[str, list[tuple[int, ...]]] = {
        "anchor_recent": [tuple(recent)],
        "k1_all_drop_positions": [],
        "k2_exhaustive_old_pairs": [],
        "b2_all_pairs": [],
    }
    if budget > point.n_candidates:
        return out

    # (c) k=1,四个丢弃位置全覆盖 —— Gate 3 只穷举了 drop=最老那一支。
    for dropped in recent:
        kept = [step for step in recent if step != dropped]
        for old in olds:
            out["k1_all_drop_positions"].append(tuple(sorted(kept + [old])))

    # (b) k=2,固定保留最新的 B-2 张 recent,穷举**全部**旧图对。
    kept_tail = recent[budget - 2 :] if budget >= 2 else []
    for first, second in itertools.combinations(olds, 2):
        out["k2_exhaustive_old_pairs"].append(tuple(sorted(kept_tail + [first, second])))

    # (a) B=2 的无偏 pool mean:全部 C(n,2)。
    for pair in itertools.combinations(point.candidates, 2):
        out["b2_all_pairs"].append(tuple(sorted(pair)))

    for key, sets in out.items():
        out[key] = sorted({steps for steps in sets if len(steps) == len(set(steps))})
    return out


def main() -> None:
    parser = build_parser()
    parser.add_argument(
        "--only-format",
        default="official_style_sparse_multiturn",
        choices=list(FORMATS),
        help="renderer to score; the format is frozen so scoring both wastes half the run",
    )
    parser.add_argument("--budget", type=int, default=MAIN_BUDGET)
    parser.add_argument(
        "--families",
        default="all",
        help=(
            "comma-separated family names to score, or 'all'. matched single-replacement "
            "sweep uses 'anchor_recent,k1_all_drop_positions' -- O(B*n) per group, so it "
            "stays affordable out to B=8 where C(n,B) never would"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="where to write the requested-set manifest (defaults next to --output)",
    )
    args = parser.parse_args()

    if args.score_cache is None:
        raise SystemExit("--score-cache is required: this probe only fills the cache")
    if args.shard_count > 1 and not args.skip_reduction:
        raise SystemExit("pass --skip-reduction on every shard (this probe never reduces)")

    config = load_config(args.config)
    samples = load_sparse_samples(resolve_sample_files(args.dataset_root))
    payload = json.loads(args.episode_filter.read_text(encoding="utf-8"))
    if args.episode_filter_key not in payload:
        raise SystemExit(f"--episode-filter-key {args.episode_filter_key!r} absent")
    allowed = set(payload[args.episode_filter_key])
    dev_rows = group_dev_rows(samples, allowed)
    chosen, sample_manifest = stratified_sample(
        dev_rows, count=args.probe_groups, seed=args.sample_seed
    )
    if args.max_groups:
        from scripts.probe_oracle_headroom import spread_subset

        chosen = spread_subset(chosen, dev_rows, args.max_groups)

    all_groups = list(chosen)
    if args.shard_count > 1:
        chosen = chosen[args.shard_index :: args.shard_count]
        if not chosen:
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

    # note (luojiaxuan): fingerprint 必须与 Gate 3 逐字段一致,否则既有缓存会被判为
    # 异源、白跑一遍全部前向。formats 保留完整列表(缓存本身是两格式共用的)。
    cache_fingerprint = {
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
    }
    cache = SetScoreCache(
        args.score_cache,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fingerprint=cache_fingerprint,
    )
    scorer = FrozenSetScorer(
        args=args, config=config, image_root=image_root, cache=cache, torch=torch
    )

    heartbeat_path = None if args.heartbeat is None else per_shard(args.heartbeat)
    started = time.time()
    requested: dict[str, dict[str, int]] = {}
    annotation_cache: dict[str, list[dict]] = {}

    def beat(done: int, total: int) -> None:
        state = {
            "stage": "scoring_replacement_families",
            "groups_done": done,
            "groups_total": total,
            "forwards": scorer.forwards,
            "cache_hits": scorer.cache_hits,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        print(json.dumps({"probe_progress": state}), flush=True)
        if heartbeat_path is not None:
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")

    for position, pair_group in enumerate(chosen, start=1):
        point = DecisionPoint(
            dev_rows[pair_group],
            annotations=args.annotations,
            annotation_cache=annotation_cache,
            image_root=image_root,
        )
        families = families_for(point, args.budget)
        if args.families != "all":
            wanted = {name.strip() for name in args.families.split(",") if name.strip()}
            unknown = wanted - set(families)
            if unknown:
                raise SystemExit(f"unknown families {sorted(unknown)}; have {sorted(families)}")
            families = {name: sets for name, sets in families.items() if name in wanted}
        counts = {name: len(sets) for name, sets in families.items()}
        requested[pair_group] = counts
        for name, sets in families.items():
            for steps in sets:
                scorer.score(point, steps, args.only_format)
        beat(position, len(chosen))

    cache.close()

    manifest_path = args.manifest or per_shard(args.output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "causalcache.replacement_set_probe.v1",
                "format": args.only_format,
                "budget": args.budget,
                "groups_in_full_sample": len(all_groups),
                "groups_in_this_shard": len(chosen),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "forwards": scorer.forwards,
                "cache_hits": scorer.cache_hits,
                "requested_per_group": requested,
                "stratification": sample_manifest,
                "cache_fingerprint": cache_fingerprint,
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "forwards": scorer.forwards}), flush=True)


if __name__ == "__main__":
    main()
