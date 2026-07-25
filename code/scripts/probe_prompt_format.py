#!/usr/bin/env python3
"""Frozen-model prompt-format probe: is the sparse **single-turn** renderer worth keeping?

# note (luojiaxuan): Gate 1 的证据脚本。背景是 v5 五臂的实测分解——
#
#     format_effect          = R0 - N0 = -0.0217  [-0.0303, -0.0131]
#     frozen_selection_effect = S0 - R0 = +0.0335  [+0.0267, +0.0403]
#
# 也就是我们自造的 sparse **单轮** 格式比官方多轮格式**差** 0.0217 nats,把可用的
# 稀疏选点收益吃掉了约 65%。本脚本在**冻结 policy**(不训练、不注入也不加载任何
# adapter)上,把"格式"与"选点"这两个因子拆成 2×2 再加官方基线,一次性量出来:
#
#     N0        原生官方多轮      + Recent-K   (build_official_messages,语料里现成的)
#     R_single  当前单轮          + Recent-K   (build_sparse_history_messages)
#     S_single  当前单轮          + Sparse-K
#     R_multi   official-style 多轮 + Recent-K (build_sparse_multiturn_messages,本次新增)
#     S_multi   official-style 多轮 + Sparse-K
#
# 三个待答的量(外加旧格式的 S_single - N0 以便直接对比):
#     R_multi - N0        换格式后还剩多少格式成本(新渲染器相对原生的净开销)
#     S_multi - R_multi   换格式后稀疏选点收益是否仍在
#     S_multi - N0        相对原生 Recent 的净效果 —— 这才是"该不该换"的判据
#
# 三条必须守住的方法学约束:
#   1. **只用 dev**。confirmation 那 254 组连分数都不许算出来——所以 --episode-filter/
#      --episode-filter-key 在本脚本里是**必填**(打分脚本里它们是可选的),且过滤发生在
#      --max-groups 截断与分片**之前**,否则被排除的组仍会进入某个分片的工作集。
#   2. **同一条编码路径**。五臂全部走 encode_sample + mean_target_logprob,与
#      score_sparse_history_arms.py 共用同一份实现;差异只能来自 messages 本身。
#   3. **像素不变**。R_multi 与 N0/R0 用同一组 recent 图,S_multi 与 S0 用同一组 sparse
#      图,逐路径断言。否则量出来的就不只是格式效应。
#
# 关于 full_responses:v5 样本行**没有**存这个字段(它只有 action_texts 这些裸描述,
# 完整响应只以 assistant 轮的形式躺在 N0 的 messages 里,因而只覆盖 recent-K 那几步)。
# 稀疏选点普遍落在更早的步上,拿不到完整响应,而"用裸描述凑数"正是审计 P0-1 修过的坑
# (实测 0.133 nats)。所以本脚本从 **GUI-Odyssey 原始 annotation 重建**完整响应:
# 复用语料构建器自己的 target_arguments() + official_response(),再拿重建结果与 N0
# messages 里现成的 assistant 轮**逐字节对账**(见 rebuild_full_responses);对不上就
# 整组 fail-closed 丢弃,绝不退回裸描述。dev 240 组实测:0 处不符、0 步重建不出。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_sparse_multiturn import (
    SPARSE_MULTITURN_PROMPT_FORMAT,
    build_sparse_multiturn_messages,
)
from scripts.build_sparse_history_dataset import official_response, target_arguments
from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from scripts.score_sparse_history_arms import (
    EpisodeClusterBootstrap,
    ScoreCache,
    load_sparse_samples,
    resolve_sample_files,
    sha256_of,
    shard_cache_path,
)
from scripts.train_success_sft_lora import (
    encode_sample,
    load_config,
    mean_target_logprob,
)

REPORT_SCHEMA = "causalcache.prompt_format_probe.v1"
SHARD_REPORT_SCHEMA = "causalcache.prompt_format_probe_shard.v1"
# 冻结模型专用的 cache 命名空间:本脚本一条 adapter 权重都不装,所以只有这一个键。
FROZEN_ADAPTER_KEY = "frozen_no_adapter"

# 五臂:slot -> (语料里的来源 slot | None, 选点模式, prompt_format)
# 来源 slot 为 None 表示 messages 由本脚本现渲染。
ARM_SPECS: tuple[tuple[str, str | None, str, str], ...] = (
    ("N0", "N0", "recent", "official_multiturn"),
    ("R_single", "R0", "recent", "sparse_single_turn"),
    ("S_single", "S0", "sparse", "sparse_single_turn"),
    ("R_multi", None, "recent", SPARSE_MULTITURN_PROMPT_FORMAT),
    ("S_multi", None, "sparse", SPARSE_MULTITURN_PROMPT_FORMAT),
)
ARM_SLOTS = tuple(spec[0] for spec in ARM_SPECS)

# 派生量的代数式。前三个是本次要回答的问题,后三个把旧格式摆在同一张表上直接对比。
DERIVED_QUANTITIES: dict[str, tuple[str, str]] = {
    "format_cost_multi": ("R_multi", "N0"),
    "sparse_gain_multi": ("S_multi", "R_multi"),
    "net_effect_multi": ("S_multi", "N0"),
    "format_cost_single": ("R_single", "N0"),
    "sparse_gain_single": ("S_single", "R_single"),
    "net_effect_single": ("S_single", "N0"),
}


# ---------------------------------------------------------------------------
# 语料装载与分组
# ---------------------------------------------------------------------------
def group_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """{pair_group: {arm_slot: row}},只保留本探针实际要用的三个来源臂。"""
    needed = {source for _slot, source, _mode, _fmt in ARM_SPECS if source}
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        slot = sample.get("arm_slot")
        if slot not in needed:
            continue
        bucket = grouped.setdefault(sample["pair_group"], {})
        if slot in bucket:
            raise ValueError(f"pair-group {sample['pair_group']!r} carries two {slot!r} rows")
        bucket[slot] = sample
    complete = {group: rows for group, rows in grouped.items() if set(rows) == needed}
    missing = sorted(set(grouped) - set(complete))
    if missing:
        # note (luojiaxuan): 缺臂的组必须显式报出来再丢,不能静默跳过——分母悄悄变小
        # 是本仓库反复出现的那类偏差。
        raise ValueError(
            f"{len(missing)} pair-groups lack one of {sorted(needed)}: {missing[:5]}"
        )
    return complete


def rebuild_full_responses(
    rows: dict[str, dict[str, Any]], annotations: Path, cache: dict[str, list[dict]]
) -> list[str]:
    """Rebuild every completed step's verbatim assistant response, then reconcile.

    重建来源是 GUI-Odyssey 的原始 annotation(动作参数)加样本行里的 action_texts
    (动作描述),两者经语料构建器自己的 official_response() 合成——与语料生成时
    **逐字节同一条代码路径**。合成完再与 N0 messages 里现成的 assistant 轮对账:
    N0 的保留轮覆盖 recent-K 那几步,对得上就说明这条重建链在这一组上是正确的。
    任何一步重建不出来、或对账不符,都抛 ValueError 让调用方整组丢弃。
    """
    n0 = rows["N0"]
    episode = n0["episode"]
    current_step = int(n0["decision_step"])
    if episode not in cache:
        payload = json.loads((annotations / f"{episode}.json").read_text(encoding="utf-8"))
        steps = list(payload.get("steps") or [])
        # GUI-Odyssey 的 annotation 比动作多一条 COMPLETE 收尾记录,不对应任何模型动作。
        if steps and str((steps[-1] or {}).get("action", "")).upper() == "COMPLETE":
            steps = steps[:-1]
        cache[episode] = steps
    annotated = cache[episode]
    action_texts = list(n0["action_texts"])
    if len(annotated) < current_step - 1 or len(action_texts) < current_step - 1:
        raise ValueError(f"{n0['pair_group']}: annotation shorter than the history")
    rebuilt: list[str] = []
    for index in range(current_step - 1):
        arguments = target_arguments(annotated[index])
        if arguments is None:
            raise ValueError(
                f"{n0['pair_group']}: Step{index + 1} has no reconstructable tool call; "
                "the probe never falls back to a bare description"
            )
        rebuilt.append(official_response(action_texts[index], arguments))
    # 对账:N0 的 assistant 轮 == 重建结果在 N0.selected_steps 上的取值,逐字节。
    observed = [
        message["content"][0]["text"]
        for message in n0["messages"]
        if message["role"] == "assistant"
    ]
    expected = [rebuilt[step - 1] for step in n0["selected_steps"]]
    if observed != expected:
        raise ValueError(
            f"{n0['pair_group']}: rebuilt full responses disagree with the official "
            "arm's assistant turns; the reconstruction chain is not trustworthy here"
        )
    return rebuilt


def render_multiturn_row(
    rows: dict[str, dict[str, Any]],
    *,
    slot: str,
    source_slot: str,
    full_responses: list[str],
) -> dict[str, Any]:
    """Build one official-style sparse multiturn row from an existing arm's selection."""
    source = rows[source_slot]
    messages = build_sparse_multiturn_messages(
        instruction=source["instruction"],
        action_texts=source["action_texts"],
        full_responses=full_responses,
        selected_steps=source["selected_steps"],
        selected_images=[{"__path__": path} for path in source["selected_images"]],
        current_step=source["decision_step"],
        current_image={"__path__": source["current_image"]},
    )
    serialised = [
        {
            "role": message["role"],
            "content": [
                {"type": "image", "path": part["image"]["__path__"]}
                if part.get("type") == "image"
                else dict(part)
                for part in message["content"]
            ],
        }
        for message in messages
    ]
    return {
        **{k: v for k, v in source.items() if k not in {"messages", "variant"}},
        "sample_id": f"{source['pair_group']}|{slot}",
        "arm_slot": slot,
        "arm_id": slot,
        "prompt_format": SPARSE_MULTITURN_PROMPT_FORMAT,
        "adapter_mode": "bypass",
        "messages": serialised,
        "variant": f"{slot}_{source['selection_mode']}{source['budget']}",
    }


def build_arm_rows(
    rows: dict[str, dict[str, Any]], *, annotations: Path, cache: dict[str, list[dict]]
) -> dict[str, dict[str, Any]]:
    """Assemble all five arms for one pair-group, or raise for a fail-closed drop."""
    full_responses = rebuild_full_responses(rows, annotations, cache)
    arms: dict[str, dict[str, Any]] = {}
    for slot, source_slot, mode, prompt_format in ARM_SPECS:
        if source_slot is not None:
            row = rows[source_slot]
            if row["selection_mode"] != mode or row["prompt_format"] != prompt_format:
                raise ValueError(
                    f"{row['pair_group']}: source arm {source_slot!r} declares "
                    f"({row['selection_mode']}, {row['prompt_format']}), expected "
                    f"({mode}, {prompt_format})"
                )
            arms[slot] = row
        else:
            donor = "R0" if mode == "recent" else "S0"
            arms[slot] = render_multiturn_row(
                rows, slot=slot, source_slot=donor, full_responses=full_responses
            )
    assert_pixels_are_held_fixed(arms)
    return arms


def image_paths(row: dict[str, Any]) -> list[str]:
    return [
        part["path"]
        for message in row["messages"]
        for part in message["content"]
        if part.get("type") == "image"
    ]


def assert_pixels_are_held_fixed(arms: dict[str, dict[str, Any]]) -> None:
    """Every arm must show the same pixels as its same-selection peers, in the same order.

    # note (luojiaxuan): 这是整个探针的效度前提。如果 R_multi 悄悄比 N0 多看/少看一张
    # 图,或者顺序不同,那 ``R_multi - N0`` 就不再是格式效应而是格式 + 视觉预算的混合,
    # 而这种错位不会让任何一步崩掉,只会安静地改变结论。
    """
    recent = {slot: image_paths(arms[slot]) for slot in ("N0", "R_single", "R_multi")}
    sparse = {slot: image_paths(arms[slot]) for slot in ("S_single", "S_multi")}
    for label, bucket in (("recent", recent), ("sparse", sparse)):
        reference_slot, reference = next(iter(bucket.items()))
        for slot, paths in bucket.items():
            if paths != reference:
                raise ValueError(
                    f"{arms['N0']['pair_group']}: {label} arm {slot!r} shows different "
                    f"images than {reference_slot!r}; the probe would no longer be "
                    "measuring format alone"
                )
    budget = int(arms["N0"]["budget"])
    for slot, row in arms.items():
        if len(image_paths(row)) != budget + 1:
            raise ValueError(
                f"{row['pair_group']}: arm {slot!r} carries "
                f"{len(image_paths(row))} images, expected K+1={budget + 1}"
            )
        if image_paths(row)[-1] != arms["N0"]["current_image"]:
            raise ValueError(
                f"{row['pair_group']}: arm {slot!r} does not end on the current screenshot"
            )


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
class FrozenScorer:
    """Teacher-forced mean log-prob under the frozen policy, with no adapter at all.

    # note (luojiaxuan): 与 score_sparse_history_arms.ArmScorer 的关键区别:那里会
    # inject_history_gated_kv 再靠 adapter_mode=bypass 让 hook 变成恒等;这里**根本不
    # 注入**,所以"探针跑在冻结模型上"是结构性的,不依赖任何运行期开关。runtime 与
    # checkpoint 无关,故 cache key 里的 adapter 段是常量。
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        config: dict[str, Any],
        image_root: Path,
        cache: ScoreCache,
        torch: Any,
    ) -> None:
        self.args = args
        self.config = config
        self.image_root = image_root
        self.cache = cache
        self.torch = torch
        self.runtime: Any = None
        self.forwards = 0
        self.cache_hits = 0

    def _prepare(self) -> Any:
        if self.runtime is None:
            # 惰性构造:全命中缓存时(纯归约)一张卡都不占。
            self.runtime = GUIOwlV21OfficialToolsRuntime(
                model_dir=self.args.model_dir,
                expected_snapshot_manifest=(
                    self.args.repository_root / self.config["policy_snapshot_manifest"]
                ),
                device=self.args.device,
                target_effective_visual_tokens_per_image=(
                    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
                ),
            )
            model = self.runtime.model
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.config.use_cache = False
            model.eval()
        return self.runtime

    def score(self, sample: dict[str, Any], *, slot: str) -> float:
        cache_key = f"{FROZEN_ADAPTER_KEY}|{sample['pair_group']}|{slot}|bypass"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.args.require_cached:
            raise SystemExit(
                f"--require-cached is set but {cache_key!r} is missing from the score "
                f"cache {self.cache.path}; some shard did not finish, or wrote to a "
                "different cache path"
            )
        runtime = self._prepare()
        encoded = encode_sample(
            runtime, sample, dataset_root=self.image_root, torch=self.torch
        )
        if encoded is None:
            raise ValueError(
                f"sample {sample['sample_id']!r} produced an empty target encoding; "
                "the scored denominator must stay reconciled"
            )
        with self.torch.no_grad():
            value = float(
                mean_target_logprob(runtime.model, encoded, torch=self.torch).detach()
            )
        self.forwards += 1
        self.cache.put(cache_key, value)
        return value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score five prompt-format arms on the dev split under the frozen policy "
            "and report the format / selection decomposition with episode-cluster CIs."
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="root the sample image paths resolve against (default: --dataset-root)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help=(
            "GUI-Odyssey raw annotation directory (<episode>.json); the only source "
            "from which history full responses can be rebuilt"
        ),
    )
    # note (luojiaxuan): 打分脚本里这两个是可选的,本脚本里是**必填**。探针的全部意义
    # 是在 dev 上做格式决策,而 confirmation 的契约是"只允许读一次"——一次手滑省掉过滤
    # 就会把 254 组 confirmation 的分数算进缓存,把"只读一次"降级成"只承诺不看一次"。
    parser.add_argument("--episode-filter", type=Path, required=True)
    parser.add_argument("--episode-filter-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-reduction", action="store_true")
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="smoke run: keep only the first N dev groups (0 = every dev group)",
    )
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit(
            f"--shard-index {args.shard_index} must lie inside "
            f"[0, --shard-count={args.shard_count})"
        )
    if args.shard_count > 1 and args.score_cache is None:
        raise SystemExit(
            "--shard-count > 1 requires --score-cache: shards hand their work to the "
            "reduction pass through the cache and nothing else"
        )
    if args.shard_count > 1 and not args.skip_reduction:
        # 分片进程只握着 1/M 的 dev 组,在上面跑 bootstrap 会产出一份结构与完整报告
        # 一模一样、分母却小了 M 倍的 JSON。硬失败,不给警告。
        raise SystemExit(
            "a sharded process only holds 1/--shard-count of the dev split; running "
            "the bootstrap on that slice would silently shrink the denominator. Pass "
            "--skip-reduction on every shard, then run one unsharded reduction pass."
        )
    if args.require_cached and args.score_cache is None:
        raise SystemExit("--require-cached is meaningless without --score-cache")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise SystemExit("--bootstrap-confidence must lie strictly inside (0, 1)")

    import torch

    config = load_config(args.config)
    sample_files = resolve_sample_files(args.dataset_root)
    samples = load_sparse_samples(sample_files)
    grouped = group_samples(samples)

    payload = json.loads(args.episode_filter.read_text(encoding="utf-8"))
    if args.episode_filter_key not in payload:
        available = sorted(k for k, v in payload.items() if isinstance(v, list))
        raise SystemExit(
            f"--episode-filter-key {args.episode_filter_key!r} absent from "
            f"{args.episode_filter}; list-valued keys are {available}"
        )
    allowed = set(payload[args.episode_filter_key])
    if not allowed:
        raise SystemExit("episode filter selected an empty episode set")
    kept = {
        group: rows
        for group, rows in grouped.items()
        if rows["N0"]["episode"] in allowed
    }
    if not kept:
        raise SystemExit("episode filter removed every group; wrong corpus?")
    print(
        json.dumps(
            {
                "episode_filter": str(args.episode_filter),
                "episode_filter_key": args.episode_filter_key,
                "episodes_allowed": len(allowed),
                "groups_kept": len(kept),
                "groups_dropped": len(grouped) - len(kept),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    groups = kept
    if args.max_groups:
        groups = dict(sorted(groups.items())[: args.max_groups])
    reduction_groups = groups
    if args.shard_count > 1:
        assigned = sorted(groups)[args.shard_index :: args.shard_count]
        groups = {group: groups[group] for group in assigned}
        if not groups:
            raise SystemExit(
                f"shard {args.shard_index}/{args.shard_count} covers no pair-group"
            )

    image_root = args.image_root or args.dataset_root

    def per_shard(path: Path) -> Path:
        if args.shard_count <= 1:
            return path
        return path.with_name(
            f"{path.stem}.shard{args.shard_index:03d}"
            f"-of-{args.shard_count:03d}{path.suffix}"
        )

    output_path = per_shard(args.output)
    heartbeat_path = None if args.heartbeat is None else per_shard(args.heartbeat)

    cache_fingerprint = {
        "config_sha256": sha256_of(args.config),
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "image_root": str(image_root),
        "annotations": str(args.annotations),
        "arms": list(ARM_SLOTS),
        "adapter": "none",
    }
    cache = ScoreCache(
        args.score_cache,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fingerprint=cache_fingerprint,
    )
    scorer = FrozenScorer(
        args=args, config=config, image_root=image_root, cache=cache, torch=torch
    )

    started = time.time()
    every = max(int(args.progress_every), 0)

    def progress(position: int, total: int) -> None:
        if position != total and (every <= 0 or position % every):
            return
        state = {
            "stage": "scoring",
            "groups_done": position,
            "groups_total": total,
            "forwards": scorer.forwards,
            "cache_hits": scorer.cache_hits,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        print(json.dumps({"probe_progress": state}), flush=True)
        if heartbeat_path is not None:
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_path.write_text(
                json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
            )

    annotation_cache: dict[str, list[dict]] = {}
    per_group_scores: dict[str, dict[str, float]] = {}
    try:
        for position, (pair_group, rows) in enumerate(sorted(groups.items()), start=1):
            arms = build_arm_rows(
                rows, annotations=args.annotations, cache=annotation_cache
            )
            per_group_scores[pair_group] = {
                slot: scorer.score(arms[slot], slot=slot) for slot in ARM_SLOTS
            }
            progress(position, len(groups))
    finally:
        cache.close()

    provenance = {
        "device": args.device,
        "computed_forwards": scorer.forwards > 0,
        "device_name": None,
    }
    if scorer.runtime is not None:
        try:
            provenance["device_name"] = torch.cuda.get_device_name(
                scorer.runtime.model.device
            )
        except Exception:  # noqa: BLE001 - provenance must never break a run
            pass

    common = {
        "repository_root": str(args.repository_root),
        "config": str(args.config),
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "annotations": str(args.annotations),
        "episode_filter": str(args.episode_filter),
        "episode_filter_key": args.episode_filter_key,
        "sample_files": [
            {"path": str(path), "sha256": sha256_of(path)} for path in sample_files
        ],
        "arms": {
            slot: {"selection_mode": mode, "prompt_format": fmt, "source_arm": source}
            for slot, source, mode, fmt in ARM_SPECS
        },
        "adapter": "none (frozen policy; no injection, no checkpoint load)",
        "cache_fingerprint": cache_fingerprint,
        "cache": cache.stats(),
        "provenance": provenance,
        "forward_passes": scorer.forwards,
        "cache_hits": scorer.cache_hits,
        "smoke_run": bool(args.max_groups),
        "group_limit_applied": args.max_groups or None,
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if args.skip_reduction:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": SHARD_REPORT_SCHEMA,
                    "shard": f"{args.shard_index}/{args.shard_count}",
                    "pair_groups_scored": len(per_group_scores),
                    "score_cache_shard": (
                        None
                        if args.score_cache is None
                        else str(
                            shard_cache_path(
                                args.score_cache, args.shard_index, args.shard_count
                            )
                        )
                    ),
                    **common,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "shard_complete": f"{args.shard_index}/{args.shard_count}",
                    "pair_groups": len(per_group_scores),
                    "output": str(output_path),
                }
            ),
            flush=True,
        )
        return

    # ---- 归约:分片跑法下这一遍必须重新组装完整 dev 集(全部命中缓存) ----
    if len(per_group_scores) != len(reduction_groups):
        for pair_group, rows in sorted(reduction_groups.items()):
            if pair_group in per_group_scores:
                continue
            arms = build_arm_rows(
                rows, annotations=args.annotations, cache=annotation_cache
            )
            per_group_scores[pair_group] = {
                slot: scorer.score(arms[slot], slot=slot) for slot in ARM_SLOTS
            }

    per_quantity: dict[str, dict[str, float]] = {slot: {} for slot in ARM_SLOTS}
    for name in DERIVED_QUANTITIES:
        per_quantity[name] = {}
    for pair_group, scores in per_group_scores.items():
        for slot in ARM_SLOTS:
            per_quantity[slot][pair_group] = scores[slot]
        for name, (left, right) in DERIVED_QUANTITIES.items():
            per_quantity[name][pair_group] = scores[left] - scores[right]

    group_episode = {
        pair_group: rows["N0"]["episode"]
        for pair_group, rows in reduction_groups.items()
    }
    bootstrap = EpisodeClusterBootstrap(
        per_quantity,
        group_episode,
        replicates=args.bootstrap_replicates,
        confidence=args.bootstrap_confidence,
        seed=args.bootstrap_seed,
    )
    names = list(ARM_SLOTS) + list(DERIVED_QUANTITIES)
    point = bootstrap.point_estimate()
    intervals = bootstrap.intervals(bootstrap.replicate_aggregates(), names)
    summary = {
        name: {
            "point": point.get(name),
            **intervals[name],
            "support": bootstrap.support.get(name),
        }
        for name in names
    }

    report = {
        "schema_version": REPORT_SCHEMA,
        "pair_groups": len(per_group_scores),
        "episodes": len(bootstrap.episodes),
        "bootstrap": {
            "cluster_unit": "episode",
            "replicates": args.bootstrap_replicates,
            "confidence": args.bootstrap_confidence,
            "seed": args.bootstrap_seed,
        },
        "derived_algebra": {
            name: f"{left} - {right}" for name, (left, right) in DERIVED_QUANTITIES.items()
        },
        "summary": summary,
        "per_group_scores": {
            group: per_group_scores[group] for group in sorted(per_group_scores)
        },
        **common,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("")
    print(f"pair-groups: {len(per_group_scores)}   episodes: {len(bootstrap.episodes)}")
    if args.max_groups:
        print(f"*** SMOKE RUN (--max-groups {args.max_groups}) — not a dev-wide result ***")
    print("")
    print(f"{'arm':<20}{'mean logprob':>14}{'ci_low':>12}{'ci_high':>12}")
    for slot in ARM_SLOTS:
        row = summary[slot]
        print(
            f"{slot:<20}{row['point']:>14.4f}{row['ci_low']:>12.4f}"
            f"{row['ci_high']:>12.4f}"
        )
    print("")
    print(f"{'quantity':<20}{'algebra':<22}{'point':>10}{'ci_low':>12}{'ci_high':>12}")
    for name, (left, right) in DERIVED_QUANTITIES.items():
        row = summary[name]
        print(
            f"{name:<20}{left + ' - ' + right:<22}{row['point']:>10.4f}"
            f"{row['ci_low']:>12.4f}{row['ci_high']:>12.4f}"
        )
    print("")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
