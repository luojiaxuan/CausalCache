#!/usr/bin/env python3
"""Execute the sparse-history checkpoint-selection gates on the heldout split.

# note (luojiaxuan): 审计第 9 条的修复。此前 config.gates 只被 schema 校验:must_pass
# 的 "> 0" / "< 0.02" 是从不被解析的字符串,composite_score / selection_rule 只被断言
# "非空",全仓库没有任何代码在留出集上给 N0/S0/RA 打过分。于是"验收标准"完全靠人肉
# 执行,config 写什么都不会让任何一次 run 失败。本脚本把这一段补上:
#
#   1. 按样本的 ``split`` 字段取留出组(权威只有这一个,和 trainer 共用同一个分桶函数);
#   2. 每组 teacher-forced 打五臂 N0/R0/S0/RA/SA,三个负样本 **active 与 bypass 各一次**
#      (后者是 drift 的锚点,与训练损失里 P1-4 的 per-negative 锚点定义一致);
#   3. 按 config.gates.derived_quantities 的代数式求七个派生量,再求
#      SA_minus_SA_neg_<kind> 与 <kind>_drift_abs;
#   4. **解析并执行** must_pass 的比较式、composite_score 表达式与 selection_rule
#      (解析器从 trainer 导入,两侧不各写一份);
#   5. 置信区间用 **episode-cluster bootstrap**:重采样单位是 episode,不是 pair-group。
#      同一条轨迹上多个决策点的分数高度相关,把每组当独立样本会把区间压窄好几倍,
#      让一个其实来自三四条轨迹的效应看起来显著。
#
# bypass 臂(N0/R0/S0 与负样本的 bypass 锚点)的分数与 checkpoint 无关——
# history_gated_lora 的 hook 在 ctx=None 时原样返回冻结输出,一条乘法都不执行。
# 所以它们只在开头算一次并缓存复用;每个 checkpoint 只需再跑 RA/SA 与三个负样本的
# active 前向。这既省掉大半算力,也顺带把"bypass 必须与 checkpoint 无关"这条不变量
# 写进了流程本身。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from scripts.train_success_sft_lora import (
    GateQuantityUnavailable,
    SPARSE_GATE_VOCABULARY,
    SPARSE_HELDOUT_REQUIRED_SLOTS,
    SPARSE_NEGATIVE_KINDS,
    SPARSE_POSITIVE_SLOT,
    adapter_context_for_sample,
    adapter_settings,
    build_sparse_history_heldout_units,
    compile_sparse_gates,
    encode_sample,
    evaluate_gate_expression,
    load_config,
    mean_target_logprob,
    validate_sparse_gates,
)

REPORT_SCHEMA = "causalcache.sparse_history_gate_report.v1"
BOOTSTRAP_CLUSTER_UNIT = "episode"
FROZEN_ADAPTER_KEY = "frozen"
IDENTITY_ADAPTER_KEY = "identity"
# 只有这三臂在契约里是 adapter bypass,因而与 checkpoint 无关、可以只算一次。
FROZEN_ARM_SLOTS = ("N0", "R0", "S0")
ACTIVE_ARM_SLOTS = ("RA", SPARSE_POSITIVE_SLOT)
if sorted(FROZEN_ARM_SLOTS + ACTIVE_ARM_SLOTS) != sorted(
    SPARSE_HELDOUT_REQUIRED_SLOTS
):
    # note (luojiaxuan): 这两条元组是"哪些臂进冻结通道、哪些臂每个 checkpoint 重算"的
    # 唯一划分,必须恰好覆盖五臂。若将来契约加了一臂而这里没跟上,漏掉的臂会安静地
    # 不出现在任何派生量里,报告照样生成——所以在 import 时就炸掉。
    raise RuntimeError(
        "frozen/active arm split drifted from the frozen five-arm contract "
        f"{SPARSE_HELDOUT_REQUIRED_SLOTS}"
    )


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def resolve_sample_files(dataset_root: Path) -> list[Path]:
    """Resolve samples.jsonl or a part directory into an ordered file list.

    # note (luojiaxuan): 分片语料很容易出现"少读了一个 part 却照常出报告"的事故,
    # 所以这里的候选来源写死成有序的几种,选中哪一种会连同每个文件的 sha256 一起写进
    # 报告——留出集的分母必须能被外部对上号。
    """
    if dataset_root.is_file():
        return [dataset_root]
    single = dataset_root / "samples.jsonl"
    if single.is_file():
        return [single]
    for subdirectory in ("parts", "shards"):
        candidate = dataset_root / subdirectory
        if candidate.is_dir():
            files = sorted(candidate.glob("*.jsonl"))
            if files:
                return files
    files = sorted(dataset_root.glob("*.jsonl"))
    if files:
        return files
    raise ValueError(
        f"{dataset_root} carries neither samples.jsonl nor any *.jsonl part files"
    )


def load_sparse_samples(files: list[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number} is not JSON: {error}")
    if not samples:
        raise ValueError(f"no samples found in {[str(path) for path in files]}")
    return samples


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
class ScoreCache:
    """Append-only (adapter, group, arm, mode) -> score cache with resume.

    # note (luojiaxuan): 打分是一个几小时量级的多 checkpoint GPU 作业,共享机器上被
    # 别的 session 收掉容器、被 OOM 打断都是常态。每条前向算完就落盘,重启时按
    # cache_key 跳过已完成项,重跑从断点继续而不是从头开始。
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.entries: dict[str, float] = {}
        self.handle = None
        if path is None:
            return
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.entries[record["cache_key"]] = float(record["score"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def get(self, cache_key: str) -> float | None:
        return self.entries.get(cache_key)

    def put(self, cache_key: str, score: float) -> None:
        self.entries[cache_key] = score
        if self.handle is not None:
            self.handle.write(
                json.dumps({"cache_key": cache_key, "score": score}) + "\n"
            )
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class ArmScorer:
    """Teacher-forced mean log-prob of one arm under one adapter state."""

    def __init__(
        self,
        runtime: Any,
        *,
        image_root: Path,
        merge_size: int,
        torch: Any,
        cache: ScoreCache,
    ) -> None:
        self.runtime = runtime
        self.image_root = image_root
        self.merge_size = merge_size
        self.torch = torch
        self.cache = cache
        self.forwards = 0
        self.cache_hits = 0

    def score(
        self,
        sample: dict[str, Any],
        *,
        slot: str,
        adapter_key: str,
        adapter_mode: str | None = None,
    ) -> float:
        from causalcache.policy.history_adapter_context import history_adapter_scope

        mode = sample["adapter_mode"] if adapter_mode is None else adapter_mode
        cache_key = f"{adapter_key}|{sample['pair_group']}|{slot}|{mode}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        encoded = encode_sample(
            self.runtime, sample, dataset_root=self.image_root, torch=self.torch
        )
        if encoded is None:
            # note (luojiaxuan): 留出集上不允许静默跳过——一条编码不出来的样本会让整组
            # 退出分母,而"分母悄悄变小"是最难在报告里被发现的评测偏差。
            raise ValueError(
                f"heldout sample {sample['sample_id']!r} produced an empty target "
                "encoding; the scored denominator must stay reconciled"
            )
        # adapter 开关只读 adapter_mode 字段(或调用方显式传入的 bypass 覆盖),
        # 与 trainer 共用同一个解析入口,绝不按 arm_slot 名字前缀猜。
        context = adapter_context_for_sample(
            encoded,
            sample,
            merge_size=self.merge_size,
            adapter_mode=adapter_mode,
            slot=slot,
        )
        with history_adapter_scope(context):
            with self.torch.no_grad():
                value = float(
                    mean_target_logprob(
                        self.runtime.model, encoded, torch=self.torch
                    ).detach()
                )
        self.forwards += 1
        self.cache.put(cache_key, value)
        return value


def negative_slots(samples: list[dict[str, Any]], group: dict[str, int]) -> dict[str, str]:
    """Return {negative_kind: arm_slot} for the negatives this group actually has.

    # note (luojiaxuan): K=1 组的 step_shuffled / duplicate 与 SA 逐字节相同因而不入库,
    # 所以负样本相关的量只在**实际存在**的组上有支撑;支撑组数会写进报告,让"这条 gate
    # 是靠几组算出来的"始终可见。
    """
    found: dict[str, str] = {}
    for slot, index in group.items():
        sample = samples[index]
        if sample["role"] != "negative":
            continue
        kind = sample["negative_kind"]
        if kind in found:
            raise ValueError(
                f"pair-group {sample['pair_group']!r} carries two {kind!r} negatives"
            )
        found[kind] = slot
    return found


def score_frozen_arms(
    scorer: ArmScorer,
    samples: list[dict[str, Any]],
    groups: dict[str, dict[str, int]],
    *,
    progress: Any,
) -> dict[str, dict[str, float]]:
    """Score every adapter-bypassed forward once; these do not depend on a checkpoint."""
    frozen: dict[str, dict[str, float]] = {}
    for position, (pair_group, group) in enumerate(sorted(groups.items()), start=1):
        scores: dict[str, float] = {}
        for slot in FROZEN_ARM_SLOTS:
            sample = samples[group[slot]]
            if sample["adapter_mode"] != "bypass":
                # 缓存复用的正确性完全建立在"这三臂确实 bypass"上,所以显式断言。
                raise ValueError(
                    f"arm {slot!r} of {pair_group!r} declares adapter_mode "
                    f"{sample['adapter_mode']!r}; the frozen pass assumes bypass"
                )
            scores[slot] = scorer.score(
                sample, slot=slot, adapter_key=FROZEN_ADAPTER_KEY
            )
        for slot in sorted(negative_slots(samples, group).values()):
            # 负样本的 bypass 锚点:同一条负样本在冻结 policy 上的分数(drift 的基准)。
            scores[f"{slot}@bypass"] = scorer.score(
                samples[group[slot]],
                slot=slot,
                adapter_key=FROZEN_ADAPTER_KEY,
                adapter_mode="bypass",
            )
        frozen[pair_group] = scores
        progress("frozen", position, len(groups))
    return frozen


def score_active_arms(
    scorer: ArmScorer,
    samples: list[dict[str, Any]],
    groups: dict[str, dict[str, int]],
    *,
    adapter_key: str,
    label: str,
    progress: Any,
) -> dict[str, dict[str, float]]:
    """Score RA/SA and the three negatives with the adapter active."""
    active: dict[str, dict[str, float]] = {}
    for position, (pair_group, group) in enumerate(sorted(groups.items()), start=1):
        scores: dict[str, float] = {}
        for slot in ACTIVE_ARM_SLOTS:
            scores[slot] = scorer.score(
                samples[group[slot]], slot=slot, adapter_key=adapter_key
            )
        for slot in sorted(negative_slots(samples, group).values()):
            scores[slot] = scorer.score(
                samples[group[slot]], slot=slot, adapter_key=adapter_key
            )
        active[pair_group] = scores
        progress(label, position, len(groups))
    return active


# ---------------------------------------------------------------------------
# 派生量
# ---------------------------------------------------------------------------
def group_quantities(
    samples: list[dict[str, Any]],
    groups: dict[str, dict[str, int]],
    frozen: dict[str, dict[str, float]],
    active: dict[str, dict[str, float]],
    *,
    derived_algebra: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Return {quantity: {pair_group: value}} for the per-group quantities.

    # note (luojiaxuan): 七个派生量直接按 **config.gates.derived_quantities 的代数式**
    # 求值,而不是在这里第三次手抄一遍 "SA - RA"。validate_sparse_gates 已断言该块与
    # 冻结的臂代数逐字相等,所以执行 config 的字符串既安全又让那一块真正被使用——
    # 一个从来没被执行过的声明块,迟早会和代码里的算式分叉。
    """
    per_quantity: dict[str, dict[str, float]] = {}
    for pair_group, group in sorted(groups.items()):
        arm_scores: dict[str, float | None] = {
            slot: frozen[pair_group][slot] for slot in FROZEN_ARM_SLOTS
        }
        arm_scores.update(
            {slot: active[pair_group][slot] for slot in ACTIVE_ARM_SLOTS}
        )
        for quantity, algebra in sorted(derived_algebra.items()):
            value = evaluate_gate_expression(algebra, arm_scores)
            per_quantity.setdefault(quantity, {})[pair_group] = value
        positive = arm_scores[SPARSE_POSITIVE_SLOT]
        for kind, slot in sorted(negative_slots(samples, group).items()):
            negative_active = active[pair_group][slot]
            negative_bypass = frozen[pair_group][f"{slot}@bypass"]
            per_quantity.setdefault(f"SA_minus_SA_neg_{kind}", {})[pair_group] = (
                positive - negative_active
            )
            per_quantity.setdefault(f"{kind}_drift", {})[pair_group] = (
                negative_active - negative_bypass
            )
    return per_quantity


def with_drift_abs(values: dict[str, float | None]) -> dict[str, float | None]:
    """Add ``<kind>_drift_abs`` on top of the aggregated signed drift.

    # note (luojiaxuan): 按 config.gates.drift_definition 的字面定义执行——
    # ``<kind>_drift`` 是留出组上 (active - bypass) 的**均值**,``<kind>_drift_abs``
    # 是这个均值的绝对值,不是逐组绝对值的均值。后者更严(正负漂移不会互相抵消),
    # 但收紧一条已冻结的验收阈值是科学决策,不能在打分脚本里顺手改掉。
    """
    enriched = dict(values)
    for kind in SPARSE_NEGATIVE_KINDS:
        drift = enriched.get(f"{kind}_drift")
        enriched[f"{kind}_drift_abs"] = None if drift is None else abs(drift)
    return enriched


def gate_namespace(aggregate: dict[str, float | None]) -> dict[str, float | None]:
    """Restrict the aggregate to the vocabulary gate expressions may reference."""
    return {name: aggregate.get(name) for name in sorted(SPARSE_GATE_VOCABULARY)}


# ---------------------------------------------------------------------------
# episode-cluster bootstrap
# ---------------------------------------------------------------------------
def percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


class EpisodeClusterBootstrap:
    """Percentile CIs from resampling **episodes**, not pair-groups.

    # note (luojiaxuan): 聚类单位是 episode。一条轨迹上抽了多个决策点,这些组共享
    # 指令、界面、模型对该任务的固有偏好,残差高度相关;按组做 i.i.d. bootstrap 会把
    # 有效样本量当成组数而不是轨迹数,区间宽度大致缩水 sqrt(每条轨迹的组数) 倍,
    # 于是一个其实只来自三四条轨迹的 SA-RA 效应会显示成"置信区间不含 0"。
    # 重采样时整条 episode 连同它的全部组一起进出,这才与数据的生成过程一致。
    """

    def __init__(
        self,
        per_quantity: dict[str, dict[str, float]],
        group_episode: dict[str, str],
        *,
        replicates: int,
        confidence: float,
        seed: int,
    ) -> None:
        self.quantities = sorted(per_quantity)
        self.replicates = replicates
        self.confidence = confidence
        self.seed = seed
        self.episodes = sorted({group_episode[group] for group in group_episode})
        # totals[quantity][episode] = [sum, count] —— 预聚合到 episode 级别,
        # 每个 bootstrap 复本只需按抽中的 episode 求和,不必重扫每一组。
        self.totals: dict[str, dict[str, list[float]]] = {}
        for quantity, per_group in per_quantity.items():
            bucket: dict[str, list[float]] = {}
            for pair_group, value in per_group.items():
                entry = bucket.setdefault(group_episode[pair_group], [0.0, 0.0])
                entry[0] += value
                entry[1] += 1.0
            self.totals[quantity] = bucket
        self.support = {
            quantity: {
                "groups": len(per_group),
                "episodes": len(self.totals[quantity]),
            }
            for quantity, per_group in per_quantity.items()
        }
        # <kind>_drift_abs 是 <kind>_drift 聚合之后的变换,支撑组数与后者相同。
        for kind in SPARSE_NEGATIVE_KINDS:
            drift_support = self.support.get(f"{kind}_drift")
            if drift_support is not None:
                self.support[f"{kind}_drift_abs"] = dict(drift_support)

    def aggregate(self, episode_counts: Counter) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for quantity in self.quantities:
            per_episode = self.totals[quantity]
            total = 0.0
            count = 0.0
            for episode, multiplicity in episode_counts.items():
                entry = per_episode.get(episode)
                if entry is None:
                    continue
                total += entry[0] * multiplicity
                count += entry[1] * multiplicity
            values[quantity] = (total / count) if count else None
        return with_drift_abs(values)

    def point_estimate(self) -> dict[str, float | None]:
        return self.aggregate(Counter(self.episodes))

    def replicate_aggregates(self) -> list[dict[str, float | None]]:
        rng = random.Random(self.seed)
        drawn: list[dict[str, float | None]] = []
        for _ in range(self.replicates):
            counts = Counter(
                rng.choice(self.episodes) for _ in range(len(self.episodes))
            )
            drawn.append(self.aggregate(counts))
        return drawn

    def intervals(
        self, replicates: list[dict[str, float | None]], names: list[str]
    ) -> dict[str, dict[str, float | None]]:
        tail = (1.0 - self.confidence) / 2.0
        result: dict[str, dict[str, float | None]] = {}
        for name in names:
            drawn = sorted(
                value
                for replicate in replicates
                for value in (replicate.get(name),)
                if value is not None
            )
            result[name] = {
                "ci_low": percentile(drawn, tail),
                "ci_high": percentile(drawn, 1.0 - tail),
                "bootstrap_replicates_used": len(drawn),
            }
        return result


# ---------------------------------------------------------------------------
# gate 执行与选点
# ---------------------------------------------------------------------------
def statistic_value(
    statistic: str, point: float | None, interval: dict[str, float | None]
) -> float | None:
    if statistic == "point":
        return point
    return interval.get(statistic)


def evaluate_gates(
    comparisons: dict[str, Any],
    point: dict[str, float | None],
    intervals: dict[str, dict[str, float | None]],
    *,
    gate_statistic: str,
) -> tuple[dict[str, Any], bool]:
    """Run every parsed must_pass comparison and report each verdict."""
    verdicts: dict[str, Any] = {}
    all_passed = True
    for quantity, comparison in sorted(comparisons.items()):
        interval = intervals.get(quantity, {})
        declared = comparison.statistic
        conservative = comparison.conservative_statistic()
        chosen = declared if gate_statistic == "declared" else conservative
        value = statistic_value(chosen, point.get(quantity), interval)
        passed = comparison.evaluate(value)
        verdicts[quantity] = {
            **comparison.as_dict(),
            "applied_statistic": chosen,
            "point": point.get(quantity),
            "ci_low": interval.get("ci_low"),
            "ci_high": interval.get("ci_high"),
            "compared_value": value,
            "passed": passed,
            # 同时报告保守(CI 端点)判定,便于人工判断这条 gate 过得有多勉强。
            "passed_conservative": comparison.evaluate(
                statistic_value(conservative, point.get(quantity), interval)
            ),
        }
        all_passed = all_passed and passed
    return verdicts, all_passed


def select_checkpoint(
    entries: list[dict[str, Any]], rule: dict[str, str]
) -> dict[str, Any]:
    """Apply the structured selection_rule to the scored checkpoints."""
    eligible = [
        entry
        for entry in entries
        if entry["selectable"]
        and entry["all_must_pass"]
        and entry["composite_score"]["value"] is not None
    ]
    selection: dict[str, Any] = {
        "rule": dict(rule),
        "candidates": [entry["label"] for entry in entries if entry["selectable"]],
        "eligible": [entry["label"] for entry in eligible],
        "selected": None,
        "selected_composite_score": None,
        "reason": None,
    }
    if not eligible:
        selection["reason"] = (
            "no checkpoint satisfies every must_pass gate; the run has no releasable "
            "checkpoint under the frozen selection rule"
        )
        return selection
    # tie_break 的 "earliest" 指命令行给出的顺序(约定为训练顺序),不解析文件名——
    # 从 checkpoint 名字里猜 step 与"从臂名前缀猜 adapter"是同一类错误。
    descending = rule["objective"] == "maximize"
    prefer_late = rule["tie_break"] == "latest_checkpoint"
    ranked = sorted(
        eligible,
        key=lambda entry: (
            -entry["composite_score"]["value"]
            if descending
            else entry["composite_score"]["value"],
            -entry["order_index"] if prefer_late else entry["order_index"],
        ),
    )
    winner = ranked[0]
    selection["selected"] = winner["label"]
    selection["selected_composite_score"] = winner["composite_score"]["value"]
    selection["reason"] = (
        f"{rule['objective']} {rule['quantity']} among {len(eligible)} checkpoints "
        f"passing all must_pass gates, tie_break={rule['tie_break']}"
    )
    return selection


def parse_checkpoint_argument(raw: str) -> tuple[str, Path]:
    """Accept ``path`` or ``label=path``; the label only names the report row."""
    if "=" in raw:
        label, _, path = raw.partition("=")
        label = label.strip()
        if not label:
            raise ValueError(f"checkpoint {raw!r} has an empty label")
        return label, Path(path.strip())
    path = Path(raw)
    return path.stem, path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score the sparse-history five arms on the heldout split and execute "
            "the config.gates checkpoint-selection contract."
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="directory holding samples.jsonl / parts/*.jsonl, or one .jsonl file",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="root the sample image paths resolve against (default: --dataset-root)",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="[LABEL=]PATH",
        help=(
            "adapter checkpoint to score; repeatable. Pass them in training order — "
            "the selection tie_break reads that order, not the file name."
        ),
    )
    parser.add_argument(
        "--include-identity-baseline",
        action="store_true",
        help=(
            "also score the freshly injected (zero-init, identity) adapter as a "
            "non-selectable row; SA_minus_RA must then equal frozen_selection_effect"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, default=None)
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument(
        "--gate-statistic",
        choices=("declared", "ci_conservative"),
        default="declared",
        help=(
            "declared: each must_pass entry uses the statistic it names (bare "
            "expressions mean @point). ci_conservative: force every gate onto the "
            "CI end that makes it harder to pass."
        ),
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="smoke runs only; truncates the heldout set and flags the report",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint and not args.include_identity_baseline:
        raise SystemExit("pass at least one --checkpoint (or --include-identity-baseline)")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise SystemExit("--bootstrap-confidence must lie strictly inside (0, 1)")

    import torch

    config = load_config(args.config)
    if not bool(config["training"].get("sparse_history", False)):
        raise SystemExit("this scorer only applies to training.sparse_history configs")
    adapter_type, adapter_options = adapter_settings(config)
    if adapter_type != "history_gated_kv":
        raise SystemExit("sparse_history requires adapter_type history_gated_kv")
    gates = validate_sparse_gates(config)
    compiled = compile_sparse_gates(gates)

    sample_files = resolve_sample_files(args.dataset_root)
    samples = load_sparse_samples(sample_files)
    groups = build_sparse_history_heldout_units(samples)
    if not groups:
        raise SystemExit("the corpus carries no heldout pair-groups to score")
    if args.max_groups:
        groups = dict(sorted(groups.items())[: args.max_groups])
    group_episode = {
        pair_group: samples[group[SPARSE_POSITIVE_SLOT]]["episode"]
        for pair_group, group in groups.items()
    }

    image_root = args.image_root or args.dataset_root
    started = time.time()
    heartbeat_state: dict[str, Any] = {}

    every = max(int(args.progress_every), 0)

    def progress(stage: str, position: int, total: int) -> None:
        if position != total and (every <= 0 or position % every):
            return
        heartbeat_state.update(
            {
                "stage": stage,
                "groups_done": position,
                "groups_total": total,
                "elapsed_seconds": round(time.time() - started, 1),
            }
        )
        print(json.dumps({"scoring_progress": heartbeat_state}), flush=True)
        if args.heartbeat is not None:
            args.heartbeat.parent.mkdir(parents=True, exist_ok=True)
            args.heartbeat.write_text(
                json.dumps(heartbeat_state, sort_keys=True) + "\n", encoding="utf-8"
            )

    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / config["policy_snapshot_manifest"]
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    model = runtime.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.eval()

    from causalcache.policy.history_gated_lora import (
        inject_history_gated_kv,
        load_history_gated_state_dict,
    )

    wrapped = inject_history_gated_kv(
        model,
        layer_count=adapter_options["layer_count"],
        rank=adapter_options["rank"],
        alpha=adapter_options["alpha"],
    )
    for lora in wrapped.values():
        lora.lora_a.requires_grad_(False)
        lora.lora_b.requires_grad_(False)

    cache = ScoreCache(args.score_cache)
    scorer = ArmScorer(
        runtime,
        image_root=image_root,
        merge_size=int(runtime.processor.image_processor.merge_size),
        torch=torch,
        cache=cache,
    )

    try:
        # 1) 冻结通道:所有 bypass 前向只算一次,与 checkpoint 无关。
        frozen = score_frozen_arms(scorer, samples, groups, progress=progress)

        # 2) 每个 checkpoint 只跑 active 前向。
        planned: list[tuple[str, Path | None, str]] = []
        if args.include_identity_baseline:
            planned.append((IDENTITY_ADAPTER_KEY, None, IDENTITY_ADAPTER_KEY))
        for raw in args.checkpoint:
            label, path = parse_checkpoint_argument(raw)
            planned.append((label, path, sha256_of(path)))
        # note (luojiaxuan): label 是报告里指认"选中哪个 checkpoint"的唯一句柄
        # (selection.selected 只写 label)。重名会让选点结果指向两行中的哪一行完全
        # 无法判定,所以在跑几小时前向之前先拒掉;--checkpoint LABEL=PATH 可消歧。
        labels = [entry[0] for entry in planned]
        duplicates = sorted({name for name in labels if labels.count(name) > 1})
        if duplicates:
            raise SystemExit(
                f"checkpoint labels must be unique, got duplicates {duplicates}; "
                "disambiguate with --checkpoint LABEL=PATH"
            )

        entries: list[dict[str, Any]] = []
        for order_index, (label, path, adapter_key) in enumerate(planned):
            if path is not None:
                load_history_gated_state_dict(
                    wrapped, torch.load(path, map_location="cpu")
                )
            active = score_active_arms(
                scorer,
                samples,
                groups,
                adapter_key=adapter_key,
                label=label,
                progress=progress,
            )
            per_quantity = group_quantities(
                samples,
                groups,
                frozen,
                active,
                derived_algebra=gates["derived_quantities"],
            )
            bootstrap = EpisodeClusterBootstrap(
                per_quantity,
                group_episode,
                replicates=args.bootstrap_replicates,
                confidence=args.bootstrap_confidence,
                seed=args.bootstrap_seed,
            )
            point = bootstrap.point_estimate()
            replicates = bootstrap.replicate_aggregates()
            reported = sorted(set(point) | set(gate_namespace(point)))
            intervals = bootstrap.intervals(replicates, reported)

            composite_expression = compiled["composite_score"]
            try:
                composite_point = evaluate_gate_expression(
                    composite_expression, gate_namespace(point)
                )
            except GateQuantityUnavailable as error:
                composite_point = None
                composite_reason = str(error)
            else:
                composite_reason = None
            composite_draws: list[float] = []
            for replicate in replicates:
                try:
                    composite_draws.append(
                        evaluate_gate_expression(
                            composite_expression, gate_namespace(replicate)
                        )
                    )
                except GateQuantityUnavailable:
                    continue
            composite_draws.sort()
            tail = (1.0 - args.bootstrap_confidence) / 2.0

            verdicts, all_passed = evaluate_gates(
                compiled["comparisons"],
                point,
                intervals,
                gate_statistic=args.gate_statistic,
            )
            entries.append(
                {
                    "label": label,
                    "order_index": order_index,
                    "path": None if path is None else str(path),
                    "adapter_key": adapter_key,
                    "checkpoint_sha256": None if path is None else adapter_key,
                    # identity 行只做流水线自检,不参与选点。
                    "selectable": path is not None,
                    "derived_quantities": {
                        name: {
                            "value": point.get(name),
                            "ci_low": intervals[name]["ci_low"],
                            "ci_high": intervals[name]["ci_high"],
                            "support_groups": bootstrap.support.get(name, {}).get(
                                "groups"
                            ),
                            "support_episodes": bootstrap.support.get(name, {}).get(
                                "episodes"
                            ),
                        }
                        for name in reported
                    },
                    "gates": verdicts,
                    "all_must_pass": all_passed,
                    "composite_score": {
                        "expression": composite_expression,
                        "value": composite_point,
                        "ci_low": percentile(composite_draws, tail),
                        "ci_high": percentile(composite_draws, 1.0 - tail),
                        "bootstrap_replicates_used": len(composite_draws),
                        "unavailable_reason": composite_reason,
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "checkpoint_scored": label,
                        "all_must_pass": all_passed,
                        "composite_score": composite_point,
                        "main_claim": point.get(gates["main_claim_quantity"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        selection = select_checkpoint(entries, compiled["selection_rule"])
        if args.max_groups:
            selection["reason"] = (
                f"SMOKE RUN (--max-groups {args.max_groups}): {selection['reason']}"
            )

        identity_consistency = None
        for entry in entries:
            if entry["adapter_key"] != IDENTITY_ADAPTER_KEY:
                continue
            main_claim = entry["derived_quantities"]["SA_minus_RA"]["value"]
            frozen_selection = entry["derived_quantities"]["frozen_selection_effect"][
                "value"
            ]
            identity_consistency = {
                "SA_minus_RA": main_claim,
                "frozen_selection_effect": frozen_selection,
                "abs_difference": (
                    None
                    if main_claim is None or frozen_selection is None
                    else abs(main_claim - frozen_selection)
                ),
                "note": (
                    "a zero-init adapter is the identity, so SA-RA must collapse onto "
                    "S0-R0; a large difference means the adapter is not actually "
                    "bypassed or the arms are mis-paired"
                ),
            }

        report = {
            "schema_version": REPORT_SCHEMA,
            "config_path": str(args.config),
            "config_sha256": sha256_of(args.config),
            "gate_schema": gates["gate_schema"],
            "main_claim_quantity": gates["main_claim_quantity"],
            "reference_arm_id": gates["reference_arm_id"],
            "dataset_root": str(args.dataset_root),
            "image_root": str(image_root),
            "sample_files": [
                {"path": str(path), "sha256": sha256_of(path)}
                for path in sample_files
            ],
            "sample_count": len(samples),
            "heldout": {
                "pair_groups": len(groups),
                "episodes": len(set(group_episode.values())),
                "group_limit_applied": args.max_groups or None,
            },
            "smoke_run": bool(args.max_groups),
            "gate_statistic": args.gate_statistic,
            "bootstrap": {
                "method": "percentile",
                "cluster_unit": BOOTSTRAP_CLUSTER_UNIT,
                "clusters": len(set(group_episode.values())),
                "replicates": args.bootstrap_replicates,
                "confidence": args.bootstrap_confidence,
                "seed": args.bootstrap_seed,
                "rationale": (
                    "pair-groups inside one episode share instruction, UI and the "
                    "policy's task-level bias; resampling groups i.i.d. would shrink "
                    "every interval by roughly sqrt(groups per episode)"
                ),
            },
            "forward_passes": scorer.forwards,
            "cache_hits": scorer.cache_hits,
            "elapsed_seconds": round(time.time() - started, 1),
            "checkpoints": entries,
            "identity_consistency": identity_consistency,
            "selection": selection,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "gate_report": str(args.output),
                    "selected_checkpoint": selection["selected"],
                    "eligible_checkpoints": selection["eligible"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        cache.close()


if __name__ == "__main__":
    main()
