#!/usr/bin/env python3
"""Argmax 一致率探针:换成 oracle 集合之后,模型对目标动作的 **argmax** 翻了多少?

# note (luojiaxuan): 到目前为止全部证据都是 teacher-forced **平均 log-prob**。实测
# oracle 单替换相对 Recent 的 cross-fit 增益是 +0.0077/token(B=4,60 组 dev),目标
# 动作平均 42 token,折合每动作 +0.32 nats、几率比 1.37x。但 log-prob 改善**只有在
# 翻转 argmax 时才会改变闭环行为**,而这两者之间没有可靠换算率:同样的 +0.32 nats,
# 既可能来自"把一个本来就排第一的 token 推得更高"(闭环行为完全不变),也可能来自
# "把第二名顶成第一名"(闭环行为改变)。本脚本直接量后者。
#
# 判据:若 oracle 在 argmax 层面几乎不翻,那 selector 再好也没有闭环价值,整条线可以
# 停;若在 **oracle 增益最大的那一层**(top decile)显著翻,则值得继续,并且 selector
# 的目标应当改成"识别出这一层",而不是"最大化平均 log-prob"。
#
# ---------------------------------------------------------------------------
# 与既有打分逐位同源
# ---------------------------------------------------------------------------
# 复用 ``probe_oracle_headroom`` 的 ``DecisionPoint`` / ``FrozenSetScorer`` /
# ``SetScoreCache``,以及 ``train_success_sft_lora.encode_sample``,渲染格式固定
# ``official_style_sparse_multiturn``(已冻结,见 docs/renderer_freeze_v1.md)。
# ``target_token_stats`` 是 ``probe_oracle_headroom.target_token_logprobs`` 的逐行副本,
# 只多返回一个 ``argmax == target`` 的布尔向量 —— 两者共用**同一次前向、同一个
# log_softmax 张量**,所以"agree 与 logprob 来自同一分布"是结构性的而不是约定。
# 启动时的 ``verify_agreement_parity`` 会把三条路径(本副本 / target_token_logprobs /
# mean_target_logprob)的均值实测比对,漂了就 fail-closed。
#
# ---------------------------------------------------------------------------
# oracle 集合的来源:cross-fit,不是 in-sample
# ---------------------------------------------------------------------------
# 本脚本**不做任何集合搜索**,oracle 集合一律从既有产物里读:
#
#   --oracle-source labels             读 generate_replacement_labels 产出的 labels.jsonl
#   --oracle-source replacement-cache  读既有 per-set 分数缓存,用
#                                      generate_replacement_labels.crossfit_gain 归约
#
# 两种来源都只取 ``k1.directions.{even,odd}.steps``,即**两折各自选出的集合**,评估时
# 两折各算一次再平均。``in_sample_best_steps`` 是同一批 token 上取 max 的结果,会高估,
# 所以它只作为一个显式标注的**乐观上界**臂出现(``*_insample``),绝不进主结论。
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
import time
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_sparse_multiturn import SPARSE_MULTITURN_PROMPT_FORMAT
from scripts.budget_replacement_sweep import cluster_bootstrap
from scripts.generate_replacement_labels import crossfit_gain
from scripts.probe_oracle_headroom import (
    BEAM_WIDTH,
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
    target_token_logprobs,
)
from scripts.train_success_sft_lora import encode_sample, mean_target_logprob

REPORT_SCHEMA = "causalcache.argmax_agreement_probe.v1"
SHARD_REPORT_SCHEMA = "causalcache.argmax_agreement_probe_shard.v1"
FROZEN_FORMAT = SPARSE_MULTITURN_PROMPT_FORMAT

# 前 k 个目标 token 是否全中。动作类型(``click`` / ``scroll`` / ``type`` ...)通常落在
# 开头几个 token,所以前缀一致率比整段一致率更接近"闭环里会不会走岔路"。
FIRST_K: tuple[int, ...] = (1, 4, 8)
CROSSFIT_DIRECTIONS: tuple[tuple[str, str], ...] = (("even", "odd"), ("odd", "even"))
DECILE = 0.10

# note (luojiaxuan): 目标串长这样:
#     Action: <自然语言描述>\n<tool_call>\n{"name": ..., "arguments": {"action": "click",
#     "coordinate": [136, 825]}}\n</tool_call>
# 也就是说**决策不在开头**:开头 9 个 token 是自由文本描述("Action" 这个 token 在
# Recent 下 100% argmax 命中,first-1 一致率恒等于 1,没有可改进空间),真正决定闭环
# 行为的是末尾 tool_call 里的 ``action`` 类型和 ``coordinate``。所以除了整段与前缀
# 一致率,再按语义段各报一份;``action_type`` 是唯一"翻了就一定换分支"的量。
SPAN_NAMES: tuple[str, ...] = ("description", "tool_call", "action_type", "coordinate")
TOOL_CALL_MARKER = "<tool_call>"
ACTION_KEY = '"action": "'
COORDINATE_KEY = '"coordinate": '


# ---------------------------------------------------------------------------
# 前向:同一次前向同时拿 logprob 与 argmax
# ---------------------------------------------------------------------------
def target_token_stats(model: Any, encoded: dict[str, Any], *, torch: Any) -> Any:
    """Per-token target log-probs **and** per-token ``argmax == target`` hits.

    # note (luojiaxuan): 与 probe_oracle_headroom.target_token_logprobs 逐行同构,只多
    # 一行 argmax。刻意在 ``log_probs``(而不是 ``logits``)上取 argmax:两者数学上等价
    # (log_softmax 单调),但写成同一个张量就杜绝了"gather 用的是 A、argmax 用的是 B"
    # 这类日后重构时的漂移。
    """
    labels = encoded.pop("labels")
    targets_full = labels[:, 1:]
    token_count = int((targets_full != -100).sum())
    try:
        outputs = model(**encoded, logits_to_keep=token_count + 1)
    except TypeError:
        outputs = model(**encoded)
    logits = outputs.logits[:, -(token_count + 1) : -1].float()
    targets = targets_full[:, -token_count:]
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)[0]
    predicted = log_probs.argmax(dim=-1)[0]
    return gathered, predicted == targets[0]


# ---------------------------------------------------------------------------
# 纯函数:记录与指标(无 GPU,可单测)
# ---------------------------------------------------------------------------
def target_char_spans(target_text: str) -> dict[str, tuple[int, int]]:
    """Character ranges of the semantically distinct pieces of one target action."""
    spans: dict[str, tuple[int, int]] = {}
    call = target_text.find(TOOL_CALL_MARKER)
    if call >= 0:
        if call > 0:
            spans["description"] = (0, call)
        spans["tool_call"] = (call, len(target_text))
    start = target_text.find(ACTION_KEY)
    if start >= 0:
        value = start + len(ACTION_KEY)
        end = target_text.find('"', value)
        if end > value:
            spans["action_type"] = (value, end)
    start = target_text.find(COORDINATE_KEY)
    if start >= 0:
        value = start + len(COORDINATE_KEY)
        end = target_text.find("]", value)
        if end > value:
            spans["coordinate"] = (value, end + 1)
    return spans


def target_token_spans(
    tokenizer: Any, target_text: str, token_count: int
) -> dict[str, tuple[int, int]]:
    """Map the char spans onto ``[start, end)`` target-token index ranges.

    # note (luojiaxuan): 用 fast tokenizer 的 ``offset_mapping`` 而不是"切前缀再分别
    # 编码"—— 后者在边界上不保证与整串编码一致,一旦错位就会把 action 类型的命中算到
    # 别的 token 上。token 数与 ``encode_sample`` 那一次编码不一致时整条 span 直接放弃
    # (返回空 dict),宁可少一个指标也不产出对不上的 span。
    """
    try:
        encoding = tokenizer(
            target_text, add_special_tokens=False, return_offsets_mapping=True
        )
    except Exception:  # noqa: BLE001 - slow tokenizers have no offsets; degrade quietly
        return {}
    offsets = encoding.get("offset_mapping")
    if offsets is None or len(encoding["input_ids"]) != token_count:
        return {}
    spans: dict[str, tuple[int, int]] = {}
    for name, (low, high) in target_char_spans(target_text).items():
        members = [
            index
            for index, (begin, end) in enumerate(offsets)
            if begin < high and end > low
        ]
        if members:
            spans[name] = (members[0], members[-1] + 1)
    return spans


def agreement_record(
    values: list[float],
    hits: list[bool],
    *,
    total_tokens: int,
    spans: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Fold one scored sequence into the six-ish scalars the cache stores.

    # note (luojiaxuan): 不存 per-token 向量。``lead_hits``(开头连续命中的长度)一个
    # 整数就能还原任意 k 的 first-k 一致率,所以未来想加 first-2 / first-16 不必重跑
    # 前向。奇偶两折的命中数是为了给 argmax 也配一个 held-out 版本:集合是在一折的
    # logprob 上选出来的,那一折上的 argmax 命中同样带着选择偏差。
    """
    n_tok = len(values)
    if n_tok == 0:
        raise ValueError("agreement_record needs at least one target token")
    if len(hits) != n_tok:
        raise ValueError(f"{len(hits)} hits for {n_tok} target tokens")
    lead = 0
    for hit in hits:
        if not hit:
            break
        lead += 1
    even = [index for index in range(n_tok) if index % 2 == 0]
    odd = [index for index in range(n_tok) if index % 2 == 1]
    record = {
        "mean": math.fsum(values) / n_tok,
        "n_tok": n_tok,
        "n_hit": sum(1 for hit in hits if hit),
        "lead_hits": lead,
        "sum_even": math.fsum(values[index] for index in even),
        "n_even": len(even),
        "hit_even": sum(1 for index in even if hits[index]),
        "sum_odd": math.fsum(values[index] for index in odd),
        "n_odd": len(odd),
        "hit_odd": sum(1 for index in odd if hits[index]),
        "prompt_tokens": total_tokens - n_tok,
        "total_tokens": total_tokens,
    }
    for name, (start, end) in (spans or {}).items():
        if not 0 <= start < end <= n_tok:
            raise ValueError(f"span {name!r}={start, end} escapes {n_tok} target tokens")
        record[f"n_{name}"] = end - start
        record[f"hit_{name}"] = sum(1 for index in range(start, end) if hits[index])
    return record


def metrics_from_record(record: dict[str, Any]) -> dict[str, float]:
    """The four families the report is built on, all in [0, 1]."""
    n_tok = int(record["n_tok"])
    metrics = {
        "token_agree": record["n_hit"] / n_tok,
        "all_agree": 1.0 if record["n_hit"] == n_tok else 0.0,
    }
    for k in FIRST_K:
        # note (luojiaxuan): 目标短于 k 时 first-k 退化成 all_agree(``hits[:k].all()``
        # 的语义),两个臂退化方式相同所以 delta 仍然可读;报告里另外统计
        # ``n_tok < 8`` 的组数,免得 first8 被当成一个与 first4 独立的量。
        metrics[f"first{k}_agree"] = (
            1.0 if int(record["lead_hits"]) >= min(k, n_tok) else 0.0
        )
    for name in SPAN_NAMES:
        count = record.get(f"n_{name}")
        if not count:
            continue
        hit = int(record[f"hit_{name}"])
        metrics[f"{name}_agree"] = hit / int(count)
        metrics[f"{name}_all_agree"] = 1.0 if hit == int(count) else 0.0
    return metrics


def fold_token_agree(record: dict[str, Any], fold: str) -> float | None:
    count = int(record[f"n_{fold}"])
    return None if not count else record[f"hit_{fold}"] / count


def fold_logprob_mean(record: dict[str, Any], fold: str) -> float | None:
    count = int(record[f"n_{fold}"])
    return None if not count else record[f"sum_{fold}"] / count


# ---------------------------------------------------------------------------
# oracle 集合的来源
# ---------------------------------------------------------------------------
def _tuple_steps(steps: Any) -> tuple[int, ...]:
    return tuple(sorted(int(step) for step in steps))


def arms_from_k1_block(block: dict[str, Any], recent: tuple[int, ...]) -> dict[str, Any]:
    """Turn one ``k1`` block into the arms this probe scores.

    ``crossfit`` 是主结论用的两折集合;``in_sample`` 只作乐观上界。刻意**不**提供
    "把 in_sample 当 oracle"的路径 —— 上游若真想比,读 ``*_insample`` 那一族即可。
    """
    directions = block["directions"]
    return {
        "recent": recent,
        "crossfit": {
            select: _tuple_steps(directions[select]["steps"])
            for select, _ in CROSSFIT_DIRECTIONS
        },
        "in_sample": _tuple_steps(block["in_sample_best_steps"]),
        "crossfit_gain": float(block["crossfit_gain"]),
        "in_sample_gain": float(block["in_sample_gain"]),
        "pool_size": int(block["pool_size"]),
    }


def load_oracle_sets_from_labels(
    paths: list[Path], budgets: list[int]
) -> dict[str, dict[int, dict[str, Any]]]:
    """Read ``budgets[B].k1.directions.{even,odd}.steps`` out of labels.jsonl shards."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for position, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # 正在写的最后一行允许是残行;中间坏行是真损坏。
                    if line.endswith("\n"):
                        raise
                    continue
                if record.get("format") != FROZEN_FORMAT:
                    raise ValueError(
                        f"{path}:{position} was rendered as {record.get('format')!r}, "
                        f"but this probe is pinned to {FROZEN_FORMAT!r}"
                    )
                group = record["pair_group"]
                for budget in budgets:
                    block = record["budgets"].get(str(budget))
                    if block is None or block.get("k1") is None:
                        continue
                    out.setdefault(group, {})[budget] = arms_from_k1_block(
                        block["k1"], _tuple_steps(block["recent_steps"])
                    )
    return out


def load_score_cache(paths: list[Path]) -> dict[str, dict[tuple[int, ...], dict]]:
    scores: dict[str, dict[tuple[int, ...], dict]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if line.endswith("\n"):
                        raise
                    continue
                key = record.get("cache_key")
                if not key:
                    continue
                group, prompt_format, raw = key.split("|")
                if prompt_format != FROZEN_FORMAT:
                    continue
                steps = () if raw in ("empty", "") else tuple(int(x) for x in raw.split("-"))
                scores.setdefault(group, {})[steps] = record["value"]
    return scores


def load_oracle_sets_from_cache(
    paths: list[Path], budgets: list[int]
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Reduce an existing per-set score cache into the same arms the labels carry.

    # note (luojiaxuan): 这条路径**不产生任何新前向** —— 它只是把已经算完的分数按
    # generate_replacement_labels.crossfit_gain(直接 import,不重写)再归约一次。之所以
    # 需要它:B=4 的标签根本没生成(标签作业只跑了 --budgets 1,2),而 +0.0077/token 这个
    # 触发本次测量的数字恰恰是 B=4 的。动作族与标签完全一致:
    #     F_B = {(R_B \\ {r_i}) ∪ {o_j}},即 k1_drop_positions="all"。
    # 池不完整的 (组, 预算) 直接丢弃并计数,不做部分池上的 argmax(那会悄悄换掉口径)。
    """
    scores = load_score_cache(paths)
    out: dict[str, dict[int, dict[str, Any]]] = {}
    skipped: dict[str, int] = {}
    for group, pool_all in scores.items():
        current = int(group.split(":")[1])
        for budget in budgets:
            recent = tuple(range(current - budget, current))
            olds = [step for step in range(1, current - budget)]
            anchor = pool_all.get(recent)
            if anchor is None or not olds:
                skipped[f"b{budget}_no_anchor_or_no_old"] = (
                    skipped.get(f"b{budget}_no_anchor_or_no_old", 0) + 1
                )
                continue
            pool: dict[tuple[int, ...], dict] = {}
            for dropped in recent:
                kept = [step for step in recent if step != dropped]
                for old in olds:
                    steps = tuple(sorted(kept + [old]))
                    if steps in pool_all:
                        pool[steps] = pool_all[steps]
            if len(pool) != budget * len(olds):
                skipped[f"b{budget}_incomplete_pool"] = (
                    skipped.get(f"b{budget}_incomplete_pool", 0) + 1
                )
                continue
            block = crossfit_gain(pool, anchor)
            if block is None:
                continue
            out.setdefault(group, {})[budget] = arms_from_k1_block(block, recent)
    return out, {"skipped": skipped, "groups_in_cache": len(scores)}


# ---------------------------------------------------------------------------
# 打分器
# ---------------------------------------------------------------------------
class ArgmaxSetScorer(FrozenSetScorer):
    """``FrozenSetScorer`` plus the argmax hits from the very same forward pass."""

    span_failures = 0

    def verify_agreement_parity(
        self, point: DecisionPoint, steps: tuple[int, ...]
    ) -> dict[str, Any]:
        """Assert this file's per-token path still equals the two frozen references.

        # note (luojiaxuan): target_token_stats 是 target_token_logprobs 的副本,而
        # target_token_logprobs 本身又是 mean_target_logprob 的副本。三份副本里任何一份
        # 漂了,本探针的 delta 就不再能和既有 logprob 表摆在同一张图上。所以每次跑实测
        # 一遍三者的均值,把差值写进报告,不靠"我抄的时候很小心"。
        """
        runtime = self._prepare()
        sample = point.render(steps, FROZEN_FORMAT)
        encodings = [
            encode_sample(runtime, sample, dataset_root=self.image_root, torch=self.torch)
            for _ in range(3)
        ]
        with self.torch.no_grad():
            vector, hits = target_token_stats(
                runtime.model, encodings[0], torch=self.torch
            )
            mine = float(vector.detach().mean())
            frozen_vector = float(
                target_token_logprobs(
                    runtime.model, encodings[1], torch=self.torch
                ).detach().mean()
            )
            frozen_mean = float(
                mean_target_logprob(
                    runtime.model, encodings[2], torch=self.torch
                ).detach()
            )
        deltas = {
            "vs_target_token_logprobs": abs(mine - frozen_vector),
            "vs_mean_target_logprob": abs(mine - frozen_mean),
        }
        if max(deltas.values()) >= 1e-4:
            raise SystemExit(
                "target_token_stats has drifted from the frozen scoring path "
                f"({mine!r} vs {frozen_vector!r} / {frozen_mean!r}); the argmax probe "
                "would no longer be measuring the same forward as the log-prob tables"
            )
        self.mean_parity = {
            "sample_id": sample["sample_id"],
            "per_token_mean": mine,
            "target_token_logprobs_mean": frozen_vector,
            "mean_target_logprob": frozen_mean,
            "abs_deltas": deltas,
            "hits_shape": int(hits.shape[0]),
        }
        return self.mean_parity

    def agree(self, point: DecisionPoint, steps: tuple[int, ...]) -> dict[str, Any]:
        cache_key = f"{point.pair_group}|{FROZEN_FORMAT}|{set_key(steps)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.args.require_cached:
            raise SystemExit(
                f"--require-cached is set but {cache_key!r} is missing from the "
                f"agreement cache {self.cache.path}; some shard did not finish, or "
                "wrote to a different cache path"
            )
        runtime = self._prepare()
        sample = point.render(steps, FROZEN_FORMAT)
        encoded = encode_sample(
            runtime, sample, dataset_root=self.image_root, torch=self.torch
        )
        if encoded is None:
            raise ValueError(
                f"sample {sample['sample_id']!r} produced an empty target encoding; "
                "the scored denominator must stay reconciled"
            )
        total_tokens = int(encoded["input_ids"].shape[1])
        with self.torch.no_grad():
            vector, hits = target_token_stats(runtime.model, encoded, torch=self.torch)
            values = [float(value) for value in vector.detach().cpu()]
            flags = [bool(flag) for flag in hits.detach().cpu()]
        spans = target_token_spans(
            runtime.processor.tokenizer, sample["target_text"], len(values)
        )
        if not spans:
            self.span_failures += 1
        record = agreement_record(
            values, flags, total_tokens=total_tokens, spans=spans
        )
        self.max_prompt_tokens = max(self.max_prompt_tokens, record["prompt_tokens"])
        self.max_total_tokens = max(self.max_total_tokens, total_tokens)
        self.forwards += 1
        self.cache.put(cache_key, record)
        return record


# ---------------------------------------------------------------------------
# 归约
# ---------------------------------------------------------------------------
def group_quantities(
    arms: dict[str, Any], records: dict[tuple[int, ...], dict[str, Any]]
) -> dict[str, float]:
    """All per-(group, budget) quantities, from already-scored records.

    主结论一族(``delta_*``)用的是 **cross-fit 两折集合各算一次再平均**;
    ``delta_*_insample`` 是同一批 token 上取 max 的乐观上界,单独命名以免混用。
    """
    recent = records[arms["recent"]]
    recent_metrics = metrics_from_record(recent)
    quantities: dict[str, float] = {
        f"recent_{name}": value for name, value in recent_metrics.items()
    }
    quantities["recent_mean_logprob"] = recent["mean"]

    crossfit_metrics: dict[str, list[float]] = {}
    heldout_oracle: list[float] = []
    heldout_recent: list[float] = []
    oracle_logprob: list[float] = []
    for select, evaluate in CROSSFIT_DIRECTIONS:
        chosen = records[arms["crossfit"][select]]
        for name, value in metrics_from_record(chosen).items():
            crossfit_metrics.setdefault(name, []).append(value)
        oracle_logprob.append(chosen["mean"])
        # note (luojiaxuan): 集合是在 select 折的 logprob 上挑出来的,所以那一折上的
        # argmax 命中同样带着选择偏差。held-out 版本只看**没参与选择**的那一折,是
        # cross-fit logprob 增益在 argmax 上的严格对应物。
        chosen_heldout = fold_token_agree(chosen, evaluate)
        recent_heldout = fold_token_agree(recent, evaluate)
        if chosen_heldout is not None and recent_heldout is not None:
            heldout_oracle.append(chosen_heldout)
            heldout_recent.append(recent_heldout)

    for name, values in crossfit_metrics.items():
        if name not in recent_metrics:
            continue
        mean = math.fsum(values) / len(values)
        quantities[f"oracle_{name}"] = mean
        quantities[f"delta_{name}"] = mean - recent_metrics[name]
    quantities["oracle_mean_logprob"] = math.fsum(oracle_logprob) / len(oracle_logprob)
    quantities["delta_mean_logprob"] = (
        quantities["oracle_mean_logprob"] - recent["mean"]
    )
    if heldout_oracle:
        quantities["oracle_token_agree_heldout"] = math.fsum(heldout_oracle) / len(
            heldout_oracle
        )
        quantities["recent_token_agree_heldout"] = math.fsum(heldout_recent) / len(
            heldout_recent
        )
        quantities["delta_token_agree_heldout"] = (
            quantities["oracle_token_agree_heldout"]
            - quantities["recent_token_agree_heldout"]
        )
        # cross-fit 的 logprob 增益(与标签里的 crossfit_gain 同定义),用来对账。
        held = []
        for select, evaluate in CROSSFIT_DIRECTIONS:
            chosen = records[arms["crossfit"][select]]
            chosen_value = fold_logprob_mean(chosen, evaluate)
            recent_value = fold_logprob_mean(recent, evaluate)
            if chosen_value is not None and recent_value is not None:
                held.append(chosen_value - recent_value)
        if held:
            quantities["delta_crossfit_logprob"] = math.fsum(held) / len(held)

    in_sample = records[arms["in_sample"]]
    for name, value in metrics_from_record(in_sample).items():
        if name not in recent_metrics:
            continue
        quantities[f"oracle_{name}_insample"] = value
        quantities[f"delta_{name}_insample"] = value - recent_metrics[name]
    quantities["delta_mean_logprob_insample"] = in_sample["mean"] - recent["mean"]
    return quantities


_CORE: tuple[str, ...] = (
    "token_agree",
    "all_agree",
    "first1_agree",
    "first4_agree",
    "first8_agree",
    *[f"{span}_agree" for span in SPAN_NAMES],
    *[f"{span}_all_agree" for span in SPAN_NAMES],
)
DELTA_NAMES: tuple[str, ...] = (
    *[f"delta_{name}" for name in _CORE],
    "delta_token_agree_heldout",
    "delta_mean_logprob",
    "delta_crossfit_logprob",
)
BASELINE_NAMES: tuple[str, ...] = (
    *[f"recent_{name}" for name in _CORE],
    "recent_mean_logprob",
    *[f"oracle_{name}" for name in _CORE],
)
INSAMPLE_NAMES: tuple[str, ...] = (
    *[f"delta_{name}_insample" for name in _CORE],
    "delta_mean_logprob_insample",
)


def split_strata(ranked: list[tuple[float, str]]) -> dict[str, list[str]]:
    """Bottom decile / middle / top decile by the group's oracle log-prob gain.

    # note (luojiaxuan): 增益分布极度重尾(dev 实测 p50=+0.0070、p90=+0.054、
    # p10=-0.059/token),所以总体均值几乎必然贴着零,而"top decile 会不会翻 argmax"
    # 才是判据。**分层量与被评估量取自同一批 token**,top decile 因此吃了 winner's
    # curse:它的 crossfit_gain 是被高估的。报告里显式写明,读法是"条件于增益很大时
    # argmax 翻不翻",不是"能提前挑出这一层"。
    """
    ranked = sorted(ranked)
    total = len(ranked)
    edge = max(1, int(round(total * DECILE)))
    if 2 * edge >= total:
        return {"all": [group for _, group in ranked]}
    return {
        "bottom_decile": [group for _, group in ranked[:edge]],
        "middle": [group for _, group in ranked[edge : total - edge]],
        "top_decile": [group for _, group in ranked[total - edge :]],
    }


def reduce_stratum(
    groups: list[str],
    per_group: dict[str, dict[str, float]],
    episode: dict[str, str],
    names: tuple[str, ...],
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for name in names:
        pairs = [
            (episode[group], per_group[group][name])
            for group in groups
            if name in per_group[group]
        ]
        result = cluster_bootstrap(pairs)
        if result is not None:
            block[name] = result
    return block


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.add_argument(
        "--oracle-source",
        choices=("labels", "replacement-cache"),
        required=True,
        help=(
            "where the cross-fit oracle sets come from; this probe never searches "
            "for sets itself"
        ),
    )
    parser.add_argument(
        "--oracle-paths",
        required=True,
        help="glob for the labels.jsonl shards or the per-set score-cache shards",
    )
    parser.add_argument("--budgets", default="1,2,4")
    parser.add_argument(
        "--reference-cache",
        default=None,
        help=(
            "glob for an existing per-set log-prob cache; the reduction reconciles "
            "this probe's mean_logprob against it set by set"
        ),
    )
    return parser


def expand(pattern: str) -> list[Path]:
    paths = sorted(Path(match) for match in globlib.glob(pattern))
    if not paths:
        raise SystemExit(f"pattern {pattern!r} matched no file")
    return paths


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit(
            f"--shard-index {args.shard_index} must lie inside "
            f"[0, --shard-count={args.shard_count})"
        )
    if args.shard_count > 1 and args.score_cache is None:
        raise SystemExit("--shard-count > 1 requires --score-cache")
    if args.shard_count > 1 and not args.skip_reduction:
        raise SystemExit(
            "a sharded process only holds 1/--shard-count of the sample; running the "
            "bootstrap on that slice would silently shrink the denominator. Pass "
            "--skip-reduction on every shard, then run one unsharded reduction pass."
        )
    if args.require_cached and args.score_cache is None:
        raise SystemExit("--require-cached is meaningless without --score-cache")
    budgets = [int(part) for part in args.budgets.split(",") if part.strip()]
    if not budgets or any(budget < 1 for budget in budgets):
        raise SystemExit(f"--budgets must be positive integers, got {args.budgets!r}")

    config = load_config(args.config)
    sample_files = resolve_sample_files(args.dataset_root)
    samples = load_sparse_samples(sample_files)

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
    rows = group_dev_rows(samples, allowed)
    if not rows:
        raise SystemExit("episode filter removed every group; wrong corpus?")

    oracle_paths = expand(args.oracle_paths)
    source_stats: dict[str, Any] = {}
    if args.oracle_source == "labels":
        oracle_sets = load_oracle_sets_from_labels(oracle_paths, budgets)
    else:
        oracle_sets, source_stats = load_oracle_sets_from_cache(oracle_paths, budgets)

    # note (luojiaxuan): 组名单由 **oracle 来源** 决定并与语料求交,不用
    # stratified_sample 重新抽 —— 那会让"打的分"和"集合的出处"来自两个可能不同的子样本。
    outside = sorted(set(oracle_sets) - set(rows))
    reduction_groups = sorted(set(oracle_sets) & set(rows))
    if not reduction_groups:
        raise SystemExit(
            "the oracle source and the episode filter share no pair-group; the labels "
            "were probably produced on a different split"
        )
    per_budget_groups = {
        budget: [g for g in reduction_groups if budget in oracle_sets[g]]
        for budget in budgets
    }
    plan = {
        "episode_filter_key": args.episode_filter_key,
        "oracle_source": args.oracle_source,
        "oracle_paths": [str(path) for path in oracle_paths],
        "groups_in_oracle_source": len(oracle_sets),
        "groups_outside_episode_filter": len(outside),
        "groups_used": len(reduction_groups),
        "episodes_used": len({rows[g]["N0"]["episode"] for g in reduction_groups}),
        "groups_per_budget": {str(b): len(v) for b, v in per_budget_groups.items()},
        "source_stats": source_stats,
    }
    print(json.dumps({"argmax_probe_plan": plan}, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    import torch

    chosen = list(reduction_groups)
    if args.shard_count > 1:
        chosen = chosen[args.shard_index :: args.shard_count]
        if not chosen:
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
        "probe": REPORT_SCHEMA,
        # note (luojiaxuan): span 计数是后加的字段,旧缓存里没有。把它写进 fingerprint,
        # 让旧缓存直接 fail-closed,而不是安静地少掉一半指标。
        "record_fields": "v2_target_spans",
        "config_sha256": sha256_of(args.config),
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "image_root": str(image_root),
        "annotations": str(args.annotations),
        "format": FROZEN_FORMAT,
        "beam_width": BEAM_WIDTH,
        "adapter": "none",
    }
    cache = SetScoreCache(
        args.score_cache,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fingerprint=cache_fingerprint,
    )
    scorer = ArgmaxSetScorer(
        args=args, config=config, image_root=image_root, cache=cache, torch=torch
    )

    started = time.time()
    every = max(int(args.progress_every), 0)

    def progress(position: int, total: int) -> None:
        if position != total and (every <= 0 or position % every):
            return
        state = {
            "stage": "scoring_argmax",
            "groups_done": position,
            "groups_total": total,
            "forwards": scorer.forwards,
            "cache_hits": scorer.cache_hits,
            "max_prompt_tokens": scorer.max_prompt_tokens,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        print(json.dumps({"argmax_progress": state}), flush=True)
        if heartbeat_path is not None:
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_path.write_text(
                json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
            )

    annotation_cache: dict[str, list[dict]] = {}
    per_group: dict[str, dict[int, dict[str, float]]] = {}
    per_group_context: dict[str, dict[int, dict[str, Any]]] = {}

    def run_group(pair_group: str) -> None:
        point = DecisionPoint(
            rows[pair_group],
            annotations=args.annotations,
            annotation_cache=annotation_cache,
            image_root=image_root,
        )
        for budget in budgets:
            arms = oracle_sets[pair_group].get(budget)
            if arms is None:
                continue
            if arms["recent"] != point.recent(budget):
                raise SystemExit(
                    f"{pair_group}: the oracle source anchors budget {budget} at "
                    f"{arms['recent']} but the corpus decision step gives "
                    f"{point.recent(budget)}; the two artefacts disagree about the "
                    "decision point"
                )
            wanted = {arms["recent"], arms["in_sample"], *arms["crossfit"].values()}
            for steps in sorted(wanted):
                if len(steps) != budget:
                    raise SystemExit(
                        f"{pair_group}: arm {steps} has size {len(steps)} under budget "
                        f"{budget}; the pixel budget would not be matched"
                    )
            records = {steps: scorer.agree(point, steps) for steps in sorted(wanted)}
            per_group.setdefault(pair_group, {})[budget] = group_quantities(arms, records)
            per_group_context.setdefault(pair_group, {})[budget] = {
                "episode": point.episode,
                "n_candidates": point.n_candidates,
                "n_tok": int(records[arms["recent"]]["n_tok"]),
                "source_crossfit_gain": arms["crossfit_gain"],
                "source_in_sample_gain": arms["in_sample_gain"],
                "pool_size": arms["pool_size"],
                "recent_steps": list(arms["recent"]),
                "crossfit_steps": {k: list(v) for k, v in arms["crossfit"].items()},
                "in_sample_steps": list(arms["in_sample"]),
                "crossfit_equals_in_sample": all(
                    v == arms["in_sample"] for v in arms["crossfit"].values()
                ),
            }

    try:
        if not args.require_cached:
            probe_point = DecisionPoint(
                rows[chosen[0]],
                annotations=args.annotations,
                annotation_cache=annotation_cache,
                image_root=image_root,
            )
            print(
                json.dumps(
                    {"encoder_parity": scorer.verify_agreement_parity(
                        probe_point, probe_point.recent(1)
                    )}
                ),
                flush=True,
            )
        for position, pair_group in enumerate(chosen, start=1):
            run_group(pair_group)
            progress(position, len(chosen))
        if not args.skip_reduction:
            for pair_group in reduction_groups:
                if pair_group not in per_group:
                    run_group(pair_group)
    finally:
        cache.close()

    totals = [
        int(entry["total_tokens"])
        for entry in cache.entries.values()
        if isinstance(entry, dict) and "total_tokens" in entry
    ]
    provenance = {
        "device": args.device,
        "computed_forwards": scorer.forwards > 0,
        "device_name": None,
        "max_total_tokens": max(totals, default=scorer.max_total_tokens),
        "sequences_measured": len(totals),
        "encoder_parity": scorer.mean_parity,
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
        "format": FROZEN_FORMAT,
        "budgets": budgets,
        "adapter": "none (frozen policy; no injection, no checkpoint load)",
        "oracle_arm": (
            "cross-fit: the size-B single-replacement set argmax-ed on one token fold, "
            "scored on both folds, averaged over the two directions. "
            "in_sample_best_steps is only reported as the *_insample optimistic bound."
        ),
        "plan": plan,
        "cache_fingerprint": cache_fingerprint,
        "cache": cache.stats(),
        "provenance": provenance,
        "forward_passes": scorer.forwards,
        "cache_hits": scorer.cache_hits,
        "span_failures": scorer.span_failures,
        "span_names": list(SPAN_NAMES),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    if args.skip_reduction:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": SHARD_REPORT_SCHEMA,
                    "shard": f"{args.shard_index}/{args.shard_count}",
                    "pair_groups_scored": len(per_group),
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
                    "pair_groups": len(per_group),
                    "forwards": scorer.forwards,
                    "output": str(output_path),
                }
            ),
            flush=True,
        )
        return

    # ---- 归约 ----
    reconciliation: dict[str, Any] = {"checked": 0, "max_abs_delta": None}
    if args.reference_cache:
        reference = load_score_cache(expand(args.reference_cache))
        worst = 0.0
        checked = 0
        for cache_key, value in cache.entries.items():
            if not isinstance(value, dict) or "mean" not in value:
                continue
            group, _, raw = cache_key.split("|")
            steps = () if raw in ("empty", "") else tuple(int(x) for x in raw.split("-"))
            other = reference.get(group, {}).get(steps)
            if other is None:
                continue
            checked += 1
            worst = max(worst, abs(float(other["mean"]) - float(value["mean"])))
        reconciliation = {
            "reference_cache": args.reference_cache,
            "checked": checked,
            "max_abs_delta": worst if checked else None,
        }

    budget_blocks: dict[str, Any] = {}
    for budget in budgets:
        groups = [g for g in per_budget_groups[budget] if g in per_group]
        if not groups:
            continue
        quantities = {g: per_group[g][budget] for g in groups}
        context = {g: per_group_context[g][budget] for g in groups}
        episode = {g: context[g]["episode"] for g in groups}
        ranked = [(context[g]["source_crossfit_gain"], g) for g in groups]
        strata = split_strata(ranked)
        strata["all"] = sorted(groups)
        block: dict[str, Any] = {
            "groups": len(groups),
            "episodes": len(set(episode.values())),
            "short_targets_under_8_tokens": sum(
                1 for g in groups if context[g]["n_tok"] < 8
            ),
            "crossfit_equals_in_sample_rate": (
                sum(1 for g in groups if context[g]["crossfit_equals_in_sample"])
                / len(groups)
            ),
            "gain_quantiles": {
                name: sorted(value for value, _ in ranked)[
                    min(len(ranked) - 1, int(fraction * len(ranked)))
                ]
                for name, fraction in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90))
            },
            "strata": {},
        }
        for stratum, members in strata.items():
            block["strata"][stratum] = {
                "n": len(members),
                "episodes": len({episode[g] for g in members}),
                "gain_range": [
                    min(context[g]["source_crossfit_gain"] for g in members),
                    max(context[g]["source_crossfit_gain"] for g in members),
                ],
                "delta": reduce_stratum(members, quantities, episode, DELTA_NAMES),
                "baseline": reduce_stratum(members, quantities, episode, BASELINE_NAMES),
                "in_sample_upper_bound": reduce_stratum(
                    members, quantities, episode, INSAMPLE_NAMES
                ),
            }
        budget_blocks[str(budget)] = block

    report = {
        "schema_version": REPORT_SCHEMA,
        "bootstrap": {
            "cluster_unit": "episode",
            "replicates": 10000,
            "confidence": 0.95,
            "implementation": "scripts.budget_replacement_sweep.cluster_bootstrap",
        },
        "quantity_algebra": {
            "token_agree": "fraction of target positions where argmax == target token",
            "all_agree": "1 if every target token is argmax-correct (teacher-forced exact-match proxy)",
            "firstK_agree": "1 if the first K target tokens are all argmax-correct",
            "delta_X": "oracle(cross-fit) - recent, averaged over the two fold directions",
            "delta_token_agree_heldout": (
                "token_agree restricted to the fold that did NOT pick the set; the "
                "argmax counterpart of the cross-fit log-prob gain"
            ),
            "delta_X_insample": (
                "same delta but with the in-sample argmax set; an OPTIMISTIC UPPER "
                "BOUND, never the headline"
            ),
            "strata": (
                "groups ranked by the source crossfit_gain for that budget; the top and "
                "bottom deciles are cut at 10%. The stratifier and the outcome share "
                "the same tokens, so the top decile carries a winner's curse: read it "
                "as 'conditional on a large measured gain', not as 'a decile a selector "
                "could pick out in advance'."
            ),
        },
        "reconciliation": reconciliation,
        # note (luojiaxuan): 键名不能叫 "budgets" —— ``**common`` 里已经有一个
        # ``"budgets": [1, 2, 4]``(请求的预算列表),后展开会把整块结果覆盖成一个
        # 三元素列表,而控制台输出用的是局部变量所以看不出来。
        "by_budget": budget_blocks,
        "per_group": {
            group: {
                str(budget): {
                    "quantities": per_group[group][budget],
                    "context": per_group_context[group][budget],
                }
                for budget in per_group[group]
            }
            for group in sorted(per_group)
        },
        **common,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def show(entry: dict[str, Any] | None) -> str:
        if entry is None:
            return "n/a".rjust(34)
        star = "*" if (entry["ci_low"] > 0 or entry["ci_high"] < 0) else " "
        return "%+.4f [%+.4f, %+.4f] n=%-4d%s" % (
            entry["point"], entry["ci_low"], entry["ci_high"], entry["n"], star
        )

    for budget, block in budget_blocks.items():
        print("")
        print(
            f"=== B={budget}  groups={block['groups']}  episodes={block['episodes']}  "
            f"gain p10/p50/p90="
            f"{block['gain_quantiles']['p10']:+.4f}/"
            f"{block['gain_quantiles']['p50']:+.4f}/"
            f"{block['gain_quantiles']['p90']:+.4f} ==="
        )
        for stratum in ("bottom_decile", "middle", "top_decile", "all"):
            entry = block["strata"].get(stratum)
            if entry is None:
                continue
            print(f"-- {stratum}  n={entry['n']}  episodes={entry['episodes']}")
            headline = (
                "delta_token_agree",
                "delta_token_agree_heldout",
                "delta_action_type_agree",
                "delta_tool_call_all_agree",
                "delta_first4_agree",
                "delta_all_agree",
                "delta_crossfit_logprob",
            )
            for name in headline:
                print(f"   {name:<32}{show(entry['delta'].get(name))}")
            for name in (
                "recent_token_agree",
                "recent_action_type_agree",
                "recent_tool_call_all_agree",
                "recent_all_agree",
            ):
                print(f"   {name:<32}{show(entry['baseline'].get(name))}")
    print("")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
