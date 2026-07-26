#!/usr/bin/env python3
"""Gate 3: dual-format **oracle** headroom probe on the frozen policy.

# note (luojiaxuan): Gate 1 已经证明"冻结模型的稀疏选点收益 S0-R0=+0.0335 是**单轮
# renderer 的产物**":换成 official-style sparse multiturn 之后它归零
# (S_multi-R_multi=+0.0027,CI 跨零)。但那个 +0.0027 测的是 ``choose_sparse`` 的
# **随机**非相邻选点,所以当时只知道"随机稀疏 ≈ 最近窗口",不知道"**最好的**稀疏是否
# ≈ 最近窗口"。本脚本把这两件事分开:候选池取决策点之前的**全部**历史步,在**两种
# 格式各自独立**搜出 oracle 集合,量出
#
#     Q(S_oracle_B) - Q(Recent_B)          <- Gate 3 判据(B=4)
#     Q(S_random_sparse_B) - Q(Recent_B)   <- 随机稀疏对照
#     Q(Recent_B) - Q(B0)                  <- 有历史 vs 无历史,给尺度
#
# 其中 ``Q(S) = log p(a* | Render(S))``,teacher-forced,冻结模型,不加载任何 adapter。
#
# 三条方法学约束沿用 Gate 1:只用 dev(--episode-filter 必填)、同一条 encode_sample +
# 目标 logprob 编码路径、像素预算按 B 对齐。
#
# ---------------------------------------------------------------------------
# 关于"oracle 必然赢"这件事(本脚本最重要的设计决定)
# ---------------------------------------------------------------------------
# oracle 是在**同一批**决策点上、用**同一个**冻结模型、**看着 a* 本身**搜出来的,所以
# 只要 Recent_B 落在搜索空间里,``Q(oracle_B) - Q(Recent_B) >= 0`` 就是**恒等式**,
# 它的 CI 下界大于零几乎是同义反复,不构成证据。本脚本的处理是把这件事**摆到明面上**
# 而不是回避它:
#
#   1. **保证 Recent 在搜索空间内**。beam 每层除了按 Q 取 top-4,再无条件把 Recent_d
#      塞进 beam(``beam_d = top4 ∪ {Recent_d}``,宽度 <=5)。于是 Recent_1..Recent_4
#      必被打分,raw oracle 是一个**真上界**,而不是"搜索能力"与"headroom"的混合物。
#      代价是每组每格式的前向从 13n-24 涨到 16n-30。
#   2. raw ``oracle_gain`` 因此被标注为 **optimistic upper bound**,报告里显式写明它的
#      CI 下界大于零是构造性的,**不是** Gate 3 的实际证据。
#   3. 真正承重的是 **cross-fit(held-out)oracle**:把目标 token 按奇偶分成两折,用
#      A 折的 logprob 选集合、在 B 折上评估同一集合(再反过来),取两折平均。它不额外
#      花任何前向(同一次前向里把 per-token logprob 拆成两折的和与计数)。如果 raw
#      headroom 只是"在 C(n,B) 个集合上取 max"的选择性噪声,cross-fit 会塌回 0;如果
#      是"这个集合真的让 a* 更可预测",它会活下来。
#      **B=1 的 cross-fit 是干净的**:深度 1 穷举了全部 n 个单元素集合,池子本身没有被
#      按 Q 剪枝过,所以选择只发生在 A 折上。B=2/B=4 的候选池经过按全量 Q 的 beam 剪枝,
#      存在轻微泄漏,报告里分开标。
#   4. 附带的免费诊断(全部来自深度 1 的穷举扫描,无额外前向):Recent_1 在 n 个单元素
#      集合里的**分位**、``mean_singleton - Recent_1``。如果 Recent_1 本来就在第 90
#      分位,那"选择"这件事本身就没什么可做的。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_sparse_history import build_sparse_history_messages
from causalcache.policy.gui_owl_sparse_multiturn import (
    SPARSE_MULTITURN_PROMPT_FORMAT,
    build_sparse_multiturn_messages,
)
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from scripts.build_sparse_history_dataset import choose_sparse, max_sparse_budget
from scripts.probe_prompt_format import rebuild_full_responses
from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from scripts.score_sparse_history_arms import (
    EpisodeClusterBootstrap,
    cache_member_paths,
    load_sparse_samples,
    resolve_sample_files,
    sha256_of,
    shard_cache_path,
)
from scripts.train_success_sft_lora import encode_sample, load_config

REPORT_SCHEMA = "causalcache.oracle_headroom_probe.v1"
SHARD_REPORT_SCHEMA = "causalcache.oracle_headroom_probe_shard.v1"
CACHE_FINGERPRINT_KEY = "cache_fingerprint"

SINGLE_TURN_FORMAT = "sparse_single_turn"
FORMATS = (SINGLE_TURN_FORMAT, SPARSE_MULTITURN_PROMPT_FORMAT)
BUDGETS = (1, 2, 4)
BEAM_WIDTH = 4
# 分层抽样的 n=cur-1 边界(左闭右闭)。dev 实测 n∈[9,32],中位数 15。
STRATA_EDGES: tuple[tuple[int, int], ...] = ((9, 12), (13, 16), (17, 20), (21, 10**9))


# ---------------------------------------------------------------------------
# 集合的规范键
# ---------------------------------------------------------------------------
def set_key(steps: tuple[int, ...]) -> str:
    return "-".join(str(step) for step in steps) if steps else "empty"


def canonical(steps: object) -> tuple[int, ...]:
    return tuple(sorted(int(step) for step in steps))  # type: ignore[union-attr]


def stable_seed(*parts: object) -> int:
    """Deterministic int seed. ``random.Random("str")`` is **not** reproducible across
    processes because str hashing is salted per interpreter (PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


# ---------------------------------------------------------------------------
# per-set 分数缓存(值是 dict,不是标量,故不能直接复用 ScoreCache)
# ---------------------------------------------------------------------------
class SetScoreCache:
    """Append-only ``(group, format, set)`` -> score record cache with resume.

    # note (luojiaxuan): 分片/锁/fingerprint/残尾容忍全部照抄
    # score_sparse_history_arms.ScoreCache 的语义(并复用它的 shard_cache_path /
    # cache_member_paths),唯一的区别是 value 从 float 变成一条小 dict:cross-fit 需要
    # 目标 token 按奇偶两折的和与计数,只存均值就没法在归约阶段重算 held-out 量,而为了
    # 它重跑三万次前向是不可接受的。存的是 6 个标量而不是完整 per-token 向量,体积仍与
    # 原 cache 同量级。
    """

    def __init__(
        self,
        path: Path | None,
        *,
        shard_index: int = 0,
        shard_count: int = 1,
        fingerprint: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.shard_path = (
            None if path is None else shard_cache_path(path, shard_index, shard_count)
        )
        self.entries: dict[str, dict[str, Any]] = {}
        self.own: dict[str, dict[str, Any]] = {}
        self.sources: list[dict[str, Any]] = []
        self.handle = None
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        for member in cache_member_paths(path):
            self.sources.append(self._load_member(member))
        started_empty = (
            not self.shard_path.exists() or self.shard_path.stat().st_size == 0
        )
        self.handle = self.shard_path.open("a", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise SystemExit(
                f"another live process already owns score-cache shard "
                f"{self.shard_path}; give every concurrent process a distinct "
                "--shard-index"
            )
        if fingerprint is not None and started_empty:
            self._write(self.handle, {CACHE_FINGERPRINT_KEY: fingerprint})

    @staticmethod
    def _write(handle: Any, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()

    def _load_member(self, member: Path) -> dict[str, Any]:
        with member.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        mine = member == self.shard_path
        loaded = 0
        torn_tail = False
        fingerprinted = False
        for position, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                # 只有**没有换行结尾的最后一行**允许是残行(正在写 / 被 SIGKILL 掐断);
                # 中间的坏行是真损坏,静默跳过等于把一批分数换成"没算过"。
                if position == len(lines) and not line.endswith("\n"):
                    torn_tail = True
                    break
                raise ValueError(f"{member}:{position} is not JSON: {error}")
            if CACHE_FINGERPRINT_KEY in record:
                self._check_fingerprint(member, record[CACHE_FINGERPRINT_KEY])
                fingerprinted = True
                continue
            value = record["value"]
            self.entries[record["cache_key"]] = value
            if mine:
                self.own[record["cache_key"]] = value
            loaded += 1
        return {
            "path": str(member),
            "entries": loaded,
            "torn_tail": torn_tail,
            "fingerprinted": fingerprinted,
        }

    def _check_fingerprint(self, member: Path, recorded: dict[str, Any]) -> None:
        if self.fingerprint is None or recorded == self.fingerprint:
            return
        differing = sorted(
            key
            for key in set(recorded) | set(self.fingerprint)
            if recorded.get(key) != self.fingerprint.get(key)
        )
        raise SystemExit(
            f"score cache {member} was produced under a different scoring setup "
            f"(differing fields: {differing}); merging it would fold another "
            "model/corpus/config's log-probs into this report"
        )

    def get(self, cache_key: str) -> dict[str, Any] | None:
        return self.entries.get(cache_key)

    def put(self, cache_key: str, value: dict[str, Any]) -> None:
        self.entries[cache_key] = value
        self.own[cache_key] = value
        if self.handle is not None:
            self._write(self.handle, {"cache_key": cache_key, "value": value})

    def stats(self) -> dict[str, Any]:
        return {
            "path": None if self.path is None else str(self.path),
            "shard_path": None if self.shard_path is None else str(self.shard_path),
            "merged_entries": len(self.entries),
            "own_entries": len(self.own),
            "unfingerprinted_sources": [
                source["path"] for source in self.sources if not source["fingerprinted"]
            ],
            "sources": self.sources,
        }

    def close(self) -> None:
        if self.handle is None:
            return
        if self.shard_path is not None:
            temporary = self.shard_path.with_name(self.shard_path.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                if self.fingerprint is not None:
                    handle.write(
                        json.dumps(
                            {CACHE_FINGERPRINT_KEY: self.fingerprint}, sort_keys=True
                        )
                        + "\n"
                    )
                for cache_key in sorted(self.own):
                    handle.write(
                        json.dumps(
                            {"cache_key": cache_key, "value": self.own[cache_key]},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.shard_path)
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


# ---------------------------------------------------------------------------
# 决策点:从语料行重建"完整历史候选池"
# ---------------------------------------------------------------------------
class DecisionPoint:
    """One dev decision point with its **full** history candidate pool.

    候选池是 1..current_step-1 的全部历史步,不是语料里已选的那 K 个。语料只存了
    recent-K / sparse-K 的图路径,所以其余步的图路径按 ``obs-{step-1:03d}.png`` 推导;
    推导规则用语料里现成的 (step, path) 对**逐条对账**(实测 dev 上 3340/3340 一致),
    并逐个断言文件存在,推不出来就整组 fail-closed 丢弃。
    """

    def __init__(
        self,
        rows: dict[str, dict[str, Any]],
        *,
        annotations: Path,
        annotation_cache: dict[str, list[dict]],
        image_root: Path,
    ) -> None:
        n0 = rows["N0"]
        self.pair_group: str = n0["pair_group"]
        self.episode: str = n0["episode"]
        self.current_step: int = int(n0["decision_step"])
        self.instruction: str = n0["instruction"]
        self.action_texts: list[str] = list(n0["action_texts"])
        self.target_text: str = n0["target_text"]
        self.corpus_budget: int = int(n0["budget"])
        self.candidates: tuple[int, ...] = tuple(range(1, self.current_step))
        self.n_candidates = len(self.candidates)

        # 完整响应:从 GUI-Odyssey 原始 annotation 重建,再与 N0 的 assistant 轮逐字节
        # 对账(Gate 1 已验证 dev 240/240 通过)。多轮 renderer 对**每个被选中的步**都
        # 要求非空完整响应,而 oracle 候选遍布整条历史,所以这里必须覆盖 1..cur-1 全部。
        self.full_responses = rebuild_full_responses(rows, annotations, annotation_cache)
        if len(self.full_responses) != self.current_step - 1:
            raise ValueError(
                f"{self.pair_group}: rebuilt {len(self.full_responses)} responses for "
                f"{self.current_step - 1} completed steps"
            )
        for step, response in enumerate(self.full_responses, start=1):
            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"{self.pair_group}: Step{step} has no full assistant response; the "
                    "probe never falls back to a bare description"
                )

        current_image = Path(n0["current_image"])
        self.image_dir = current_image.parent
        if current_image.name != f"obs-{self.current_step - 1:03d}.png":
            raise ValueError(
                f"{self.pair_group}: current image {current_image} does not follow the "
                f"obs-{{step-1:03d}}.png convention at step {self.current_step}"
            )
        if self.image_dir.name != self.episode:
            raise ValueError(
                f"{self.pair_group}: image directory {self.image_dir} is not the episode"
            )
        self.current_image = str(current_image)

        # 对账:语料里现成的每一对 (step, path) 都必须与推导规则一致。负样本臂
        # (SA_neg_irrelevant / SA_neg_step_shuffled)故意错位,不参与对账。
        self.reconciled_pairs = 0
        for slot in ("N0", "R0", "S0"):
            row = rows.get(slot)
            if row is None:
                continue
            for step, path in zip(row["selected_steps"], row["selected_images"]):
                if path != self.step_image(int(step)):
                    raise ValueError(
                        f"{self.pair_group}: corpus arm {slot!r} maps Step{step} to "
                        f"{path!r} but the derivation gives {self.step_image(int(step))!r}"
                    )
                self.reconciled_pairs += 1
        for step in self.candidates:
            if not (image_root / self.step_image(step)).is_file():
                raise ValueError(
                    f"{self.pair_group}: Step{step} screenshot "
                    f"{self.step_image(step)} is absent from {image_root}"
                )
        self.corpus_sparse_steps: tuple[int, ...] = canonical(rows["S0"]["selected_steps"])

    def step_image(self, step: int) -> str:
        return str(self.image_dir / f"obs-{step - 1:03d}.png")

    def recent(self, budget: int) -> tuple[int, ...]:
        return tuple(range(self.current_step - budget, self.current_step))

    def render(self, steps: tuple[int, ...], prompt_format: str) -> dict[str, Any]:
        images = [{"__path__": self.step_image(step)} for step in steps]
        if prompt_format == SINGLE_TURN_FORMAT:
            messages = build_sparse_history_messages(
                instruction=self.instruction,
                action_texts=self.action_texts,
                selected_steps=list(steps),
                selected_images=images,
                current_step=self.current_step,
                current_image={"__path__": self.current_image},
            )
        elif prompt_format == SPARSE_MULTITURN_PROMPT_FORMAT:
            messages = build_sparse_multiturn_messages(
                instruction=self.instruction,
                action_texts=self.action_texts,
                full_responses=self.full_responses,
                selected_steps=list(steps),
                selected_images=images,
                current_step=self.current_step,
                current_image={"__path__": self.current_image},
            )
        else:
            raise ValueError(f"unknown prompt format {prompt_format!r}")
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
        images_in_prompt = sum(
            1
            for message in serialised
            for part in message["content"]
            if part["type"] == "image"
        )
        if images_in_prompt != len(steps) + 1:
            raise ValueError(
                f"{self.pair_group}: {prompt_format} rendered {images_in_prompt} images "
                f"for |S|={len(steps)}, expected {len(steps) + 1}"
            )
        return {
            "sample_id": f"{self.pair_group}|{prompt_format}|{set_key(steps)}",
            "pair_group": self.pair_group,
            "episode": self.episode,
            "prompt_format": prompt_format,
            "target_text": self.target_text,
            "messages": serialised,
        }


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
def target_token_logprobs(model: Any, encoded: dict[str, Any], *, torch: Any) -> Any:
    """Per-token teacher-forced target log-probs.

    # note (luojiaxuan): 与 train_success_sft_lora.mean_target_logprob **逐行同构**,
    # 唯一的区别是返回 gather 之后的向量而不是 ``sum/token_count``。cross-fit 需要按
    # token 分折,而分折只能在向量上做;冒烟阶段会断言 ``vector.mean()`` 与
    # ``mean_target_logprob`` 逐位相等,防止这份副本悄悄漂移。
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
    return log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)[0]


class FrozenSetScorer:
    """Teacher-forced target log-prob under the frozen policy, with no adapter at all.

    与 probe_prompt_format.FrozenScorer 同一条路径:**根本不注入** HGKV,也不加载任何
    checkpoint,所以"跑在冻结模型上"是结构性的,不依赖运行期开关。
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        config: dict[str, Any],
        image_root: Path,
        cache: SetScoreCache,
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
        self.max_prompt_tokens = 0
        self.max_total_tokens = 0
        self.mean_parity: dict[str, Any] | None = None

    def verify_mean_parity(
        self, point: DecisionPoint, steps: tuple[int, ...], prompt_format: str
    ) -> dict[str, Any]:
        """Assert the per-token path reproduces ``mean_target_logprob`` exactly.

        # note (luojiaxuan): target_token_logprobs 是 mean_target_logprob 的一份副本
        # (只把 sum/count 换成向量),而副本是会漂的。Gate 1 与本探针要能摆在同一张表
        # 上,前提是这两条路径算的是同一个数,所以每次跑都实测一次,把差值写进报告,
        # 不靠"我抄的时候很小心"。
        """
        from scripts.train_success_sft_lora import mean_target_logprob

        runtime = self._prepare()
        sample = point.render(steps, prompt_format)
        first = encode_sample(
            runtime, sample, dataset_root=self.image_root, torch=self.torch
        )
        second = encode_sample(
            runtime, sample, dataset_root=self.image_root, torch=self.torch
        )
        with self.torch.no_grad():
            vector = target_token_logprobs(runtime.model, first, torch=self.torch)
            vector_mean = float(vector.detach().mean())
            reference = float(
                mean_target_logprob(runtime.model, second, torch=self.torch).detach()
            )
        delta = abs(vector_mean - reference)
        if not delta < 1e-4:
            raise SystemExit(
                "target_token_logprobs has drifted from mean_target_logprob "
                f"({vector_mean!r} vs {reference!r}); the probe would no longer be "
                "measuring the same quantity as the Gate-1 format probe"
            )
        self.mean_parity = {
            "sample_id": sample["sample_id"],
            "per_token_mean": vector_mean,
            "mean_target_logprob": reference,
            "abs_delta": delta,
        }
        return self.mean_parity

    def _prepare(self) -> Any:
        if self.runtime is None:
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

    def score(
        self, point: DecisionPoint, steps: tuple[int, ...], prompt_format: str
    ) -> dict[str, Any]:
        cache_key = f"{point.pair_group}|{prompt_format}|{set_key(steps)}"
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
        sample = point.render(steps, prompt_format)
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
            vector = target_token_logprobs(runtime.model, encoded, torch=self.torch)
            values = [float(value) for value in vector.detach().cpu()]
        token_count = len(values)
        if token_count == 0:
            raise ValueError(f"sample {sample['sample_id']!r} scored zero target tokens")
        even = [value for index, value in enumerate(values) if index % 2 == 0]
        odd = [value for index, value in enumerate(values) if index % 2 == 1]
        record = {
            "mean": math.fsum(values) / token_count,
            "n_tok": token_count,
            "sum_even": math.fsum(even),
            "n_even": len(even),
            "sum_odd": math.fsum(odd),
            "n_odd": len(odd),
            "prompt_tokens": total_tokens - token_count,
            "total_tokens": total_tokens,
        }
        self.max_prompt_tokens = max(self.max_prompt_tokens, record["prompt_tokens"])
        self.max_total_tokens = max(self.max_total_tokens, total_tokens)
        self.forwards += 1
        self.cache.put(cache_key, record)
        return record


def fold_mean(record: dict[str, Any], fold: str) -> float | None:
    """Mean target log-prob restricted to one token fold (``even`` / ``odd``)."""
    count = record[f"n_{fold}"]
    return None if not count else record[f"sum_{fold}"] / count


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------
def search_sets(
    point: DecisionPoint,
    prompt_format: str,
    scorer: FrozenSetScorer,
    *,
    exact_max_candidates: int,
) -> dict[str, Any]:
    """Score a family of candidate sets, exact when tiny, seeded beam-4 otherwise.

    返回 ``{"scores": {set_key: record}, "sizes": {B: [set_key,...]}, "mode": ...}``。
    beam 每层的 active prefix 是 ``top-BEAM_WIDTH ∪ {Recent_d}``,见模块 docstring 的
    设计说明:塞进 Recent 是为了让 raw oracle 成为**真上界**,代价是每组多约 3n 次前向。
    """
    scores: dict[str, dict[str, Any]] = {}
    sizes: dict[int, set[str]] = {budget: set() for budget in range(0, max(BUDGETS) + 1)}

    def evaluate(steps: tuple[int, ...]) -> float:
        key = set_key(steps)
        if key not in scores:
            scores[key] = scorer.score(point, steps, prompt_format)
            if len(steps) in sizes:
                sizes[len(steps)].add(key)
        return scores[key]["mean"]

    evaluate(())  # B0
    candidates = point.candidates
    exact = point.n_candidates <= exact_max_candidates
    if exact:
        for budget in BUDGETS:
            if budget > point.n_candidates:
                continue
            for combination in itertools.combinations(candidates, budget):
                evaluate(combination)
        mode = "exact"
        singleton_pool_exhaustive = True
    else:
        mode = f"beam{BEAM_WIDTH}_recent_seeded"
        singleton_pool_exhaustive = True
        # depth 1:穷举全部单元素集合(所以 B=1 的 oracle 与 cross-fit 都是精确的)
        level = [(evaluate((step,)), (step,)) for step in candidates]
        beam = [steps for _, steps in sorted(level, key=lambda item: -item[0])[:BEAM_WIDTH]]
        recent_prefix = point.recent(1)
        if recent_prefix not in beam:
            beam.append(recent_prefix)
        for depth in range(2, max(BUDGETS) + 1):
            children: list[tuple[float, tuple[int, ...]]] = []
            seen: set[tuple[int, ...]] = set()
            for prefix in beam:
                for step in candidates:
                    if step in prefix:
                        continue
                    child = canonical(prefix + (step,))
                    if child in seen:
                        continue
                    seen.add(child)
                    children.append((evaluate(child), child))
            beam = [
                steps for _, steps in sorted(children, key=lambda item: -item[0])[:BEAM_WIDTH]
            ]
            recent_prefix = point.recent(depth)
            if recent_prefix not in beam:
                evaluate(recent_prefix)
                beam.append(recent_prefix)
    return {
        "scores": scores,
        "sizes": {budget: sorted(keys) for budget, keys in sizes.items()},
        "mode": mode,
        "singleton_pool_exhaustive": singleton_pool_exhaustive,
        "sets_scored": len(scores),
    }


def analyse_point(
    point: DecisionPoint,
    prompt_format: str,
    scorer: FrozenSetScorer,
    *,
    exact_max_candidates: int,
) -> dict[str, Any]:
    """All Gate-3 quantities for one (decision point, format)."""
    search = search_sets(
        point, prompt_format, scorer, exact_max_candidates=exact_max_candidates
    )
    scores = search["scores"]

    def record_for(steps: tuple[int, ...]) -> dict[str, Any]:
        key = set_key(steps)
        if key not in scores:
            scores[key] = scorer.score(point, steps, prompt_format)
            if len(steps) in search["sizes"]:
                search["sizes"][len(steps)] = sorted(
                    set(search["sizes"][len(steps)]) | {key}
                )
        return scores[key]

    b0 = record_for(())
    quantities: dict[str, float] = {}
    diagnostics: dict[str, Any] = {
        "search_mode": search["mode"],
        "sets_scored": search["sets_scored"],
        "n_candidates": point.n_candidates,
        "b0": b0["mean"],
    }

    # 语料里 choose_sparse 的那一组(原生 budget),用来把本子样本与 Gate 1 对齐。
    corpus = record_for(point.corpus_sparse_steps)
    corpus_recent = record_for(point.recent(point.corpus_budget))
    quantities["corpus_sparse_gain_native"] = corpus["mean"] - corpus_recent["mean"]
    diagnostics["corpus_budget"] = point.corpus_budget

    for budget in BUDGETS:
        recent_steps = point.recent(budget)
        # note (luojiaxuan): 必须在 record_for 之前问"搜索本身有没有碰过 Recent_B",
        # 否则这个检查是自我实现的——record_for 会把它补进池子,断言永远通过,而
        # "Recent 在搜索空间内"恰恰是 raw oracle 能被称为上界的前提。beam 每层塞
        # Recent_d 之后这里应当恒为 True,报告里把实测比例写出来供核对。
        recent_in_search_space = set_key(recent_steps) in set(search["sizes"][budget])
        recent = record_for(recent_steps)
        pool_keys = sorted(set(search["sizes"][budget]))
        pool = {key: scores[key] for key in pool_keys}
        diagnostics[f"recent_in_search_space_b{budget}"] = recent_in_search_space
        oracle_key = max(pool, key=lambda key: pool[key]["mean"])
        oracle = pool[oracle_key]

        rng = random.Random(stable_seed(point.pair_group, budget, "random_sparse"))
        random_steps: tuple[int, ...] = ()
        if budget <= max_sparse_budget(point.current_step):
            random_steps = tuple(choose_sparse(rng, point.current_step, budget))
        if random_steps:
            random_record = record_for(random_steps)
            quantities[f"random_gain_b{budget}"] = (
                random_record["mean"] - recent["mean"]
            )
            diagnostics[f"random_steps_b{budget}"] = list(random_steps)
        else:
            diagnostics[f"random_steps_b{budget}"] = None

        quantities[f"oracle_gain_b{budget}"] = oracle["mean"] - recent["mean"]
        quantities[f"recent_over_b0_b{budget}"] = recent["mean"] - b0["mean"]
        quantities[f"oracle_over_b0_b{budget}"] = oracle["mean"] - b0["mean"]

        # cross-fit:A 折选集合、B 折评估,再反过来,取两折平均。
        held: list[float] = []
        # 目标只有一个 token 时奇偶折其中一折为空,cross-fit 无定义;整组跳过而不是让
        # max(..., key=...) 在 None 上崩掉(归约是最后一步,炸在这里等于白跑一小时)。
        if all(
            entry["n_even"] and entry["n_odd"] for entry in list(pool.values()) + [recent]
        ):
            for select_fold, evaluate_fold in (("even", "odd"), ("odd", "even")):
                chosen = max(pool, key=lambda key: fold_mean(pool[key], select_fold))
                chosen_value = fold_mean(pool[chosen], evaluate_fold)
                recent_value = fold_mean(recent, evaluate_fold)
                held.append(chosen_value - recent_value)
                diagnostics[f"heldout_pick_{select_fold}_b{budget}"] = chosen
        means = [entry["mean"] for entry in pool.values()]
        pool_mean_gap = math.fsum(means) / len(means) - recent["mean"]
        if held:
            quantities[f"oracle_heldout_gain_b{budget}"] = sum(held) / len(held)
            # note (luojiaxuan): cross-fit 的正确零假设**不是** 0。若池子里各集合的真实
            # 效用相同、A/B 两折只差噪声,则在 A 折取 argmax 后在 B 折上的期望值就是池子
            # 均值,于是 held-out 会等于 pool_mean − Recent(Recent 本来就好时该值为负)。
            # 所以"选择是否携带可迁移信息"要看 held-out **减去** pool mean 这一项:
            # 它 >0 才说明在半份证据上做的选择好过从同一池子里随便抓一个。
            quantities[f"heldout_over_poolmean_b{budget}"] = (
                quantities[f"oracle_heldout_gain_b{budget}"] - pool_mean_gap
            )
        below = sum(1 for value in means if value < recent["mean"])
        ties = sum(1 for value in means if value == recent["mean"])
        diagnostics[f"pool_size_b{budget}"] = len(pool)
        diagnostics[f"oracle_steps_b{budget}"] = [int(s) for s in oracle_key.split("-")]
        diagnostics[f"oracle_is_recent_b{budget}"] = oracle_key == set_key(recent_steps)
        diagnostics[f"recent_percentile_b{budget}"] = (
            (below + 0.5 * ties) / len(means) if means else None
        )
        quantities[f"pool_mean_minus_recent_b{budget}"] = pool_mean_gap
    return {"quantities": quantities, "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# 分层抽样
# ---------------------------------------------------------------------------
def stratum_of(n_candidates: int) -> int:
    for index, (low, high) in enumerate(STRATA_EDGES):
        if low <= n_candidates <= high:
            return index
    raise ValueError(f"history length {n_candidates} falls outside {STRATA_EDGES}")


def stratified_sample(
    dev_rows: dict[str, dict[str, dict[str, Any]]], *, count: int, seed: int
) -> tuple[list[str], dict[str, Any]]:
    """Pick ``count`` dev groups stratified by history length, distinct episodes first.

    # note (luojiaxuan): 只取短历史会系统性偏向"容易穷举"的样本,而搜索代价恰恰随 n
    # 增长——那等于用最便宜的一批样本回答一个关于全体的问题。按 n 分层并按各层在 dev
    # 中的占比配额(最大余数法),层内优先取**互不相同的 episode**,让 episode-cluster
    # bootstrap 的簇数尽量接近组数。
    """
    per_group_n = {
        group: int(rows["N0"]["decision_step"]) - 1 for group, rows in dev_rows.items()
    }
    buckets: dict[int, list[str]] = {index: [] for index in range(len(STRATA_EDGES))}
    for group in sorted(per_group_n):
        buckets[stratum_of(per_group_n[group])].append(group)
    total = len(per_group_n)
    exact_quota = {index: len(groups) * count / total for index, groups in buckets.items()}
    quota = {index: int(value) for index, value in exact_quota.items()}
    remainder = count - sum(quota.values())
    for index in sorted(
        exact_quota, key=lambda i: (-(exact_quota[i] - quota[i]), i)
    )[:remainder]:
        quota[index] += 1

    chosen: list[str] = []
    used_episodes: set[str] = set()
    detail: dict[str, Any] = {}
    for index in sorted(buckets):
        pool = list(buckets[index])
        random.Random(stable_seed(seed, "stratum", index)).shuffle(pool)
        picked: list[str] = []
        for pass_number in (0, 1):
            for group in pool:
                if len(picked) >= quota[index]:
                    break
                if group in picked:
                    continue
                episode = dev_rows[group]["N0"]["episode"]
                if pass_number == 0 and episode in used_episodes:
                    continue
                picked.append(group)
                used_episodes.add(episode)
        if len(picked) < quota[index]:
            raise ValueError(
                f"stratum {STRATA_EDGES[index]} holds {len(pool)} groups but the quota "
                f"asks for {quota[index]}"
            )
        chosen.extend(picked)
        detail[f"{STRATA_EDGES[index][0]}-{STRATA_EDGES[index][1]}"] = {
            "dev_groups": len(pool),
            "quota": quota[index],
            "selected": len(picked),
        }
    chosen.sort()
    manifest = {
        "method": (
            "proportional (largest-remainder) allocation over history-length strata; "
            "within a stratum a deterministic shuffle picks groups, preferring "
            "previously unused episodes so the cluster bootstrap keeps its clusters"
        ),
        "strata": detail,
        "seed": seed,
        "selected_groups": len(chosen),
        "selected_episodes": len({dev_rows[g]["N0"]["episode"] for g in chosen}),
    }
    return chosen, manifest


def spread_subset(groups: list[str], dev_rows: dict[str, Any], limit: int) -> list[str]:
    """A smoke subset that is a genuine **subset** of the full sample, spread over n.

    # note (luojiaxuan): 冒烟必须落在正式样本内部,否则冒烟花掉的前向进了缓存却对正式
    # 跑没有任何用处;按 n 排序后等距抽,又能保证 5 组里长短都有。
    """
    if limit <= 0 or limit >= len(groups):
        return groups
    ordered = sorted(
        groups, key=lambda g: (int(dev_rows[g]["N0"]["decision_step"]), g)
    )
    if limit == 1:
        return [ordered[0]]
    positions = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return sorted({ordered[position] for position in positions})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search per-decision-point oracle history subsets under two renderers on "
            "the frozen policy and report Gate-3 headroom with episode-cluster CIs."
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, required=True)
    # dev-only 契约与 probe_prompt_format 一致:这两个是**必填**,confirmation 的 254 组
    # 连分数都不许算出来。
    parser.add_argument("--episode-filter", type=Path, required=True)
    parser.add_argument("--episode-filter-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-cache", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-reduction", action="store_true")
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--probe-groups", type=int, default=60)
    parser.add_argument("--sample-seed", type=int, default=20260727)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="smoke run: keep a spread subset of the stratified sample (0 = all)",
    )
    parser.add_argument(
        "--exact-max-candidates",
        type=int,
        default=8,
        help="exhaustive subset search when the candidate pool is at most this large",
    )
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def group_dev_rows(
    samples: list[dict[str, Any]], allowed: set[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    needed = {"N0", "R0", "S0"}
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        if sample["episode"] not in allowed or sample.get("arm_slot") not in needed:
            continue
        bucket = grouped.setdefault(sample["pair_group"], {})
        if sample["arm_slot"] in bucket:
            raise ValueError(
                f"pair-group {sample['pair_group']!r} carries two "
                f"{sample['arm_slot']!r} rows"
            )
        bucket[sample["arm_slot"]] = sample
    incomplete = sorted(g for g, rows in grouped.items() if set(rows) != needed)
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} pair-groups lack one of {sorted(needed)}: {incomplete[:5]}"
        )
    return grouped


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
        raise SystemExit("--shard-count > 1 requires --score-cache")
    if args.shard_count > 1 and not args.skip_reduction:
        raise SystemExit(
            "a sharded process only holds 1/--shard-count of the sample; running the "
            "bootstrap on that slice would silently shrink the denominator. Pass "
            "--skip-reduction on every shard, then run one unsharded reduction pass."
        )
    if args.require_cached and args.score_cache is None:
        raise SystemExit("--require-cached is meaningless without --score-cache")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise SystemExit("--bootstrap-confidence must lie strictly inside (0, 1)")

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

    dev_rows = group_dev_rows(samples, allowed)
    if not dev_rows:
        raise SystemExit("episode filter removed every group; wrong corpus?")
    chosen, sample_manifest = stratified_sample(
        dev_rows, count=args.probe_groups, seed=args.sample_seed
    )
    if args.max_groups:
        chosen = spread_subset(chosen, dev_rows, args.max_groups)
        sample_manifest["smoke_subset"] = len(chosen)
    length_histogram: dict[str, int] = {}
    for group in chosen:
        key = str(int(dev_rows[group]["N0"]["decision_step"]) - 1)
        length_histogram[key] = length_histogram.get(key, 0) + 1
    sample_manifest["history_length_histogram"] = dict(
        sorted(length_histogram.items(), key=lambda item: int(item[0]))
    )
    projected = sum(
        max(16 * (int(dev_rows[g]["N0"]["decision_step"]) - 1) - 30, 0) + 6
        for g in chosen
    ) * len(FORMATS)
    print(
        json.dumps(
            {
                "episode_filter_key": args.episode_filter_key,
                "dev_groups_available": len(dev_rows),
                "groups_selected": len(chosen),
                "episodes_selected": len({dev_rows[g]["N0"]["episode"] for g in chosen}),
                "projected_forwards_upper_bound": projected,
                "stratification": sample_manifest,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.plan_only:
        return

    import torch

    reduction_groups = list(chosen)
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

    started = time.time()
    every = max(int(args.progress_every), 0)

    def progress(position: int, total: int) -> None:
        if position != total and (every <= 0 or position % every):
            return
        state = {
            "stage": "searching",
            "groups_done": position,
            "groups_total": total,
            "forwards": scorer.forwards,
            "cache_hits": scorer.cache_hits,
            "max_prompt_tokens": scorer.max_prompt_tokens,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        print(json.dumps({"probe_progress": state}), flush=True)
        if heartbeat_path is not None:
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_path.write_text(
                json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
            )

    annotation_cache: dict[str, list[dict]] = {}
    per_group: dict[str, dict[str, dict[str, Any]]] = {}
    reconciled_pairs = 0

    def run_group(pair_group: str) -> None:
        nonlocal reconciled_pairs
        point = DecisionPoint(
            dev_rows[pair_group],
            annotations=args.annotations,
            annotation_cache=annotation_cache,
            image_root=image_root,
        )
        reconciled_pairs += point.reconciled_pairs
        per_group[pair_group] = {
            prompt_format: analyse_point(
                point,
                prompt_format,
                scorer,
                exact_max_candidates=args.exact_max_candidates,
            )
            for prompt_format in FORMATS
        }

    try:
        if not args.require_cached:
            probe_point = DecisionPoint(
                dev_rows[chosen[0]],
                annotations=args.annotations,
                annotation_cache=annotation_cache,
                image_root=image_root,
            )
            print(
                json.dumps(
                    {
                        "encoder_parity": scorer.verify_mean_parity(
                            probe_point,
                            probe_point.recent(1),
                            SPARSE_MULTITURN_PROMPT_FORMAT,
                        )
                    }
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

    # note (luojiaxuan): token 长度必须从 **cache** 上统计,不能只用本进程 scorer 的
    # 计数器——归约那一遍是纯缓存命中,scorer.max_prompt_tokens 恒为 0,于是"有没有
    # 发生截断"这条证据会在最终报告里静默变成 0。
    totals = [
        int(entry["total_tokens"])
        for entry in cache.entries.values()
        if isinstance(entry, dict) and "total_tokens" in entry
    ]
    prompts = [
        int(entry["prompt_tokens"])
        for entry in cache.entries.values()
        if isinstance(entry, dict) and "prompt_tokens" in entry
    ]
    provenance = {
        "device": args.device,
        "computed_forwards": scorer.forwards > 0,
        "device_name": None,
        "max_prompt_tokens": max(prompts, default=scorer.max_prompt_tokens),
        "max_total_tokens": max(totals, default=scorer.max_total_tokens),
        "mean_total_tokens": (math.fsum(totals) / len(totals)) if totals else None,
        "sequences_measured": len(totals),
        "model_max_position_embeddings": None,
        "encoder_parity": scorer.mean_parity,
    }
    if scorer.runtime is not None:
        try:
            provenance["device_name"] = torch.cuda.get_device_name(
                scorer.runtime.model.device
            )
        except Exception:  # noqa: BLE001 - provenance must never break a run
            pass
        try:
            text_config = getattr(scorer.runtime.model.config, "text_config", None)
            provenance["model_max_position_embeddings"] = int(
                getattr(text_config, "max_position_embeddings", 0)
            ) or None
        except Exception:  # noqa: BLE001
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
        "formats": list(FORMATS),
        "budgets": list(BUDGETS),
        "search": {
            "beam_width": BEAM_WIDTH,
            "recent_seeded": True,
            "exact_max_candidates": args.exact_max_candidates,
            "per_format_independent": True,
        },
        "adapter": "none (frozen policy; no injection, no checkpoint load)",
        "stratification": sample_manifest,
        "image_path_pairs_reconciled": reconciled_pairs,
        "cache_fingerprint": cache_fingerprint,
        "cache": cache.stats(),
        "provenance": provenance,
        "forward_passes": scorer.forwards,
        "cache_hits": scorer.cache_hits,
        "smoke_run": bool(args.max_groups),
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
    summary: dict[str, dict[str, Any]] = {}
    diagnostics_summary: dict[str, dict[str, Any]] = {}
    group_episode = {
        group: dev_rows[group]["N0"]["episode"] for group in reduction_groups
    }
    for prompt_format in FORMATS:
        per_quantity: dict[str, dict[str, float]] = {}
        for group in reduction_groups:
            for name, value in per_group[group][prompt_format]["quantities"].items():
                per_quantity.setdefault(name, {})[group] = value
        bootstrap = EpisodeClusterBootstrap(
            per_quantity,
            group_episode,
            replicates=args.bootstrap_replicates,
            confidence=args.bootstrap_confidence,
            seed=args.bootstrap_seed,
        )
        names = sorted(per_quantity)
        point_estimate = bootstrap.point_estimate()
        intervals = bootstrap.intervals(bootstrap.replicate_aggregates(), names)
        summary[prompt_format] = {
            name: {
                "point": point_estimate.get(name),
                **intervals[name],
                "support": bootstrap.support.get(name),
            }
            for name in names
        }
        diagnostics = [per_group[group][prompt_format]["diagnostics"] for group in reduction_groups]
        block: dict[str, Any] = {
            "search_modes": {
                mode: sum(1 for d in diagnostics if d["search_mode"] == mode)
                for mode in sorted({d["search_mode"] for d in diagnostics})
            },
            "mean_sets_scored": (
                math.fsum(d["sets_scored"] for d in diagnostics) / len(diagnostics)
            ),
        }
        for budget in BUDGETS:
            ranks = [d[f"recent_percentile_b{budget}"] for d in diagnostics]
            block[f"b{budget}"] = {
                "mean_pool_size": (
                    math.fsum(d[f"pool_size_b{budget}"] for d in diagnostics)
                    / len(diagnostics)
                ),
                "oracle_is_recent_rate": (
                    sum(1 for d in diagnostics if d[f"oracle_is_recent_b{budget}"])
                    / len(diagnostics)
                ),
                "recent_in_search_space_rate": (
                    sum(1 for d in diagnostics if d[f"recent_in_search_space_b{budget}"])
                    / len(diagnostics)
                ),
                "mean_recent_percentile_in_pool": math.fsum(ranks) / len(ranks),
            }
        diagnostics_summary[prompt_format] = block

    gate = {}
    for prompt_format in FORMATS:
        row = summary[prompt_format].get("oracle_gain_b4", {})
        held = summary[prompt_format].get("oracle_heldout_gain_b4", {})
        gate[prompt_format] = {
            "criterion": "CI_low(Q(S_oracle_B4) - Q(Recent_B4)) > 0",
            "oracle_gain_b4": row.get("point"),
            "oracle_gain_b4_ci_low": row.get("ci_low"),
            "raw_verdict": (
                None if row.get("ci_low") is None else bool(row["ci_low"] > 0)
            ),
            "raw_verdict_is_near_tautological": True,
            "heldout_gain_b4": held.get("point"),
            "heldout_gain_b4_ci_low": held.get("ci_low"),
            "heldout_verdict": (
                None if held.get("ci_low") is None else bool(held["ci_low"] > 0)
            ),
        }

    report = {
        "schema_version": REPORT_SCHEMA,
        "pair_groups": len(reduction_groups),
        "episodes": len(set(group_episode.values())),
        "bootstrap": {
            "cluster_unit": "episode",
            "replicates": args.bootstrap_replicates,
            "confidence": args.bootstrap_confidence,
            "seed": args.bootstrap_seed,
        },
        "quantity_algebra": {
            "oracle_gain_bB": "Q(S_oracle_B) - Q(Recent_B)  [in-sample optimistic upper bound]",
            "oracle_heldout_gain_bB": (
                "token-fold cross-fit: select S on one half of the target tokens, "
                "evaluate against Recent on the other half, averaged over both "
                "directions. B1 is leak-free (exhaustive singleton pool); B2/B4 "
                "inherit a mild pool-selection leak from the full-Q beam pruning."
            ),
            "random_gain_bB": "Q(S_random_sparse_B) - Q(Recent_B)",
            "recent_over_b0_bB": "Q(Recent_B) - Q(B0)",
            "oracle_over_b0_bB": "Q(S_oracle_B) - Q(B0)",
            "pool_mean_minus_recent_bB": (
                "mean over the scored size-B sets - Q(Recent_B). Only B1 is an "
                "unbiased 'average frame' baseline: its pool is the exhaustive "
                "singleton sweep. The B2/B4 pools are beam children and are biased "
                "upward, so read those two as 'Recent vs the sets the beam liked', "
                "not as 'Recent vs a random set'."
            ),
            "heldout_over_poolmean_bB": (
                "oracle_heldout_gain_bB - pool_mean_minus_recent_bB. This is the "
                "cross-fit's correct null: if every set in the pool had the same "
                "true utility, selecting on one token fold would land on the pool "
                "mean in the other, so >0 is what 'the selection carried "
                "transferable information' actually looks like."
            ),
            "corpus_sparse_gain_native": (
                "Q(corpus choose_sparse set) - Q(Recent) at the group's native budget; "
                "reconciles this subsample against the Gate-1 numbers"
            ),
        },
        "summary": summary,
        "diagnostics": diagnostics_summary,
        "gate3": gate,
        "per_group": {
            group: {
                prompt_format: per_group[group][prompt_format]
                for prompt_format in FORMATS
            }
            for group in reduction_groups
        },
        **common,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("")
    print(
        f"pair-groups: {len(reduction_groups)}   episodes: "
        f"{len(set(group_episode.values()))}   forwards: {scorer.forwards}"
    )
    if args.max_groups:
        print(f"*** SMOKE RUN (--max-groups {args.max_groups}) — not a formal result ***")
    for prompt_format in FORMATS:
        print("")
        print(f"=== {prompt_format} ===")
        print(f"{'quantity':<34}{'point':>10}{'ci_low':>12}{'ci_high':>12}")
        for name in sorted(summary[prompt_format]):
            row = summary[prompt_format][name]
            if row["point"] is None:
                continue
            print(
                f"{name:<34}{row['point']:>10.4f}{row['ci_low']:>12.4f}"
                f"{row['ci_high']:>12.4f}"
            )
    print("")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
