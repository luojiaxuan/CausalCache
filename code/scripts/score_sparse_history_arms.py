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
#
# note (luojiaxuan): 494 个留出组 × 三个 checkpoint 单卡约四小时,所以本脚本支持
# **只对前向分片、绝不对统计分片**的两段式跑法:
#
#     # 第一段:M 个进程各占一张卡,只算自己那一份前向,写进共享 --score-cache
#     for i in 0..M-1:
#         score_sparse_history_arms.py ... --shard-index i --shard-count M \
#             --skip-reduction --score-cache CACHE --device cuda:i
#     # 第二段:一个不分片的进程,全部命中缓存后只做归约
#     score_sparse_history_arms.py ... --score-cache CACHE --require-cached
#
# 分片只决定"这个进程去算哪些留出组的前向",而 bootstrap、gate 判定、composite 与
# 选点**只在第二段的完整留出集上发生一次**,与单进程跑法逐值相同。分片进程被硬性
# 要求带 --skip-reduction:在 1/M 的切片上跑一遍 bootstrap 不只是白算,它会产出一份
# 长得和验收报告一模一样、分母却小了 M 倍的 JSON,而"分母悄悄变小"正是本脚本从第一天
# 起就在防的那类偏差。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
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
    SPARSE_ALL_NEGATIVE_KINDS,
    SPARSE_ARM_CONTRACTS,
    SPARSE_GATE_VOCABULARY,
    SPARSE_HELDOUT_REQUIRED_SLOTS,
    SPARSE_HELDOUT_REQUIRED_SLOTS_BY_SCHEMA,
    SPARSE_POSITIVE_SLOT,
    SPARSE_SUPPORTED_ADAPTER_TYPES,
    adapter_scope_for_sample,
    adapter_settings,
    build_sparse_history_heldout_units,
    compile_sparse_gates,
    encode_sample,
    evaluate_gate_expression,
    inject_lora,
    load_config,
    load_lora_state_dict,
    lora_state_dict,
    mean_target_logprob,
    normalize_desktop_splits,
    sparse_config_sample_schema,
    sparse_diagnostic_keys,
    validate_sparse_gates,
)

REPORT_SCHEMA = "causalcache.sparse_history_gate_report.v1"
# 分片进程写的是"我算完了这些前向"的回执,不是验收报告——schema 不同,任何按
# REPORT_SCHEMA 取数的下游都不会误把切片当成留出集。
SHARD_REPORT_SCHEMA = "causalcache.sparse_history_score_shard.v1"
CACHE_FINGERPRINT_KEY = "cache_fingerprint"
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


def arm_partition_for_schema(schema: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(bypass slots, active slots) — one schema's heldout measurement partition.

    # note (luojiaxuan): 划分从契约表的 adapter_mode 列**推导**,不再手抄第二份:
    # v2/v6 推导结果与上面的冻结常量逐字相同(N0/R0/S0 + RA/SA,由断言锁住),
    # 桌面 schema 没有 N0,得到 R0/S0 + RA/SA。负样本臂不在此列(role 字段决定)。
    """
    contract = SPARSE_ARM_CONTRACTS[schema]
    required = SPARSE_HELDOUT_REQUIRED_SLOTS_BY_SCHEMA[schema]
    frozen = tuple(slot for slot in required if contract[slot][4] == "bypass")
    active = tuple(slot for slot in required if contract[slot][4] == "active")
    return frozen, active


if arm_partition_for_schema("causalcache.sparse_history_sample.v2") != (
    FROZEN_ARM_SLOTS,
    ACTIVE_ARM_SLOTS,
):
    raise RuntimeError("schema-derived arm partition drifted from the v2 constants")


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
def shard_cache_path(path: Path, shard_index: int, shard_count: int) -> Path:
    """Return the one file this process may write, given the logical cache path.

    # note (luojiaxuan): 不分片时仍旧原样写 ``path``,老 cache 与老命令行完全不受影响;
    # 分片时每个进程只写自己的 ``<stem>.shardIII-of-CCC<suffix>``。"一个 writer 一个
    # 文件"是这里唯一的并发模型——共享一个 append 句柄看起来也能用,但两个进程的部分
    # 写一旦交错,坏掉的是一条**分数**,而分数错了报告照样生成。
    """
    if shard_count <= 1:
        return path
    return path.with_name(
        f"{path.stem}.shard{shard_index:03d}-of-{shard_count:03d}{path.suffix}"
    )


def cache_member_paths(path: Path) -> list[Path]:
    """Every file belonging to one logical cache, in a stable order."""
    members = [path] if path.is_file() else []
    members.extend(sorted(path.parent.glob(f"{path.stem}.shard*-of-*{path.suffix}")))
    return members


class ScoreCache:
    """Append-only (adapter, group, arm, mode) -> score cache with resume.

    # note (luojiaxuan): 打分是一个几小时量级的多 checkpoint GPU 作业,共享机器上被
    # 别的 session 收掉容器、被 OOM 打断都是常态。每条前向算完就落盘,重启时按
    # cache_key 跳过已完成项,重跑从断点继续而不是从头开始。
    #
    # cache_key = ``<adapter_key>|<pair_group>|<arm_slot>|<adapter_mode>``,其中
    # adapter_key 是 checkpoint 文件的 sha256(或 ``frozen`` / ``identity`` 两个哨兵)。
    # 这四段已经足以支撑分片:checkpoint 之间靠 sha256 区分,同一条负样本的 active 与
    # bypass 两次前向靠 adapter_key + mode 区分,不会互相覆盖。**唯一缺的**是"这份
    # cache 是在哪套 config/模型/语料下算出来的"——单进程时这靠人记着,多进程共享一份
    # cache 之后,一个参数敲错的分片会把别的模型的分数悄悄混进同一份报告。所以每个分片
    # 文件的首行写一条 fingerprint,合并时对不上就直接拒绝。
    #
    # 写入分两层:热路径是 append + flush(每条前向即刻可恢复,不必等收尾),收尾时再用
    # 临时文件 + os.replace 把**本进程自己**那份原子地重发一次(去重、排序、fsync)。
    # 逐条前向都做一次全文件 rename 是 O(n²) 的无谓 IO,而这两层合起来给出的保证是一样
    # 的:别的进程要么读到旧的完整文件,要么读到新的完整文件;中途崩溃最多在 append 日志
    # 末尾留半行,而末行残行在读取时是被容忍并跳过的。
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
        # entries 是"我看得见的全部分数"(自己的 + 别的分片已落盘的);own 只装本进程
        # 这个文件的内容,收尾原子重发时只能写 own,否则每个分片都会把别人的条目抄进
        # 自己的文件,cache 体积按分片数平方膨胀。
        self.entries: dict[str, float] = {}
        self.own: dict[str, float] = {}
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
                f"{self.shard_path}; two writers on one shard file lose entries — "
                "give every concurrent process a distinct --shard-index"
            )
        if fingerprint is not None and started_empty:
            self._write(self.handle, {CACHE_FINGERPRINT_KEY: fingerprint})

    @staticmethod
    def _write(handle: Any, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()

    def _load_member(self, member: Path) -> dict[str, Any]:
        """Read one cache file into the merged view, tolerating a torn final line."""
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
                # note (luojiaxuan): 只有**没有换行结尾的最后一行**允许是残行——那是
                # 一个分片正在写、或被 SIGKILL 掐在半路的正常状态,丢掉它重算一条前向
                # 就行。中间出现坏行则是真的损坏(比如两个 writer 抢过同一个文件),
                # 这时静默跳过等于把一批分数换成"没算过",必须炸掉。
                if position == len(lines) and not line.endswith("\n"):
                    torn_tail = True
                    break
                raise ValueError(f"{member}:{position} is not JSON: {error}")
            if CACHE_FINGERPRINT_KEY in record:
                self._check_fingerprint(member, record[CACHE_FINGERPRINT_KEY])
                fingerprinted = True
                continue
            score = float(record["score"])
            self.entries[record["cache_key"]] = score
            if mine:
                self.own[record["cache_key"]] = score
            loaded += 1
        # note (luojiaxuan): 本次改动之前写的 cache 没有 fingerprint 行,拒绝它们会让
        # 已有的断点全部作废,所以照常接收——但"这份文件没法被校验"必须留在报告里,
        # 否则一份来路不明的 cache 混进归约后,报告上看不出任何痕迹。
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

    def get(self, cache_key: str) -> float | None:
        return self.entries.get(cache_key)

    def put(self, cache_key: str, score: float) -> None:
        self.entries[cache_key] = score
        self.own[cache_key] = score
        if self.handle is not None:
            self._write(self.handle, {"cache_key": cache_key, "score": score})

    def stats(self) -> dict[str, Any]:
        return {
            "path": None if self.path is None else str(self.path),
            "shard_path": None if self.shard_path is None else str(self.shard_path),
            "merged_entries": len(self.entries),
            "own_entries": len(self.own),
            "unfingerprinted_sources": [
                source["path"]
                for source in self.sources
                if not source["fingerprinted"]
            ],
            "sources": self.sources,
        }

    def close(self) -> None:
        """Atomically republish this process's own shard, then release the lock."""
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
                            {"cache_key": cache_key, "score": self.own[cache_key]},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.shard_path)
        self.handle.close()
        self.handle = None


class LazyPolicyEngine:
    """Own the policy runtime and the injected adapter, built on first real forward.

    # note (luojiaxuan): 两段式跑法的第二段是"全部命中缓存后只做归约"。如果归约进程
    # 照旧先把 8B policy 装进显存、再逐个 torch.load checkpoint,那一步既要排队等一张
    # 空卡、又会和还在跑的分片抢显存,而它其实一次前向都不做。把 runtime 构造、adapter
    # 注入与 state_dict 装载全部推迟到**第一次真正的 cache miss**,归约就退化成一个几秒
    # 钟的纯 CPU 作业;配合 --require-cached,"归约没有偷偷重算任何一条前向"这件事也就
    # 从"事后看 forward_passes"变成了硬失败。
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        config: dict[str, Any],
        adapter_options: dict[str, Any] | None,
        torch: Any,
        adapter_type: str = "history_gated_kv",
    ) -> None:
        if adapter_type not in SPARSE_SUPPORTED_ADAPTER_TYPES:
            raise ValueError(f"unsupported adapter_type {adapter_type!r}")
        self.adapter_type = adapter_type
        self.args = args
        self.config = config
        self.adapter_options = adapter_options
        self.torch = torch
        self.runtime: Any = None
        self.merge_size: int | None = None
        self.wrapped: dict[str, Any] | None = None
        self.identity_state: dict[str, Any] | None = None
        self.built = False
        self._requested_state: Path | None = None
        self._applied_state: Path | None = None
        self._state_is_current = True

    def select_adapter_state(self, path: Path | None) -> None:
        """Declare which checkpoint the next forwards must see (None = identity)."""
        self._requested_state = path
        self._state_is_current = False

    def prepare(self) -> Any:
        """Materialise everything the next forward needs and return the runtime."""
        if not self.built:
            self._build()
        if not self._state_is_current:
            self._apply_state()
        return self.runtime

    def _build(self) -> None:
        args = self.args
        self.runtime = GUIOwlV21OfficialToolsRuntime(
            model_dir=args.model_dir,
            expected_snapshot_manifest=(
                args.repository_root / self.config["policy_snapshot_manifest"]
            ),
            device=args.device,
            target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
        )
        model = self.runtime.model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.config.use_cache = False
        model.eval()

        # adapter 注入按类型分派(交接 §8.4 的三行 ablation 共用本 scorer);
        # 三种类型的 identity 语义相同:刚注入的零初始化 adapter。
        if self.adapter_type == "history_gated_kv":
            from causalcache.policy.history_gated_lora import (
                history_gated_state_dict,
                inject_history_gated_kv,
            )

            self.wrapped = inject_history_gated_kv(
                model,
                layer_count=self.adapter_options["layer_count"],
                rank=self.adapter_options["rank"],
                alpha=self.adapter_options["alpha"],
            )
            self._state_dict = history_gated_state_dict
        elif self.adapter_type == "ungated_kv_lora":
            self.wrapped = inject_lora(
                model,
                rank=self.adapter_options["rank"],
                alpha=self.adapter_options["alpha"],
                target_modules=("k_proj", "v_proj"),
                torch=self.torch,
                last_layer_count=self.adapter_options["layer_count"],
            )
            self._state_dict = lora_state_dict
        else:
            self.wrapped = inject_lora(
                model,
                rank=self.config["lora"]["rank"],
                alpha=self.config["lora"]["alpha"],
                target_modules=tuple(self.config["lora"]["target_modules"]),
                torch=self.torch,
            )
            self._state_dict = lora_state_dict
        for lora in self.wrapped.values():
            lora.lora_a.requires_grad_(False)
            lora.lora_b.requires_grad_(False)
        # note (luojiaxuan): identity 行的定义是"刚注入的零初始化 adapter"。之前它靠
        # "identity 永远排在 checkpoint 前面"这条排序巧合成立;打分一旦可以从缓存里
        # 任意跳过若干行,顺序就不再保证,于是把这份原始权重显式存下来,回到 identity
        # 时装回去,而不是依赖"还没人覆盖过它"。
        self.identity_state = self._state_dict(self.wrapped)
        self.merge_size = int(self.runtime.processor.image_processor.merge_size)
        self.built = True

    def provenance(self) -> dict[str, Any]:
        """What actually ran the forwards, or None when nothing was computed.

        # note (luojiaxuan): 分数在不同型号的卡上未必逐 bit 相同,而分片让"一份报告里
        # 的分数来自几张不同的卡"第一次成为常态。这不进 cache fingerprint——因为换卡
        # 续跑是完全正当的操作,拿它做硬校验会把断点续跑一并否掉——但每个分片跑在什么
        # 卡上必须留痕,否则日后发现末位不一致时已经无从追查。
        """
        if not self.built:
            return {"device": self.args.device, "computed_forwards": False}
        name = None
        try:
            name = self.torch.cuda.get_device_name(self.runtime.model.device)
        except Exception:  # noqa: BLE001 - provenance must never break a run
            name = None
        return {
            "device": self.args.device,
            "device_name": name,
            "computed_forwards": True,
        }

    def _apply_state(self) -> None:
        if self.adapter_type == "history_gated_kv":
            from causalcache.policy.history_gated_lora import (
                load_history_gated_state_dict as load_state,
            )
        else:
            load_state = load_lora_state_dict

        requested = self._requested_state
        if requested is None:
            if self._applied_state is not None:
                load_state(self.wrapped, self.identity_state)
        else:
            load_state(self.wrapped, self.torch.load(requested, map_location="cpu"))
        self._applied_state = requested
        self._state_is_current = True


class ArmScorer:
    """Teacher-forced mean log-prob of one arm under one adapter state."""

    def __init__(
        self,
        engine: LazyPolicyEngine,
        *,
        image_root: Path,
        torch: Any,
        cache: ScoreCache,
        require_cached: bool = False,
    ) -> None:
        self.engine = engine
        self.image_root = image_root
        self.torch = torch
        self.cache = cache
        self.require_cached = require_cached
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
        mode = sample["adapter_mode"] if adapter_mode is None else adapter_mode
        cache_key = f"{adapter_key}|{sample['pair_group']}|{slot}|{mode}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.require_cached:
            # note (luojiaxuan): 归约进程本来就该一条前向都不跑。默默重算的代价不是慢
            # 一点,而是几小时之后才发现某个分片其实没写完、或者写去了别的 --score-cache,
            # 而报告已经按一份缺角的留出集出完了。报错里带上 cache 的文件清单:少一个
            # 分片文件和"某一组算崩了"是两种完全不同的故障,而这一行就能区分。
            inventory = ", ".join(
                f"{source['path'].rsplit('/', 1)[-1]}={source['entries']}"
                for source in self.cache.sources
            )
            raise SystemExit(
                f"--require-cached is set but {cache_key!r} is missing from the score "
                f"cache {self.cache.path}; some shard did not finish, or wrote to a "
                f"different cache path. Cache files seen: [{inventory or 'none'}]"
            )
        runtime = self.engine.prepare()
        encoded = encode_sample(
            runtime, sample, dataset_root=self.image_root, torch=self.torch
        )
        if encoded is None:
            # note (luojiaxuan): 留出集上不允许静默跳过——一条编码不出来的样本会让整组
            # 退出分母,而"分母悄悄变小"是最难在报告里被发现的评测偏差。
            raise ValueError(
                f"heldout sample {sample['sample_id']!r} produced an empty target "
                "encoding; the scored denominator must stay reconciled"
            )
        # adapter 开关只读 adapter_mode 字段(或调用方显式传入的 bypass 覆盖),
        # 与 trainer 共用同一个分派入口(adapter_scope_for_sample,按 adapter_type
        # 选择 HGKV mask 上下文或 plain-LoRA bypass 开关),绝不按名字前缀猜。
        scope = adapter_scope_for_sample(
            self.engine.adapter_type,
            encoded,
            sample,
            merge_size=self.engine.merge_size,
            adapter_mode=adapter_mode,
            slot=slot,
        )
        with scope:
            with self.torch.no_grad():
                value = float(
                    mean_target_logprob(
                        runtime.model, encoded, torch=self.torch
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
    frozen_slots: tuple[str, ...] = FROZEN_ARM_SLOTS,
) -> dict[str, dict[str, float]]:
    """Score every adapter-bypassed forward once; these do not depend on a checkpoint."""
    frozen: dict[str, dict[str, float]] = {}
    for position, (pair_group, group) in enumerate(sorted(groups.items()), start=1):
        scores: dict[str, float] = {}
        for slot in frozen_slots:
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
    active_slots: tuple[str, ...] = ACTIVE_ARM_SLOTS,
) -> dict[str, dict[str, float]]:
    """Score RA/SA and this schema's negatives with the adapter active."""
    active: dict[str, dict[str, float]] = {}
    for position, (pair_group, group) in enumerate(sorted(groups.items()), start=1):
        scores: dict[str, float] = {}
        for slot in active_slots:
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
    frozen_slots: tuple[str, ...] = FROZEN_ARM_SLOTS,
    active_slots: tuple[str, ...] = ACTIVE_ARM_SLOTS,
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
            slot: frozen[pair_group][slot] for slot in frozen_slots
        }
        arm_scores.update(
            {slot: active[pair_group][slot] for slot in active_slots}
        )
        for quantity, algebra in sorted(derived_algebra.items()):
            value = evaluate_gate_expression(algebra, arm_scores)
            per_quantity.setdefault(quantity, {})[pair_group] = value
        positive = arm_scores[SPARSE_POSITIVE_SLOT]
        for kind, slot in sorted(negative_slots(samples, group).items()):
            negative_active = active[pair_group][slot]
            negative_bypass = frozen[pair_group][f"{slot}@bypass"]
            # 量名记法 SA_minus_<arm_slot>(v2/v6 的 slot 本就是 SA_neg_<kind>,
            # 取值逐字节不变;桌面负臂 WA 由此得到 SA_minus_WA)。
            gap_key, _drift_abs_key = sparse_diagnostic_keys(kind, arm_slot=slot)
            per_quantity.setdefault(gap_key, {})[pair_group] = (
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
    # note (luojiaxuan): 遍历**全部** kind(含 v6 的 age_matched),不是只有 v5 那三个。
    # 少一个 kind 的后果不是报错,而是它的 <kind>_drift_abs 永远算不出来,于是留出集
    # 报告里这一项静默变成 None —— 与"这个负样本没被推动"长得一模一样。v5 语料里不存在
    # 的 kind 本来就取不到 <kind>_drift,照样得到 None,行为逐字不变。
    enriched = dict(values)
    for kind in SPARSE_ALL_NEGATIVE_KINDS:
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
        for kind in SPARSE_ALL_NEGATIVE_KINDS:
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
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help=(
            "which slice of the heldout pair-groups this process scores; sharding "
            "splits forwards only, never the statistics"
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help=(
            "number of scoring processes sharing one --score-cache (default 1 = no "
            "sharding). Groups are cut as sorted(pair_group)[index::count]."
        ),
    )
    parser.add_argument(
        "--skip-reduction",
        action="store_true",
        help=(
            "score and cache the forwards, then stop: no bootstrap, no gates, no "
            "composite, no selection. Required on every sharded process."
        ),
    )
    parser.add_argument(
        "--require-cached",
        action="store_true",
        help=(
            "fail on the first cache miss instead of running the forward; the "
            "reduction pass over a shard-filled cache should never compute anything"
        ),
    )
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
        "--episode-filter",
        type=Path,
        default=None,
        help=(
            "JSON file carrying an episode list; only groups whose episode appears in "
            "it are scored at all. Keeps the confirmation split physically untouched "
            "while monitoring on dev."
        ),
    )
    parser.add_argument(
        "--episode-filter-key",
        default=None,
        help="key inside --episode-filter holding the episode list, e.g. dev",
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
            "reduction pass through the cache and nothing else, so without one every "
            "forward they compute is thrown away"
        )
    if args.shard_count > 1 and not args.skip_reduction:
        # note (luojiaxuan): 这条必须是硬失败而不是警告。一个分片进程手里只有 1/M 的
        # 留出组,在它上面跑 bootstrap/gate/选点会产出一份**结构上与验收报告完全一样、
        # 分母却小了 M 倍**的 JSON;等它被误当成结论,已经没有任何字段能提示区别。
        raise SystemExit(
            "a sharded process only holds 1/--shard-count of the heldout set; running "
            "the bootstrap, gates and selection on that slice would silently shrink "
            "the statistical denominator. Pass --skip-reduction on every shard, then "
            "run one unsharded pass over the shared --score-cache to reduce."
        )
    if args.require_cached and args.score_cache is None:
        raise SystemExit("--require-cached is meaningless without --score-cache")

    import torch

    config = load_config(args.config)
    if not bool(config["training"].get("sparse_history", False)):
        raise SystemExit("this scorer only applies to training.sparse_history configs")
    adapter_type, adapter_options = adapter_settings(config)
    if adapter_type not in SPARSE_SUPPORTED_ADAPTER_TYPES:
        raise SystemExit(
            f"sparse_history requires adapter_type in {SPARSE_SUPPORTED_ADAPTER_TYPES}"
        )
    gates = validate_sparse_gates(config)
    compiled = compile_sparse_gates(gates)

    sample_files = resolve_sample_files(args.dataset_root)
    samples = load_sparse_samples(sample_files)
    # 桌面语料 dev→heldout / test 剥离,与 trainer 同一入口;旧语料原样直通。
    samples, desktop_split_counters = normalize_desktop_splits(samples)
    if any(desktop_split_counters.values()):
        print(
            json.dumps(
                {"desktop_split_normalization": desktop_split_counters},
                sort_keys=True,
            ),
            flush=True,
        )
    # config 声明的语料 schema 必须与样本实际 schema 一致 —— 拿 v2 的 config 去打
    # 桌面语料会在这里当场断开,而不是在派生量表里静默错位。
    sample_schemas = {sample["schema_version"] for sample in samples}
    config_schema = sparse_config_sample_schema(config)
    if sample_schemas != {config_schema}:
        raise SystemExit(
            f"config declares corpus schema {config_schema!r} but the dataset "
            f"carries {sorted(sample_schemas)}"
        )
    frozen_slots, active_slots = arm_partition_for_schema(config_schema)
    groups = build_sparse_history_heldout_units(samples)
    if not groups:
        raise SystemExit("the corpus carries no heldout pair-groups to score")
    # note (luojiaxuan): confirmation split 的契约是"只读一次"。若 dev 评测照常打全部
    # 494 组,confirmation 的分数就已经躺进缓存了 —— 即便不去看,那也把"只读一次"
    # 降级成了"只承诺不看一次"。过滤必须在 --max-groups 截断与分片**之前**生效,
    # 否则被排除的组仍会进入某个分片的工作集。
    if args.episode_filter is not None:
        if not args.episode_filter_key:
            raise ValueError("--episode-filter requires --episode-filter-key")
        payload = json.loads(args.episode_filter.read_text(encoding="utf-8"))
        if args.episode_filter_key not in payload:
            available = sorted(k for k, v in payload.items() if isinstance(v, list))
            raise ValueError(
                f"--episode-filter-key {args.episode_filter_key!r} absent from "
                f"{args.episode_filter}; list-valued keys are {available}"
            )
        allowed = set(payload[args.episode_filter_key])
        if not allowed:
            raise ValueError("episode filter selected an empty episode set")
        kept = {
            pair_group: group
            for pair_group, group in groups.items()
            if samples[group[SPARSE_POSITIVE_SLOT]]["episode"] in allowed
        }
        if not kept:
            raise ValueError(
                "episode filter removed every heldout group; check that the filter "
                "file describes the same corpus"
            )
        print(json.dumps({
            "episode_filter": str(args.episode_filter),
            "episode_filter_key": args.episode_filter_key,
            "episodes_allowed": len(allowed),
            "groups_kept": len(kept),
            "groups_dropped": len(groups) - len(kept),
        }, ensure_ascii=False), flush=True)
        groups = kept
    if args.max_groups:
        groups = dict(sorted(groups.items())[: args.max_groups])
    # note (luojiaxuan): 分片切在 --max-groups 截断**之后**,所以 M 个分片的并集与
    # 同样参数的单进程 run 是**逐组相同**的集合,不多不少。切法是排序后的
    # [index::count],纯确定性、不读环境、不掷随机数——换一台机器、换一个启动顺序,
    # 第 i 个分片拿到的组一定还是这一批,断点续跑才谈得上"接着算"。
    reduction_groups = groups
    if args.shard_count > 1:
        assigned = sorted(groups)[args.shard_index :: args.shard_count]
        groups = {pair_group: groups[pair_group] for pair_group in assigned}
        if not groups:
            raise SystemExit(
                f"shard {args.shard_index}/{args.shard_count} covers no pair-group; "
                f"only {len(reduction_groups)} groups are in play, so --shard-count "
                "is larger than the heldout set"
            )
    group_episode = {
        pair_group: samples[group[SPARSE_POSITIVE_SLOT]]["episode"]
        for pair_group, group in reduction_groups.items()
    }

    image_root = args.image_root or args.dataset_root

    def per_shard(path: Path) -> Path:
        """Give each shard its own file so concurrent writers cannot clobber.

        # note (luojiaxuan): --output 与 --heartbeat 都是"一个进程一个文件"的东西。
        # 分片时让四个进程共用同一个路径,报告会互相覆盖(活下来那份看上去完全正常),
        # 心跳会互相刷新(于是一个已经死掉的分片的心跳被别人续着,监控永远发现不了)。
        # 与其指望调用方每次都记得手工加后缀,不如在这里改名并把结果打印出来。
        """
        if args.shard_count <= 1:
            return path
        return path.with_name(
            f"{path.stem}.shard{args.shard_index:03d}"
            f"-of-{args.shard_count:03d}{path.suffix}"
        )

    output_path = per_shard(args.output)
    heartbeat_path = None if args.heartbeat is None else per_shard(args.heartbeat)
    if args.shard_count > 1:
        print(
            json.dumps(
                {
                    "shard": f"{args.shard_index}/{args.shard_count}",
                    "pair_groups": len(groups),
                    "output": str(output_path),
                    "heartbeat": None if heartbeat_path is None else str(heartbeat_path),
                    "score_cache_shard": str(
                        shard_cache_path(
                            args.score_cache, args.shard_index, args.shard_count
                        )
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
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
        if heartbeat_path is not None:
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_path.write_text(
                json.dumps(heartbeat_state, sort_keys=True) + "\n", encoding="utf-8"
            )

    engine = LazyPolicyEngine(
        args=args,
        config=config,
        adapter_options=adapter_options,
        torch=torch,
        adapter_type=adapter_type,
    )

    # note (luojiaxuan): fingerprint 只覆盖"换了它分数就该变"的输入。它的作用不是版本
    # 管理,而是挡住分片流程新引入的那一类事故:四条命令行里有一条把 --model-dir 或
    # --dataset-root 敲错,它照样能算出合法的浮点数并写进同一份共享 cache,归约进程
    # 全部命中、报告正常生成,而其中 1/4 的分数来自另一个模型。
    cache_fingerprint = {
        "config_sha256": sha256_of(args.config),
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "image_root": str(image_root),
        "adapter_type": adapter_type,
        "adapter_options": json.loads(
            json.dumps(adapter_options, sort_keys=True, default=str)
        ),
    }
    cache = ScoreCache(
        args.score_cache,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fingerprint=cache_fingerprint,
    )
    scorer = ArmScorer(
        engine,
        image_root=image_root,
        torch=torch,
        cache=cache,
        require_cached=args.require_cached,
    )

    try:
        # 1) 冻结通道:所有 bypass 前向只算一次,与 checkpoint 无关。
        frozen = score_frozen_arms(
            scorer, samples, groups, progress=progress, frozen_slots=frozen_slots
        )

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
            engine.select_adapter_state(path)
            active = score_active_arms(
                scorer,
                samples,
                groups,
                adapter_key=adapter_key,
                label=label,
                progress=progress,
                active_slots=active_slots,
            )
            if args.skip_reduction:
                print(
                    json.dumps(
                        {
                            "checkpoint_scored": label,
                            "reduction": "skipped",
                            "shard": f"{args.shard_index}/{args.shard_count}",
                            "pair_groups": len(groups),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            per_quantity = group_quantities(
                samples,
                groups,
                frozen,
                active,
                derived_algebra=gates["derived_quantities"],
                frozen_slots=frozen_slots,
                active_slots=active_slots,
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

        sharding = {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "scored_pair_groups": len(groups),
            "group_slice": (
                "all"
                if args.shard_count == 1
                else f"sorted(pair_group)[{args.shard_index}::{args.shard_count}]"
            ),
            "reduction_ran": not args.skip_reduction,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_reduction:
            # note (luojiaxuan): 分片进程的产物是一张"我把这些组的前向算完并落盘了"的
            # 回执,schema 与验收报告不同,里面没有任何 derived_quantities / gates /
            # selection 字段——不是省略,是根本不存在,所以没有人能从这份 JSON 里读出
            # 一个分母只有 1/M 的结论。真正的交付物是 --score-cache 里的分数。
            shard_report = {
                "schema_version": SHARD_REPORT_SCHEMA,
                "config_path": str(args.config),
                "config_sha256": sha256_of(args.config),
                "dataset_root": str(args.dataset_root),
                "image_root": str(image_root),
                "sharding": sharding,
                "scored_labels": [entry[0] for entry in planned],
                "pair_groups_scored": sorted(groups),
                "score_cache": cache.stats(),
                "cache_fingerprint": cache_fingerprint,
                "scoring_device": engine.provenance(),
                "smoke_run": bool(args.max_groups),
                "group_limit_applied": args.max_groups or None,
                "forward_passes": scorer.forwards,
                "cache_hits": scorer.cache_hits,
                "elapsed_seconds": round(time.time() - started, 1),
                "note": (
                    "forward-only shard: bootstrap, gates, composite and selection are "
                    "deliberately absent and must be produced by one unsharded pass "
                    "over the shared score cache"
                ),
            }
            output_path.write_text(
                json.dumps(shard_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "shard_report": str(output_path),
                        "shard": f"{args.shard_index}/{args.shard_count}",
                        "pair_groups_scored": len(groups),
                        "forward_passes": scorer.forwards,
                        "cache_hits": scorer.cache_hits,
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

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
            "sharding": sharding,
            "score_cache": cache.stats(),
            "scoring_device": engine.provenance(),
            "require_cached": args.require_cached,
            "forward_passes": scorer.forwards,
            "cache_hits": scorer.cache_hits,
            "elapsed_seconds": round(time.time() - started, 1),
            "checkpoints": entries,
            "identity_consistency": identity_consistency,
            "selection": selection,
        }
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "gate_report": str(output_path),
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
