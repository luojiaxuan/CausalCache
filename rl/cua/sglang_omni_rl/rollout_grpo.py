# note (luojiaxuan): ROLLOUT_MODULE shim —— run_grpo.sh 以
# ROLLOUT_MODULE=sglang_omni_rl.rollout_grpo 解析三个 dotted 入口。
# import 副作用注册 gui_owl family(每个 Ray actor 都会 import 本模块)。
# convert 在 slime 原版(组优势归一化)之上:①旁路落盘 episode returns
# (selector RLOO trainer 与决策日志按 cc_episode 对账);②增量维护
# 逐任务混合组率统计(task_stats.json)。
# generate_rollout 在 lite 原版之上装**难度优先任务采样**(v2 recipe:
# 均匀地板+按混合组率加权):替换 data_source.get_samples,直接从
# dataset.samples 按优先级选 prompt 并镜像上游组装语义(该 data source
# 的 add_samples 只读拒绝,超采回还路线不可行)。无统计时严格退化为
# 均匀(与原行为一致);CC_TASK_PRIORITY=0 可整体关闭。
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

from sglang_omni_rl import registration  # noqa: F401 — 注册副作用
from sglang_omni_rl import task_priority as _tp
from lite.train.rollout.grpo import (  # noqa: F401 — dotted-path 再导出
    convert_samples_to_train_data as _base_convert,
    generate,
    generate_rollout as _base_generate_rollout,
)

__all__ = ["generate", "convert_samples_to_train_data", "generate_rollout"]

logger = logging.getLogger(__name__)


def _iter_samples(samples: Any):
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def _ret_dir() -> str:
    """CC_RET_DIR 解析:env 优先;Ray worker 不继承 driver shell,故回退读
    挂载内哨兵文件(launcher 写入,所有 worker 可见)。"""
    d = os.environ.get("CC_RET_DIR", "")
    if d:
        return d
    # realpath:包经 symlink 挂进 CUA_LITE_ROOT 时,__file__ 的字面路径停在
    # cua-lite 侧;哨兵在 cc_recipe 真身旁,必须解析 symlink 后再上跳。
    sentinel = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "CC_RET_DIR.path")
    try:
        return open(sentinel).read().strip()
    except OSError:
        return ""


def _stats_path() -> str:
    d = _ret_dir()
    return os.path.join(d, "task_stats.json") if d else ""


def _sample_env_key(s: Any) -> str | None:
    md = getattr(s, "metadata", None) or {}
    return md.get("env_key") if isinstance(md, dict) else None


def _dump_episode_returns(samples: Any) -> None:
    out_dir = _ret_dir()
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    seen: dict[str, dict[str, Any]] = {}
    for s in _iter_samples(samples):
        md = getattr(s, "metadata", None) or {}
        others = md.get("others", {}) if isinstance(md, dict) else {}
        ep = others.get("cc_episode")
        if not ep or ep in seen:
            continue
        seen[ep] = {
            "episode": ep,
            "return": others.get("episode_return"),
            "group_index": getattr(s, "group_index",
                                   getattr(s, "group_id", None)),
            "frame_policy": others.get("cc_frame_policy"),
            "env_key": _sample_env_key(s),
        }
    if not seen:
        logger.warning("CC_RET_DIR 设定但本轮无 cc_episode 样本 —— "
                       "adapter 注册或 metadata 通路断了,selector 将拿不到奖励")
        return
    path = os.path.join(out_dir, "episode_returns.jsonl")
    with open(path, "a") as f:
        for rec in seen.values():
            f.write(json.dumps(rec) + "\n")

    # ── 逐任务混合组率统计(难度优先采样的数据源;也是全量交接要的
    #    "逐任务混合组率表"本身)──
    sp = _stats_path()
    if not sp:
        return
    by_group: dict[Any, dict[str, Any]] = {}
    for rec in seen.values():
        g = rec["group_index"]
        if g is None or rec["return"] is None or not rec["env_key"]:
            continue
        ent = by_group.setdefault(g, {"key": rec["env_key"], "rewards": []})
        ent["rewards"].append(float(rec["return"]))
    batch_groups: dict[str, list[list[float]]] = {}
    for ent in by_group.values():
        batch_groups.setdefault(ent["key"], []).append(ent["rewards"])
    if batch_groups:
        stats = _tp.update_stats(_tp.load_stats(sp), batch_groups)
        _tp.save_stats(sp, stats)


def convert_samples_to_train_data(args: Any, samples) -> dict:
    _dump_episode_returns(samples)
    return _base_convert(args, samples)


# ---------------------------------------------------------------------------
# 难度优先任务采样(装在 data_source.get_samples 上)
# ---------------------------------------------------------------------------


def _install_prioritized_sampling(data_source: Any, rollout_id: int) -> None:
    """替换 get_samples 为难度优先版本。
    不走"超采+add_samples 回还"——该 data source 的 add_samples 是
    只读拒绝(raise);改为绕过 offset 机制,直接从 dataset.samples 按
    优先级选 prompt,并逐字节镜像上游的组装语义(deepcopy + 用
    data_source 自身的 sample_group_index/sample_index 计数器),批量
    记账与 Sample 身份完全不变。批内对任务无放回(≤1 组/任务/批)。"""
    if getattr(data_source, "_cc_prioritized", False):
        return
    sp = _stats_path()
    if not sp:
        return
    orig = data_source.get_samples
    rng = random.Random(20260825 + rollout_id)

    def prioritized(n: int):
        import copy as _copy

        stats = _tp.load_stats(sp)
        ds = getattr(data_source, "dataset", None)
        prompts = getattr(ds, "samples", None) if ds is not None else None
        if not stats or not prompts or n <= 0:
            return orig(n)  # 冷启动/形状不符:严格均匀(原行为)
        keys = [_sample_env_key(p) or "?" for p in prompts]
        picked = _tp.choose(keys, n, stats, rng)
        groups = []
        for i in picked:
            group = []
            for _ in range(data_source.args.n_samples_per_prompt):
                s = _copy.deepcopy(prompts[i])
                s.group_index = data_source.sample_group_index
                s.index = data_source.sample_index
                data_source.sample_index += 1
                group.append(s)
            data_source.sample_group_index += 1
            groups.append(group)
        logger.info("CC_PRIORITY 选中 %s", [keys[i] for i in picked])
        return groups

    data_source.get_samples = prioritized
    data_source._cc_prioritized = True


def generate_rollout(args: Any, rollout_id: int, data_source: Any,
                     evaluation: bool = False):
    if not evaluation and os.environ.get("CC_TASK_PRIORITY", "1") == "1":
        try:
            _install_prioritized_sampling(data_source, rollout_id)
        except Exception:  # noqa: BLE001 — 安装失败不阻塞 rollout
            logger.warning("难度优先采样安装失败,继续均匀采样", exc_info=True)
    return _base_generate_rollout(args, rollout_id, data_source, evaluation)
