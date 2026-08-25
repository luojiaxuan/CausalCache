# note (luojiaxuan): ROLLOUT_MODULE shim —— run_grpo.sh 以
# ROLLOUT_MODULE=sglang_omni_rl.rollout_grpo 解析三个 dotted 入口。
# import 副作用注册 gui_owl family(每个 Ray actor 都会 import 本模块)。
# convert 在 slime 原版(组优势归一化)之上旁路落盘 episode returns,
# 供 selector RLOO trainer 与 rollout 决策日志(cc_episode 键)对账。
from __future__ import annotations

import json
import logging
import os
from typing import Any

from sglang_omni_rl import registration  # noqa: F401 — 注册副作用
from lite.train.rollout.grpo import (  # noqa: F401 — dotted-path 再导出
    convert_samples_to_train_data as _base_convert,
    generate,
    generate_rollout,
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
        }
    if not seen:
        logger.warning("CC_RET_DIR 设定但本轮无 cc_episode 样本 —— "
                       "adapter 注册或 metadata 通路断了,selector 将拿不到奖励")
        return
    path = os.path.join(out_dir, "episode_returns.jsonl")
    with open(path, "a") as f:
        for rec in seen.values():
            f.write(json.dumps(rec) + "\n")


def convert_samples_to_train_data(args: Any, samples) -> dict:
    _dump_episode_returns(samples)
    return _base_convert(args, samples)
