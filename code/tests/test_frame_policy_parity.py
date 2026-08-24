"""回归测试:recent 策略必须与官方 cut_current_messages 逐字节等价。

# note (luojiaxuan): 这是把干预面收缩到"一个函数"之后的安全带。整套论证依赖
# "除帧选择外与官方逐字节相同";若 fork 时 recent 分支就偏了,四臂对照的底座
# 即失效,而这种偏差在成绩上只表现为几个点漂移、极难察觉 —— 2026-08-23 那
# 16pt 的自研 harness 缺口正是这类问题。故用官方原函数作 oracle 做穷举比对。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causalcache.frame_policy import (  # noqa: E402
    OraclePolicy, RandomPolicy, RecentPolicy,
)


def official_cut(messages: list[dict], last_image: int) -> list[dict]:
    """官方 gui_owl.py cut_current_messages 的逐字复制(作为对照 oracle)。

    # note (luojiaxuan): 逐字复制而非 import —— 官方文件在 emulator 镜像里、
    # 不在本仓库;复制体保留原始语义(含 index==1 只留首个 text 项的特例)。
    """
    non_empty_user_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user" and msg.get("content") and len(msg["content"]) > 0:
            non_empty_user_indices.append(i)
    if len(non_empty_user_indices) > last_image:
        indices_to_clear = non_empty_user_indices[:-last_image]
    else:
        indices_to_clear = []
    for index in indices_to_clear:
        if index == 1:
            messages[index]["content"] = [messages[index]["content"][0]]
        else:
            messages[index]["content"] = []
    return messages


def build(n_images: int) -> list[dict]:
    """构造 system + (text+image) + [assistant, user(image)]* 的官方形状。"""
    msgs: list[dict] = [{"role": "system", "content": [{"type": "text", "text": "sys"}]}]
    msgs.append({"role": "user", "content": [{"type": "text", "text": "instr"},
                                             {"type": "image_url",
                                              "image_url": {"url": "0"}}]})
    for k in range(1, n_images):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"a{k}"}]})
        msgs.append({"role": "user",
                     "content": [{"type": "image_url", "image_url": {"url": str(k)}}]})
    return msgs


@pytest.mark.parametrize("n_images", range(1, 9))
@pytest.mark.parametrize("budget", range(1, 7))
def test_recent_matches_official(n_images: int, budget: int) -> None:
    official = official_cut(build(n_images), budget)
    kept_official = [i for i, m in enumerate(official)
                     if m.get("role") == "user" and m.get("content")
                     and any(it.get("type") == "image_url" for it in m["content"])]

    msgs = build(n_images)
    candidates = [i for i, m in enumerate(msgs)
                  if m.get("role") == "user" and m.get("content")]
    kept_ours = RecentPolicy().select(candidates, budget)

    assert kept_ours == kept_official, (
        f"n_images={n_images} budget={budget}: ours={kept_ours} official={kept_official}"
    )


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_all_policies_keep_current_frame(budget: int) -> None:
    """当前观测帧不可丢 —— 环境约束,非策略自由度。"""
    cands = [1, 3, 5, 7, 9, 11]
    for pol in (RecentPolicy(), RandomPolicy(seed=1), OraclePolicy()):
        keep = pol.select(cands, budget, oracle_indices=[3, 5])
        assert cands[-1] in keep, f"{pol.name} dropped the current frame"
        assert len(keep) == min(budget, len(cands)), f"{pol.name} budget mismatch"
        assert keep == sorted(keep), f"{pol.name} must return ascending indices"


def test_oracle_pads_to_budget() -> None:
    """oracle 帧不足时补最近帧,保证各臂看到的帧数一致(对比公平)。"""
    cands = [1, 3, 5, 7, 9]
    keep = OraclePolicy().select(cands, 4, oracle_indices=[3])
    assert len(keep) == 4
    assert 3 in keep and cands[-1] in keep
