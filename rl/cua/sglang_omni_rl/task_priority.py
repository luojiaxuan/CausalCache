# note (luojiaxuan): 难度优先任务采样(v2 recipe:均匀地板 + 按混合组率
# 加权)。纯逻辑模块,无 slime/lite 依赖,便于本地单测;IO 与接线在
# rollout_grpo.py。
#
# 统计:task_stats[env_key] = {"groups": 总组数, "mixed": 混合组数}
# (由 convert 每批增量更新;混合组 = 组内奖励非全同 = 有梯度的组)。
# 优先分:P(t) = mixed_hat(t) + c/sqrt(groups(t)+1)
#   mixed_hat 用 Laplace 平滑 (mixed+1)/(groups+2);第二项是探索加成,
#   访问少的任务分高(冷启动时全体同分 → 等价均匀)。
# 采样:每个名额以 FLOOR 概率均匀抽(地板,保证"现在全败"的任务不被
# 永久放弃——selector 变强后它们可能翻成混合组,那正是最想要的信号),
# 否则按优先分加权无放回抽。
from __future__ import annotations

import json
import math
import os
import random
import tempfile

FLOOR = 0.25          # 均匀地板占比(v2 recipe 口径 20-30%)
EXPLORE_C = 0.5       # 探索加成系数
OVERSAMPLE = 3        # 候选超采倍数(受缓冲余量约束)


def load_stats(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_stats(path: str, stats: dict) -> None:
    """原子写(同目录 tmp + rename),避免 reader 读到半个 JSON。"""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(stats, f)
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def update_stats(stats: dict, batch_groups: dict[str, list[float]]) -> dict:
    """batch_groups: env_key -> 该批每组的奖励列表(每组一个 list)。
    调用方按组聚合后传入;这里只累计 groups/mixed 计数。"""
    for key, groups in batch_groups.items():
        ent = stats.setdefault(key, {"groups": 0, "mixed": 0})
        for rewards in groups:
            ent["groups"] += 1
            if len(set(rewards)) > 1:
                ent["mixed"] += 1
    return stats


def priority(stats: dict, key: str) -> float:
    ent = stats.get(key, {"groups": 0, "mixed": 0})
    g, m = ent["groups"], ent["mixed"]
    mixed_hat = (m + 1) / (g + 2)          # Laplace 平滑
    return mixed_hat + EXPLORE_C / math.sqrt(g + 1)


def choose(
    candidate_keys: list[str], n: int, stats: dict, rng: random.Random,
    floor: float = FLOOR,
) -> list[int]:
    """从候选(env_key 列表,含重复)中选 n 个下标,无放回。
    每名额先掷 floor 决定均匀/加权;返回选中下标(原顺序无关)。"""
    remaining = list(range(len(candidate_keys)))
    picked: list[int] = []
    while remaining and len(picked) < n:
        if rng.random() < floor:
            i = rng.randrange(len(remaining))
        else:
            weights = [priority(stats, candidate_keys[j]) for j in remaining]
            total = sum(weights)
            r = rng.random() * total
            acc = 0.0
            i = len(remaining) - 1
            for k, w in enumerate(weights):
                acc += w
                if r <= acc:
                    i = k
                    break
        picked.append(remaining.pop(i))
    return picked
