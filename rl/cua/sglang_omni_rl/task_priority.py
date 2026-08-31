# note (luojiaxuan): 难度优先任务采样。纯逻辑模块,无 slime/lite 依赖,
# 便于本地单测;IO 与接线在 rollout_grpo.py。
#
# 统计:task_stats[env_key] = {groups, mixed, attempts, successes}
# (由 convert 每批增量更新;混合组 = 组内奖励非全同 = 有梯度的组)。
# 优先分:P(mixed | p_hat) = 1 - p_hat^G - (1-p_hat)^G,G = 组内 rollout 数。
#   这是"下一组会出现混合结果(=有梯度)的概率"的解析式,由任务成功率
#   p_hat 推出。比直接用历史混合组率省样本:p_hat 的估计用了 G×groups 个
#   episode,而混合组率只用了 groups 个。它自然把"太难"(p→0)与"太易"
#   (p→1)同时降权——p=0.5 时几乎每组都有信息(0.99),p=0.01 时只有 7.7%。
# 采样三层(外审 20260830):
#   MAIN     按 P(mixed) 加权 —— 预算主体给有梯度的任务;
#   FRONTIER 零成功但尝试次数少的 —— 还没探索够,保留发现机会;
#   AUDIT    零成功且尝试次数多的 —— 极小配额,防"模型变强后永久放弃"。
#   不设纯均匀地板(均匀会把预算平摊给已证明无梯度的任务)。
from __future__ import annotations

import json
import math
import os
import random
import tempfile

FLOOR = 0.25          # 兼容旧调用签名;三层口径下等于 FRONTIER+AUDIT
FRONTIER = 0.20       # 零成功且欠探索任务的配额
AUDIT = 0.05          # 零成功且已充分尝试任务的极小配额
EXPLORE_C = 0.5       # 探索加成系数
OVERSAMPLE = 3        # 候选超采倍数(受缓冲余量约束)
GROUP_SIZE = 8        # G:组内 rollout 数,进 P(mixed) 解析式
AUDIT_ATTEMPTS = 24   # 尝试数超过此值仍零成功 -> 归 AUDIT 层


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
        ent.setdefault("attempts", 0)
        ent.setdefault("successes", 0)
        for rewards in groups:
            ent["groups"] += 1
            if len(set(rewards)) > 1:
                ent["mixed"] += 1
            ent["attempts"] += len(rewards)
            ent["successes"] += sum(1 for r in rewards if r > 0)
    return stats


def p_mixed(p_hat: float, group_size: int = GROUP_SIZE) -> float:
    """一组 group_size 次伯努利(成功率 p_hat)出现混合结果的概率。"""
    return 1.0 - p_hat ** group_size - (1.0 - p_hat) ** group_size


def success_rate(ent: dict) -> float:
    """Laplace 平滑的成功率;旧 stats 无 attempts 时退回混合率的粗略代理。"""
    a = ent.get("attempts")
    if a:
        return (ent.get("successes", 0) + 1) / (a + 2)
    g, m = ent.get("groups", 0), ent.get("mixed", 0)
    return (m + 1) / (2 * (g + 2))


def priority(stats: dict, key: str) -> float:
    ent = stats.get(key, {"groups": 0, "mixed": 0})
    g = ent.get("groups", 0)
    return p_mixed(success_rate(ent)) + EXPLORE_C / math.sqrt(g + 1)


def _layer(ent: dict) -> str:
    """有过成功 -> main;零成功但欠探索 -> frontier;否则 audit。"""
    if ent.get("successes", 0) > 0 or ent.get("mixed", 0) > 0:
        return "main"
    # 旧格式 stats 无 attempts,用 groups*G 折算,否则已试过很多组的死任务
    # 会被误判为"欠探索"而占走 FRONTIER 配额。
    attempts = ent.get("attempts") or ent.get("groups", 0) * GROUP_SIZE
    if attempts < AUDIT_ATTEMPTS:
        return "frontier"
    return "audit"


def choose(
    candidate_keys: list[str], n: int, stats: dict, rng: random.Random,
    floor: float = FLOOR,
) -> list[int]:
    """从候选(env_key 列表,含重复)中选 n 个下标,无放回。
    每名额先按三层配额掷层,未命中则按 P(mixed) 优先分加权。"""
    remaining = list(range(len(candidate_keys)))
    picked: list[int] = []
    frontier_p = FRONTIER * (floor / FLOOR)
    audit_p = AUDIT * (floor / FLOOR)
    while remaining and len(picked) < n:
        roll = rng.random()
        layer_pick = ("frontier" if roll < frontier_p else
                      "audit" if roll < frontier_p + audit_p else None)
        if layer_pick:
            pool = [k for k, j in enumerate(remaining)
                    if _layer(stats.get(candidate_keys[j], {})) == layer_pick]
            if pool:
                picked.append(remaining.pop(pool[rng.randrange(len(pool))]))
                continue
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
