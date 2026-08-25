# note (luojiaxuan): 难度优先采样纯逻辑单测(无 slime/lite 依赖,本地可跑)。
import collections
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sglang_omni_rl.task_priority import (  # noqa: E402
    choose, priority, update_stats,
)


def test_update_stats_counts_mixed():
    stats = update_stats({}, {
        "mobileworld@A": [[1.0, 0.0], [0.0, 0.0]],   # 1 混合 + 1 全败
        "mobileworld@B": [[1.0, 1.0]],               # 全成,非混合
    })
    assert stats["mobileworld@A"] == {"groups": 2, "mixed": 1}
    assert stats["mobileworld@B"] == {"groups": 1, "mixed": 0}


def test_priority_orders_by_mixed_rate_and_explores_unseen():
    stats = {
        "hot": {"groups": 20, "mixed": 12},    # 混合率高
        "dead": {"groups": 20, "mixed": 0},    # 全败常客
    }
    assert priority(stats, "hot") > priority(stats, "dead")
    # 从未见过的任务靠探索加成不至于垫底
    assert priority(stats, "unseen") > priority(stats, "dead")


def test_choose_no_replacement_and_full_coverage_when_n_equals_pool():
    keys = [f"t{i}" for i in range(8)]
    picked = choose(keys, 8, {}, random.Random(0))
    assert sorted(picked) == list(range(8))


def test_choose_biases_toward_mixed_but_floor_keeps_coverage():
    keys = ["hot", "dead1", "dead2", "dead3"]
    stats = {
        "hot": {"groups": 30, "mixed": 20},
        "dead1": {"groups": 30, "mixed": 0},
        "dead2": {"groups": 30, "mixed": 0},
        "dead3": {"groups": 30, "mixed": 0},
    }
    rng = random.Random(1)
    counts = collections.Counter()
    for _ in range(4000):
        for i in choose(keys, 1, stats, rng):
            counts[keys[i]] += 1
    # hot 应显著超过均匀份额(0.25),dead 靠地板仍有覆盖
    assert counts["hot"] / 4000 > 0.5
    for k in ("dead1", "dead2", "dead3"):
        assert counts[k] > 100  # 地板保证不被饿死


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} OK")
