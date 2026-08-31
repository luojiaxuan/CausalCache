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
    assert stats["mobileworld@A"]["groups"] == 2
    assert stats["mobileworld@A"]["mixed"] == 1
    assert stats["mobileworld@B"]["groups"] == 1
    assert stats["mobileworld@B"]["mixed"] == 0


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


def test_p_mixed_peaks_at_half():
    """P(mixed) 应在 p=0.5 取最大,并对称地把 p→0 与 p→1 降权。"""
    from sglang_omni_rl.task_priority import p_mixed
    assert p_mixed(0.5) > p_mixed(0.25) > p_mixed(0.05) > p_mixed(0.01)
    assert abs(p_mixed(0.1) - p_mixed(0.9)) < 1e-9
    assert p_mixed(0.01) < 0.1 and p_mixed(0.5) > 0.98


def test_priority_prefers_mixed_over_saturated():
    """半数成功的任务优先分应高于恒成功与恒失败的任务。"""
    from sglang_omni_rl.task_priority import priority
    stats = {
        "half": {"groups": 5, "mixed": 4, "attempts": 40, "successes": 20},
        "always": {"groups": 5, "mixed": 0, "attempts": 40, "successes": 40},
        "never": {"groups": 5, "mixed": 0, "attempts": 40, "successes": 0},
    }
    assert priority(stats, "half") > priority(stats, "always")
    assert priority(stats, "half") > priority(stats, "never")


def test_layers_split_zero_success_by_attempts():
    from sglang_omni_rl.task_priority import _layer, AUDIT_ATTEMPTS
    assert _layer({"attempts": 8, "successes": 0}) == "frontier"
    assert _layer({"attempts": AUDIT_ATTEMPTS + 1, "successes": 0}) == "audit"
    assert _layer({"attempts": 40, "successes": 3}) == "main"


def test_update_stats_tracks_episode_level():
    from sglang_omni_rl.task_priority import update_stats
    st = update_stats({}, {"t": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]})
    assert st["t"]["groups"] == 2 and st["t"]["mixed"] == 1
    assert st["t"]["attempts"] == 8 and st["t"]["successes"] == 1
