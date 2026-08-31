# note (luojiaxuan): 策略版本过滤的回归测试。热换只在轮次边界发生,故一轮
# 窗口内的决策与本轮起始权重同版本;有效性单位是 episode——跨了 reload 的
# episode 即使只留新版本那几条也不干净(状态由旧版本动作诱导),整条弃用。
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _select(by_ep, adv, cur_pv):
    """复刻 train_round 的资格判定。"""
    pending, n_stale, n_mixed = [], 0, 0
    for ep, recs in by_ep.items():
        vers = {r.get("pv") for r in recs}
        if vers != {cur_pv}:
            n_stale += len(recs)
            if len(vers) > 1:
                n_mixed += 1
            continue
        pending.extend((r, adv[ep], ep) for r in recs)
    return pending, n_stale, n_mixed


def test_same_version_kept_regardless_of_wall_clock():
    # 同版本、墙钟跨度一小时:全保留(旧的按龄过滤会丢掉大半)
    eps = {"e1": [{"pv": 7, "t": 1000.0 + 600 * i} for i in range(6)]}
    pending, stale, mixed = _select(eps, {"e1": 1.0}, 7)
    assert len(pending) == 6 and stale == 0 and mixed == 0


def test_mixed_version_episode_dropped_whole():
    # 跨 reload 的 episode 整条弃用,不保留其中的新版本决策
    eps = {"e1": [{"pv": 6}, {"pv": 6}, {"pv": 7}, {"pv": 7}]}
    pending, stale, mixed = _select(eps, {"e1": 1.0}, 7)
    assert pending == [] and stale == 4 and mixed == 1


def test_wholly_old_episode_dropped_but_not_counted_mixed():
    eps = {"e1": [{"pv": 6}, {"pv": 6}]}
    pending, stale, mixed = _select(eps, {"e1": 1.0}, 7)
    assert pending == [] and stale == 2 and mixed == 0


def test_unstamped_records_dropped():
    # 改造前写下的记录无 pv,属于更早的部署
    pending, stale, mixed = _select({"e1": [{"t": 5.0}]}, {"e1": 1.0}, 0)
    assert pending == [] and stale == 1


def test_max_age_flag_removed():
    # 墙钟阈值不再是判据,残留参数会让运维以为它还在生效
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "sglang_omni_rl", "selector", "trainer.py")).read()
    assert "--max-age" not in src and "max_age" not in src


def test_version_activated_only_after_successful_reload():
    # 版本文件必须写在 urlopen 之后:失败的 reload 不得推进版本号,
    # 否则下一轮把全部决策判为不匹配而清空数据面
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "sglang_omni_rl", "selector", "trainer.py")).read()
    body = src[src.index("new_pv = cur_pv + 1"):src.index("reload_ok = True")]
    assert body.index("urlopen") < body.index('open(pv_file, "w")')
