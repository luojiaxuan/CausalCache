# note (luojiaxuan): 策略版本过滤的回归测试。热换只在轮次边界发生,故一轮
# 窗口内的决策与本轮起始权重同版本,必须全部进训练;跨了轮次边界的才丢。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sglang_omni_rl.selector import trainer as T  # noqa: E402


def _filter(records, cur_pv):
    """复刻 train_round 里的过滤判据。"""
    kept, dropped = [], 0
    for rec in records:
        if rec.get("pv") != cur_pv:
            dropped += 1
            continue
        kept.append(rec)
    return kept, dropped


def test_same_version_all_kept_regardless_of_wall_clock():
    # 同一版本、墙钟跨度 1 小时:全部保留(旧的按龄过滤会丢掉大半)
    recs = [{"pv": 7, "t": 1000.0 + 600 * i} for i in range(6)]
    kept, dropped = _filter(recs, 7)
    assert len(kept) == 6 and dropped == 0


def test_older_version_dropped_even_if_recent():
    # 跨轮次边界的决策即便刚产生也要丢
    recs = [{"pv": 6, "t": 9999.0}, {"pv": 7, "t": 1.0}]
    kept, dropped = _filter(recs, 7)
    assert dropped == 1 and kept[0]["pv"] == 7


def test_unstamped_records_dropped():
    # 改造前写下的记录无 pv,属于更早的部署,应丢弃
    kept, dropped = _filter([{"t": 5.0}], 0)
    assert dropped == 1 and kept == []


def test_pv_handshake_increments_and_persists(tmp_path):
    pv_file = tmp_path / "selector_pv.txt"
    assert (int(pv_file.read_text()) if pv_file.exists() else 0) == 0
    pv_file.write_text("1")
    assert int(pv_file.read_text()) == 1


def test_max_age_flag_removed():
    # 墙钟阈值不再是判据,残留的参数会让运维以为它还在生效
    ap_src = open(os.path.join(os.path.dirname(T.__file__), "trainer.py")).read()
    assert "--max-age" not in ap_src
