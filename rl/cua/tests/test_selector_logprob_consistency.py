# note (luojiaxuan): 采样路径与训练期重算路径的一致性回归测试。
# 判据来自外审:同一权重、零优化下,重算 logprob 与采样时记录的 logprob
# 必须严格相等,否则 PPO 比率在"策略根本没变"时就已偏离 1,裁剪掉的是
# 记账误差而非真实策略更新。
import torch

from sglang_omni_rl.selector.model import FrameSelector


def test_sample_logprob_matches_recompute():
    torch.manual_seed(0)
    sel = FrameSelector()
    for trial in range(64):
        T = int(torch.randint(3, 12, (1,)))
        feats = torch.randn(T, 256)
        pos = torch.arange(T)
        cur = T
        idx, logp, _ = sel.sample(feats, pos, cur, budget=2)
        recomputed = sel.slate_logprob(feats, pos, cur, idx)
        assert torch.allclose(recomputed, logp, atol=1e-5), (
            f"trial {trial}: sample logp {float(logp):.6f} != "
            f"recompute {float(recomputed):.6f} (chosen={idx})"
        )


def test_plackett_luce_is_order_sensitive():
    """守住修复的前提:PL 联合概率确实依赖抽取顺序,所以顺序必须被保留。"""
    torch.manual_seed(1)
    sel = FrameSelector()
    feats = torch.randn(6, 256)
    pos = torch.arange(6)
    ab = sel.slate_logprob(feats, pos, 6, [4, 1])
    ba = sel.slate_logprob(feats, pos, 6, [1, 4])
    assert not torch.allclose(ab, ba, atol=1e-3)
