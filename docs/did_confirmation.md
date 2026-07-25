# RA-aware 差中差 pilot:confirmation split 一次性判定

日期:2026-07-26。**confirmation split(89 episodes / 254 groups)**,该 split 在此之前
**从未被读取**,由 `--episode-filter` 在物理上排除于所有 dev 评测之外。这是它唯一一次使用。

## 结果:全部 gate 通过

| 量 | 值 | CI95 | gate |
|---|---|---|---|
| `did_select` = A_c − A_r | **+0.00638** | [+0.00513, +0.00764] | > 0 @ci_low ✅ |
| `adapter_on_sparse` A_c | +0.00834 | [+0.00741, +0.00933] | > 0 ✅ |
| `adapter_on_recent` A_r | +0.00196 | [+0.00108, +0.00280] | — |
| `adapter_on_recent_abs` | +0.00579 | [+0.00528, +0.00633] | < 0.02 ✅ |
| `step_shuffled_drift_abs` | +0.00789 | [+0.00688, +0.00894] | < 0.02 ✅ |
| `irrelevant_drift_abs` | +0.00668 | [+0.00583, +0.00756] | < 0.02 ✅ |
| `duplicate_drift_abs` | +0.00707 | [+0.00617, +0.00803] | < 0.02 ✅ |
| `SA_minus_RA` | +0.03984 | [+0.03036, +0.04980] | ≥ S0−R0 ✅ |

`all_must_pass: true`,`selected_checkpoint: s50`。与 dev 上的
`did_select = +0.00548 [+0.00425, +0.00672]` 高度一致 —— 独立 split 上复现。

## 这证明了什么 —— 以及**没有**证明什么

**证明了**:在**单轮 sparse 格式下**,RA-aware 差中差目标能训出真正的选择性增益。
adapter 对稀疏选点的增量(+0.0083)显著大于对最近窗口的增量(+0.0020),
而且这不是靠"整体放大"刷出来的(三个 drift 都在 0.007 量级、远低于 0.02 上限)。

对比 v5 的旧目标:`SA − R0` 从 +0.034 涨到 +0.118 而真实的 `SA − RA` 归零。
**换了正确的目标之后,同一个架构、同一份语料、同样的 identity 初始化,结论反转。**
所以 v5 的失败是**目标写错了**,不是接口不行 —— 这一点现在有 confirmation 级别的证据。

**没有证明**:这个结果适用于最终部署格式。

它是在**单轮 renderer** 下测的,而 Gate 1(2026-07-26)随后测出该 renderer 相对官方
多轮格式有 −0.0208 的成本,并且**冻结模型的稀疏选点优势 `S0 − R0 = +0.0335` 正是这个
renderer 的产物** —— 换成 official-style sparse multiturn 后它归零(+0.0027,CI 跨零,
K=1/2/3/4 无一显著)。

因此:

1. 换 renderer 则 `S0/R0/SA/RA` 四臂全变,**本 checkpoint 需要重训**;
2. 更根本的是,换格式后 adapter 要学的东西从"放大一个已存在的 +0.0335 优势"
   变成"在 ≈0 的基线上凭空创造优势",**难度不同,不能假定结论会平移**;
3. 所以本节的正确表述是 **"在单轮格式下 DiD 目标有效"**,而不是
   "sparse-history adapter 有效"。

## 状态

`NOT_FINAL_PENDING_RENDERER_FREEZE`。是否重训、以及在哪种格式下重训,取决于
Gate 3 的 oracle headroom 结果。
