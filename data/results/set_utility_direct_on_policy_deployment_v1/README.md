# Direct on-policy deployment truth v1

状态：`COMPLETED_DEVELOPMENT_DEPLOYMENT_TRUTH_AND_FROZEN_CANDIDATE`。

本轮在 trajectory-disjoint 的 train-heldout development denominator 上，为 256 个 states / 78 条
trajectories（其中 Long+ 为 32 条 trajectories）补齐实际部署 selector 所选 subsets 的精确 policy
restoration truth。这里的“精确”指所选 subset 直接由冻结 policy rerun 得到真实 utility，不代表枚举全部
subsets 的 exact oracle。未读取 evaluation role 数据。

## Direct 与 0.001 hybrid

主指标是 trajectory-equal B1--B4 macro normalized recovery；区间是 10,000 次 trajectory-paired
bootstrap 的 95% percentile CI。

| 方法 | Macro | Long+ | 相对 recent 的 Macro delta（95% CI） |
|---|---:|---:|---:|
| Recent | 0.37329757 | 0.38001933 | -- |
| Set Transformer direct | 0.39258351 | 0.37374657 | +0.01928594 [-0.00988560, 0.06293650] |
| Structured DeepSets direct | 0.39057983 | 0.37532659 | +0.01728226 [-0.01108342, 0.06221528] |
| Set Transformer hybrid, threshold=0.001 | 0.37329757 | 0.38001933 | 0；所有选择与 truth 均和 recent 完全相同 |
| Structured DeepSets hybrid, threshold=0.001 | 0.37335555 | 0.37842064 | +0.00005798 [-0.00542764, 0.00483995] |

两条 direct selector 的 macro 点估计都高于 recent，但 paired CI 均跨 0；固定 0.001 confidence fallback
没有产生可用增益。Set hybrid 完全退回 recent，DeepSets hybrid 与 recent 的差异近乎为 0。
DeepSets hybrid 的 Long+ delta 也为负（`-0.00159869`）。

逐预算对照如下：

| 方法 | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| Recent | 0.18320309 | 0.35564526 | 0.42122277 | 0.53311916 |
| Structured DeepSets direct | 0.18320309 | 0.31722076 | 0.49327303 | 0.56862244 |

DeepSets 的直接选择在 B2 明显弱于 recent，但在 B3/B4 更强；因此后续 development candidate 不再让一个
固定 fallback threshold 同时处理所有 budgets。

## 冻结的 development candidate

冻结 route 为：

- B1、B2 使用 recent；
- B3、B4 使用 Structured DeepSets direct conditional-marginal selector；
- 每个预算内部仍执行 at-most-`B` selection，不要求跨预算 nested；
- evaluation 后不得重新选择 budgets 或 threshold。

该 route 的 development truth 为：

| Slice | Candidate | Recent | Delta（95% CI） |
|---|---:|---:|---:|
| B1--B4 macro | 0.40018596 | 0.37329757 | +0.02688839 [0.00038743, 0.07046829] |
| Long+ macro | 0.38200244 | 0.38001933 | +0.00198311 [-0.01660415, 0.02120196] |

这是在已消费的 train-heldout development truth 上冻结 future evaluation 输入，不是 untouched evaluation
结果，也不能据此声称 closed-loop success 提升。

Structured DeepSets 在 H200 上的 warm shared-encoder selector latency 为 p50/p95=
`8.9204/9.2974 ms`；cold standalone p95=`31.3340 ms`。B1/B2 走 recent，因此部署时跳过 predictor。

## Artifacts

以下 hash 均为 artifact 内声明的 `content_sha256`：

- selector：Hyper00
  `/data02/jaxan/runs/causalcache-direct-on-policy-unified-eval-v3-bdc396a/selections.json`，
  content SHA256=`a59d33a8cce87589f5ad441dd28a499104bb053da1aba97adf8867338c20a4b7`；
- exact selected-subset truth result：Hyper00
  `/data02/jaxan/runs/causalcache-direct-on-policy-unified-eval-v3-bdc396a/post-selection-truth-result-v2.json`，
  content SHA256=`ca8497a0610dc37ff906671a92cba1f7634b24bafd04df763b178cb275121ec1`；
- supplemental truth manifest：Hyper00
  `/data02/jaxan/runs/causalcache-post-selection-truth-labels-v1-13e2967/formal-truth-manifest.json`，
  content SHA256=`4aebc006f8420d56b720faf1ee4036cfec95faacfebd7c0b72d519343fdc52b3`；
- supplemental truth receipt：Hyper00
  `/data02/jaxan/runs/causalcache-post-selection-truth-labels-v1-13e2967/formal-truth-receipt.json`，
  content SHA256=`b062ee8bb3f6c6f012580584b084be66339cf8462cfc8b45bcec9a6fc7cdbd27`。

Supplemental truth 只补原 truth roots 缺失的 11 states / 13 policy forwards；它不是把 256 states 全部重跑
一遍。selector、truth payload 与 checkpoint 当前仍在 persistent storage，状态 `PENDING_HF_UPLOAD`。

机器可读数字见 [`summary.json`](summary.json)。
