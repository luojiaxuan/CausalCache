# Direct marginal v3 Stage-A 门槛复审

状态：`ADOPTED AS ONE POST_HOC DEVELOPMENT-GATE REPAIR`。本文提供 Stage-A
`singleton Spearman > 0.5` 门槛的复审证据；它不改写 `NO_GO_DIRECT_MARGINAL_V3_STAGE_A` 的既有记录。
执行合同另立于 [`Stage-B v1`](set_utility_direct_marginal_v3_stage_b_v1.md)。由于阈值是在看到 v1 数字后
提出，它必须透明标记为 post-hoc，只用于决定是否做一次训练，不能作为 paper evidence。

## 证据一:标签是确定性的,门槛并非被噪声压低

用 immutable long-oracle payload(tag `set-utility-long-oracle-v1-179b0d8`)中每个 state 在
wave 1--4 的四次独立 `D(∅)` 重测计算测量噪声:250 states 的归一化噪声中位数为 `0.0000`
(漂移 ~1e-8),2σ 下不可分辨的 singleton 对比例为 `0.000`;若学生能复现真值,全列表 Spearman
天花板为 `1.0`。因此**不能**以"标签噪声导致门槛不可达"为修订理由;此前会话中的该猜想被本计算否定。

## 证据二:门槛测的量与部署任务脱钩,且与决策质量存在已记录的 tradeoff

- 部署合同是 at-most-`B`(`B<=4`)选择:只消费排名头部与 STOP 决策;尾部 30--40 个近零边际候选
  之间的相对顺序不进入任何部署决策。这些顺序由策略在约 3 万 token 上下文上的确定性计算产生,
  可复现但对不模拟策略的学生近似不可预测。
- 冻结记录中的直接证据(v1 → rank repair,均在任何修订提议之前产生):Spearman `0.2384 → 0.2905`
  (+0.052),同时 top-1 `0.9040 → 0.8407`、B1 recovery `0.4732 → 0.4375`、top-4 recall
  `0.9873 → 0.9515`——优化全列表排序**因果性地损害**决策头部质量。
- 本项目历史上已两次确立"proxy 指标改善不代表 selection 改善"(contextual v3 的 tune loss、
  L64/S4 的不可比 loss);全列表 Spearman 属于同类 proxy。

## 建议的 Stage-A' 门槛(决策对齐、合取)

| 条件 | 阈值 | v1 冻结记录 |
|---|---:|---:|
| top-1 exact-best rate | `>= 0.80` | 0.9040 |
| true-best top-4 recall | `>= 0.95` | 0.9873 |
| learned B1 recovery / oracle B1 | `>= 0.90` | 0.944 |
| STOP accuracy | `= 1.0` | 1.0000 |

v1 数字先于任何修订提议被冻结,不存在为门槛调参;rank-repair 方向(加大全列表 rank loss)已被
证据否定,不再采用。

## 不可再退让的边界

1. v1/v2 探针的 `NO_GO` 记录与本复审并存,均不改写;
2. **本次是唯一一次 Stage-A 指标修订**;Stage-B 完整 conditional-marginal 训练后,仍以原封不动的
   decision-v2 fixed-tune gate(B1--B4 逐点 > recent、macro paired bootstrap 下界 > 0、Long+ 点估计
   > recent)判定;该 gate 任何条款不因本复审改变;
3. Stage-B fixed-tune 若 `NO_GO`,learned general-`B` 路线终止,不进行第四次学生迭代;
4. tune long-oracle labels(`labels_reusable_for_training=false`)与 untouched evaluation 的防火墙
   不受本复审影响。
