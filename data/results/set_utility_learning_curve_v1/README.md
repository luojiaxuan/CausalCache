# Set Utility learning curve v1

状态：`INPUTS_FROZEN_READY_TO_RUN`。

Formal variable-history labels 已完成：11,746/11,746 terminal states，其中 11,721 completed、25 skipped。
只从 completed train/tune states 构建 predictor input，不加载 evaluation。

| Train fraction | Train trajectories | Train states | Tune trajectories | Tune states | Content SHA256 |
|---:|---:|---:|---:|---:|---|
| 10% | 100 | 1,037 | 100 | 1,063 | `10680375...6d3982` |
| 25% | 250 | 2,640 | 100 | 1,063 | `7e4663f7...8d282c` |
| 50% | 500 | 5,403 | 100 | 1,063 | `d1e3cb8b...554533` |
| 100% | 1,000 | 10,658 | 100 | 1,063 | `8a530cc6...dac6fa` |

四个 train trajectory sets 使用同一 SHA256 ranking，已验证 10% ⊂ 25% ⊂ 50% ⊂ 100%；tune 完全相同。
两台 Hyper 的 full-token cache 独立 finalization 后均为 `ef82b077...23385`，覆盖 16,146 visual 与 17,152
text sequences。

每个 fraction 只训练 25% tuning 冻结的两套配置，各跑 seed `20260720`：

- DeepSets `d512/l16/r2/lr3e-4`；
- Set Transformer `d256/l8/r1/s2/lr1e-4`。

本 learning curve 仍只使用 train/tune，用于判断数据拐点；不构成 held-out selector 或 deployment 结论。
执行映射见 [`causalcache_set_utility_learning_curve_execution_v1.json`](../../../code/configs/causalcache_set_utility_learning_curve_execution_v1.json)。
