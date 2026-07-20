# Small-history B4 oracle diagnostic v1

状态：`SUPERSEDED_BEFORE_LABEL_OUTPUT`。

本版 schedule 已生成并尝试启动，但两个 host 在停止时均为 `0 states / 0 completed batches`，没有产生可用
label。复用旧 B1/B2 truth 会让不同 reference session 的 distance 混入同一张 exact table，因此由 v2 的
self-contained `|S|≤4` truth 取代；本版仅保留为执行历史。

## 问题

Formal held-out v1 只对 B1/B2 做 exact enumeration，因此 0.517/0.685 的 oracle recovery 不能判断四个高保真
events 的真实恢复上限。本诊断在不查看 distance 的前提下，选择 exact track 中全部 `5≤n_t≤8` states，对
所有 triples/quads 生成 restoration truth，再与既有 empty/singleton/pair truth 合并，计算 at-most-B4 exact
subset oracle。

## 冻结范围

- selection rule：`exact_oracle` track 且 `5≤n_t≤8`；
- 103 states、73 trajectories；
- 新增 5,922 个 triples/quads forwards；每个 state 另含 full reference anchor；
- existing B1/B2 truth 不重算，reducer 按 state identity 合并；
- exact oracle 与 true-conditional-greedy 同时报，区分 budget ceiling 和 search gap；
- 比较 formal Set Transformer、DeepSets、recent、OCR/RGB 到 B4 exact oracle 的 gap。

配置见
[`causalcache_set_utility_b4_oracle_diagnostic_v1.json`](../code/configs/causalcache_set_utility_b4_oracle_diagnostic_v1.json)，
执行配置见
[`causalcache_set_utility_b4_oracle_execution_v1.json`](../code/configs/causalcache_set_utility_b4_oracle_execution_v1.json)。

## Claim 边界

这是 formal v1 `NO_GO` 后的小历史上限诊断，不代表全体 long-history states；不能改变 formal winner，也不能单独
授权 policy replay 或 closed-loop。其用途是回答：`B=4` 是否已经足以恢复大部分冻结策略行为，以及 student 还剩
多大的 representation/distillation gap。
