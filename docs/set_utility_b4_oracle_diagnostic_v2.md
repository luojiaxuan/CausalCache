# Small-history B4 oracle diagnostic v2

状态：`LABELS_RUNNING`。

## 问题

Formal held-out v1 的 B1/B2 exact truth 与新 B3/B4 truth 若来自不同 reference session，可能受 reference
distribution 或 logits 数值漂移影响，不能严谨地组成一张 exact table。v2 在同一次 reference session 中重算
全部 `|S|≤4` coalitions，直接测量四个高保真 events 的真实恢复上限。

## 冻结范围

- selection rule：`exact_oracle` track 且 `5≤n_t≤8`；
- 103 states、73 trajectories；
- 每个 state 完整计算 empty、singleton、pair、triple、quad，并写 full reference anchor；
- 共 8,525 个 policy forwards、8,628 个 schedule coalitions；
- exact oracle 与 true-conditional-greedy 同时报，区分 budget ceiling 和 search gap；
- 比较 formal Set Transformer、DeepSets、recent、OCR/RGB 到 exact oracle 的 gap。

配置见
[`causalcache_set_utility_b4_oracle_diagnostic_v2.json`](../code/configs/causalcache_set_utility_b4_oracle_diagnostic_v2.json)，
执行配置见
[`causalcache_set_utility_b4_oracle_execution_v2.json`](../code/configs/causalcache_set_utility_b4_oracle_execution_v2.json)。

## Claim 边界

这是 formal v1 `NO_GO` 后的小历史上限诊断，不代表全体 long-history states；不能改变 formal winner，也不能单独
授权 policy replay 或 closed-loop。其用途是区分 B4 budget ceiling、search gap 与 representation/distillation gap。

## 执行记录

- frozen source revision：`59e6a7341523b67c9d7dff8e0bce2677b0bbe5bf`；
- schedule：103 states、73 trajectories、8,525 forwards、8,628 total coalitions；summary file
  SHA256=`247cc6a0ac753209c2c64f08b3ff19dc922182cbdf3c36bfa9e595b3f4f1c807`；
- Hyper00 使用 H200 `0--5`、Hyper01 使用 H200 `2--7`，每卡两个 state-lane worker；
- 两端 output root：`/data02/jaxan/runs/causalcache-set-utility-b4-oracle-labels-v2-59e6a73`；
- output 以 state 与 microbatch 为原子断点，已完成 state 会在重启时跳过；
- reducer 独立报告 exact B1--B4、true-conditional-greedy、formal learned/heuristic selector、search gap、
  distillation gap 与 B2-to-B4 ceiling gain。

标签和最终 result 在完成前为 `PENDING_HF_UPLOAD`；Git 只保存轻量 summary，完整 truth table 上传私有 HF dataset。
