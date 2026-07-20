# Small-history B4 oracle diagnostic v2

状态：`COMPLETED_SET_UTILITY_B4_ORACLE_EVALUATION`。

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

## 结果

| 方法 | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| Exact subset oracle | 0.5304 | 0.6999 | 0.7823 | 0.8248 |
| True conditional greedy | 0.5304 | 0.6686 | 0.7329 | 0.7677 |
| Set Transformer v1 | 0.1997 | 0.4386 | 0.5947 | 0.7128 |
| DeepSets v1 | 0.1427 | 0.4485 | 0.5854 | 0.7064 |
| OCR/RGB | 0.1499 | 0.4340 | 0.5635 | 0.6942 |
| Recent | 0.1486 | 0.4237 | 0.5823 | 0.6718 |

B2→B4 exact gain 为 `+0.1249`；exact-minus-greedy B4 gap 为 `0.0571`；exact-minus-Set-Transformer B4
gap 为 `0.1121`。因此 B4 的 budget ceiling 已足够高，greedy 不是主要误差源，当前优先问题是
representation/distillation。Set Transformer 在 B4 的点估计最好，但本诊断不能改变 formal v1 `NO_GO` 或授权
closed-loop。

完整 aggregate 位于 Hyper00
`/data02/jaxan/runs/causalcache-set-utility-b4-oracle-labels-v2-aggregate-59e6a73`；result content
SHA256=`195bcbb57d915ca81a40f1c7bf66acab6a28fd2998d52452ea23a3891ba0473b`，result file
SHA256=`b14896f0113f7aa839b4c864287003cc1d6769a3ae06887b95e3b684785ce1b6`。完整 truth table、schedule、
worker receipts 与 configs 已发布到 private HF dataset revision
[`9b53ec82c12fefaba571233e5e0d78d0f6c599c0`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/9b53ec82c12fefaba571233e5e0d78d0f6c599c0/artifacts/set-utility-b4-oracle-v2-195bcbb)，
tag=`set-utility-b4-oracle-v2-195bcbb`；force-download 的 `result.json` SHA 与本地逐字节一致。
