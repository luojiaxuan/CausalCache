# Fixed-tune Long+ conditional-greedy oracle v1

状态：`COMPLETED / HEADROOM_CONFIRMED_AUTHORIZE_DIRECT_MARGINAL_V3`。正式结果见
[`data/results/archive/set_utility_tune_long_oracle_v1/`](../../data/results/archive/set_utility_tune_long_oracle_v1/README.md)。

## 目的

当前 scalar `U(S)` student 在 fixed-tune Long+ 上仅比 recent 高约 `+0.024`，但 train Long+
oracle headroom 为 `+0.467`。两者 denominator 不同，不能据此判定蒸馏失败还是 tune 人群本身没有
headroom。本诊断只回答：在已经消费的 **同一个 fixed-tune Long+ denominator** 上，true
conditional-greedy restoration oracle 是否仍显著超过 recent。

## 冻结 denominator 与执行

- 上游 result content SHA256：`043d3e12...d4c86`；
- 249 states / 35 trajectories：long 241、very-long 8；
- state manifest：[`data/manifests/set_utility_tune_long_oracle_v1_states.json`](../../data/manifests/set_utility_tune_long_oracle_v1_states.json)，SHA256 `084d3c8e...24523`；
- 每个 state 保留完整 variable-size history candidate set，不做 recent truncation；
- wave 1 标注 empty/full、全部 singleton、recent/random anchors；wave 2--4 标注当前 true-greedy prefix
  的全部 one-event expansions；
- `oracle_greedy@B` 是 conditional-greedy 路径前 `B` 个 prefix 中真实 recovery 的最大值，因此遵循
  at-most-`B`，加入有害事件时允许停在此前 prefix；
- trajectory-clustered paired bootstrap，10,000 resamples；比较同 state 上
  `oracle_greedy - recent` 的 B1--B4 macro。

配置：[`causalcache_set_utility_tune_long_oracle_v1.json`](../../code/configs/causalcache_set_utility_tune_long_oracle_v1.json)。
四个 wave 均使用可断点续跑的 state/microbatch terminal；每个 wave 完成后才物化下一 wave。

## 预注册决策

- 若 macro point estimate `> 0.10` 且 95% CI lower `> 0.03`：
  `HEADROOM_CONFIRMED_AUTHORIZE_DIRECT_MARGINAL_V3`；只授权一次 direct conditional-marginal + STOP v3；
- 若 95% CI upper `< 0.03`：
  `HEADROOM_INSUFFICIENT_RECONSIDER_DATA_AND_CLAIM`；不再把失败主要归因于 student；
- 其余：`INCONCLUSIVE_TUNE_LONG_ORACLE`，不自动授权 v3。

## Firewall

这是对已消费 tune 的 failure decomposition，不是新的 GO evaluation。tune truth 不得进入任何训练、
checkpoint 选择或超参数搜索；untouched evaluation、policy replay、closed-loop 继续锁定。若 v3 被授权，
250-state train in-sample probe 只作为优化门，最终仍须按新的冻结合同评估。

## 结果

249/249 states、四 wave 0 skip；oracle/recent B1--B4 macro=`0.6949/0.3199`，paired delta=
`+0.3750 [0.2627,0.5656]`。point 与 CI lower 均以大幅裕量通过预注册阈值，因此只授权一次 direct
conditional-marginal + explicit STOP v3。现有 scalar `U(S)` 参数化不再继续调参。
