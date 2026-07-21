# Direct conditional-marginal v3 Stage-B v1

状态：`TRAINING_COMPLETED / FIXED_TUNE_PENDING`。

Hyper00 8×H200 完成 12 epochs，用时 785.7s。checkpoint 只按 100 条 train trajectories 的 holdout
decision regret 选择，tune/evaluation labels 未进入训练或模型选择。

| metric | train | train holdout |
|---|---:|---:|
| action top-1 | 0.6129 | 0.2982 |
| decision regret | 0.1186 | 0.2581 |
| conditional listwise | 1.1211 | 3.9697 |
| STOP accuracy | 0.0000 | 0.0000 |
| teacher STOP rate | 0.0227 | 0.0229 |

最佳 epoch=12，checkpoint SHA256=
`d8abbe8cf7bd04324220030b040f09e0d9a09402c87c0e17ca78c4047631dd23`。optimization 正常，但
train→holdout gap 和 STOP 未学会是明确风险；这些指标不替代真实 selection truth，也不触发额外调参。
下一步只运行一字不改的 fixed-tune B1--B4/macro-CI/Long+ gate。

## Source of Truth

- [`summary.json`](summary.json)：Git 轻量摘要，SHA256=
  `d6aad40288504715f7cf330356c6beae78a2aec6d7ad140e71ee45b63f080735`；
- source=`main@8ccdb5c`，config SHA256=
  `41bdefc9be8171123caa23ddf99e24592ba1b95608e546ba88f7e69a3a8be2c3`；
- Hyper00 full run：
  `/data02/jaxan/runs/causalcache-direct-marginal-v3-stage-b-v1-8ccdb5c`；
- full summary content/file SHA256=`f03bba42...db173` / `d24a53e2...57892`；
- checkpoint 状态=`PENDING_HF_UPLOAD`，intended model repo=
  `gavinlaw/causalcache-set-utility-predictors-mobile`。

合同：[`docs/set_utility_direct_marginal_v3_stage_b_v1.md`](../../../docs/set_utility_direct_marginal_v3_stage_b_v1.md)。
