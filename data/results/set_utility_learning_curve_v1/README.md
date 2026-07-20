# Set Utility learning curve v1

状态：`COMPLETED_TRAIN_TUNE_ONLY`。

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
text sequences。nested input 只允许复用其 manifest 直接 parent 所绑定的 cache；其他 cache/split 组合 fail
closed。

每个 fraction 只训练 25% tuning 冻结的两套配置，各跑 seed `20260720`：

- DeepSets `d512/l16/r2/lr3e-4`；
- Set Transformer `d256/l8/r1/s2/lr1e-4`。

本 learning curve 仍只使用 train/tune，用于判断数据拐点；不构成 held-out selector 或 deployment 结论。
执行映射见 [`causalcache_set_utility_learning_curve_execution_v1.json`](../../../code/configs/causalcache_set_utility_learning_curve_execution_v1.json)。

## 结果

| Train fraction | DeepSets tune total | Set Transformer tune total | DeepSets termination | Set Transformer termination |
|---:|---:|---:|---|---|
| 10% | 0.32270 | 0.32057 | early stop | early stop |
| 25% | 0.31731 | 0.32363 | early stop | early stop |
| 50% | 0.32065 | 0.32039 | non-finite after best | early stop |
| 100% | **0.31171** | **0.31577** | non-finite after best | early stop |

从 10% 到 100%，DeepSets 与 Set Transformer 的 tune total 分别改善 0.01098 与 0.00480；但中间点不单调，
不能把单 seed 曲线解释成精确 power law。100% 对两个 family 都优于 50%，因此目前没有清晰饱和证据；同时
DeepSets 的高 LR 在 50/100% 重现 non-finite，不能据此直接选为部署模型。

8 个训练容器均为 exit 0、无 OOM；镜像 digest、软件版本、GPU/driver、完整 argv 与起止时间见
[`execution_summary.json`](execution_summary.json)。

下一步不立即盲目扩充 labels。先冻结 100% checkpoints，在 trajectory-disjoint evaluation 上比较真实
at-most-`B` selector utility、OCR/RGB/recent 与 latency；若 held-out gain 成立且数据曲线仍改善，再增加
trajectories 或 interaction-dense labels。机器可读结果见 [`summary.json`](summary.json)。

Artifacts：

- dataset/splits/cache/compact label archives：`gavinlaw/causalcache-set-utility-variable-history-mobile@02a05ab11fd3a5036b62244bf04f37aa5e41db59`，414 files / 77.4GB；
- 8 checkpoints：`gavinlaw/causalcache-set-utility-predictors-mobile@5409e846cc45a26b2ae617e39b3ebf7462d180e6`，17 files；
- path：`artifacts/set-utility-learning-curve-v1-9863f43`。
