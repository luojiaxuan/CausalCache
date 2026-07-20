# Held-out selector scaling diagnostic v1

状态：`COMPLETED_SET_UTILITY_HELDOUT_SCALING_DIAGNOSTIC_WITH_SKIPS`。

## 问题

Train/tune learning curve 的 loss 不能直接回答更多同分布数据是否改善 selector。本诊断把已经冻结的
10%/25%/50%/100% DeepSets 与 Set Transformer checkpoints 全部运行在同一 805-state label-blind feature
snapshot 上，并只使用 exact track 已存在的 `|S|<=2` truth 比较 B1/B2：

- held-out trajectory-equal normalized recovery；
- exact-oracle regret；
- 相对 recent 与 OCR/RGB 的 paired trajectory bootstrap；
- 10% 到 100% 的 family 内 scaling trend；
- tune objective 排名是否与 held-out selector 排名一致。

## 冻结输入

- 8 个 checkpoints 均来自 train/tune-only immutable HF revision `5409e846...180e6`；
- evaluation inventory、full-history candidate universe、feature/token cache、conditional-greedy at-most-`B`
  search 与 formal held-out v1 相同；
- exact track 为 320 states；formal v1 已有一个 exact-track strict-parser skip，其余 319 states 的全部
  empty/singleton/pair distances 已在 checkpoint 选择前冻结；
- 配置见
  [`causalcache_set_utility_heldout_scaling_diagnostic_v1.json`](../code/configs/causalcache_set_utility_heldout_scaling_diagnostic_v1.json)。

## Claim 边界

这是 formal v1 `NO_GO` 后的 post-hoc failure decomposition，不是新模型选择合同：

- 不生成新 restoration labels；
- 不使用 B3/B4，因为新 checkpoints 的 B3/B4 selected subsets 未在原 schedule 中全部覆盖；
- 不追认或改变 formal v1 winner；
- 无论结果如何，都不能直接授权 policy replay 或 closed-loop；
- 结果只决定下一步优先扩同分布 trajectories，还是优先改 representation/on-policy supervision。

## 执行

每个 checkpoint 在 Hyper00 与 Hyper01 的 disjoint feature/cache partition 上各运行一次，再按 805-state identity
合并。8 个 merged candidate outputs 与 canonical held-out truth 一次性进入 reducer；不得按中间 evaluation 结果
追加、删除或重训 checkpoint。

## 结果与决策

完整结果见
[`data/results/set_utility_heldout_scaling_diagnostic_v1/`](../data/results/set_utility_heldout_scaling_diagnostic_v1/README.md)。
Set Transformer 的 B1/B2 macro recovery 为 0.2392/0.2460/0.2456/0.2629，10%→100% paired delta 为
+0.0237，95% CI `[-0.0050, 0.0529]`；DeepSets 为 0.2490/0.2436/0.2579/0.2405，endpoint delta
-0.0085。两条曲线均不单调。

结论是同分布扩数对 Set Transformer 有方向性收益，但证据不足以把问题归结为纯数据量；后续优先做 B4 oracle
上限和 contextualized multi-latent representation，不继续盲目堆同分布 trajectories。
