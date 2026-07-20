# Set utility train-side on-policy enrichment v1

## 目的

Full-data Set Transformer 在固定 tune truth 上达到 `0.44985`，与 recent=`0.45192` 基本持平；B3/B4 已胜
recent，但 B1/B2 与 long-history 仍弱。下一步不增加同分布 trajectory，也不访问 evaluation，而是补当前
predictor conditional-greedy search 实际访问的 train coalitions。

## Firewall

- selector 输入、targeted state selection、restoration labels 与重训全部只使用 `role=train`；
- 当前 1,063-state tune truth 只用于重训后的固定判定，不进入 optimizer；
- evaluation、policy replay 与 closed-loop 保持锁定，直到新 checkpoint 在固定 tune truth 上胜过 recent。

## Target states

从 10,658 train states 中选择 20%，目标 2,132 states。历史 bin 配额为 short/medium/long/very-long=
`10%/25%/60%/5%`；very-long 数量或 trajectory diversity 不足时把余量按确定性规则转给其他 bin。每条
trajectory 严格最多 3 states，不为填满某个 bin 放宽。

bin 内优先级依次为：

1. DeepSets 与 Set Transformer conditional-greedy path 分歧；
2. Set Transformer 与 recent path 分歧；
3. conditional top-1/top-2 predicted utility margin 小；
4. 选择集合覆盖更老事件；
5. state-id hash 稳定 tie-break。

## Coalition coverage

Train selector 在每个 greedy step 保存 top-3 candidate subsets。每个 targeted state 的 restoration schedule
包含：

- empty/full anchors；
- DeepSets 与 Set Transformer B1--B4 prefixes；
- 两模型每一步的 top-3 conditional candidates；
- recent B1--B4。

重复 coalition 去重。这样监督同时包含部署 path、近邻错误选择、模型分歧与 conditional marginal，而不是只给
最终 subset。

## 实现与验收

- config：[`causalcache_set_utility_train_on_policy_enrichment_v1.json`](../code/configs/causalcache_set_utility_train_on_policy_enrichment_v1.json)；
- targeting：[`set_utility_train_on_policy.py`](../code/causalcache/set_utility_train_on_policy.py)；
- schedule CLI：[`materialize_set_utility_train_on_policy_schedules.py`](../code/scripts/materialize_set_utility_train_on_policy_schedules.py)；
- 5 个相关 tests 通过。

重训后只接受以下判定：Set Transformer 在相同固定 tune truth 上 primary macro 高于 recent，且 paired
trajectory bootstrap 不显示稳定退化；否则继续 `NO_GO`，不进入 policy experiments。
