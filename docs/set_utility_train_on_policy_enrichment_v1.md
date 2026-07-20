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
trajectory 严格最多 5 states，不为填满某个 bin 放宽。cap=3 的 dry run 只能保留 910 个 long states；cap=5
可保留 1,307 个 long、76 个 very-long、534 个 medium 与 215 个 short，同时覆盖 680 条 trajectories。

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
- label merge：[`set_utility_contextual_enrichment.py`](../code/causalcache/set_utility_contextual_enrichment.py)；
- enriched snapshot CLI：[`materialize_set_utility_contextual_enriched_inputs.py`](../code/scripts/materialize_set_utility_contextual_enriched_inputs.py)；
- 相关 tests 通过。

Enriched snapshot 只向 selected train states 增加新 coalition rows；原 broad label 与新 terminal 重叠时保留原值，
并要求绝对差不超过 `1e-6`。manifest 绑定 schedule、label terminals、source revision 与 parent input SHA；
contextual hidden cache 通过 parent binding 复用，不重复运行 GUI-Owl encoder。Tune states 逐字节保持不变。

重训目标在原 raw/normalized subset utility 与 within-state ranking 之外，显式加入所有已标注
`S -> S union {j}` one-event expansion 的 normalized conditional-marginal regression，权重为 `1.0`。这使
targeted labels 直接约束部署时 conditional greedy 的后续步骤。固定 tune reducer 允许候选 checkpoint 来自
不同训练 config，但仍要求相同 contextual input、hidden cache 和 state inventory，并逐模型记录 config SHA。

重训后只接受以下判定：Set Transformer 在相同固定 tune truth 上 primary macro 高于 recent，且 paired
trajectory bootstrap 不显示稳定退化；否则继续 `NO_GO`，不进入 policy experiments。
