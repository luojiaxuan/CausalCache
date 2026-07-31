# At-most-B 与 closed-loop memory learning 方法决策 v1

日期：2026-07-31

状态：`WORKING / METHOD_DECISION`

观测分支：`paper-draft`

观测 revision：`163fbf1f84c0226200d4135b16d16c7477699a9b`

本文记录两个相互关联、但必须分开判断的问题：

1. 旧 set-utility 路线为什么没有学会 at-most-`B` 的 STOP；
2. Desktop offline DiD 是否足以支持 closed-loop，以及下一版是否应直接优化
   closed-loop memory policy。

本文是方法决策与后续实验设计，不改变当前 AAAI working paper 的 exact-`B`
matched-budget 主结果，也不是新实验结果。

## 0. 结论

1. **At-most-`B` 的 oracle 定义没有问题。** 对同一个真实效用函数 `Q`，
   `max_{|S|<=B} Q(S)` 一定不低于 `max_{|S|=B} Q(S)`。旧路线失败的是学习目标和
   搜索实现，不是这个集合包含关系。
2. **不应复活旧的 greedy marginal + learned/fixed STOP。** 它把全局、带剩余预算的
   set optimization 错写成局部的 `max_j Delta(j|S) <= 0`，在存在 interaction 时不能
   表达正确决策；旧模型又没有 remaining-budget input，因此相同 `S` 在不同剩余预算下
   只能给出同一个 STOP 判断。
3. **Offline DiD 是 policy-interface 的局部识别/正则目标，不是 closed-loop
   objective。** 它能回答“在同一个 teacher-forced 决策点，打开 adapter 后，相关图是否
   比 recent/wrong 图更能提高 gold action 的 likelihood”，但不能回答换图导致动作变化后，
   agent 会进入什么新状态、最终是否成功。
4. **下一版优先做 frozen-action-policy 下的 closed-loop selector learning。** 先固定
   action policy，只让 memory policy 在 `0..B` 张图之间选择，并直接用 rollout return
   学习；验证 selector 本身成立以后，再考虑更新 SFT/RL policy。不要第一步就让 policy
   与 selector 共同变化。
5. **STOP 不再作为一个自由 token 单独学习。** 对所有已搜索到的 `|S|=0..B` 完整集合
   比较预测 closed-loop value，选择最大者；若目标还包含自适应 compute/memory，应明确
   优化 `return - lambda * |S|` 或 average-budget constraint，而不是期待无成本的
   at-most-`B` 自发大量少用图。

## 1. 旧 at-most-B 为什么没有训好

### 1.1 Oracle 更强，不等于 learned optimizer 更容易

令 `Q(S)` 为完整集合 `S` 在冻结 policy 下的真实效用，则

```text
Q*_{<=B} = max_{|S|<=B} Q(S) >= max_{|S|=B} Q(S) = Q*_{=B}.
```

这个结论只约束 oracle。learned at-most-`B` 同时要学习：

- 候选图的相对排序；
- 不同 cardinality 之间可比较的绝对 value；
- interaction；
- 何时少选一张；
- 一个能找到非局部组合的搜索过程。

Exact-`B` 只要求在固定 cardinality 下排序或替换，统计与搜索难度都更低。因此更大的
feasible set 降低了 oracle bias，却可能增加 estimation error、calibration error 和
search error。

### 1.2 STOP 标签稀少，而且由最大假正例决定

旧 direct on-policy optimizer inventory 有 `84,441` 个 candidate-complete groups，
其中 `stop_all_negative=true` 只有 `2,124` 个，即 **2.52%**。完整输入的候选数中位数
约为 `11`，P90 约为 `21`。这里“某些 addition 有害”并不等于应该 STOP；STOP 要求
**所有**剩余候选都无正收益。

旧推理规则为：

```text
STOP iff max_j predicted_marginal(j | S) <= 0.
```

当一个 state 有十几到几十个候选时，只要留下一个小的 false-positive marginal，最大值
就会超过零。旧 loss 对 non-STOP candidates 做平均 marginal regression，并以 listwise/
regret 辅助；它没有直接控制 all-negative state 上最危险的最大假正例。对应实现见
[`train_set_utility_direct_marginal_v3.py`](../code/scripts/train_set_utility_direct_marginal_v3.py)。

最终行为与这一故障完全一致：fixed-tune 的 `1,063/1,063` states 在 B1--B4 全部选满，
显式 STOP 一次未触发；direct marginal v3 相对 Recent 的 macro delta 为 `-0.00609`，
95% CI `[-0.03012,+0.01642]`。证据见
[`set_utility_direct_marginal_v3_fixed_tune_v1`](../data/results/archive/set_utility_direct_marginal_v3_fixed_tune_v1/README.md)。

### 1.3 旧 STOP 是 myopic rule，不能处理 complementarity

存在 interaction 时，可能出现：

```text
Delta(a | S) < 0
Delta(b | S) < 0
Q(S union {a,b}) > Q(S).
```

此时 remaining budget 为 1 应该停止，但 remaining budget 为 2 可能应该接受第一步的
局部负收益以获得 pair synergy。旧 Stage-B 配置明确设置
`budget_input_to_model=false`，而 selector 一旦在较小预算停止，就把同一个停止结果复用到
所有更大预算。配置与实现分别见
[`causalcache_set_utility_direct_marginal_v3_stage_b_v1.json`](../code/configs/causalcache_set_utility_direct_marginal_v3_stage_b_v1.json)
和
[`run_set_utility_direct_marginal_tune_selectors.py`](../code/scripts/run_set_utility_direct_marginal_tune_selectors.py)。

因此，在允许 coalition interaction 的问题定义下，旧模型缺少作出正确 STOP 所必需的
remaining-budget 信息，旧 greedy search 也无法跨过负 marginal prefix。这不是简单增加
训练 epoch 或 backbone 就能修复的错误。

### 1.4 后续 learned STOP 仍表现为零点漂移

On-policy repair 扩大了 candidate-complete coverage，并加入 STOP 分层采样与 learned
STOP head，但 heldout rollout 在相邻 epoch 间出现：几乎全部填满、几乎全部在空集停止、
再次全部填满、再变成大多数只选一张。这说明 decision boundary 主要受整体 score offset
漂移控制，而不是稳定学会了 cardinality choice。旧记录见
[`set_utility_direct_on_policy_v1.md`](archive/set_utility_direct_on_policy_v1.md)。

结论：**旧路线实际学的是不稳定的绝对零点，而不是全局 at-most-`B` value。**

## 2. Offline DiD 能证明什么，不能证明什么

Desktop DiD 对同一个决策点、同一个 target action 和 matched image budget 比较
relevant/recent/wrong arms。它的优点是标签密、方差低，并能通过 per-arm frozen anchor
排除“adapter 对所有历史图统一放大”的假改进。现有结果支持 HGKV 在该受控条件下形成
selectivity，并保持 recent/wrong drift cap；证据见
[`desktop_did_policy_v1`](../data/results/desktop_did_policy_v1/README.md)。

但它固定了数据中的 history `h_t` 和 gold action `a*_t`。它没有识别：

```text
memory allocation
  -> generated action
  -> environment transition
  -> future state distribution
  -> future memory candidates
  -> eventual task return.
```

换句话说，offline DiD 估计的是 logged-state 上的局部、policy-relative surrogate；
closed-loop 关心的是该 memory policy 诱导的新 occupancy measure 下的累计回报。动作只要
改变一次，后续 state、候选图、错误恢复机会和 episode 长度都会变化，teacher-forced
margin 不能自动外推过去。

当前项目证据也要求维持这个边界：Desktop offline selectivity 已成立，而 MobileWorld
closed-loop 的 full-roster 总体对照仍需要按配对区间谨慎表述，收益主要集中在
construction-defined memory-critical split。二者不矛盾；它们测的是不同层级。

因此正确裁决不是“DiD 方法错误”，而是：

- DiD 适合作为 HGKV/interface 的 pretraining 与 invariance regularizer；
- selector 的 headline objective 应升级为 closed-loop return；
- offline margin 继续作为机制诊断和低成本辅助指标，不能作为 success 的替代物。

## 3. Template 探索是正确方向，但不能只收成功轨迹做 SFT

用参数化 task templates 生成实例，让 agent 探索并发现成功 trajectory，方向上比纯
offline DiD 更接近目标。问题不在于 trajectory 受 SFT policy 影响——任何 on-policy
数据都会依赖当前 policy——而在于是否把这种依赖显式纳入算法和 claim。

建议把对象写成联合系统：

```text
J(pi, mu) = E[episode return | action policy pi, memory policy mu].
```

“哪张图有用”本来就是相对 `pi` 的；不存在脱离 action policy 的 universal image
utility。若 `pi` 更新，旧 selector labels 可能失效，需要重新 rollout/relabel。这是 policy
iteration，不是无法消除的污染。

但仅把成功 trajectory 拿来做 SFT 有三个不足：

1. 成功轨迹只展示被选中的 memory set，没有说明未选集合的 counterfactual return；
2. SFT 容易学习动作序列或 template shortcut，不能单独归因到 high-fidelity images；
3. 失败通常来自较早的 memory/action 决策，episode success 对具体图片的 credit 很稀疏。

所以成功轨迹可以用于 warm start action policy 或产生 candidate witness，但不能替代
memory allocation 的对照探索。

## 4. 推荐的 closed-loop 主方法

### 4.1 第一阶段固定 action policy，只学习 memory policy

先冻结当前可执行的 action policy `pi_0`，让 memory selector `mu` 成为唯一学习变量。
在每个决策点，`mu` 从完整 summarized history 选择一个集合
`S_t, |S_t|<=B`，再由 `pi_0` 产生动作。优化目标为：

```text
maximize E[R_task - lambda * sum_t |S_t| - eta * latency_t].
```

若论文只关心 hard capacity，可令 `lambda=eta=0`；但这时最优解经常会用满预算，不能把
“实际少用图”当作必然结果。若目标包含 adaptive compute/memory，必须显式给每张图或
每次 second pass 定价，或约束 episode-level average image budget。

固定 `pi_0` 的好处是任何 return 变化都能归因于 memory policy，而不是 action policy
同时变强/变弱。这个阶段最适合检验 CausalCache 的独立方法价值。

### 4.2 不训练自由 STOP；联合比较 0..B 的完整集合

候选 selector 应预测完整集合的 continuation value，例如：

```text
V(h_t, S, r) = expected closed-loop return with selected set S
               and r remaining memory slots.
```

对 B<=4，可用 beam/branch-and-bound 搜索各 cardinality 的候选集合，并在所有访问过的
`|S|=0..B` 中选择：

```text
argmax_S [V_hat(h_t, S, B-|S|) - lambda * |S|].
```

最佳集合的 cardinality 就是停止位置，不再需要一个独立、易漂移的 STOP score。若仍采用
sequential policy，输入必须包含 remaining budget，训练 target 必须是 lookahead
continuation advantage，而不是 immediate marginal。

### 4.3 Template-level closed-loop data collection

建议的最小闭环数据循环：

1. 按 **template** 划分 train/dev/test；同 template 的不同参数实例不得跨 split，避免
   selector 记住操作脚本或 UI 字段。
2. 对训练 templates 生成多个初始状态/参数实例，固定 action policy 与环境版本。
3. 每个到达状态构造少量有意义的完整集合 arms：Recent-`B`、当前 selector、
   random/hard-negative、training-only witness/oracle proposal，以及不同 cardinality。
4. 若环境支持 snapshot/fork，从同一 pre-action state 对 2--4 个 allocation 做短 horizon
   或完整 episode branch rollout；否则用重复 instance/seed 做配对 rollout。
5. 用相对 Recent-`B` 的 n-step/episode return 训练 set-value 或 pairwise preference，
   同时保留 offline DiD/margin 作为 auxiliary loss。
6. 用更新后的 selector 重新采集 on-policy states，做 DAgger/policy-iteration 式迭代；
   checkpoint 只按 template-disjoint closed-loop return 与 cost 选择。
7. 在完全未参与训练、threshold、prompt 和 early stopping 的 templates/platform 上做最终
   closed-loop evaluation。

Branch rollout 很重要：它在相同环境状态下改变 memory allocation，比“比较两条自然成功
轨迹”更接近真正的 allocation intervention。若只能完整重跑，也应按 task instance/seed
配对，而不是把不同 trajectory 当作独立样本。

### 4.4 第二阶段再决定是否更新 action policy

只有 frozen-`pi_0` 的 selector 已显示 closed-loop headroom 后，才进入联合改进：

1. 用成功与高-return rollout 对 `pi_0` 做有限 SFT/RL，得到 `pi_1`；
2. 冻结 `pi_1`，废止把 `pi_0` utility 当作当前真值；
3. 重新采集/重标 memory data，训练 `mu_1`；
4. 每轮分别报告 policy-only、memory-only 和 joint delta。

这种交替更新比从第一天就 joint RL 更慢，但能避免 policy 与 selector 相互补偿后无法判断
论文贡献来自哪里。若最终目标是 agent RL paper，可在 selector-only 成立后把交替过程
推广成 joint policy optimization。

## 5. 最小 Go/No-Go 路线

### Gate A：closed-loop memory oracle headroom

- 先取一小批 train templates，在相同 snapshot/instance 上比较 Recent-`B`、若干手工/
  witness set 与不同 `|S|`；
- 只问“存在可重复的 closed-loop allocation headroom 吗”；
- 如果 oracle/witness allocation 都不能稳定改善 return，停止训练 selector。

### Gate B：frozen-policy learned selector

- 只训练 memory policy；
- primary 对照为同 policy、同最大 `B`、同 instance 的 Recent-`B`；
- 同时报实际图片数、latency/second-pass cost 和 template-clustered CI。

### Gate C：at-most-B / adaptive-cost curve

- 比较 exact-`B`、hard at-most-`B`、`return-lambda|S|` 或 average-budget constrained
  policy；
- 报告 success versus actual images 的 Pareto frontier，而不是只报 capacity `B`；
- 验证收益是否来自更好的 set identity、合理少用图，还是仅仅固定选满。

### Gate D：可选的 policy improvement

- Gate B/C 成立后才做 SFT/RL policy update；
- 每次 policy revision 后重新做 selector calibration 与 closed-loop evaluation；
- 不把旧 policy 的 offline DiD 数值当作新 policy 的 transferable guarantee。

## 6. 对当前论文的影响

- 当前 AAAI working paper 继续使用 exact-`B` matched-budget reallocation 作为最干净的
  causal/protocol contrast；它避免 image-count confound，已有直接证据。
- At-most-`B` 在新的 closed-loop + cost-aware protocol 通过前，只作为方法扩展与未来
  路线，不应把旧 direct-marginal 负结果改写成正面证据。
- 若 frozen-policy closed-loop selector learning、variable-cardinality Pareto curve 和
  on-policy iteration 都成立，这会形成一条比当前 offline DiD 更接近 ICLR/agent-RL 的
  独立工作：核心问题从“offline conditional utility estimation”升级为
  “policy-relative visual-memory control under distribution shift”。

## 7. Source of Truth

- 本方法决策：当前文件，Git canonical；状态 `WORKING`。
- 旧 at-most-`B` 训练合同与修复记录：
  [`docs/archive/set_utility_direct_marginal_v3.md`](archive/set_utility_direct_marginal_v3.md)、
  [`docs/archive/set_utility_direct_on_policy_v1.md`](archive/set_utility_direct_on_policy_v1.md)。
- 旧 fixed-tune 负结果：
  [`data/results/archive/set_utility_direct_marginal_v3_fixed_tune_v1/`](../data/results/archive/set_utility_direct_marginal_v3_fixed_tune_v1/README.md)。
- 当前 Desktop DiD 证据：
  [`data/results/desktop_did_policy_v1/`](../data/results/desktop_did_policy_v1/README.md)。
- 当前 MobileWorld closed-loop 证据：
  [`data/results/mobileworld_hgkv_selected_b4/`](../data/results/mobileworld_hgkv_selected_b4/README.md)。
- 本文未生成新的 dataset、checkpoint 或 raw rollout；无新增 HF upload 状态。
