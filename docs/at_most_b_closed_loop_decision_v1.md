# At-most-B 与 closed-loop memory learning 方法决策 v1

日期：2026-07-31

状态：`WORKING / METHOD_DECISION`

观测分支：`paper-draft`

观测 revision：`72363dcd6fb40e68e82156f7bbf3795ccd5d47d3`

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
4. **下一版应把 memory selection 与 environment action 写成一个分层联合策略。** 冻结
   action-policy 权重并不能冻结它的行为：selector 改变输入图片的身份、数量和布局后，
   同一组权重也会产生新的 action distribution。永久冻结 action policy 只是在评估一个
   固定模型对新 observation policy 的兼容性，不能解决 selector--policy co-adaptation。
   正确路线是先让 action interface 覆盖 `0..B` 的 promotion-mask 分布，再用 on-policy
   rollout 对 memory policy 与 action policy 做小步交替或联合优化。
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

### 2.1 冻结权重不等于冻结 action policy 的行为

令 renderer 为 `R`，selector 为 `mu`，action-policy 权重为 `theta`。真实 action
distribution 是：

```text
p(a_t | h_t) = pi_theta(a_t | R(h_t, S_t)),  S_t ~ mu(. | h_t).
```

即使 `theta` 完全冻结，改变 `mu` 仍会改变送给 `pi_theta` 的 observation，进而改变
action distribution。Frozen weights 只能排除“action model 参数也在更新”，不能排除：

- 图片数量和 token/turn layout 的格式漂移；
- recent-dense 到 non-contiguous/sparse images 的语义分布漂移；
- action policy 与 selector 形成新的 feedback loop；
- selector 利用 policy 的格式脆弱性，而不是真正找到有用视觉证据。

这不是纯理论风险。既有 renderer audit 已观察到：旧单轮 sparse 格式下 Random restoration
也能显著胜 Recent，而 official-style multiturn 下 Random 回到零附近。对应结论是“赢的是
格式，不是选点”，见 [`renderer_freeze_v1.md`](renderer_freeze_v1.md)。

当前 exact-`B` matched-budget 设计能控制图片数量、主要 prompt structure 与分辨率，因而
较适合隔离 image identity。真正的 at-most-`B` 会进一步改变 active image count；若不先
训练/验证 variable-cardinality interface，closed-loop return 无法区分“少图更好”与“模型
不适应这种格式”。因此，frozen-policy branch rollout 仍可作诊断，但不能成为推荐的最终
训练算法或单独的 attribution 结论。

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

### 4.1 先建立 variable-cardinality-compatible action interface

在学习 selector 之前，先让 action policy 见过部署时可能出现的 observation family，而不
只习惯“action summaries + dense Recent-`B` images”。训练分布应覆盖：

- `m=0..B` 的实际 promoted-image count；
- recent、relevant、mixed 与 non-contiguous allocation；
- 保持 event-summary trace、renderer、图片位置语义和 action serialization 一致；
- 明确的 cardinality/promotion-mask 条件，而不是让模型从 token 长度暗猜协议。

可用 SFT/behavior regularization 做 interface warm start，但同一个 gold action 只应用于
该 action 在当前 observation 下仍可判定或经 rollout 验证正确的 allocation；不能强迫
缺失关键证据的 wrong/random arm 复述 gold action。DiD、B0 parity、recent/wrong drift
cap 继续作为 auxiliary constraints，防止 policy 把“任何稀疏格式”都学成统一增益。

这一步不是在学习最终 selector，而是在让 action policy 对 selector 将要产生的输入分布
可用。对应的必要对照是：在相同 image identity 下只改变 `m`/layout，以及在相同 `m`/
layout 下只改变 image identity。

### 4.2 将 memory action 与 environment action 联合建模

每个决策点包含两个相连的 action：

```text
S_t ~ mu_phi(. | h_t)                         # memory/acquisition action
a_t ~ pi_theta(. | h_t, R(S_t))               # environment action
```

联合目标为：

```text
maximize E[R_task - lambda * sum_t |S_t| - eta * latency_t].
```

若论文只关心 hard capacity，可令 `lambda=eta=0`；但这时最优解经常会用满预算，不能把
“实际少用图”当作必然结果。若目标包含 adaptive compute/memory，必须显式给每张图或
每次 second pass 定价，或约束 episode-level average image budget。

这里 headline method 是 `(mu_phi, pi_theta)` 的组合，而不是声称 `mu` 脱离 action policy
仍有普适 utility。为保留归因，训练和结果中应另报：固定 joint checkpoint 后替换 selector
的 memory-policy delta、固定 selector 后替换 action checkpoint 的 policy delta，以及完整
joint delta。

### 4.3 不训练自由 STOP；联合比较 0..B 的完整集合

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

### 4.4 Template-level closed-loop data collection 与交替更新

建议的最小闭环数据循环：

1. 按 **template** 划分 train/dev/test；同 template 的不同参数实例不得跨 split，避免
   selector 记住操作脚本或 UI 字段。
2. 对训练 templates 生成多个初始状态/参数实例，冻结环境版本，以 interface-randomized
   SFT policy 初始化当前 action policy。
3. 每个到达状态构造少量有意义的完整集合 arms：Recent-`B`、当前 selector、
   random/hard-negative、training-only witness/oracle proposal，以及不同 cardinality。
4. 若环境支持 snapshot/fork，从同一 pre-action state 对 2--4 个 allocation 做短 horizon
   或完整 episode branch rollout；否则用重复 instance/seed 做配对 rollout。
5. 用相对 Recent-`B` 的 n-step/episode return 更新 set-value/memory policy，同时保留
   offline DiD/margin 作为 auxiliary loss。
6. 在当前 selector 诱导的 observation distribution 上，用成功/high-return rollouts 对
   action policy 做受 KL/behavior anchor 约束的小步 SFT/RL 更新；不能只训练 selector。
7. 重新 rollout 当前 `(mu,pi)`，交替执行短暂的 memory-policy update 与 action-policy
   update；每轮旧 utility labels 都标记为上一 policy revision 的历史数据，而非当前真值。
8. checkpoint 只按 template-disjoint closed-loop return、actual images 与 cost 选择。
9. 在完全未参与训练、threshold、prompt 和 early stopping 的 templates/platform 上做最终
   closed-loop evaluation。

Branch rollout 很重要：它在相同环境状态下改变 memory allocation，比“比较两条自然成功
轨迹”更接近真正的 allocation intervention。若只能完整重跑，也应按 task instance/seed
配对，而不是把不同 trajectory 当作独立样本。

训练时短暂冻结一侧只是一种 coordinate-update 工程手段，不是科学假设：`mu` 更新几步后
必须让 `pi` 在新 observation distribution 上适配，`pi` 更新后又必须重新采集 memory
return。为防止两者互相补偿到不可解释，每轮更新幅度要受 trust region/KL 限制，并保留
上面的三类固定组件评测；不能把“永久冻结 action policy”作为解决鸡生蛋问题的答案。

## 5. 最小 Go/No-Go 路线

### Gate A：closed-loop memory oracle headroom

- 先取一小批 train templates，在相同 snapshot/instance 上比较 Recent-`B`、若干手工/
  witness set 与不同 `|S|`；
- 只问“存在可重复的 closed-loop allocation headroom 吗”；
- 如果 oracle/witness allocation 都不能稳定改善 return，停止训练 selector。

### Gate B：variable-cardinality interface gate

- 在 content-matched 条件下检查 `m=0..B`/layout 是否导致系统性 action drift；
- 在 cardinality-matched 条件下检查 relevant/recent/wrong 的内容选择性；
- Random 不能因为格式变化而稳定胜 Recent，B0/summary-only 路径保持明确的 parity/drift
  边界。

### Gate C：joint closed-loop policy iteration

- 用当前 `(mu,pi)` 采集 template-level paired/branch rollouts并交替更新；
- primary 对照为同 action-policy checkpoint、同最大 `B`、同 instance 的 Recent-`B`，
  另报 policy-only 与 full-joint delta；
- 每次 `pi` revision 后重新做 selector calibration，不复用旧 policy utility 作当前标签。

### Gate D：at-most-B / adaptive-cost curve

- 比较 exact-`B`、hard at-most-`B`、`return-lambda|S|` 或 average-budget constrained
  policy；
- 报告 success versus actual images 的 Pareto frontier，而不是只报 capacity `B`；
- 验证收益是否来自更好的 set identity、合理少用图，还是仅仅固定选满。

## 6. 对当前论文的影响

- 当前 AAAI working paper 继续使用 exact-`B` matched-budget reallocation 作为最干净的
  causal/protocol contrast；它避免 image-count confound，已有直接证据。
- At-most-`B` 在新的 closed-loop + cost-aware protocol 通过前，只作为方法扩展与未来
  路线，不应把旧 direct-marginal 负结果改写成正面证据。
- 若 variable-cardinality interface、joint closed-loop policy learning 与
  variable-cardinality Pareto curve 都成立，这会形成一条比当前 offline DiD 更接近 ICLR/agent-RL 的
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
