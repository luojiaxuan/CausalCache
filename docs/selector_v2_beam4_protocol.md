# HGKV Selector V2 Beam-4 冻结协议

状态：`FROZEN_BEFORE_V2_DEVELOPMENT_READ`  
冻结日期：2026-07-24  
正式 config：`code/configs/hgkv_selector_v2_beam4.json`

## 1. Supersession 与研究问题

V1 终态为 `NO_GO_HGKV_SELECTOR_V1`：Stage-1 singleton selector 显著输给 Recent；
set-conditioned B1/第一步被 Stage-1 固定；candidate inventory 截断为 recent-8；
conditional prefix 由 singleton proxy 产生；edge3 缺失。V1 scoring 已停止，V1
Stage-2 未启动。V1 config、源码和已有 artifact 保留为 provenance。

V2 不沿用 V1 两阶段决策结构。唯一主方法定义为：

```text
完整真实历史候选
+ 从空集合开始
+ true-U teacher beam-4
+ edge0 / edge1 / edge2 / edge3
+ fresh unified set-conditioned student
+ learned beam-4 inference
+ STOP
```

预算固定为 `B∈{1,2,4}`；B8 只属于 policy memory-dose stress test，不进入 selector
V2。

## 2. Utility、集合与 STOP

对 decision state 的历史事件集合 `S`：

```text
U(S) =
  log p_hg-s100(success action | restored set S)
  - log p_frozen(success action | B0)
```

`U(empty)=0` 由已认证 B0 parity 定义。统一监督：

```text
Δ(j | S) = U(S ∪ {j}) - U(S)
```

四个深度分别为：

```text
edge0: |S|=0
edge1: |S|=1
edge2: |S|=2
edge3: |S|=3
```

STOP 的真实 marginal 固定为 0；`max_j Δ(j|S) <= 0` 时 STOP 最优。teacher 仍渲染
负 marginal child 作为有害添加监督，但不把它解释成 teacher 应继续选择。

## 3. Full-history state inventory

对每条成功 GUI-Odyssey trajectory 只选择一个 state：

1. 只保留真实标注 decision；
2. 排除 synthetic terminal state；
3. action 必须能序列化为正式 GUI-Owl target；
4. 选择拥有最长 candidate history 的 eligible decision；
5. 并列时选择更大的 decision step；
6. candidate 少于 2 时排除 trajectory；
7. candidate 是完整真实 history 中的所有事件，减去最新 current-equivalent event；
8. 禁止 recent-8 或其他 `max_candidates` 截断。

分别生成 Git-tracked lightweight manifest：

```text
data/manifests/hgkv_selector_v2_train_states.jsonl
data/manifests/hgkv_selector_v2_dev_states.jsonl
```

inventory 必须审计 trajectory/state 数、candidate count min/median/p90/p95/max、
`candidate_count>=4` 比例、action/app 分布、history histogram，并硬检查 synthetic
terminal=0、recent-8 truncation=0。

Odyssey development 只在 train-only epoch selection 和 full-train refit 后读一次，不得
进入训练或 width/architecture/loss 修改。

## 4. Coalition score cache

V2 cache canonical key：

```text
(
  pair_group,
  restored_set_key,
  target_action_sha256,
  prompt_revision,
  hgkv_checkpoint_sha256,
  b0_policy_sha256
)
```

value 至少包含 `target_logprob_mean`、`u_act`、`source_artifact`、`source_version`。
只摄取 frozen B0、V1 hg-s100 singleton、V1 已完成 conditional score 和其他完整
selected-set score 的 exact-key 命中；冲突值 fail closed，近似相同不得复用。

同一完整集合的真实 `U(S)` 与产生它的旧 selector path 无关，因此 score 数值可以复用；
V1 recent-8/singleton-proxy 的 prefix sampling distribution 不可复用。

## 5. Full-history singleton 与 feature schema

V2 对 inventory 中所有候选补齐 singleton render、hg-s100 singleton score 和 HGKV
readout。旧 `(pair_group,event_step_id)` feature exact hit 直接复用，只补新 state 或
recent-8 之外的早期候选。

冻结 feature 为：

```text
1280-d HGKV counterfactual readout
+ relative_age
+ normalized_age
+ normalized_position
+ is_most_recent
+ log_history_length
= 1285 dims
```

禁止 app ID、AndroidWorld/OSWorld 或 benchmark-specific feature。V2 student fresh init，
不得读取 V1 Stage-1 checkpoint。

## 6. True-U teacher beam-4

teacher beam width 固定为 4，所有 prefix 按真实 `U(S)` 排序：

```text
beam_0 = {empty}
beam_1 = top4 unique singletons by true U
beam_2 = top4 unique pairs by true U
beam_3 = top4 unique triples by true U
depth 3 expands all remaining candidates to edge3
```

每层对每个 active prefix 展开全部剩余 full-history candidates；canonical set 去重，同一
coalition 只打一次。禁止按 singleton utility 求和或 predicted score 产生 teacher beam。

候选数 `n` 时未去重上限为 `13n-24` sets/state。在 `n<=8` 的 development validation
subset 同时运行 exact subset search，报告 beam recovery、regret、top-B Jaccard 和
B1/B2/B4 utility gap。beam width 不因该 development 结果调整，第一版不并行尝试 beam-8。

## 7. Training groups、budget replication 与 weighting

一个训练样本是一个 prefix group，保存 selected/candidate event IDs、feature indices、
全部 marginal targets、remaining budget 与 depth。budget replication：

```text
depth0: remaining budget 1,2,4
depth1: remaining budget 1,3
depth2: remaining budget 2
depth3: remaining budget 1
```

candidate width 由当前 batch 动态 padding；selected width 固定最大 3。loss 先在
candidate 维度平均，再在 prefix group 维度平均，避免长 history 获得更大总权重。

## 8. Unified student 与 checkpoint

student 从头初始化：candidate encoder、selected-set self-attention、
candidate-to-selected-context attention、remaining-budget embedding，以及 marginal、
rank、positive、STOP 四头。空集合由 learned `empty_selected` token 表示。

总 loss：

```text
L = L_marginal + 0.5 L_rank + 0.25 L_positive + 0.25 L_STOP
```

- marginal：真实 `Δ(j|S)` 的 SmoothL1；
- rank：同 prefix candidate-candidate pairwise logistic；
- candidate-STOP rank：按 marginal 正负；
- STOP：`max marginal <= 0`；
- candidate loss 先 group 内平均。

STOP 必须报告 balanced accuracy、precision/recall/F1、AUROC、predicted/true stop rate，
并按 depth 0/1/2/3 分层；raw accuracy 不是主指标。

checkpoint selection 冻结为 train trajectory GroupKFold(5)，以 teacher-forced
learned-beam4 realized U 选择 epoch；五折 epoch 中位数用于 full-train refit；然后
development 只读一次。

## 9. Learned beam-4 inference

每个预算从 `(empty, cumulative_predicted_U=0)` 开始。每层：

1. 每个 active prefix 预测全部 remaining candidates；
2. 生成所有 candidate child 和一个 STOP terminal child；
3. candidate child 累计 calibrated predicted marginal；
4. STOP child 保持当前累计值；
5. canonical set 去重；
6. 保留累计 predicted U 最高的 4 条 path；
7. 达预算或全 terminal 时结束。

最终返回累计 predicted U 最高的 terminal/nonterminal path。跨 prefix pruning 只使用
calibrated marginal 累计值；raw rank 只在同 prefix 内排序以及局部 candidate-vs-STOP。
同时输出 greedy V2 ablation 和逐步 beam trace。

## 10. Formal selected-set gate

比较：

```text
Random
Recent
OCR/RGB Similarity
V1 singleton selector
V2 greedy
V2 beam-4 (main)
teacher beam-4 (offline upper bound)
exact oracle (n<=8 subset only)
```

每个 B1/B2/B4 完整集合都必须重渲染、送入 hg-s100、exact-join frozen B0，再计算真实
`U(S)`。预测 marginal、累计 predicted U 和 singleton sum 不得进入正式 utility。

统计单位为 episode：先在 episode 内平均 state，再做 10,000 次 paired
episode-cluster bootstrap。冻结 gate：

- B1：`V2 beam-4 − Recent` 的 95% paired CI upper bound `>=0`，解释为没有显著劣化；
- B2：相对 Recent/Similarity/Random 的全部 CI lower bound `>0`；
- B4：相对 Recent/Similarity/Random 的全部 CI lower bound `>0`；
- `Avg(B1,B2,B4)` 相对 Recent 的 CI lower bound `>0`。

同时报告 beam-4−greedy、beam-4−V1 singleton、STOP rate、mean selected size、teacher
beam recovery 与 exact-oracle recovery。只有完整 offline gate PASS 才允许接入
AndroidWorld/OSWorld closed loop。

## 11. Definition of Done

V2 只有同时满足以下条件才完成：

```text
V1 已封存且不会自动恢复
每条 train trajectory 使用最长真实 eligible decision
candidate 覆盖完整真实历史
edge0/edge1/edge2/edge3 全部存在
teacher beam 只按真实 U 排序
student 从空集合 fresh init
inference 使用 learned beam-4 + STOP
B1/B2/B4 完整 selected set 真实重打分
offline gate PASS 后才进入 closed loop
```
