# Direct conditional-marginal student v3

状态：`ORIGINAL STAGE-A NO-GO PRESERVED / VERSIONED STAGE-A-PRIME REPAIR AUTHORIZED`。原授权依据为
[`fixed-tune Long+ oracle v1`](../data/results/set_utility_tune_long_oracle_v1/README.md)：同 denominator
oracle--recent macro=`+0.3750 [0.2627,0.5656]`。唯一一次 rank-loss repair 仍未通过冻结 Stage-A gate；
原合同按 stop rule 终止，NO-GO 记录不可改写。后续复审认定 full-list Spearman 与 at-most-4 部署任务
错位，另立一次透明的 post-hoc Stage-A′ development gate；执行边界见
[`Stage-B v1`](set_utility_direct_marginal_v3_stage_b_v1.md)，不是对原结果的追认。

## 1. 修复的合同错配

v2 回归 scalar `U(S)`，再用差分得到 marginal。它在 250 个 train states 上表现为：预测 utility
250/250 随集合基数严格上升、at-most-`B` 250/250 选满、singleton Spearman 仅 0.11；真值却有
56.6% 的第二次加入有害。v3 不再学习 scalar utility，直接学习：

\[
\hat\Delta_j(q,C,S)\approx U(S\cup\{j\})-U(S),\qquad \Delta_{\mathrm{STOP}}=0.
\]

推理时每步在 `STOP` 和所有未选事件中取最大 predicted marginal；选到 STOP 即终止，因此天然是
at-most-`B`。模型不输入 `B`，head 不输入显式 cardinality feature，也不含会结构性鼓励“more is better”
的 scalar offset。

## 2. 架构与延迟合同

1. frozen GUI-Owl contextual visual/text hidden sequences 经共享 multimodal latent resampler；
2. 每个事件的多个 latents 用当前 query 做 attention pooling，不先做无条件 mean；
3. Set Transformer 在所有 candidate events 上只编码一次，形成 subset-independent contextual events；
4. 每个 selection step 用 candidate-to-selected attention 得到 candidate-specific coalition context，一次并行输出
   全部 `n_t` 个 marginal；STOP 固定为真实 no-op gain `0`；
5. 新增事件后只重跑轻量 selected-attention/marginal head，不重跑 GUI-Owl token encoder、resampler 或 candidate
   Set Transformer。

因此部署成本为一次 state encoding + `O(B n_t^2)` 的小 head；同 checkpoint 服务
`B∈{1,2,3,4}`。正式报告 warm shared-encoder latency、cold latency、p50/p95 和 action-policy forward 比例。

## 3. 两阶段、无 tune 泄漏的止损门

### Stage A：250-state train-only singleton fit probe

- 数据：immutable long-oracle 250 train states 的 empty + candidate-complete singleton truth；不生成新 label；
- 唯一 variant：direct marginal、hard listwise + explicit STOP、sign/marginal calibration；
- 每 epoch 评估同一 train-only probe；连续 2 epoch 同时满足
  `singleton Spearman > 0.5` 与 `true-best top-4 recall > 0.6` 即通过，最少训练 3 epochs；最多 30 epochs；
- Stage A 只回答优化/表示能否把已见排序拟合进去，不是泛化结果。

Stage-A v1 的正式结果是 Spearman=`0.238<0.5`、top-4=`0.987`、top-1=`0.904`、learned/oracle
B1=`0.473/0.501`。conjunctive gate 因 Spearman 正式未过，结果不可追认为 PASS。由于 best-event retrieval
已达到 90.4%，失败被定位为 tail pairwise ordering，而非 direct head 完全无法拟合；在不访问 tune/evaluation
的前提下，预先冻结**唯一一次** rank-loss-only repair：`within_state_ranking 1→8`，regression 权重降低，架构、
数据、seed 与 gate 均不变，最多 40 epochs。若 repair 仍未通过，停止 learned general-`B` 路线，不创建第三个
probe variant。

Rank repair 的正式结果为 Spearman=`0.291<0.5`、top-4=`0.951`、top-1=`0.841`、learned/oracle
B1=`0.437/0.501`。Spearman 相比 v1 仅提高约 `0.052`，同时 top-1 与 B1 recovery 下降；因此不能把失败继续
归因于简单 loss 权重不足。conjunctive gate 正式 FAIL，Stage B 永久未授权。轻量结果见
[`rank repair result`](../data/results/set_utility_direct_marginal_v3_fit_probe_v2_rank_repair/README.md)。

### Stage B：完整 conditional-marginal training

原 Stage A 未通过，因此本节在原合同下未授权。版本化 Stage-A′ 已另行允许一次 Stage-B v1；训练单位仍是
complete expansion group：

```text
(q, C, selected S, remaining candidate j or STOP)
    -> true conditional marginal Δ_j(S)
```

- 使用现有 train-only complete groups（当前 census 67,322 groups），按 trajectory、history bin、base
  cardinality 与 STOP/non-STOP 最优动作分层；同 state 的 groups 不当作独立 trajectory；
- primary loss：hard STOP-aware listwise classification + teacher regret；
- auxiliary loss：normalized marginal Smooth-L1、pairwise ranking、gain-sign classification；
- 不再加入 scalar `U(S)` regression，也不从 subset cardinality 预测 offset；
- checkpoint 只由 train split 内确定性 trajectory holdout 的 conditional decision regret 选择；既有 tune truth
  不进 optimizer、early stopping、超参数选择或 checkpoint 选择。

版本化 Stage-B 只能由新合同启动；fixed-tune gate 不变。untouched evaluation、policy replay、closed-loop
与 matched-NLL 在 fixed-tune GO 前继续锁定。

## 4. 实现与 frozen Stage-A config

- model：[`TokenConditionalMarginalPredictor`](../code/causalcache/set_utility_token_models.py)；
- Stage-A trainer：[`train_set_utility_long_oracle_fit_probe.py`](../code/scripts/train_set_utility_long_oracle_fit_probe.py)；
- config：[`causalcache_set_utility_long_oracle_fit_probe_v1.json`](../code/configs/causalcache_set_utility_long_oracle_fit_probe_v1.json)；
- 唯一 repair config：[`causalcache_set_utility_long_oracle_fit_probe_v2_rank_repair.json`](../code/configs/causalcache_set_utility_long_oracle_fit_probe_v2_rank_repair.json)；
- tests：`code/tests/test_set_utility_long_oracle_fit_probe.py`，覆盖固定 STOP、conditional rescoring、梯度与
  trajectory-equal gate；与 long-oracle tests 合计 13 passed。

## 5. Firewall

Stage A/B 都只读取 role=`train`。刚生成的 fixed-tune Long+ oracle labels 明确
`labels_reusable_for_training=false`；它们只用于授权本路线，禁止合并进训练输入。Stage A 的 checkpoint 也
不能直接进入 paper evaluation；只有 Stage B 的冻结 checkpoint 才可能进入新合同。
