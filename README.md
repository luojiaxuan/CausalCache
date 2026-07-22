# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

目标会议：AAAI。项目当前优先验证一件事：由 restoration utility 监督的 set predictor，能否在未参与训练的 GUI states 上超过 recent 与 OCR/RGB heuristic。

## 当前结论

- **OSWorld runner v1 已完成真实 KVM live smoke。** runner 绑定官方 pinned `DesktopEnv`，覆盖 10 domains / 369
  tasks 的 inventory、受限 desktop action 到 PyAutoGUI 的安全映射、mixed-fidelity policy HTTP boundary、
  task-level 可断点 suite sharding 与原子 episode evidence。Hyper01 的真实 reset→screenshot→WAIT→DONE→
  evaluate→close 已通过，第二次运行 `resumed_skips=1`；当前仍是跨平台 benchmark substrate，不是科学结果，
  learned selector、GUI-Owl desktop policy 和正式 OSWorld roster 尚未接入。复现见
  [`docs/osworld_runner_v1.md`](docs/osworld_runner_v1.md)，证据见
  [`data/results/osworld_runner_v1_smoke/`](data/results/osworld_runner_v1_smoke/)。
- **主线已转为 selector-side GUI-Owl LoRA；旧 budget-deferral evaluation 已在读取 truth 前停止。**
  旧 evaluation 的 `truth read=0`，partial receipts 仅作为可恢复执行记录保留，不产生结果。Teacher/action
  policy 始终是原始 frozen GUI-Owl；LoRA 只更新 selector encoder，因此现有 restoration labels 继续有效。
  GUI-Owl top-4 branch 已通过真实 context bitwise parity（max absolute difference=`0.0`）。23,714 个
  layer-32 boundary contexts 已由 Hyper00/Hyper01 的 12/12 partitions 完整提取、0 failure；Hyper00
  单机 cache 已 finalize：3,065 shards、289,753,201,728 bytes、content=`77dec757...5d16b`。LoRA trainer 与
  token-adapter control 已实现；两条 adaptation 分支首 epoch 均已完成。token-adapter 的 phase-1 truth
  已封存并回填，但 adapter-only recovery 低于 recent；LoRA-only 的 truth 也已封存，macro/Long+=
  `0.30542/0.32770`，同样低于 recent。joint token-adapter e1 又降至 macro/Long+=
  `0.23919/0.28640`，因此该诊断对照停止。joint LoRA e1 的 fixed-256 truth 已完成：macro=
  `0.40535`，比 recent 高 `+0.03205`，B3/B4 高 `+0.07471/+0.05389`；但 B2 与 Long+ 仍基本持平。
  e2 的 macro/Long+=`0.39455/0.36339`，相对 e1 下降 `0.01080/0.01622`，B2 下降 `0.02808`，
  只有 B4 继续提高。按逐 epoch heldout 资源纪律停止 e3/e4并保留 e1；当前 top-layer LoRA 没有解决
  B2/Long+，因此该 representation hypothesis 未获支持。
  [设计](docs/set_utility_selector_lora_v1.md)与
  [状态](data/results/set_utility_selector_lora_v1/README.md)。
- 正式 label rollout 已使用 Hyper00/Hyper01 各 6×H200 完成：5,108/5,108 states、33,175/33,175
  microbatches、359,189 sampled rows + 5,108 zero-cost anchors，0 skip/non-finite/duplicate/error。合并后的
  training input content=`6c9243a...1bfad`；formal optimizer inventory 为 900 trajectories、9,287 states、
  84,441 candidate-complete groups。epoch 1 的联合 held-out truth 已封存：198 states、7,681 forwards、
  manifest=`a6ba7c...efa36`，0 skip/error；Set Transformer 与 structured/DeepSets 的真实 B1--B4 macro
  recovery 分别为 `0.38508/0.39251`，Long+ 为 `0.36343/0.37929`。两者均把 epoch 1 设为首个 best 后才
  进入 epoch 2。Set epoch 2 提升至 macro/Long+=`0.39491/0.38618` 并成为新 best，epoch 3 降至
  macro=`0.00610`，因此保留 epoch 2；DeepSets epoch 2 为全 STOP、macro=`0`，因此保留 epoch 1。
  两条训练线均已按 `minimum_delta=0.005`、`patience=3` 完成真实 recovery 早停。Set Transformer 的
  epoch 1--5 macro=`0.38508/0.39491/0.00610/0.39529/0.19033`；epoch 4 相对 epoch 2 仅提升
  `0.00038`，因此最终选择 epoch 2（checkpoint `ed890219...ad3da1`）。DeepSets 的 epoch 1--4 macro=
  `0.39251/0/0.39581/0.31853`；epoch 3 相对 epoch 1 仅提升 `0.00330`，最终选择 epoch 1
  （checkpoint `5f5e21db...9cd3e`）。这确认 training loss 或最后 epoch 都不能代替真实 selector recovery；
  evaluation/test 仍密封。[完整轨迹](data/results/set_utility_direct_on_policy_training_v1/README.md)。
- **Direct conditional-marginal v3 已在最终 fixed-tune gate 正式 NO-GO，learned general-`B` 路线停止。**
  1,063/1,063 states、0 skip；direct-v3/recent macro=`0.44583/0.45192`，delta=`-0.00609`，95% CI=
  `[-0.03012,+0.01642]`。它只在 B1/B4 胜 recent，B2/B3、macro CI、Long+ 全部失败，并弱于旧
  scalar-v2=`0.47593`；1,063/1,063 states 在 B1--B4 全部选满，STOP 未学会。按唯一修订时的预承诺，
  不创建 v4、不再修改 gate、不访问 untouched evaluation/closed-loop。
  [最终结果](data/results/set_utility_direct_marginal_v3_fixed_tune_v1/README.md)。
- **Direct conditional-marginal v3 的原 Stage-A NO-GO 保留，但已通过一次透明的版本化 Stage-A′ repair
  进入 Stage-B。** 唯一 rank-loss repair 在 250 个 train-only long-oracle states 上把 singleton Spearman 从
  `0.238` 提到 `0.291`，仍远低于冻结门槛 `>0.5`；top-4 recall=`0.951` 虽通过，但 top-1 与 B1 recovery
  反而降到 `0.841/0.437`。复审确认标签确定、Spearman 天花板为 1，但 full-list tail ordering 不进入
  at-most-4 部署；v1 在修订提出前已有 top-1/top-4/B1-oracle/STOP=`0.904/0.987/0.944/1.0`。新门槛明确
  标记为 post-hoc development gate，不是 paper evidence；最终仍由一字不改的 fixed-tune gate 判生死。
  [Stage-B 合同](docs/set_utility_direct_marginal_v3_stage_b_v1.md)。
- **同 denominator 的 tune Long+ true conditional-greedy oracle 已强阳性通过，曾授权 direct-marginal v3。** scalar `U(S)`
  student 的病理已经确认：250/250 states 预测 utility 随基数严格上升、at-most-`B` 全部选满，而真实
  第二次加入有 56.6% 会降低 utility，singleton in-sample Spearman 仅 0.11。在 fixed-tune 的同一 249
  Long+ states 上，oracle/recent macro=`0.6949/0.3199`，差值=`+0.3750 [0.2627,0.5656]`，大幅通过
  point `>0.10`、CI lower `>0.03` 的门槛。结论是 student 问题，不是 tune 人群没有 headroom；该结果
  曾授权一次 direct conditional-marginal + explicit STOP v3，但其 Stage-A 后续已按门槛失败。
  [正式结果](data/results/set_utility_tune_long_oracle_v1/README.md)。
- **Decision distillation v2 首次在真实 fixed-tune selector truth 上显著超过 recent，但冻结 gate 仍为
  NO-GO。** 基础 Set Transformer 在 1,063/1,063 states 上的 B1--B4 macro recovery=`0.47593`，recent=
  `0.45192`，paired delta=`+0.02401`、95% CI=`[+0.00430,+0.04340]`；B1/B3/B4 均胜 recent。这是当前
  learned gate 最强的正信号。正式 gate 仍因 B2=`0.42930<0.43606` 与 Long+=`0.38853<0.40081` 返回
  `NO_GO_DECISION_DISTILLATION_V2`，所以 untouched evaluation、policy replay、closed-loop 与 matched-NLL
  继续锁定。[完整轻量结果](data/results/set_utility_decision_distillation_v2_epoch1_tune/README.md)。
- **更大 Set Transformer 没有自动改善 selector。** `latent64/set-layers4` epoch-1 macro=`0.45575`、
  Long+=`0.35577`、p95=`114.19ms`，均弱于基础 Set 的 `0.47593/0.38853/71.39ms`；DeepSets macro=
  `0.41636`。L64/S4 的较低 tune total 不可跨配置比较，因为其 eval batch=1、基础模型 eval batch=8，
  conditional-listwise complete-group 统计随 batch 改变。容量 run 仍可自然训练到 early stop，但当前
  epoch-1 truth 已否定“只加 latent/depth 即可闭合 gap”。
- **Held-out selector v1 正式 NO-GO。** 805/805 states 均有终态，但仅 801 completed、4 个因冻结 GUI-Owl
  strict tool-call parser 失败而 skipped，故正式状态为 `INCOMPLETE_SET_UTILITY_HELDOUT_EVALUATION`，没有合法
  deployment winner，policy replay 与 closed-loop 未获授权。
- completed states 上 Set Transformer 有方向性信号：B1--B4 primary recovery 为 0.4093，高于 recent 的
  0.4040 与 OCR/RGB 的 0.3826；相对 OCR/RGB 的 bootstrap 95% CI 为 `[0.0034, 0.0525]`，但相对 recent
  为 `[-0.0248, 0.0318]`，且 B2 exact-oracle regret 0.0281 差于 recent 0.0264 与 OCR/RGB 0.0230。
  因此这不只是 coverage 阻塞：当前 checkpoint 也没有满足冻结的 selector performance gate。
- **Small-history exact B4 证明主要风险是 distillation，而不是 restoration signal。** 在同一 reference
  session 的 103 states/73 trajectories 上，exact B1--B4 recovery 为
  `0.530/0.700/0.782/0.825`；true conditional greedy B4 为 `0.768`，search gap=`0.057`。Set Transformer
  B4 为 `0.713`，高于 DeepSets `0.706`、OCR/RGB `0.694` 与 recent `0.672`，但距 exact 仍差 `0.112`。
  因此下一步优先 contextualized multi-latent representation 与 deployment-search labels，不继续盲目堆同分布
  trajectories，也不因本诊断启动 closed-loop。
- **25% contextual representation 单独未闭合 gap。** 1,063/1,063 tune states 的 on-policy truth 上，
  contextual DeepSets/Set Transformer primary recovery=`0.4492/0.4365`，recent=`0.4519`；DeepSets 与
  recent 差值 95% CI=`[-0.0160,0.0100]`，Set Transformer 则显著更差。继续按既定要求跑 full-data 双模型，
  但 untouched evaluation、policy replay 与 closed-loop 仍未解锁。
- Restoration signal 存在，但旧版 conditional gate 与 independent gate 都没有在 untouched confirm 上稳定超过廉价 heuristic。
- GUIOdyssey 已固定 1,200 trajectories。旧 processor artifact 只物化每条轨迹的 anchor/terminal 两个 query，不能算 state-level 扩数完成。
- D2 已证明 strict-determinism runtime 可稳定复现先前的异常 state。
- 旧的 all-or-nothing throughput / stability 协议不再阻塞探索主线。新 MVP 按 state 接受：稳定 state 产标签，不稳定 state 记录并跳过。
- 旧 `scale-v1` 仅有 355 个 anchor states，现降级为 anchor-only pilot，不再作为扩数结果或继续调参依据。
- Oracle-independent `J` 在 B1/B2 恢复 exact utility 的 99.39%/90.06%，证明 independent restoration objective 在这批数据上有效；learned models 仍有明显 distillation gap。
- Dense per-step query census 已覆盖 12,792 states（10,680 train / 1,066 tune / 1,046 evaluation），但旧生成器把候选历史截断为 recent-4，改变了 long-horizon selector 的任务定义。该批 labels 与基于它的 token pilot 全部降级为 smoke-only，不进入正式 predictor 或方法结论。
- Predictor v2 已冻结为 full GUI-Owl visual/text token sequence + learned latent resampler；旧 64 维手工表示只保留为 cheap-feature baseline，不再代表主方法。
- train/tune-only recent-4 token pilot 已完成：745 states、1,043 unique images、22.78GB unpooled token cache；它只证明 optimization/data pipeline 可运行，不能验证 long-horizon selector。正式 variable-history full token sequences 已另行完成，不再使用该 pilot cache。
- 主部署口径已冻结为 warm shared-encoder：event embedding 在到达时计算一次、当前 query tokens 与 action policy 共享；正式方法还必须满足 warm selector p95 不超过 action-policy forward p95 的 10%，并完整报告 cold standalone latency。训练用 22.78GB full-token cache 不属于线上 memory。

完整历史与失败记录保留在 [`docs/progress.md`](docs/progress.md)，但不应把历史 formal contract 当成当前 MVP 的执行清单。

## 方法

完整历史策略给出参考行为。将历史事件降为摘要后，恢复 coalition `S`，得到与完整历史行为的距离 `D(S)`：

\[
U(S)=D(\varnothing)-D(S).
\]

当前 v3 student 直接预测 conditional marginal
`Δ_θ(q,C,S,j)`，并把 STOP 的 gain 固定为 0。模型始终看到全部候选和 selected mask；每加入一个事件后
重打分剩余候选，正式 selector 只使用 at-most-`B` search，不再把 fixed-`B` 当作结果口径。旧 scalar
`U_θ(q,C,m_S)` 仅保留为已失败的对照路线。

当前比较：

- pairwise-additive；
- DeepSets；
- Set Transformer；
- recent、OCR/RGB、oracle-independent `J`；
- exact subset oracle。

论文主模型与 DeepSets 共享完整 frozen GUI-Owl token cache，只替换 set aggregator；架构与当前 partial-label
pilot 合同见 [`docs/set_utility_predictor_v2.md`](docs/set_utility_predictor_v2.md)。

## 当前执行主线

- 旧 budget-deferral evaluation 已停止且 `truth read=0`；不再继续该 evaluation，也不把 partial receipts
  解释为结果。Hyper00 单机 boundary cache 与 versioned executable training config 已完成；当前顺序为
  Hyper00 6-GPU LoRA-only phase 1 已完成 fixed-256 truth 回填：macro/Long+=`0.30542/0.32770`，低于
  recent=`0.37330/0.38002`；下一步进入 joint LoRA。Hyper01 的 token-adapter phase 1 已完成 truth 回填：
  B1--B4 macro/Long+=
  `0.30893/0.30401`，低于 recent=`0.37330/0.38002`；joint adapter e1 进一步退化后停止。joint LoRA e1
  e1/e2 checkpoint 与 formal truth 均已完成；e1 macro/Long+=`0.40535/0.37961`，e2=
  `0.39455/0.36339`。最终保留 e1，停止相同配置的后续 epoch；所有 large artifacts 仍为
  `PENDING_HF_UPLOAD`。
  [LoRA 执行状态](data/results/set_utility_selector_lora_v1/README.md)。
- direct-marginal Stage-B 与 unchanged fixed-tune gate 均已完成；最终 `NO_GO` 已停止 learned
  general-`B` v3 路线。该旧合同不被追认；本轮是用户显式授权的 data-coverage/structured-fallback 新假设，
  下游 policy replay、closed-loop 与 matched-NLL 仍需由新 checkpoint 的 untouched evaluation 解锁。
  [最终结果](data/results/set_utility_direct_marginal_v3_fixed_tune_v1/README.md)。
- 同一 trajectory 的所有 eligible decision steps 保持在同一 split；
- 正式 state 使用当前决策前的全部 eligible events，`n_t=|C_t|` 是数据属性；不得做 recent-`n` 候选截断；
- 每个 state 约生成 40 个 age/interaction-stratified coalition labels，小历史可 exact，大历史只采样 subsets；
- census 见 [`data/results/set_utility_dense_v1/census.json`](data/results/set_utility_dense_v1/census.json)；
- recent-4 rollout 已在 4,033/12,792 states 时停止并封存为 smoke-only；结果见 [`data/results/set_utility_dense_recent4_smoke/`](data/results/set_utility_dense_recent4_smoke/README.md)。
- full-history source 与 256-shard full VLM hidden states 已物化；exact-grid context postflight 后直接启动 resumable 8-GPU label runner。evaluation 在模型选择冻结前不加载。
- train/tune broad schedules 已完成 256/256 shards：11,746 states、461,040 coalition labels、449,294 次非 full-anchor 样本；候选集始终为完整 `C_t`。结果见 [`data/results/set_utility_variable_history_schedules_v1/`](data/results/set_utility_variable_history_schedules_v1/README.md)。
- formal label rollout 已完成 11,746/11,746 terminal states：11,721 completed、25 skipped；state/microbatch 原子断点覆盖了 long-history OOM 与最后单 worker 恢复。运行入口见 [`data/results/set_utility_variable_history_labels_v1/`](data/results/set_utility_variable_history_labels_v1/README.md)。
- labels 并行期间的 1,501-state partial snapshot 已在 Hyper00 GPU 4/5 完成 DeepSets 与 Set Transformer 双卡训练；两者 tune objective 均明显下降，Set Transformer best=0.4714、DeepSets best=0.4853。该结果只验证优化路径，不消费 evaluation，也不做正式 GO/NO-GO；见 [`data/results/set_utility_variable_history_training_partial_v1/`](data/results/set_utility_variable_history_training_partial_v1/README.md)。
- 25% train-trajectory 单 seed 超参搜索已完成 12/12 配置：DeepSets 冻结 `d512/l16/r2/lr3e-4`（tune total 0.31137，最佳 checkpoint 后出现 non-finite），Set Transformer 冻结 `d256/l8/r1/s2/lr1e-4`（0.31169）。下一步从完成 labels 重建 nested 10/25/50/100% learning curve；[结果](data/results/set_utility_tuning25_v1/README.md)与[配置](code/configs/causalcache_set_utility_learning_curve_v1.json)。
- 正式 nested learning curve 已完成：100% 相对 10% 的 tune total，DeepSets 改善 0.01098、Set Transformer 改善 0.00480；中间点不单调，且 DeepSets 50/100% 在最佳 checkpoint 后 non-finite。当前没有明确数据饱和证据，但应先做 held-out selector/latency evaluation，再决定是否继续扩 labels。[完整结果](data/results/set_utility_learning_curve_v1/README.md)。
- held-out selector v1 已完成：label-blind selections 在 truth access 前密封；12×H200、48 lanes 生成 805/805
  terminal states（801 completed、4 strict-parser skips）。正式 reducer 判定 `NO_GO`；Set Transformer 虽在 primary、
  OCR/RGB 与 long-history 点估计上领先，但未显著超过 recent，且 B2 oracle regret 失败。根据冻结合同，不启动
  policy replay、online controller 或 closed-loop。[合同](docs/set_utility_heldout_v1.md)与[artifact](data/results/set_utility_heldout_v1/README.md)。
- post-hoc scale diagnostic 已完成：Set Transformer 的 exact-track B1/B2 recovery 从 10% 的 0.2392 提高到
  100% 的 0.2629（paired delta +0.0237，95% CI `[-0.0050, 0.0529]`），但曲线不单调；DeepSets endpoint
  反而下降 0.0085。结论是同分布扩数有弱方向性帮助，但不足以解释当前 gap，下一步优先 B4 oracle 上限和
  contextualized multi-latent representation。[结果](data/results/set_utility_heldout_scaling_diagnostic_v1/README.md)。
- small-history B4 oracle diagnostic v2 已完成：103/103 completed、0 skip；exact B4 recovery=`0.8248`，
  true-greedy=`0.7677`，Set Transformer=`0.7128`。B2→B4 exact gain=`+0.1249`，Set Transformer 到 B4 exact
  的 distillation gap=`0.1121`。[结果](data/results/set_utility_b4_oracle_diagnostic_v2/README.md)与
  [合同](docs/set_utility_b4_oracle_diagnostic_v2.md)。
- contextual multi-latent v3 source 已实现：train/tune-only entity requirements 使用冻结 GUI-Owl final-layer
  contextual hidden states，query/event 各保留多个 learned latent 进入 set interaction，不再提前压成单个
  256-d event vector。当前先生产 25% nested train + full tune cache，evaluation、policy replay 与 closed-loop
  继续锁定。[设计与执行顺序](docs/set_utility_contextual_multilatent_v3.md)。
- contextual v3 cache 已完成并严格 finalization：8,813 contexts、1,174 atomic chunks、37.77GB，visual
  positions 保持 `{448,459,464,480}` 原生变长；0 tensor-only、0 receipt-only。DeepSets/Set Transformer
  共享 cache 的双模型训练已启动。[结果与 artifact 状态](data/results/set_utility_contextual_cache_v3/README.md)。
- contextual v3 双模型训练已完成：Set Transformer tune objective=`0.30910`，优于 contextual DeepSets
  `0.32128`、旧 25% Set Transformer `0.32363` 与旧 100% `0.31577`。这只是 train/tune distillation 信号；
  两模型的 tune conditional-greedy selections 已冻结为 1,063 states / 9,814 个去重 truth coalitions；
  Hyper00/Hyper01 正以 12×H200 生成真实 `D(S)`，evaluation 仍未访问。
  [训练结果](data/results/set_utility_contextual_training_v3/README.md)。
- contextual tune on-policy truth 已完成：1,063/1,063 states、0 skip、9,814 rows。DeepSets 是 contextual
  winner 但没有超过 recent；表示改进的 tune loss 收益没有转化为更好 subset selection。
  [结果](data/results/set_utility_contextual_tune_on_policy_v3/README.md)。
- full train+tune contextual hidden cache 已完成并 finalization：11,721 states、1,100 trajectories、27,867
  contexts、3,572 atomic shards、119.16GB；Hyper00/Hyper01 各使用 4×H200，8/8 workers exit 0，
  evaluation access=false。cache content SHA256=`44405c2c...97604`。
- 100% contextual 双模型训练已完成：DeepSets/Set Transformer 都在 epoch 1 最佳，tune objective 分别为
  `0.30927/0.32233`。真实 1,063-state on-policy truth 中，Set Transformer 是 learned winner：macro=
  `0.44985`，接近但未超过 recent=`0.45192`；B3/B4 已胜 recent，B1/B2 与 long-history 仍弱。当前不访问
  evaluation；train-side on-policy enrichment v1 已实现并通过 tests，将从 10,658 train states 中按长历史、
  模型分歧和 conditional margin 选择 2,132 states / 52,744 coalitions；train-only labels 已 2,132/2,132
  完成，合并后新增 37,982 个 distance rows，重复 label 最大漂移 `1.19e-7`。固定 tune truth 上
  DeepSets/Set Transformer/recent macro=`0.44383/0.44141/0.45192`，long-history 也未胜 recent；verdict=
  `NO_GO_TRAIN_ON_POLICY_ENRICHMENT_V1`。因此不访问 untouched evaluation，不进入 policy replay、closed-loop
  或 matched-NLL。[结果](data/results/set_utility_contextual_tune_on_policy_enriched_v1/README.md)与
  [合同](docs/set_utility_train_on_policy_enrichment_v1.md)。
- decision distillation v2 已冻结：同一 budget-free utility checkpoint 使用 beam-4 为 B1--B4 独立搜索；
  train-only 一轮 DAgger 对 beam frontier 的所有 one-event expansions 生成 labels，并以 STOP-aware listwise
  与 expected-regret loss 训练。只有 B1--B4 每个预算、macro CI 与 long-history 同时通过 fixed-tune gate，才访问
  untouched evaluation。[合同](docs/set_utility_decision_distillation_v2.md)与
  [配置](code/configs/causalcache_set_utility_decision_distillation_v2.json)。
- v2 train beam traces 与 candidate-complete schedule 已完成：1,066 states / 365,043 coalitions，
  long/very-long/medium/short=`693/53/213/107`；Hyper00/Hyper01 已各用 4×H200 启动 8 个 resumable label
  partitions。
  [轻量结果](data/results/set_utility_decision_distillation_v2/README.md)。
- **long-history oracle ceiling diagnostic v1 已完成:`HEADROOM_CONFIRMED`。** 250 个 train
  long/very-long states(158 trajectories,`n_t` 17--43)的 candidate-complete singleton + true-greedy
  真值:oracle B1--B4=`0.528/0.661/0.737/0.785`(macro `0.678`),recent 只有 `0.210`(very_long 为
  `-0.418`,接近 random);`oracle-recent` macro=`+0.467`,95% CI `[0.359, 0.630]`,以 7 倍裕量过预注册
  阈值。additive top-B 在 B2--B4 平台化(`0.530`),显著低于条件贪心——长历史必须建模条件边际,纯
  singleton-additive 不够。best singleton 58.4% 落在 recent-4 之外。v2 decision distillation 以 Long+
  为主攻目标有充分依据;25,032 个 train 标签可复用为 candidate-complete supervision。
  [结果](data/results/set_utility_long_oracle_v1/README.md)与
  [合同](docs/set_utility_long_history_oracle_v1.md)。
- v2 交接文档见 [`docs/set_utility_long_oracle_v2_handoff.md`](docs/set_utility_long_oracle_v2_handoff.md):判定、对 v2 训练/gate 的含义、可复用标签 artifact 与合并规则、共存执行注意事项。
- 交接后的唯一训练顺序、multisource merge 与 decision-dominant loss 见
  [`docs/set_utility_decision_distillation_v2_long_oracle_training.md`](docs/set_utility_decision_distillation_v2_long_oracle_training.md)；
  对应 versioned config 不改写已冻结 v2 label contract。
- variable-history v1 合同见 [`docs/set_utility_variable_history_v1.md`](docs/set_utility_variable_history_v1.md)：完整 `C_t`、约 40 个 stratified subsets/state、320-state exact track、720-state large-history track，以及 coalition-microbatch 断点恢复。
- state inventory 已冻结为 [`data/manifests/set_utility_variable_history_v1_states.json`](data/manifests/set_utility_variable_history_v1_states.json)：12,792 个 variable-`n_t` states，候选数为 5–45；训练 collate 已支持 `event_mask` 与 `label_mask`，不再要求固定 event/label 数。
- full source 已在 Hyper00/Hyper01 完成：256 shards、1,200 trajectories、13.16GB。512-token profile 因 1 个 state 超限而 BLOCK；480-token v2 的 full VLM sequences 已在 8xH200 完成 256/256 token shards、约 67GB、零失败。真实 image-grid postflight 覆盖 12,792/12,792 states，最大 prompt+reserve 为 30,292/32,768，正式 labels 的 context blocker 已解除。

## Source of Truth

| 内容 | 位置 | 状态 |
|---|---|---|
| 代码、配置、论文、轻量结果 | 本 Git 仓库 `main` | canonical |
| Processor substrate | [HF dataset](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/c20bab8df424dc9e45ece1084f3d1dc035dd1ed8/artifacts/processor-freeze-v2-image-contract-repair) | immutable，23 files / 18.73 GB |
| GUI-Owl snapshot | `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc` | frozen |
| Dense image backfill | Hyper00 `/data02/jaxan/artifacts/causalcache-set-utility-dense-v1-backfill-d43a15c` | 42MB / 77 PNG；`PENDING_HF_UPLOAD` |
| recent-4 label smoke | Hyper00 `/data02/jaxan/runs/causalcache-set-utility-dense-v1-0d32187`；Hyper01 `/data02/jaxan/runs/causalcache-set-utility-dense-v1-9671e45-partition-01` | stopped；4,033 states；`DEPRECATED_SMOKE_ONLY_RECENT4` |
| Variable-history formal labels | Hyper00/Hyper01 `/data02/jaxan/runs/causalcache-set-utility-variable-history-labels-v1-969f2b9` | complete；11,721 completed + 25 skipped；`PENDING_HF_UPLOAD` |
| Variable-history full source | Hyper00 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-source-v1-a7213db` | 13.16GB / 256 shards；manifest `a37ce0b...042cde4`；`PENDING_HF_UPLOAD`；[census](data/results/set_utility_variable_history_source_context_v1/README.md) |
| Variable-history full VLM tokens | Hyper00、Hyper01 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-tokens-v2-47161ca` | 256/256 shards、约 67GB、零失败；source revision `47161cac...`；`PENDING_HF_UPLOAD` |
| Variable-history exact context | [`data/results/set_utility_variable_history_context_exact_v2/`](data/results/set_utility_variable_history_context_exact_v2/README.md) | PASS；12,792/12,792 fit；max prompt+reserve 30,292/32,768 |
| Variable-history broad schedules | [`data/results/set_utility_variable_history_schedules_v1/`](data/results/set_utility_variable_history_schedules_v1/README.md) | 256/256 shards；11,746 states；461,040 coalition labels；`PENDING_HF_UPLOAD` |
| Variable-history formal label run | [`data/results/set_utility_variable_history_labels_v1/`](data/results/set_utility_variable_history_labels_v1/README.md) | complete；11,746/11,746 terminal states；state/microbatch atomic resume |
| Learning-curve full snapshot/cache | Hyper00/Hyper01 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-training-full-merged-5e71d05`；`...learning-curve-cache-5e71d05` | 11,721 states；cache SHA `ef82b077...23385`；`PENDING_HF_UPLOAD` |
| Variable-history partial train/tune snapshot | [HF dataset@771db37c](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/771db37c9ce6d3e7057b87730c400cbae66a5398) | immutable；1,501 states；2,105 visual / 2,256 text sequences |
| Variable-history partial predictor checkpoints | [HF model@b6bd8232](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/b6bd823221e2ec4b2e68518ad40efe7e7bd5d673/artifacts/set-utility-variable-history-partial-ca0383a) | immutable；DeepSets + Set Transformer；diagnostic only；[summary](data/results/set_utility_variable_history_training_partial_v1/README.md) |
| 25% tuning snapshot/cache | [HF dataset@d8639bc8](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/d8639bc8b0a3237a8904aefee06d6d7f70ad7768/artifacts/set-utility-tuning25-v3-seeded-floor1e2-2d68154) | immutable；3,575 states；23.9GB；[summary](data/results/set_utility_tuning25_v1/README.md) |
| 25% tuning checkpoints | [HF model@d628ecf6](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/d628ecf6e1783591a039080999f36635c36049fb/artifacts/set-utility-tuning25-v3-seeded-floor1e2-2d68154) | immutable；12 configs / 481.8MB；tuning-only |
| Learning-curve dataset/cache | [HF dataset@02a05ab1](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/02a05ab11fd3a5036b62244bf04f37aa5e41db59/artifacts/set-utility-learning-curve-v1-9863f43) | immutable；414 files / 77.4GB；nested splits + compact label archives |
| Learning-curve checkpoints | [HF model@5409e846](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/5409e846cc45a26b2ae617e39b3ebf7462d180e6/artifacts/set-utility-learning-curve-v1-9863f43) | immutable；8 checkpoints；[summary](data/results/set_utility_learning_curve_v1/README.md) |
| Held-out selector v1 | [`summary`](data/results/set_utility_heldout_v1/README.md)；[HF tag `set-utility-heldout-v1-d89263c`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/set-utility-heldout-v1-d89263c/artifacts/set-utility-heldout-v1-d89263c)；Hyper00 persistent mirror | 805 terminals：801 completed + 4 skipped；`INCOMPLETE/NO_GO`；result content `d89263c...8ef49`；immutable revision `df41e1d...be8c6` |
| Held-out scale diagnostic v1 | [`data/results/set_utility_heldout_scaling_diagnostic_v1/`](data/results/set_utility_heldout_scaling_diagnostic_v1/README.md)；Hyper00 `/data02/jaxan/runs/causalcache-set-utility-heldout-scaling-v1-3b0be31` | 8 checkpoints；319 exact states；result content `c346bce...17c2`；`PENDING_HF_UPLOAD` |
| Small-history exact B4 v2 | [`summary`](data/results/set_utility_b4_oracle_diagnostic_v2/README.md)；[HF dataset@9b53ec82](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/9b53ec82c12fefaba571233e5e0d78d0f6c599c0/artifacts/set-utility-b4-oracle-v2-195bcbb)；tag `set-utility-b4-oracle-v2-195bcbb` | 103/103 completed；8,628 distance rows；result content `195bcbb...0473b`；669 files / 4.7MB |
| Contextual hidden cache v3 | [`summary`](data/results/set_utility_contextual_cache_v3/README.md)；Hyper00 `/data02/jaxan/runs/causalcache-contextual-hidden-v3-e413e5b` | 8,813 contexts；1,174 chunks；37,771,051,973 bytes；content `af2b090d...91eb0`；`PENDING_HF_UPLOAD` |
| Contextual predictor training v3 | [`summary`](data/results/set_utility_contextual_training_v3/README.md)；Hyper00 two persistent roots | DeepSets `0.32128`；Set Transformer `0.30910`；evaluation not loaded；`PENDING_HF_UPLOAD` |
| Contextual tune on-policy truth v3 | [`summary`](data/results/set_utility_contextual_tune_on_policy_v3/README.md)；[HF dataset@d7a6e97e](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/d7a6e97e1b654cf17b8a06690adc55d55e3c53f7/artifacts/set-utility-contextual-tune-on-policy-v3-f34d368) | 1,063/1,063 completed；DeepSets `0.4492`、Set Transformer `0.4365`、recent `0.4519`；immutable |
| Contextual full inputs v4 | [HF dataset@268bae32](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/268bae32792c3b651541354f74d9a18b7b97ecd2/artifacts/set-utility-contextual-inputs-full-v4-af18388e)；Hyper00/Hyper01 mirror | 11,721 states；27,867 contexts；tag `set-utility-contextual-inputs-full-v4-af18388e`；immutable |
| Contextual full hidden cache v4 | Hyper00 `/data02/jaxan/runs/causalcache-contextual-hidden-full-v4-e97f2d4` | 27,867 contexts；119.16GB；content `44405c2c...97604`；`PENDING_HF_UPLOAD` |
| Contextual full predictor training v4 | [`summary`](data/results/set_utility_contextual_training_full_v4/README.md)；[HF model@729a62fd](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/729a62fdaed53ccf04e641bea9ecac3ac7da3bcc/artifacts/set-utility-contextual-full-v4-44405c2) | DeepSets `0.30927`；Set Transformer `0.32233`；Set Transformer true-utility winner；immutable |
| Contextual full tune truth v4 | [`summary`](data/results/set_utility_contextual_tune_on_policy_full_v4/README.md)；[HF dataset@014aee81](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/014aee81e0f95199163773440658d6eaf36d3ddb/artifacts/set-utility-contextual-tune-on-policy-full-v4-fb6de0d) | 1,063/1,063；Set Transformer `0.44985` vs recent `0.45192`；`FULL_DATA_NO_GO`；immutable |
| Train on-policy enrichment v1 | [合同/进展](docs/set_utility_train_on_policy_enrichment_v1.md)；[HF dataset@9b436c9c](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/9b436c9c8ac645d20f2aa86ba0d519b14f5d6934/artifacts/set-utility-train-enrichment-v1-2711ab55) | 2,132 train states；52,744 scheduled rows；37,982 novel rows；tag `set-utility-train-enrichment-v1-2711ab55`；immutable |
| Contextual enriched fixed-tune result v1 | [`summary`](data/results/set_utility_contextual_tune_on_policy_enriched_v1/README.md)；[HF dataset@bbee1ae7](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/bbee1ae7aae2a712aaf2a897a08fa69adcef4ee6/artifacts/set-utility-contextual-tune-enriched-v1-3f73e17) | 1,063/1,063；DeepSets/Set Transformer/recent=`0.44383/0.44141/0.45192`；`NO_GO_TRAIN_ON_POLICY_ENRICHMENT_V1`；immutable |
| Decision distillation v2 | [`summary`](data/results/set_utility_decision_distillation_v2/README.md)；Hyper00/Hyper01 persistent run | 1,066/1,066 states、25,915/25,915 microbatches completed；`PENDING_HF_UPLOAD` |
| Long-history oracle diagnostic v1 | [`summary`](data/results/set_utility_long_oracle_v1/README.md)；[HF dataset@8d5a5021](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/8d5a5021d8e69999ed944574bc8e243f386c2288/artifacts/set-utility-long-oracle-v1-179b0d8)；Hyper00 mirror | 250/250 states；25,032 labels；`HEADROOM_CONFIRMED`(+0.467 [0.359,0.630]);tag `set-utility-long-oracle-v1-179b0d8`；immutable |
| Decision v2 + long-oracle training source | [执行单](docs/set_utility_decision_distillation_v2_long_oracle_training.md)；[versioned config](code/configs/causalcache_set_utility_decision_distillation_v2_long_oracle_training_v1.json)；Hyper00 `/data02/jaxan/artifacts/causalcache-decision-v2-long-oracle-training-inputs-v2-f7f6b14` | complete；content `3d011990...ad36ce`；5,550 decision-supervised states / Long+ 902；`PENDING_HF_UPLOAD` |
| Decision v2 distributed training | [DDP config](code/configs/causalcache_set_utility_decision_distillation_v2_long_oracle_ddp_v1.json)；Hyper00 `/data02/jaxan/runs/causalcache-set-transformer-decision-v2-long-oracle-ddp4-v2-ecd5579`；Hyper01 `/data02/jaxan/runs/causalcache-deepsets-decision-v2-long-oracle-ddp4-v4-1aafe4c` | complete；两模型 epoch 1 best、epoch 6 early-stop；evaluation labels loaded=false；checkpoints `PENDING_HF_UPLOAD` |
| Decision v2 epoch-best tune truth | [`summary`](data/results/set_utility_decision_distillation_v2_epoch1_tune/README.md)；Hyper00 full result `/data02/jaxan/runs/causalcache-running-best-tune-evaluation-v2-650afbe/result.json` | 1,063/1,063、0 skip；Set base/recent macro=`0.47593/0.45192`、CI lower=`+0.00430`；B2 与 Long+ fail；`NO_GO_DECISION_DISTILLATION_V2`；`PENDING_HF_UPLOAD` |
| Set capacity + DeepSets prefetch v1 | [`summary`](data/results/set_utility_capacity_and_prefetch_v1/README.md)；Hyper00 `/data02/jaxan/runs/causalcache-set-transformer-decision-v2-long-oracle-l64-s4-ddp2-v1-8a6b8a0` | L64/S4 2×H200 running；epoch-1 tune truth macro=`0.45575`、p95=`114.19ms`；prefetch elapsed -11.22%；checkpoint `PENDING_HF_UPLOAD` |
| Fixed-tune Long+ oracle v1 | [`summary`](data/results/set_utility_tune_long_oracle_v1/README.md)；Hyper00 `/data02/jaxan/runs/causalcache-tune-long-oracle-v1-b6a4643` | 249/249、0 skip；oracle/recent macro=`0.6949/0.3199`；`HEADROOM_CONFIRMED`；full payload `PENDING_HF_UPLOAD` |
| Direct marginal v3 Stage-A | [`v1`](data/results/set_utility_direct_marginal_v3_fit_probe_v1/README.md)；[`rank repair`](data/results/set_utility_direct_marginal_v3_fit_probe_v2_rank_repair/README.md)；Hyper00 persistent run | v2 Spearman/top-4=`0.291/0.951`；`NO_GO_DIRECT_MARGINAL_V3_STAGE_A`；fit-only checkpoints `PENDING_HF_UPLOAD`，不是正式模型 |
| Direct marginal v3 Stage-B v1 | [结果](data/results/set_utility_direct_marginal_v3_stage_b_v1/README.md)；[config](code/configs/causalcache_set_utility_direct_marginal_v3_stage_b_v1.json)；Hyper00 `/data02/jaxan/runs/causalcache-direct-marginal-v3-stage-b-v1-8ccdb5c` | training complete；best epoch 12 / regret `0.2581`；checkpoint `d8abbe8c...1dd23`；`PENDING_HF_UPLOAD` |
| Direct marginal v3 final fixed-tune | [结果](data/results/set_utility_direct_marginal_v3_fixed_tune_v1/README.md)；[HF dataset@76615721](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/766157217d99dc8c10d82349d9909ba30ceaa8e9/artifacts/set-utility-direct-marginal-v3-fixed-tune-794fb90) | 1,063/1,063、0 skip；direct/recent macro=`0.44583/0.45192`；CI crosses 0；B2/B3/Long+ fail；`NO_GO`；learned general-`B` stopped；immutable |
| Direct on-policy coverage v1 | [合同](docs/set_utility_direct_on_policy_v1.md)；[结果](data/results/set_utility_direct_on_policy_v1/README.md) | labels PASS：5,108 states / 359,189 sampled rows；merged input `6c9243a...1bfad`；9,287 optimizer states / 84,441 groups；`PENDING_HF_UPLOAD` |
| Per-epoch heldout training v1 | [结果](data/results/set_utility_direct_on_policy_training_v1/README.md)；[合同](docs/set_utility_direct_on_policy_v1.md)；Hyper00 Set root `...set-transformer-direct-on-policy-v2-a69c706`；Hyper01 DeepSets root `...structured-deepsets-direct-on-policy-v4-843e360` | complete；Set e2=`0.39491`、DeepSets e1=`0.39251`；均由 truth recovery 早停；evaluation/test sealed；`PENDING_HF_UPLOAD` |
| Budget-deferral candidate v1 | [冻结 config](code/configs/causalcache_set_utility_budget_deferral_v1.json)；[development truth result](data/results/set_utility_direct_on_policy_deployment_v1/README.md) | B1/B2 recent + B3/B4 DeepSets direct；development delta=`+0.02689 [0.00039,0.07047]`；不是 evaluation；payload `PENDING_HF_UPLOAD` |
| Budget-deferral evaluation Stage-A | [执行 config](code/configs/causalcache_set_utility_budget_deferral_evaluation_stage_a_v1.json)；[状态](data/results/set_utility_budget_deferral_evaluation_v1/README.md) | 用户在 selection/truth 前停止；truth read=0；Hyper00/01 保留 42/33 个 resumable receipts；转向 selector-side LoRA |
| Selector-side GUI-Owl LoRA v1 | [设计](docs/set_utility_selector_lora_v1.md)；[extraction config](code/configs/causalcache_set_utility_selector_lora_v1.json)；[training config](code/configs/causalcache_set_utility_selector_lora_training_v1.json)；[状态](data/results/set_utility_selector_lora_v1/README.md) | teacher/action policy frozen；top-4 parity PASS；boundary cache complete：23,714 contexts、3,065 shards、289.75GB、content `77dec757...5d16b`；e1 macro `+0.03205` vs recent 但 B2/Long+ 持平，e2 退化；保留 e1、停止 e3/e4；artifacts `PENDING_HF_UPLOAD` |
| Token predictor v2 partial snapshot/cache | [HF dataset@e1240bde](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/e1240bdeff500114097b71148ba65ae19e71e6e8/artifacts/set-utility-token-v2-partial-a73cc18) | 23.43GB / 1,878 files；tag `set-utility-token-v2-partial-a73cc18`；pilot 不含 evaluation；[summary](data/results/set_utility_token_predictor_v2_partial/README.md) |
| Token predictor v2 partial checkpoints | [HF model@a55666c1](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/a55666c11da6b2848a6f066b4b05980704f1bf7a/artifacts/set-utility-token-v2-partial-a73cc18) | 155.44MB / 11 files；同名 tag；4 checkpoints |
| Anchor-only pilot labels/features | [HF dataset@a95ce68b](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/a95ce68bd628daaec40a7575847c9db584f20dc4/artifacts/set-utility-scale-v1-ac0ef27) | deprecated pilot；仅保留复现 |
| Anchor-only pilot checkpoints | [HF model@365f3882](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/365f3882658b40eccb64c9565ae8639986c59e82) | deprecated pilot；仅保留复现 |

大文件若暂时无法上传 HF，必须保存在个人 persistent storage，并在本 README 或结果文档记录精确路径与 `PENDING_HF_UPLOAD`。

## 仓库结构

- `code/`：package、scripts、configs、tests；
- `data/`：轻量 manifests、summaries、结果索引；
- `docs/`：当前实验说明和历史进展；
- `paper/`：AAAI LaTeX、references、figures/tables；
- `ablations/`：interaction 与 subset-search 分析设计。

## 本地验证

```bash
PYTHONPATH=code .venv/bin/pytest -q \
  code/tests/test_set_utility_dense.py \
  code/tests/test_set_utility_mvp.py \
  code/tests/test_set_utility_variable_history.py \
  code/tests/test_set_utility_variable_history_contract.py \
  code/tests/test_set_utility_token_models.py \
  code/tests/test_set_utility_label_inputs.py

PYTHONPATH=code python3 -m compileall -q \
  code/causalcache \
  code/scripts/materialize_set_utility_variable_history_inventory.py \
  code/scripts/census_set_utility_dense_states.py \
  code/scripts/train_set_utility_mvp.py
```

## Go / No-Go

Held-out v1 已消费：没有模型满足全部冻结条件，后续 policy replay/closed-loop 保持关闭。801 个 completed states
上的诊断表明 Set Transformer 优于 OCR/RGB、long-history 点估计为正，但未可靠优于 recent，且 B2 oracle regret
仍差于两个 heuristic。下一轮应先改进 distillation/selection，再以新合同评估，不能把本轮追认为 GO。

MVP 的首要判断不是 closed-loop，而是 held-out utility selection：

- 若 variable-`n_t` predictor 稳定超过 OCR/RGB，并明显缩小 exact-oracle gap：再做 matched-NLL 与 closed-loop。
- 主模型除 held-out utility GO 外，还必须通过 warm p95 selector overhead `<=10%` 的部署门槛；否则只能作为 capacity ablation。最终选择依据是 utility--latency Pareto frontier，不以最大参数量为默认赢家。
- recent-4 partial token pilot 不执行上述 GO 判断；正式判断必须重做 variable-`n_t` labels 与训练。
- 若 oracle-independent `J` 有效但 learned models 失败：改进表示和训练数据。
- 若 exact oracle 有 signal、但所有可学习目标和简单 baseline 持平：重新审视 predictor objective。
- 新合同应按 state 报告接受率和失败类别，不因单个 near-tie / unstable state 让数据集归零；该原则不追溯改写
  已冻结且已消费的 held-out v1 coverage contract。

## Paper Claim 边界

“Causal”只指对冻结策略行为进行受控 restoration intervention 得到的 counterfactual attribution；不声称识别环境结构因果，也不把方法包装成 world model。
