# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

目标会议：AAAI。项目当前优先验证一件事：由 restoration utility 监督的 set predictor，能否在未参与训练的 GUI states 上超过 recent 与 OCR/RGB heuristic。

## 当前结论

- **Decision distillation v2 已进入 train-only label 阶段。** Beam-4 traces 已覆盖 10,658 train states；冻结
  sampler 选中 1,066 states / 486 trajectories，candidate-complete schedule 含 365,043 个 coalitions，
  long/very-long 占 70%。完成后将与 immutable long-oracle 的 250 states / 25,032 rows 做 versioned
  union；conditional listwise 与 decision-regret 是主损失，普通 scalar regression 仅做校准。evaluation 仍未
  访问；当前不是 GO/NO-GO 结果。
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

训练 predictor `U_θ(q,C,m_S)` 直接预测 subset utility。模型始终看到全部候选和 selected mask；正式 selector 只使用 at-most-`B` search，不再把 fixed-`B` 当作结果口径。

当前比较：

- pairwise-additive；
- DeepSets；
- Set Transformer；
- recent、OCR/RGB、oracle-independent `J`；
- exact subset oracle。

论文主模型与 DeepSets 共享完整 frozen GUI-Owl token cache，只替换 set aggregator；架构与当前 partial-label
pilot 合同见 [`docs/set_utility_predictor_v2.md`](docs/set_utility_predictor_v2.md)。

## 当前执行主线

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
| Decision distillation v2 | [`summary`](data/results/set_utility_decision_distillation_v2/README.md)；Hyper00/Hyper01 persistent run | 1,066 train states；365,043 candidate-complete coalitions；8×H200 labels running；`PENDING_HF_UPLOAD` |
| Long-history oracle diagnostic v1 | [`summary`](data/results/set_utility_long_oracle_v1/README.md)；[HF dataset@8d5a5021](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/8d5a5021d8e69999ed944574bc8e243f386c2288/artifacts/set-utility-long-oracle-v1-179b0d8)；Hyper00 mirror | 250/250 states；25,032 labels；`HEADROOM_CONFIRMED`(+0.467 [0.359,0.630]);tag `set-utility-long-oracle-v1-179b0d8`；immutable |
| Decision v2 + long-oracle training source | [执行单](docs/set_utility_decision_distillation_v2_long_oracle_training.md)；[versioned config](code/configs/causalcache_set_utility_decision_distillation_v2_long_oracle_training_v1.json) | source ready；预期 1,222 decision-supervised states / Long+ 902；等待 v2 labels 完成后物化与发布 |
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
