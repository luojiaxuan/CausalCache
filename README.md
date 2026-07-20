# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

目标会议：AAAI。项目当前优先验证一件事：由 restoration utility 监督的 set predictor，能否在未参与训练的 GUI states 上超过 recent 与 OCR/RGB heuristic。

## 当前结论

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
- formal label rollout 已在 reference parse-failure repair 后使用 deterministic state lanes 扩并发。36-worker burst 在 long-history tail 上有 4 条 Hyper00 lanes OOM，现已各自单卡恢复；Hyper01 继续运行。2026-07-20 07:01 UTC 已有 10,643/11,746 terminal states（90.6%），剩余 ETA 1--1.5 小时。运行入口见 [`data/results/set_utility_variable_history_labels_v1/`](data/results/set_utility_variable_history_labels_v1/README.md)。
- labels 并行期间的 1,501-state partial snapshot 已在 Hyper00 GPU 4/5 完成 DeepSets 与 Set Transformer 双卡训练；两者 tune objective 均明显下降，Set Transformer best=0.4714、DeepSets best=0.4853。该结果只验证优化路径，不消费 evaluation，也不做正式 GO/NO-GO；见 [`data/results/set_utility_variable_history_training_partial_v1/`](data/results/set_utility_variable_history_training_partial_v1/README.md)。
- 全量 labels 收尾期间已启动 25% train-trajectory 单 seed 超参搜索：保留完整 frozen tune、使用完整 variable-history GUI-Owl token sequence，先为 DeepSets/Set Transformer 各冻结一套配置，再运行正式 learning curve；契约见 [`docs/set_utility_tuning25_v1.md`](docs/set_utility_tuning25_v1.md)。
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
| Variable-history formal labels | [HF dataset](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile) | `RUNNING`；10,643/11,746 terminal states at 2026-07-20 07:01 UTC；完成后发布 immutable revision |
| Variable-history full source | Hyper00 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-source-v1-a7213db` | 13.16GB / 256 shards；manifest `a37ce0b...042cde4`；`PENDING_HF_UPLOAD`；[census](data/results/set_utility_variable_history_source_context_v1/README.md) |
| Variable-history full VLM tokens | Hyper00、Hyper01 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-tokens-v2-47161ca` | 256/256 shards、约 67GB、零失败；source revision `47161cac...`；`PENDING_HF_UPLOAD` |
| Variable-history exact context | [`data/results/set_utility_variable_history_context_exact_v2/`](data/results/set_utility_variable_history_context_exact_v2/README.md) | PASS；12,792/12,792 fit；max prompt+reserve 30,292/32,768 |
| Variable-history broad schedules | [`data/results/set_utility_variable_history_schedules_v1/`](data/results/set_utility_variable_history_schedules_v1/README.md) | 256/256 shards；11,746 states；461,040 coalition labels；`PENDING_HF_UPLOAD` |
| Variable-history formal label run | [`data/results/set_utility_variable_history_labels_v1/`](data/results/set_utility_variable_history_labels_v1/README.md) | `RUNNING`；12xH200 / 36 workers；state/microbatch atomic resume；lane runner=`7680e40` |
| Variable-history partial train/tune snapshot | [HF dataset@771db37c](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/771db37c9ce6d3e7057b87730c400cbae66a5398) | immutable；1,501 states；2,105 visual / 2,256 text sequences |
| Variable-history partial predictor checkpoints | [HF model@b6bd8232](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/b6bd823221e2ec4b2e68518ad40efe7e7bd5d673/artifacts/set-utility-variable-history-partial-ca0383a) | immutable；DeepSets + Set Transformer；diagnostic only；[summary](data/results/set_utility_variable_history_training_partial_v1/README.md) |
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

MVP 的首要判断不是 closed-loop，而是 held-out utility selection：

- 若 variable-`n_t` predictor 稳定超过 OCR/RGB，并明显缩小 exact-oracle gap：再做 matched-NLL 与 closed-loop。
- 主模型除 held-out utility GO 外，还必须通过 warm p95 selector overhead `<=10%` 的部署门槛；否则只能作为 capacity ablation。最终选择依据是 utility--latency Pareto frontier，不以最大参数量为默认赢家。
- recent-4 partial token pilot 不执行上述 GO 判断；正式判断必须重做 variable-`n_t` labels 与训练。
- 若 oracle-independent `J` 有效但 learned models 失败：改进表示和训练数据。
- 若 exact oracle 有 signal、但所有可学习目标和简单 baseline 持平：重新审视 predictor objective。
- 不因单个 near-tie / unstable state 让整个数据集归零；报告接受率和失败类别。

## Paper Claim 边界

“Causal”只指对冻结策略行为进行受控 restoration intervention 得到的 counterfactual attribution；不声称识别环境结构因果，也不把方法包装成 world model。
