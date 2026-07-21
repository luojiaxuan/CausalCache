# 项目进展

## 2026-07-20：long-oracle 监督并入 decision distillation v2 后续训练

- 已 fast-forward 到 long-oracle 交接 commit `ae08561` 并冻结后续执行单；现有 1,066-state / 365,043-
  coalition v2 labels 继续运行，不改 schedule、runner 或 scientific identity；
- long-oracle 250 states 与 v2 选中 states 重叠 94，预期合并后有 1,222 个 decision-supervised states、其中
  Long+ 902；新增 multisource merger 逐 source 校验 schedule/config/runtime/input binding，重复 `D(S)` 只在
  绝对差 `≤1e-6` 时去重，并保证 tune payload byte-identical；
- versioned training config 将 conditional marginal/listwise/decision-regret 权重设为 `1/2/2`，普通 raw/
  normalized/ranking 降为 `0.25/0.25/0.25`；listwise/regret 按 active decision rows 归一化；
- trainer 新增 fail-fast census：少于 1,100 个 complete-decision states 或少于 800 个 Long+ states 时不加载
  模型、不启动训练；训练后另报 age-bin singleton ranking 与 outside-recent4 recall，防止学习 recency shortcut；
- 2026-07-21 01:27 UTC live labels 为 575/1,066 states、13,930/25,915 microbatches，575/575 terminal
  statuses completed；过去一小时 223 states，两台 Hyper 容器均运行，短窗 ETA 约 2.2 小时；
- 后续只按原 fixed-tune B1--B4 + macro CI + Long+ 三重 gate 判定。此前所有 NO-GO 保持有效，evaluation、
  policy replay、closed-loop 和 matched-NLL 继续锁定。执行单见
  [`set_utility_decision_distillation_v2_long_oracle_training.md`](set_utility_decision_distillation_v2_long_oracle_training.md)。

## 2026-07-20：decision distillation v2 冻结

- v2 保留 variable at-most-`B` 与 budget-free utility predictor，但把 nested greedy 改为 width-4 beam；B1--B4
  从各自可行集合独立选取，不再要求形成同一 prefix；
- train-only DAgger 固定一轮、10% long-heavy states、每 trajectory 最多 3 states；每个 beam frontier base
  覆盖全部 one-event expansions，不再只标旧 student top-3；
- loss 新增 STOP-aware conditional listwise 与 expected regret；最终 fixed-tune gate 要求 B1--B4 每个预算
  点估计、macro paired CI 和 long-history 同时超过 Recent；失败不访问 evaluation；
- 实现与冻结参数见 [`docs/set_utility_decision_distillation_v2.md`](set_utility_decision_distillation_v2.md)。

## 2026-07-20：decision distillation v2 candidate-complete schedule

- DeepSets 与 Set Transformer 分别绑定自己的 v1 training config/checkpoint，共享 enriched input、119GB
  contextual cache 与 v2 beam-4 search config；两份 train trace 均完成 10,658/10,658 states；
- 训练 lineage validator 已修正为只要求 input/cache/search contract 一致，并在 schedule manifest 分别记录
  training config SHA；tests=`9 passed, 1 skipped`；
- 冻结 sampler 选中 1,066 states / 486 trajectories，long/very-long/medium/short=
  `693/53/213/107`，每 trajectory 最多 3 states；
- candidate-complete schedule 含 365,043 个去重 coalitions，content SHA=
  `d5e508c2...10187a`；fleet preflight 后使用 Hyper00 GPU 0--3 与 Hyper01 GPU 2--5 启动 8 个 partitions，
  每卡一个 worker；state/microbatch 原子断点开启，output=
  `/data02/jaxan/runs/causalcache-decision-v2-labels-v1-87bea18`，evaluation 仍锁定。
- 单 lane 的 10 秒启动采样中，forward phase 接近 100%，但 generation/preprocess 间隙使多数卡平均约
  70%--80%；以 versioned workers-v2 切为每卡 2 个 deterministic state lanes，保持 8 GPUs、相同 output
  和 scientific identity，已完成的原子 records 全部复用。

## 2026-07-20：train-side on-policy enrichment v1 fixed-tune NO-GO

- 2,132/2,132 train states、52,744 targeted coalitions 完成，合并后新增 37,982 个 rows；重复 label 最大绝对
  漂移 `1.19e-7 < 1e-6`，1,063 个 tune states 未进入 optimizer；
- DeepSets/Set Transformer 都在 epoch 1 最佳；Set Transformer 首次 backward 的 cuDNN MHA runtime failure
  通过 committed backend repair 解决，模型/seed/labels 不变；
- fixed tune truth 完成 1,063/1,063 states、11,692 coalitions、0 skip；DeepSets / Set Transformer / recent
  macro=`0.44383/0.44141/0.45192`，long+=`0.32980/0.34226/0.40081`；
- learned-minus-recent 95% CI 分别为 `[-0.02673,0.00954]` 与 `[-0.03491,0.01237]`；verdict=
  `NO_GO_TRAIN_ON_POLICY_ENRICHMENT_V1`；不访问 untouched evaluation，不启动 policy replay、closed-loop 或
  matched-NLL；
- 轻量结果见
  [`data/results/set_utility_contextual_tune_on_policy_enriched_v1/`](../data/results/set_utility_contextual_tune_on_policy_enriched_v1/README.md)；
  完整证据已发布到 private HF dataset revision
  [`bbee1ae7`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/bbee1ae7aae2a712aaf2a897a08fa69adcef4ee6/artifacts/set-utility-contextual-tune-enriched-v1-3f73e17)，
  tag=`set-utility-contextual-tune-enriched-v1-3f73e17`。

## 2026-07-20：train-side on-policy enrichment labels 启动

- 10,658-state train selector traces 已完成：DeepSets / Set Transformer content SHA=
  `a5fa6252...dc6f / 62acad22...a83d`；每步保存 top-3 conditional candidates；
- final schedule=`2,132 states / 52,744 coalitions`，long/very-long/medium/short=
  `1307/76/534/215`，680 trajectories、每轨迹最多 5；schedule content SHA=`adc4eab7...2bd1`；
- 首次 launch 的 source-revision 参数手工展开错误，在有效 run 前停止并删除自己的不完整 output；随后用
  `git rev-parse` 的完整 SHA=`5714b529...a817` 全新启动，不复用错误 lineage；
- 正式 run 使用 Hyper00 GPU 0--3、Hyper01 GPU 2--5，每卡 2 lanes、每 host 仍最多 4 GPUs；output=
  `/data02/jaxan/runs/causalcache-contextual-train-on-policy-labels-enrichment-v1-5714b52`。

## 2026-07-20：train-side on-policy enrichment v1 实现

- selector CLI 新增 train role 与每个 conditional-greedy step top-k candidate trace；tune 默认行为与 status
  保持兼容；
- targeted sampler 固定 20% states、long-history-heavy bin quotas、trajectory cap，并按 cross-model path
  disagreement、recent disagreement、top1/top2 margin 与 old-event span 排序；
- schedule dry run 发现若强行填满 very-long quota 会把少数 trajectory 放宽到 13 states；GPU labels 尚未启动，
  已改为严格 cap。cap=3 又会把 long quota 从 1,281 压到 910；最终冻结 cap=5，得到 long/very-long/
  medium/short=`1307/76/534/215`，覆盖 680 trajectories，最大每轨迹 5 states；
- schedule 同时覆盖两模型 B1--B4 paths、每步 top-3 candidates、recent 与 anchors；只允许 train role，
  evaluation access=false；
- 相关 5 tests passed；下一步在 full checkpoints 上生成 10,658-state train selector traces，再物化约 2,132
  states 的 targeted restoration schedules。

## 2026-07-20：full contextual deployment-search truth 完成

- 先以 8 GPUs / 8 workers 启动；5 秒利用率约 39%--78% 后，在不增加 GPU 的前提下切换为每卡 2 lanes；
  atomic state terminals 全部复用，随后利用率大多为 91%--100%；
- Hyper00 501 + Hyper01 562 = 1,063/1,063 states，16/16 lane workers exit 0，0 skip；10,368 个 schedule
  coalitions 全部获得真实 `D(S)`；
- Set Transformer B1/B2/B3/B4 recovery=`0.1822/0.4120/0.5569/0.6484`，macro=`0.44985`；recent=
  `0.45192`；差值 `-0.00207`，95% CI=`[-0.01339,0.00964]`；
- Set Transformer 是 learned winner 且 B3/B4 已超过 recent，但 B1/B2 与 long-history=`0.3542` 仍弱；
  DeepSets macro=`0.42470`，显著低于 recent；
- verdict=`FULL_DATA_NO_GO`：不访问 untouched evaluation、policy replay 或 closed-loop。下一步只从 train split
  生成 on-policy/conditional-marginal enrichment，并复用当前固定 tune truth 判定。
- raw terminals、workers、selections、schedules 与 full result 已发布到 private HF dataset revision
  `014aee81e0f95199163773440658d6eaf36d3ddb`，tag=
  `set-utility-contextual-tune-on-policy-full-v4-fb6de0d`，1,597 files。

## 2026-07-20：full contextual 双模型完成，deployment-search truth 启动

- fleet preflight 返回 Hyper00 0--7 全空闲；仅选择 GPU 0/1，严格低于每 host 4 GPUs 上限；
- DeepSets=`deepsets_contextual_d256_l16_r2_lr3e4`，Set Transformer=
  `set_transformer_contextual_d256_l16_r2_s2_lr1e4`，共享 full cache、seed 与 20-epoch early-stopping contract；
- 两模型均在 epoch 1 最佳并于 epoch 6 early-stop：DeepSets tune=`0.3092726`、checkpoint SHA=
  `bae1ead5...9e32`；Set Transformer tune=`0.3223304`、checkpoint SHA=`aafe3627...c044`；evaluation
  records loaded=false；
- 两个 frozen checkpoints 已生成完整 1,063-state at-most-B conditional-greedy paths；selection content SHA
  分别为 `fd526c20...8f86` 与 `372ebc76...6ab9`；
- 新 schedule 含 10,368 个去重 coalitions，content SHA=`6efe9238...2683`，同时覆盖 learned paths、recent、
  empty/full anchors；
- fleet preflight 后在 Hyper00 GPU 0--3 与 Hyper01 GPU 2--5 启动 8 个 resumable truth partitions，每台
  严格最多 4 GPUs。output root=
  `/data02/jaxan/runs/causalcache-contextual-tune-on-policy-labels-full-v4-e97f2d4`；仍未访问 evaluation。
- 两个 checkpoints、summaries 与 frozen config 已发布到 private HF model revision
  `729a62fdaed53ccf04e641bea9ecac3ac7da3bcc`，tag=`set-utility-contextual-full-v4-44405c2`。

## 2026-07-20：full contextual extraction 完成

- frozen input 为 11,721 train+tune states / 27,867 contexts，evaluation access=false；
- fleet preflight 后使用 Hyper00 0--3 与 Hyper01 2--5，共 8×H200、每 host 4 workers；
- 8 个 logical partitions 与每 8 contexts 的 atomic chunks 支持断点续跑；source revision=`e97f2d4...49fe`，
  output root=`/data02/jaxan/runs/causalcache-contextual-hidden-full-v4-e97f2d4`。
- 两台各 4 workers 均 exit 0；Hyper00 1,772 + Hyper01 1,800 = 3,572/3,572 atomic shards，随后通过
  `10.0.32.247 -> 10.0.32.246` 内网流式合并，避免经 Mac 中转；
- finalizer 验证 3,572 tensors、3,572 receipts、8 worker receipts、27,867 contexts，byte count=
  `119,159,482,488`，status=`COMPLETED_SET_UTILITY_CONTEXTUAL_HIDDEN_CACHE`；
- cache content SHA256=`44405c2cf96f468e6d0f087e8249bdb504c7efd150673e9bda836d2f52597604`，input content
  SHA256=`af18388e86406a7d3921e6f3d8e02c9b18cdeaacca3250f29f2f4fed102139c1`；下一步启动 full-data
  DeepSets 与 Set Transformer，并重新生成 deployment-search tune truth。

## 2026-07-20：contextual tune on-policy truth 完成

- 1,063/1,063 tune states、9,814 distance rows、0 skip；两台 Hyper 初始 12 workers，用户收缩后保持每台
  最多 4 active GPUs，主动停止的 partition 5 从原子断点补完；
- DeepSets / Set Transformer / recent 的 trajectory-equal B1--B4 macro recovery 为
  `0.4492 / 0.4365 / 0.4519`；DeepSets 是 contextual winner，但没有超过 recent；
- DeepSets-minus-recent 95% CI=`[-0.0160,0.0100]`；Set Transformer-minus-recent=
  `[-0.0314,-0.0004]`。representation-only 为 `NO_GO`，不启动 untouched evaluation 或 policy replay；
- 仍按用户冻结的执行顺序，用已发布 full inputs 训练 100% DeepSets 与 Set Transformer，再以真实 selector
  truth 判定；若仍失败，下一步是 train-side on-policy enrichment。
- raw labels、selections、schedules 与 full result 已发布到 private HF dataset revision
  `d7a6e97e...c53f7`，tag=`set-utility-contextual-tune-on-policy-v3-f34d368`；remote 1,601 entries 已回读。

## 2026-07-20：contextual tune on-policy truth 启动

- DeepSets 与 Set Transformer 已在完整 1,063 个 tune states 上生成 at-most-B conditional-greedy paths；
- 去重后的 truth schedule 含 9,814 coalitions，包括 learned paths、recent B1--B4 与 empty/full anchors；
- Hyper00 GPU 0--5、Hyper01 GPU 2--7 共 12×H200 按 12 个独立 partitions 启动，state/microbatch 原子结果
  支持中断续跑；evaluation split 未加载；
- reducer 已实现并通过 10 tests（另 7 个无 PyTorch 环境时 skip）：只用真实 `D(S)` 比较 trajectory-equal
  B1--B4 recovery、paired recent delta、long-history 与 selector latency，不以 tune loss 选 winner。
- 等待 GPU truth 时已完成 full train+tune requirements：11,721 states / 27,867 contexts，Hyper00/Hyper01
  manifest SHA256 均为 `0e87364b...f3b4b`；GPU extraction 仍等待 tune truth 判定。
- 用户将两台 Hyper 的并发上限收缩为各 4 GPUs；当前 truth run 已中断额外 worker 并保留原子进度，后续 full
  extraction 固定为 8 个 logical partitions、每台 4 GPUs。
- full contextual input snapshot 已发布到 private HF dataset revision `268bae32...ecd2`，immutable tag=
  `set-utility-contextual-inputs-full-v4-af18388e`；remote revision 已回读验证。

## 2026-07-20：small-history self-contained exact B4 完成

- v1 在 `0 states / 0 completed batches` 时停止；v2 不再跨 reference session 合并旧 B1/B2，而是在同一
  session 内重算全部 `|S|≤4` truth；
- Hyper00/H200 `0--5` 与 Hyper01/H200 `2--7` 共 12 卡、24 workers 完成 103/103 states、73
  trajectories、8,525 forwards、8,628 distance rows；0 skip、0 error，两个 supervisors 均 exit 0；
- trajectory-equal exact B1/B2/B3/B4 recovery=`0.5304/0.6999/0.7823/0.8248`，B2→B4
  gain=`+0.1249`；说明 B1/B2 确有 budget ceiling，但 B4 已能恢复约 82.5% frozen-policy behavior；
- true-conditional-greedy B4=`0.7677`，exact-minus-greedy=`0.0571`；搜索 gap 存在但小于 learned
  distillation gap；
- Set Transformer B4=`0.7128`、DeepSets=`0.7064`、OCR/RGB=`0.6942`、recent=`0.6718`；Set
  Transformer 点估计最好，但距 exact 仍差 `0.1121`，不改变 formal v1 `NO_GO`；
- 85/103 states 的 at-most-B4 exact 解使用 4 events，另有 18 states 使用更少 events，验证不能把 selector
  定义成 exactly-B；
- 下一步优先 contextualized GUI-Owl hidden states、保留多 latent tokens 进入 set interaction，并补
  deployment-search/on-policy coalition supervision；不先启动 policy replay 或 closed-loop。轻量结果见
  [`data/results/set_utility_b4_oracle_diagnostic_v2/`](../data/results/set_utility_b4_oracle_diagnostic_v2/README.md)。
- 669-file / 4.7MB 完整 artifact 已发布到 private HF dataset revision
  `9b53ec82c12fefaba571233e5e0d78d0f6c599c0`，tag=`set-utility-b4-oracle-v2-195bcbb`；remote
  force-download result SHA 与本地一致。

## 2026-07-20：held-out selector data-scaling diagnostic 完成

- 8 个 10/25/50/100% checkpoints 在 Hyper00/Hyper01 的 disjoint held-out partitions 上完成 16 个 inference
  jobs；12×H200、两台 containers 均 exit 0，未新增 restoration labels；
- exact track 为 319/320 completed states、86 trajectories；Set Transformer B1/B2 macro recovery 随 scale
  为 0.2392/0.2460/0.2456/0.2629，10%→100% delta=+0.0237，95% CI
  `[-0.0050, 0.0529]`；
- DeepSets 为 0.2490/0.2436/0.2579/0.2405，endpoint delta=-0.0085；两个 families 均非单调，DeepSets
  tune-best 100% 也不是 held-out best；
- 数据规模不是完全无效，但继续堆同分布 trajectory 不足以自动闭合 distillation gap；下一步优先小历史 B4
  strong/exact oracle，再做 contextualized multi-latent v3；
- full result content SHA256=`c346bceb...17c2`，persistent path 与轻量结果见
  [`data/results/set_utility_heldout_scaling_diagnostic_v1/`](../data/results/set_utility_heldout_scaling_diagnostic_v1/README.md)。

## 2026-07-20：held-out selector v1 正式 NO-GO

- Truth rollout 在 Hyper00/Hyper01 使用 12×H200 完成，48/48 deterministic lanes 与 12/12 containers
  均正常结束；冻结 inventory 的 805/805 states 有 terminal records，其中 801 completed、4 个 strict GUI-Owl
  tool-call parse skips，missing 0；
- formal reducer 返回 `INCOMPLETE_SET_UTILITY_HELDOUT_EVALUATION / NO_GO`，没有 deployment winner；根据冻结
  顺序，native policy replay、online controller、AndroidWorld closed-loop 与 matched-NLL 均未启动；
- completed-state 诊断中，Set Transformer primary recovery=0.4093，recent=0.4040，OCR/RGB=0.3826；Set
  Transformer 显著胜 OCR/RGB（95% CI `[0.0034, 0.0525]`），但未显著胜 recent（`[-0.0248, 0.0318]`），
  且 exact B2 regret=0.0281，差于 recent=0.0264 与 OCR/RGB=0.0230；
- DeepSets primary=0.3997，long+very-long=0.2983，整体弱于 Set Transformer；Set Transformer 的
  long+very-long=0.3912，高于 recent=0.3693，是值得保留但不足以 GO 的方向性信号；
- full result 位于 Hyper00
  `/data02/jaxan/runs/causalcache-set-utility-heldout-evaluation-v1-7113e09/result.json`，content SHA256=
  `d89263ce...8ef49`、file SHA256=`4fb4a55f...94e01`；轻量摘要见
  [`data/results/set_utility_heldout_v1/`](../data/results/set_utility_heldout_v1/README.md)，完整 artifact 已发布到
  private HF tag `set-utility-heldout-v1-d89263c`（revision `df41e1d...be8c6`）。

## 2026-07-20：10/25/50/100% learning curve 完成

- 8/8 单 seed jobs exit 0，无 OOM；evaluation records 均未加载；
- DeepSets tune total 为 0.32270/0.31731/0.32065/0.31171，Set Transformer 为
  0.32057/0.32363/0.32039/0.31577；100% 相对 10% 分别改善 0.01098/0.00480，但中间点不单调；
- DeepSets 50%/100% 在保存最佳 checkpoint 后出现 non-finite；Set Transformer 四个 scale 均稳定 early stop；
- 暂不盲目增加 labels。held-out v1 已在 evaluation label access 前冻结：先密封两个 100% checkpoints 在
  trajectory-disjoint variable-`n_t` states 上的 at-most-B selections，再生成 sparse restoration truth；若真实
  utility GO，再做 policy replay 与小规模 closed-loop。合同见 [`set_utility_heldout_v1.md`](set_utility_heldout_v1.md)。
- nested splits、full cache 与 compact raw-label archives 已发布到 private HF dataset revision
  `02a05ab1...1db59`（414 files / 77.4GB）；8 checkpoints 已发布到 model revision
  `5409e846...180e6`。

## 2026-07-20：formal labels 与 learning-curve inputs 完成

- labels 达到 11,746/11,746 terminal states：11,721 completed、25 state-level parse skips；最后 repair worker
  exit 0；
- merged train/tune snapshot 为 10,658/1,063 states、1,000/100 trajectories，content SHA256=
  `8a530cc6...dac6fa`；
- 同一 trajectory ranking 生成 10/25/50/100% nested train splits，tune 固定不变；
- Hyper00/Hyper01 独立 finalization 的 full-token cache identity 均为 `ef82b077...23385`，16,146 visual +
  17,152 text sequences；
- preflight 得到两台机器 GPU 4--7 空闲，8 个 fraction×family jobs 的固定映射已写入 execution config。

## 2026-07-20：25% trajectory 单 seed 超参冻结

- 从运行中 labels 冻结 249 条 train trajectories / 2,535 states 与完整 99 条 tune trajectories / 1,040
  states；候选为完整 variable history，不加载 evaluation；
- 修复 normalized target 的 near-zero denominator 与 model initialization 前 seed，正式 v3 完成 12/12 配置；
- DeepSets 最低 tune total 为 0.31137（d512/l16/r2/lr3e-4，最佳 checkpoint 后出现 non-finite），Set
  Transformer 为 0.31169（d256/l8/r1/s2/lr1e-4，稳定 early stop）；
- 两个 family 赢家已冻结到 `causalcache_set_utility_learning_curve_v1.json`，待 labels 完成后运行 nested
  10/25/50/100% train trajectories；本轮不作 held-out 或 deployment 结论。
- snapshot/cache 已发布到 private HF dataset revision `d8639bc8...ad7768`，12 个 checkpoints 已发布到
  private HF model revision `d628ecf6...049fb`。

## 2026-07-20：label long-history tail recovery

- 36-worker burst 的 3 processes/H200 在 short/medium states 上可运行，但 4 条 Hyper00 lanes 在更长 context
  上 OOM；失败 lanes 为 `8:lane0`、`9:lane0`、`9:lane1`、`16`，峰值已接近 H200 全部显存；
- 其他 Hyper00 workers 已完成，Hyper01 6 containers 保持运行。四条失败 lanes 依靠既有 state/microbatch
  原子断点，各自绑定 Hyper00 GPU 1--4 单进程恢复，不重算已完成 labels；
- 06:59 UTC 为 10,619/11,746 states；07:01 UTC 为 10,643 states，所有 tail recovery containers 正常。
  剩余 1,103 states，当前 ETA 1--1.5 小时；
- 执行结论：36 workers 只能作为前段 burst，正式长历史执行的安全并发上限低于 3 processes/H200。

## 2026-07-19（UTC 07-20）：label workers 24 → 36

- runner revision `7680e40` 增加 deterministic state lanes：`SHA256(state_id)` 的前 8 bytes 取模；lane
  只改变未完成 state 的执行归属，不进入 state identity，不改变 schedule、label 或已有原子断点；
- committed manifest `code/configs/causalcache_set_utility_variable_history_labels_workers_36_v1.json`
  固定 12xH200 / 36 workers；每 host 18 workers、每 GPU 3 processes，并保持原 host-partition affinity；
- 在 5,924/11,746 states 处完成切换。选择剩余量最大的 13 个 base partitions 拆为两 lanes，已完成的
  partition 6 不再加载；12/12 containers 均正常，单卡显存约 87--103GB；
- 03:54 UTC 为 5,983 states；首个稳定分钟约 27 states/min，与原 24-worker 约 26.8 states/min 接近，
  暂无显著扩并发收益。当前瓶颈更接近 GPU forward saturation；保留 36 workers 继续跑，ETA 3.5--5 小时。

## 2026-07-19（UTC 07-20）：label parse repair 与 12-GPU resume

- 旧 24-worker execution 在 4,805 terminal states 后有 6 个 partitions 因 reference tool-call 无法严格解析而
  退出；这是 runner 未实现 state-level parse admissibility，不是 restoration distance 或数据定义失败；
- revision `415f608` 仅捕获 `GUIOwlV21GenerationParseError`，原子写
  `SKIPPED_VARIABLE_HISTORY_LABEL_STATE` 后继续；测试为 `8 passed`，其他异常继续 fail-fast；
- 用户明确授权 Hyper00/Hyper01 各最多 6 张 GPU。repair run 使用 Hyper00 0/1/2/3/5/6 与 Hyper01 2--7，
  共 12xH200、24 workers、每卡 2 processes；原 host-partition affinity 保持不变，已有 state/microbatch 全复用；
- 第一次 remap 跨 host 移动了 partition，暴露两台机器只保存原 assignment 对应 shards；同时 Hyper00 GPU 4
  被新进程占满。该 attempt fail-fast、未覆盖 terminal records，随后改用 GPU 6 并恢复原 host affinity；
- 2026-07-20 03:28 UTC 为 5,317/11,746 states，12/12 containers healthy；最近 6 分钟约
  26.8 states/min，剩余 ETA 4--5.5 小时。

## 2026-07-19（UTC 07-20）：Hyper 双卡 partial predictor 训练完成

- fleet preflight 清理后 Hyper00 GPU 4/5 空闲；正式 labels 继续占用 Hyper00 GPU 0--3 与 Hyper01 GPU 2--5，
  两条工作流互不抢卡；
- 从 private HF immutable revision `771db37c9ce6d3e7057b87730c400cbae66a5398` 拉取 1,501-state
  train/tune-only snapshot，cache 复核为 2,105 visual / 2,256 text sequences，SHA256=
  `22bbc791...9b933b`；
- DeepSets d256/l8 与 Set Transformer d256/l8 分别绑定 Hyper00 GPU 4/5，均 exit 0；best tune total 分别为
  0.4853 与 0.4714，证明 full-token variable-set 训练链可优化，但不构成 held-out selector 结论；
- 两个 checkpoint、完整训练 summary、config 与 model card 已发布到 private HF model revision
  `b6bd823221e2ec4b2e68518ad40efe7e7bd5d673`；
- Taurus 首次启动因三张选定 A6000 在 launch 时已被其他进程占满而 OOM，0 epoch；失败容器已删除、日志保留；
- formal labels 于 2026-07-20 01:43 UTC 为 2,332/11,746 terminal states；24-worker 净速率约
  28.3 states/min，当前剩余 ETA 为 5.5--7 小时。

## 2026-07-19：variable-history formal labels 启动

- 完整历史 source、67GB full VLM token cache、12,792-state exact-grid context postflight 与 256-shard
  train/tune schedule 均已完成；postflight 为 12,792/12,792 fit，schedule 为 11,746 states / 461,040 labels；
- 首次 label launch 暴露 v2.1 official-tools encoder 的 historical batch=2 常量未对 subclass 多态；在 16 个
  empty-coalition microbatches 后 fail-fast、0 terminal，修复已由测试覆盖并推送；
- formal run=`969f2b9`，Hyper00/Hyper01 共 8xH200、24 workers、每卡 3 processes；初始 86 states / 3,232
  labels 全部完成，`n_t=5..9`，KL finite/non-negative；
- state/reference/coalition-microbatch 均原子落盘。16-way 到 24-way worker 重分配已复用已有 62 states 与
  335 microbatches，没有从头计算。

## 2026-07-19：recent-4 rollout 停止并降级

- 正式 selector 的候选集必须是 query 前全部 eligible events：`C_t={e_1,...,e_{t-1}}`，`n_t=|C_t|`
  随 state 变化；旧 dense-v1 把每个 state 截断为 recent-4，无法覆盖 long-range dependency；
- Hyper00/Hyper01 的 8×H200 rollout 已主动停止。封存时共处理 4,033 states，其中 3,964 completed、69
  skipped；两个容器均已停止，已有 state-level atomic records 保留；
- 旧 labels、745-state token pilot 与其 checkpoints 全部标记为 `DEPRECATED_SMOKE_ONLY_RECENT4`，只能证明
  generator、token cache 和 optimization path 可运行，不得进入正式 predictor、evaluation 或 GO/NO-GO；
- 下一版不再把 `n` 写进配置。完整历史 candidates 通过 padding/event mask 进入 DeepSets/Set Transformer；
  coalition 只做 variable-`n_t` 分层采样，小历史和少量 held-out states 才做 exact enumeration。

## 当前目标

### 2026-07-20：full source 完成，512-token census BLOCK

- full-history source 已从 pinned raw pool 与 processor terminal metadata 合并为 256 个可断点 Parquet shards，
  共 1,200 trajectories / 13.16GB；
- 512-token census 覆盖全部 12,792 states，12,791 fit；唯一超限 state 的 prompt+reserve 为 33,236；
- 不删除任何 old candidate。512 v1 保留为 `BLOCK_FULL_HISTORY_CONTEXT_CENSUS`，新建只修改视觉 token profile
  的 480-token v2，下一步重跑 census；
- source 尚未上传 HF，persistent path 与 hash 见 README，状态 `PENDING_HF_UPLOAD`。
- 480-token v2 的 tokenizer/chat-template census 已 12,792/12,792 PASS，最大 prompt+reserve=31,764；由于
  image processor 的实际 grid 可能有 rounding，visual-token extraction 后仍需 exact-grid context postflight。
- 首次 token extraction 在 7/256 shards 主动停止并删除：其 receipt 未绑定代码 revision、也未保存 exact-grid
  counts。修复后新 runner 将 Git SHA 与 counts 写入每个原子 shard receipt，旧 partial 不复用。

当前路线暂停 closed-loop，先用 1,200 条 trajectory 的 12,792 个 eligible states 生成 variable-history
restoration data，再训练 budget-agnostic `U_theta(q,C,m_S)`。每个 state 的候选是完整历史 `C_t`；只采样
coalition，禁止 recent-`n` 截断候选。train/tune broad track 约 40 labels/state；evaluation 分 320-state exact
`|S|<=2` track 与 720-state selected-subset track。reference8、old-dev5、fresh-16 和 confirm-20 不进入新训练、
调参或评估，matched-NLL 与 sealed AndroidWorld test 继续 locked。

state inventory 与动态 sampler 已实现并绑定 config/assignment hash；predictor collate 已支持 variable event/label
padding。下一步是全量 observation rematerialization、32,768-token context census、可断点 multi-state label runner，
然后才生成正式 labels 和训练 predictor。历史 D1/D2 协议只作为诊断记录，不再阻塞新的 state-level admissibility
生产合同。

### 2026-07-19：strict-determinism D2 valid PASS

- Source-A=`60fd971b1614afba2601c838a34f7952d2334eb0`，direct-child Execution-B=
  `38e7f53b83c3b943e6c5a60c15c01ac382e00576`；execution envelope 9,563 bytes / SHA256=
  `1f194b90b2bdd780478ce112d25407ca78d7a77a749713bc048aed8ffa692f53`；
- 5 秒 fleet preflight 后绑定 Hyper00 host GPU 0/1/2，在同一 image/runtime 的三个 fresh processes 并行运行；
  3/3 terminals completed、3/3 encode、6/6 generation、retry=0，无 runtime failure；
- target `029675...:decision:006` 与两个 stable controls 的 exact sequence、decoded output、canonical action 和
  frozen-input checks 全部 repeat-equal；diagnoses 分别为 `TARGET_STABILIZED_UNDER_STRICT_DETERMINISM` 与两个
  `STABLE_CONTROL_REPRODUCED_UNDER_STRICT_DETERMINISM`；
- exact aggregate 6,643 bytes / SHA256=
  `cc8ae9e18d72002d642c9111fa7bcb78574d5f3dabc68a54d67586e62c3078e5`，独立只读 rebuild 逐 byte 相同；正式
  verdict=`PASS_STRICT_DETERMINISM_REPEAT_STABILITY_DIAGNOSTIC`；
- 标准 runner 首次因 Docker GPU argument parsing 在容器创建前失败；随后保持相同 allocation/image/mount/profile，
  只修正 quoted device-list 启动语法。该 failure 发生在 B materialization/model/state process 之前，科学调用为 0；
- D2 PASS 只授权另立新的 full-roster strict-profile source。D1b NO-GO 不变，labels/training/matched-NLL/
  closed-loop 继续 locked。结果见
  [`../data/results/set_utility_action_stability_diagnostic_v3/`](../data/results/set_utility_action_stability_diagnostic_v3/)。

### 2026-07-19：strict-determinism D2 Source-A freeze

- D2 只诊断 `0296753837938323:decision:006`，并固定同 decision stratum control
  `0336706763935531:decision:006` 与 longer-context control `0268406573756492:decision:010`；三进程可同波并发，
  ceiling=3 encode / 6 generation，无 retry/top-up；
- `strict_determinism_sdpa_frozen_encoded` 保持 parent automatic loader 与实际 top/text/vision 全 SDPA；在 CUDA
  初始化前精确设置 cuBLAS workspace、PyTorch deterministic algorithms、seed、cuDNN、TF32 与 matmul precision；
  unsupported deterministic operation 只能形成 class-only `INVALID_RUNTIME_FAILURE`，不能降级 backend；
- canonical config SHA256=`4531b1c7e8c067b28c31662ee04b210dd2009031a9f92b4689128f701074b18d`；
  53-file transitive source inventory SHA256=
  `86dbd5a6ed340dd46ce93709e6a7b59c07d200a04652723407e28683b917066b`；focused verification 合计
  `40 passed`，相关 execution/aggregate/CLI v2 回归另 `9 passed`，source validator 与 `py_compile` 通过；
- Source-A 是包含本记录的 clean pushed `main`。下一步只能 fresh fleet preflight、机械生成唯一 direct-child
  Execution-B，再运行一次 3-state D2；D2 PASS 也不追认 D1b、不直接解锁 labels/training/matched-NLL/closed-loop。
  协议见 [`set_utility_action_stability_diagnostic_v3.md`](set_utility_action_stability_diagnostic_v3.md)。

### 2026-07-19：D1b formal execution valid，repeat-stability NO-GO

- Source-A=`fe6f6b69adf2acd5457026835529f2ca3e682dd2`，唯一 direct-child Execution-B=
  `db2c848af310d7559247895f418f69255465eb73`；execution envelope 19,349 bytes / SHA256=
  `7adb460bbe65d08cc7fe2bea992cf77f544d8be18f5c43f28c84d4ee4e9171f8`；
- fresh 10 秒 all-container preflight 后，在历史 D1 同一 Hyper00 container 与 host GPU `0,1,3,4` 上运行；
  wave 0 四进程、wave 1 两进程，barrier、6 个 unique process identity、attempt/terminal/artifact/source hashes
  全部通过；
- 6/6 processes completed，实际 12/12 generation、6/6 encode、retry=`0`；三个 decision-10 state 均不再
  OOM，两个 stable controls 均稳定；
- `0296753837938323:decision:006` 的 frozen encoded input 在 before/between/after 均 exact unchanged，但两次
  generation 的 exact sequence、decoded output 与 canonical action 全部不一致；formal diagnosis=
  `MIXED_D1_FRESH_PATH_AND_SDPA_CONTROL_INSTABILITY`；其他 5/6 states repeat-stable；
- exact aggregate 17,011 bytes / SHA256=
  `b419f9640cb46276b7a52e292d6feabd81311f66efb22563642ce80c4561919c`，只读 rebuild 逐 byte 相同；正式 verdict=
  `NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY`；
- startup utilization sampler 因 operator shell quoting 错误未形成可用证据，所以不作 utilization/throughput claim；
  state/terminal/aggregate 不受影响，也不因此重跑；
- 预注册 contract 只有 PASS 才允许冻结 12-state throughput v3。本次 NO-GO，故 labels、predictor training、
  matched-NLL、closed-loop 继续 locked。Git-safe 结果见
  [`../data/results/set_utility_action_stability_diagnostic_v2/`](../data/results/set_utility_action_stability_diagnostic_v2/)。

### 2026-07-19：D1b memory-safe SDPA Source-A freeze

- canonical config SHA256=`07ec807effd9a1655b0572e49a835e61aec5192503d16bd5c2ee6a3d7a18c77f`，
  54-file transitive source inventory SHA256=
  `a58cbda4db59bf6f951783c94f606e9e00bb310191e371de099256c4286c55b0`；固定 6 states、每 state fresh OS
  process、非连续 4+2 waves、12 generation / 6 encode ceiling、no retry/top-up；
- parent projection 只验证并保留 D1 的 `auto_fresh_encode` 与 `auto_frozen_encoded`；第三个 eager/OOM row、
  旧 diagnosis 与 worker index 不参与 D1b semantic projection。D1 aggregate/summary 的完整 bytes 仍由 Git commit
  与 SHA256 绑定；
- 唯一新 condition 为 `sdpa_numerical_control_frozen_encoded`：numerical controls 在 CUDA 初始化前应用，
  parent automatic loader 保持不变，模型加载后必须回读 top/text/vision 全 `sdpa`，且不声称 strict CUDA
  determinism；
- condition 内 failure 和 outer process failure 都只能序列化 class-only evidence；outer failure 投影为 0
  encode/0 generation，不伪造调用，并按最高优先级形成 `INVALID_RUNTIME_FAILURE`；
- preflight 绑定 canonical cleanup raw log、至少连续 10 秒 0% sample、四张 H200 UUID、container identity 与
  900 秒 freshness；attempt/terminal 逐字段绑定 parent candidate/model/processor inventory、source、D1、wave 与
  unique process identity；
- 四个本机 CPU worker 并发复验 runtime/core、contract、envelope/execution、aggregate/CLI 以及相关 v1 回归，
  共 `75 passed`；`py_compile`、source validator 与 diff check 通过；
- Source-A 禁止 GPU/model load/result write。下一步只允许从 clean pushed `main` 做 fresh preflight，机械生成
  direct-child Execution-B 且该 commit 唯一 changed path 必须是 execution JSON；formal D1b 尚未运行，
  labels/training/matched-NLL/closed-loop 继续 locked。详见
  [`set_utility_action_stability_diagnostic_v2.md`](set_utility_action_stability_diagnostic_v2.md)。

### 2026-07-19：D1 formal execution completed，runtime-invalid

- source A2=`121c8621bae4f56a57060ea2fb402cfa3d8c7146`，唯一 direct-child execution B=
  `5fa1fdd45dba1fa95c7a69ee84d6ddb5b3f5166e`；execution envelope SHA256=
  `093eadeea6c05e1266c681aeea31fa48a86b6dff74d83404e115ce63c3b0a6ea`；
- fresh 10 秒 preflight 后在 Hyper00 同一 container 的 host GPU `0,1,3,4` 上运行 4 个 auto worker，再在全部
  auto terminals 完成 190.222 秒后运行 4 个 eager worker；8/8 attempts、8/8 terminals、retry=`0`，stage
  barrier 与 aggregate audit 均通过；
- aggregate 14,672 bytes / SHA256=
  `2debde6de3f552e9551d0ee37d25b82fa2ca85dfb42746d389dde39eff577ba2`；实际 33/36 generation、24/24
  encode，teacher/KL/restoration/label/training/HF mutation 全为 0；
- 三个 `decision:010` state 的 `eager_frozen_encoded_control` 均在第一次 generation 返回
  `OutOfMemoryError`，包括没有前序 state 的 worker 3；因此预注册 formal verdict 为
  `INVALID_RUNTIME_FAILURE`，不能声称全局 action-instability localization；
- 完整的 decision-6 子集得到 1 个 `FRESH_VS_FROZEN_PATH_ASSOCIATION` 与 2 个
  `PARENT_MISMATCH_NOT_REPRODUCED`。另一个 decision-10 mismatch 的 auto-frozen condition 本身不稳定，但其
  eager control OOM，故该 state 仍只可归为 invalid；
- 6-state/4-worker 不等长短任务没有捕获满足每张卡 80% 的代表性 startup window，因此本 run 不提供 throughput
  claim；这不改变 metric-safe failure aggregate。结果见
  [`../data/results/set_utility_action_stability_diagnostic_v1/`](../data/results/set_utility_action_stability_diagnostic_v1/)；
- D1 封存且不得 retry/top-up。下一步另立 D1b：同 roster/input、memory-efficient SDPA numerical control、每
  state fresh OS process、4+2 两波；D1b 闭合前 labels/training/matched-NLL/closed-loop 继续 locked。

### 2026-07-18（UTC 07-19）：D1 formal source-A freeze

- canonical source config SHA256=`e72b0ba226048b6a188ec5ee35e9a0d5bffa04fcc8ee48f1af2b409151fad2c6`，
  51-file transitive source inventory SHA256=
  `3313f90e43b3f3dec44e1fba8502d796847f5bfba55bc8c05308e19f222cded0`；固定 6 states、4 workers、
  3 conditions、36 generation / 24 encode ceiling，teacher/KL/utility/labels/training/closed-loop 全为 0；
- 新增 source contract、fresh four-H200 envelope、profile-worker runner 与 exact aggregate。B 必须是 A 的唯一
  direct child 且只能新增 canonical execution JSON；parent 必须从旧 run-root canonical envelope 读取；
- auto/eager 分 OS process 和 stage；四个 eager attempt 前均强制验证全部 4 个 auto terminals、attempt hash、
  source/envelope/parent identity、roster 与实际 calls。condition failure 可在 ceiling 内聚合为
  `INVALID_RUNTIME_FAILURE`，不被误写成 action instability，也不得 retry/top-up；
- auto/eager observed attention 分别固定为 top/text/vision 全 `sdpa` 与全 `eager`；worker 的 `PYTHONPATH`
  必须精确指向 source-A worktree `code/`，避免复用旧 venv 时 import 回 parent checkout；
- 初始 pushed source `e3bb7bc` 的真实 pre-policy parent-binding smoke 在 GPU/preflight/model load 前以
  `worker 0 exact mapping/argv drifted` fail closed；根因是 parent absolute argv 必须在 parent 自身绑定的旧
  checkout 重建。repair 由 4 个 parent worker argv 与 aggregate argv 共同锁定唯一旧 checkout，先完成旧
  envelope/stat validation，再显式 rebind D1 source；初始 source 不得执行；
- focused suites 按四个本机 CPU process 并发复验，共 `41 passed`；source validator、`py_compile`、diff check
  全通过。该条是 historical source milestone；后续 A2、B 与唯一 formal run 已完成，结果见上一节。详见
  [`set_utility_action_stability_diagnostic_v1.md`](set_utility_action_stability_diagnostic_v1.md)。

### 2026-07-18（UTC 07-19）：D1 action-stability diagnostic source core

- 不修改 v2 44-file inventory 中的 frozen runtime/adapter/core，新增 prepared-input runtime、metric-safe adapter
  与 pure diagnostic reducer；parent aggregate 仍为 `35e0c250...ff33f`，v2 `NO_GO` 不改写；
- roster 固定为 4 个 mismatch states + 2 个同 stratum stable controls；三个 condition 为
  `auto_fresh_encode`、`auto_frozen_encoded`、`eager_frozen_encoded_control`，每个 condition 两次 generation，
  ceiling=`36` generation / `24` encode，teacher/KL/utility/labels/training/HF mutation 全为 0；
- frozen-input handle 只在进程内保留 processor tensor mapping 与 clone snapshot，before/between/after 检查
  keys/shape/dtype/device/`torch.equal`；serialized witness 只含三层 equality、input unchanged、counts 和安全
  error class，禁止 action/text/coordinate/token/output/logit；
- eager condition 复用已验证 numerical-control runtime，但该 runtime 明确不是 strict CUDA determinism；如它仍
  不稳，后续 D2 才能在新进程中引入 strict deterministic algorithms/CUBLAS workspace；
- auto 固定先 fresh 后 frozen，故 D1 的正结果只写作 path/profile association，不作单一 encoding/backend 因果
  归因；auto 与 eager 必须分进程分 stage。runtime/OOM/parse/input-mutation failure 统一使 diagnostic invalid；
- 3 个 focused test files 以 3 个本机 CPU worker 并行执行，共 `13 passed`，`py_compile` 通过。当前仍不是
  formal source freeze，不授权 GPU；下一步补齐 source contract/runner/inventory 后依次 push A、机械冻结 B，再在
  Hyper00 四张同构 H200 上执行。详见
  [`set_utility_action_stability_diagnostic_v1.md`](set_utility_action_stability_diagnostic_v1.md)。

### 2026-07-18（UTC 07-19）：throughput pilot v2 valid execution、selection NO-GO

- source A=`d5e0cca5c5e05d4aeeef74a1bbfae5685a4254c9` 与 direct-child execution-envelope B=
  `e5002d8820b3e4c8c89343b73ba4a5d13aa117c3` 均已 push；envelope SHA256=
  `dd0e64fb40bd39f839a1df91240a1e9a1a528a0bbfe9131d49ba726e8ecf4e3a`。fresh preflight 在 Hyper00 选择
  host GPU `0,1,3,4`，四个 worker 并发处理各自三个冻结 state；
- execution 与 aggregate 有效：4/4 attempts、4/4 terminals、retry=`0`，aggregate SHA256=
  `35e0c250232efbd2b9bdccfb1b04a7c372fbed892635342cce4f9f81097ff33f`。只读审计确认 lineage、Git/run
  envelope byte identity、canonical terminal、roster、freshness、hashes 和 no-top-up 全部一致；
- 12 个 pair 全部 attempted，8 个完整 completed；实际 42 generation + 26 teacher=`68/84` native calls。
  3 个 state 为 `MICROBATCH_1__REFERENCE_ACTION_MISMATCH`，另 1 个为
  `CROSS_VARIANT_REFERENCE_ACTION_MISMATCH`。因此预注册 selector 正式返回 `NO_GO` /
  `MICROBATCH_1_FAILURE`，selected microbatch=`null`；
- full-call reserved peak=`45,770,342,400` bytes，占每 worker 可见显存 30.49%，未触发 80% bound。8 个成功
  pair 的 mb2/mb1 teacher-wall ratio=`0.997107` 只作 incomplete-subset 描述，不能用于 formal selection；
- operationally，10 秒 active container-level window 平均利用率 84%，另有四卡同时瞬时 100% snapshot；
  worker 完成后的逐卡 0% sample 仅表示 post-completion idle，不参与 startup 或科学判定；
- 正式状态为
  `VALID_COMPLETED_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_V2_SELECTION_NO_GO_REFERENCE_ACTION_INSTABILITY`。
  它不是 processor repair failure、OOM 或 restoration scientific negative。结果见
  [`../data/results/set_utility_train_only_throughput_pilot_v2_candidate_schedule_key_repair/`](../data/results/set_utility_train_only_throughput_pilot_v2_candidate_schedule_key_repair/)；
  labels/training/matched-NLL/closed-loop 继续 locked。下一步另立 versioned action-stability diagnostic，不重跑 v2。

### 2026-07-18（UTC 07-19）：throughput pilot candidate-schedule key repair v2 source freeze

- read-only byte audit 证明 processor candidate schedule 的 raw bytes 正确：18,718,642 bytes、SHA256=
  `186f2952108273672c6cdbf963094754298d23693fd6268a72b6223e99c2299d`、strict JSON parse 成功。naive
  round-trip 的 70 个 byte differences 全部来自
  `exact_label_schedule.state_count_by_candidate_count`：producer 的 int keys `4..11` 被 JSON 转成 string 后，
  consumer 改用字典序；
- v2 仅在该唯一注册路径把 canonical positive decimal string keys 恢复为 int；producer-typed reconstruction
  必须逐 byte 等于 raw。invalid UTF-8、duplicate/non-finite、leading zero、非数字、empty mapping、conversion
  collision 与未注册 numeric-key path 继续 fail closed；HF artifact 不修改、不重发；
- 新 canonical source config SHA256=
  `b2e224147a70dbec14c389098a8fdcdd80b789b4f7f7cbafca15857102eb41ae`，44-file source inventory SHA256=
  `ea3ca6dfda529ccf2cde50e7d1c577a6a8714fbbeb90415faa3c32a16456a669`；focused suite=`55 passed`，source
  validator、py_compile 与 diff check 通过；
- v2 source 逐 byte 绑定 v1 failure summary `d4fde65f...3f45c`、parent A=`5158f2a`、B=`c7d5b31`、parent
  envelope `adf50363...228e` 与 `0` actual calls。旧 v1 config/envelope/result 原样保留，同一 run 不得重试；
- 本步仍是 source-only，policy/GPU/result/label/training/HF mutation 为 0。下一步 push source A2，再以 direct-child
  commit 新增 v2 execution envelope，fresh preflight 后四 H200 并发执行唯一 v2 attempt。

### 2026-07-18（UTC 07-19）：12-state throughput pilot v1 pre-policy contract failure

- source A=`5158f2a`、execution-envelope B=`c7d5b31` 已分别 push `main`；exact envelope SHA256=
  `adf50363d9b3623b84b9229d709f796d7b1971faf7e4056c468cfe3b55ce228e`。fresh 10-second preflight 后，
  Hyper00 四张同构 H200 分别绑定四个 worker，并在 `2026-07-19T10:40:49Z` 同时启动；
- `4/4` worker 均完成 no-retry attempt，但都在 semantic input load 阶段以 `ValueError` fail closed；只读
  decomposition 精确定位为 `candidate schedule is not canonical pretty JSON`。runtime factory/model load 未进入，
  `0/12` pairs、`0/84` native calls、`0` retries；startup utilization gate 因 CUDA compute 从未开始而不适用；
- aggregate CLI 因 failed terminals 返回 `ValueError`，没有 `aggregate.json`，没有 mb1/mb2 selection。正式状态是
  `INVALID_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_V1_CANDIDATE_SCHEDULE_CANONICALIZATION_CONTRACT_DRIFT`，
  不是 scientific/throughput `NO_GO`，也不改变 processor v2 的 VALID verdict；
- v1 evidence 原样保留且不得重跑。轻量记录见
  [`../data/results/set_utility_train_only_throughput_pilot_v1/`](../data/results/set_utility_train_only_throughput_pilot_v1/)；
  下一步另立 versioned consumer canonicalization repair，不改 immutable schedule bytes/hash、12-state roster、
  mb1/mb2、84-call budget、80% memory threshold 或 5% latency rule。labels/training/matched-NLL/closed-loop 仍为 0。

### 2026-07-18（UTC 07-19）：12-state train-only throughput pilot source freeze

- 从 Freeze-B v2 预注册了 12 个 train-only `stratum_anchor`：`decisions_6_9`、`decisions_10_17`、
  `decisions_18_plus` 各 4 个 state。四个 worker 固定为 `state_ids[index::4]`，每个 worker 恰好处理三个
  strata 各一个 state；禁止替换失败 state、top-up 或跨 host 混合 metric；
- 每个 state 在同一 policy instance/GPU 上 paired 执行 mb1→mb2；每 variant 两次 generation，mb1
  两次 teacher calls，mb2 一次 teacher call，success path 精确为 48 generation + 36 teacher =
  84 native calls。action 仅作进程内 opaque handle，result 只允许 latency、full-call CUDA peaks、counts、
  equality boolean 与 class-only failure；
- 选择规则事先固定：reserved peak 不得超过每张卡总显存 80%；只有 mb2 teacher wall
  `<=0.95*mb1` 且全部 equality/memory/success 通过才选 mb2；mb2 仅 teacher-stage failure 才可在
  mb1 全通过时 fallback，其余 `NO_GO`；
- source-only canonical config SHA256=
  `ca484ed808aea0881a48cb19edc363f149e1cf1a2117a86d15ac163749569c17`；44-file transitive source
  inventory SHA256=`78a25d74c1a714c7f48746691d70415521c5d43dd8d5355720487807d3ee1509`。
  processor immutable revision=`c20bab8df424dc9e45ece1084f3d1dc035dd1ed8`，candidate schedule SHA256=
  `186f2952108273672c6cdbf963094754298d23693fd6268a72b6223e99c2299d`；
- exact execution envelope 已实现 source-A→envelope-B 双 commit lifecycle：A 必须 clean pushed `main`，B 必须是
  A 的 direct child 且只新增 Git execution JSON；Git blob 必须与 `/data` O_EXCL envelope 逐 byte
  一致。worker/aggregate CLI 仅接受 exact envelope，不提供 model/processor/device/output 绕过参数；
- contract/pair/execution/envelope focused suite=`46 passed in 7.57s`，source validator 返回
  `VALID_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_SOURCE_V1`，`py_compile` 与 diff check 通过。本里程碑
  不访问 policy/GPU，不生成 utility/label/checkpoint/result；后续唯一 v1 formal attempt 已由上一里程碑记录并因
  consumer canonicalization contract drift 在 `0/84` calls 处 fail closed。

### 2026-07-18（UTC 07-19）：processor v2 immutable HF publication 与 Git finalization 已闭合

- private dataset `gavinlaw/causalcache-set-utility-new-development-mobile` 在 prefix
  `artifacts/processor-freeze-v2-image-contract-repair` 以单次 25-operation commit 发布；annotated tag
  `phase1-b2-processor-freeze-v2-image-contract-repair` 解析到 immutable revision
  `c20bab8df424dc9e45ece1084f3d1dc035dd1ed8`，parent main 为 census-v2 revision `c1d19eb9...eae0`；
- remote 共 25 files；其中 formal tree 为 23 files / 18,730,620,511 bytes，inventory SHA256=
  `7c2a9716...32fc1`。从 commit 与 tag 各 fresh-download 一份 retained replay，25/25 remote files 与 23/23
  formal files 均逐 byte 相同；之后独立 `validate-only` 再次 exit=`0`；
- publication receipt 为 8,136 bytes / mode `0600` / SHA256=`ad69d201...6278`；finalizer 从 clean exact
  `main@db51693` 先执行完整 remote validation，再于 `2026-07-19T09:46:48Z` atomic 写 exact-two sibling result，
  status=`FINALIZED_PROCESSOR_V2_IMMUTABLE_HF_PUBLICATION`，自身 HF mutation count=`0`；
- Git-safe finalization 见
  [`../data/results/set_utility_processor_freeze_v2_image_contract_repair_publication_v1/`](../data/results/set_utility_processor_freeze_v2_image_contract_repair_publication_v1/)。原 pending card/summary、外置 receipt 与两份 replay
  保持原字节可复验；临时 HF token 已从 Hyper00 容器和本机临时路径删除；
- processor prerequisite 现已完整闭合。restoration labels、predictor training、matched-NLL 与 closed-loop 仍为
  0/locked；下一步冻结 12-state train-only throughput pilot，从 fresh GPU preflight 后的 Hyper00 空闲 H200 中按
  有效 shard 并行，不混合异构 host throughput 指标。

### 2026-07-18（UTC 07-19）：processor v2 canonical result 进入 PENDING_HF_UPLOAD（历史里程碑）

- `e976b990` producer 用时 3,163s、exit=`0`，atomic root 后 canonical committed postflight exit=`0`；随后
  fail-closed watcher 从 clean recorder `63734097` 运行 fresh committed postflight，并于
  `2026-07-19T09:29:16Z` 原子产生 exact-two Git result；
- formal result status=`VALID_RECORDED_SET_UTILITY_PROCESSOR_FREEZE_V2_IMAGE_CONTRACT_REPAIR`，
  `scientific_eligibility=true`、`postflight_accepted_for_formal_result=true`。artifact 为 23 files / 18,730,620,511
  bytes，inventory SHA256=`7c2a9716...32fc1`，tree SHA256=`ab222778...456a6`，staging absent；
- Git-safe `README.md` 为 1,351 bytes / SHA256=`452f7d63...fb179`，`summary.json` 为 17,673 bytes /
  SHA256=`87a9a17e...30a3`；从 remote recorder 到本地逐 byte `cmp` 相同，见
  [`../data/results/set_utility_processor_freeze_v2_image_contract_repair/`](../data/results/set_utility_processor_freeze_v2_image_contract_repair/)；
- 该历史时点的 publication status=`PENDING_HF_UPLOAD`，HF mutation count 为 0。policy inference、restoration
  labels、training、matched-NLL 与 closed-loop 也为 0；后续 publication、commit/tag 双 fresh replay 与独立
  finalization 已按本页顶部最新里程碑完成。

### 2026-07-18（UTC 07-19）：12-state throughput pilot host readiness

- 对 Hyper00、Hyper01、H100 与 Aries 做只读快照；没有 GPU launch/cleanup、container stop 或远端写入。free
  定义为无 compute app 且显存 free>=99%，不是简单要求 used=0；
- Hyper00 当前有 7 张符合定义的 H200（IDs `0,1,3,4,5,6,7`），本地已有 exact GUI-Owl model
  revision `06d5faec...d04fc`、manifest SHA256=`65bd2377...4265c` 与 18,730,620,511-byte/23-file processor
  root，额外搬运为 0；GPU 2 被现有 133,458 MiB compute process 占用，明确排除；
- Hyper01 8 张 H200 当前均被占用，且缺 18.73GB processor root；H100 有 3 张空闲 80GB 卡，但缺 repo/model/
  processor、local image identity 不同，需要搬约 36.28GB 并另做 runtime/behavioral binding；Aries 无空闲卡、
  repo dirty 且本地盘紧张；
- 当前 primary 因而是 Hyper00，H100 是 cold backup，Hyper01 仅在卡释放并下载 immutable processor artifact 后
  才是 warm backup。正式 pilot 不合并异构 host throughput 指标，且 launch 前仍必须重新 preflight；届时从
  sealed commit 建新的 clean checkout，不能复用普通 dirty repo 或旧 producer checkout。轻量快照见
  [`../data/results/set_utility_throughput_host_readiness_v1/`](../data/results/set_utility_throughput_host_readiness_v1/)。

### 2026-07-18（UTC 07-19）：真实 GUI-Owl throughput adapter source core

- 新增 `set_utility_gui_owl_v2_1_throughput_adapter.py`，只接受 versioned
  `GUIOwlV21ThroughputRuntime`；generation/teacher 均显式使用 caller-owned CUDA peak，外层 measurement 从
  native encode 前开始，覆盖 H2D、preparation、forward、generation decode/parse 或 teacher logits/metadata
  disposal；
- adapter boundary 只允许 exact `GUIOwlV2Action` 作为进程内 action handle。native output、tokens、metadata、
  logits、KL 与 utility 均不能进入 metric-only payload；runtime 或 measurement exception 只投影安全 class id、
  elapsed wall 与 CUDA peak，敏感 exception message 不序列化；
- 修复 processor→label join 的真实输入缺口：`SelectedProcessorQuery` 现在显式暴露 tar allowlist 中的 exact
  image mapping，`JoinedUtilityQueryInput` 要求 inventory 恰为 initial candidates + current，并保持同一 mapping
  identity，避免在 join 后丢失 policy prompt 所需图像或静默扩大可见范围；
- adapter source SHA256=`7b8f40eab4963b3327752981a1a6ebf0d75d38f5a9e24e4970d389ee8021d1b4`；adapter、
  throughput core/runtime、label input/producer focused regression=`43 passed, 2 subtests passed`，compileall 与
  diff audit 通过。旧 throughput runtime Git blob 仍为
  `64c49b641e29b90047c0cb665e028af8f394ea7e`；
- 本步没有读取 formal artifact、运行 CUDA/policy forward、冻结 12-state roster/config、生成 label 或执行 HF
  mutation。下一步仍须等 processor Git result + immutable HF publication 后冻结 train-only pilot contract。

### 2026-07-18（UTC 07-19）：versioned 4-worker processor postflight source path

- 当前 formal 尾部串行瓶颈已定位：historical v1 structural audit 已按 4 worker 并行，但 v2 semantic overlay
  仍顺序重读四个 tar，并逐 image 做 decode/SHA/OCR canonical reconstruction；
- 历史 `set_utility_processor_postflight_v2.py` 保持 37,150 bytes / SHA256=
  `de6eb1a18ea896890efc8361e287733300c1e704887582c5f82342e69382f7cc`，旧 execution config/hash 与当前 formal
  均未修改。新增 parallel module、fail-closed source contract/config 与独立 CLI；
- 新 path 对四个 worker tar 使用 `ThreadPoolExecutor(4)` + ordered `map`，worker 内仍调用完全相同的 iterator、
  image/OCR validators；主线程按 worker 0→3 聚合 tally/coverage，并额外拒绝跨 worker path overlap 与
  stored-image coverage 逃出 terminal inventory；
- 首次真实只读启动在 41 秒后 fail closed：versioned checkout 被错误用于构造 historical context，和 artifact
  正确记录的 producer absolute path 不同，`_run_identity` 返回 `fixed runtime values drifted`。exit=1、无
  summary、formal root 无修改；该失败只否定初版 parallel contract，不是 artifact verdict；
- 修复版 config SHA256=`bea5d83324263461c6a5ef2368606e1eaf4a160aa3ce25adcc493ebee7a66365`；新增
  `--producer-repository-root`，要求 absolute/real/non-symlink/clean Git root 与 exact HEAD，historical config、
  postflight/source audit 和 context 全从 producer root 加载；versioned root 只绑定新 validator source；
- 本机 processor/result/publication/watcher focused regression=`100 passed`，另有 historical v2 contract
  regression=`14 passed`。fixture 验证 concurrent=4、逆序完成确定性、最低 index
  failure、exact tar binding、serial/parallel equivalence、全树只读及 dual-root 的 path/symlink/dirty/HEAD/
  config failures；
- fixed path 随后从 clean detached `de4c0348` 对 producer `e976b990` 的 completed root 完成真实复核：
  4 workers、ordered aggregation `[0,1,2,3]`、exit=`0`、stderr=`0B`，wall=`908.558s`；canonical
  reference=`2549.669s`，speedup=`2.8063x`；
- parallel summary 与 canonical 在 artifact shards、candidate、operation budget、tally、stored/metadata-only OCR
  counts、image/runtime identity 与 structural status 上逐字段相同。运行开始后 formal root mtime 更新文件数为
  0，两个 checkout 结束后均 clean；Git-safe 记录见
  [`../data/results/set_utility_processor_postflight_parallel_v1/`](../data/results/set_utility_processor_postflight_parallel_v1/)。
  它可作为后续合同的默认 validator，但不追溯替换 `e976b99` canonical evidence；
- `e976b99` canonical producer 与 committed postflight 已于 `2026-07-19T08:45:41Z` 双零终止，状态为
  `VALID_COMPLETED_SET_UTILITY_PROCESSOR_FREEZE_V2_IMAGE_CONTRACT_REPAIR`。fail-closed watcher 已进入旧合同要求的
  fresh recorder validation；Git-safe formal result 与 HF publication 仍在等待该 recorder terminal。parallel
  path 本身不执行 policy/GPU/label/HF mutation。

### 2026-07-18（UTC 07-19）：versioned GUI-Owl throughput measurement seam

- 只读 adapter 审计确认：现有 v2.1 generation/teacher 都在 encode/H2D/preparation 之后才 reset CUDA peak；
  外层即使提前 reset，也会被内部 reset 清零，因此不能直接声称 full-call peak；
- 没有修改 byte-pinned `gui_owl_v2_1_runtime.py`、历史 config 或 hash。新增独立
  `policy/gui_owl_v2_1_throughput_runtime.py`，只 override generation 与 teacher 两条路径；
- keyword-only `cuda_peak_measurement_owner` 只允许 `runtime|caller`，并在任何 encode 前 fail closed。默认
  `runtime` ownership 与 pinned base 的 reset、timer、metadata 和输出一致；`caller` ownership 只跳过内部
  reset，保留同步与原 native timer，使未来 adapter 可在外层测量完整调用；
- 新 source SHA256=`a2c139a1c5c73d394c34953bc1c17273656447604bdae85ce2e4a1ac0cd51c94`；本机
  generation/teacher seam、base runtime 与 throughput core 独立回归=`26 passed, 2 subtests passed`，原 pinned
  runtime SHA256 仍为 `a34b4417c23c085fb6a01e29dcc6398a00804c07eff4945812f9ba4688de6ba5`；
- 本步没有实现真实 adapter、冻结 12-state roster/config、执行 GPU/policy forward、生成 label 或修改 HF。

### 2026-07-18（UTC 07-19）：train-only metric-only throughput pilot source core

- 新增 `code/causalcache/set_utility_throughput_pilot.py`：只接受 train `UtilityQuerySpec`，固定两次 reference
  generation 与两个 logical reference teacher examples，并只允许 reference teacher microbatch 1/2；
- input builder 仅允许渲染 native messages，不得运行 processor 或接触 CUDA。真实 runtime adapter 必须从
  encode/H2D/preparation 前开始计时，并覆盖 native forward、generation decode/parse 或 teacher-logit disposal；
  主指标是完整 adapter call 的 end-to-end wall time 与同边界 CUDA allocated/reserved peaks；
- 初版审阅发现直接复用 GUI-Owl model-only metadata 会漏掉上述前后处理，且 exception 路径会丢失失败调用
  latency/peak。现已新增严格 metric-only `PilotRuntimeFailure`：只含 safe error-class identifier 与 performance，
  不含 message 或 policy output；generation/teacher 失败调用的指标也进入 aggregate；
- output schema 仅有 latency、peak memory、failure class 与 operation counts；action handle 只在内存传递，不输出
  action text、tokens、logits、KL 或 utility。敏感异常 message 与不合法 class name 均不能进入 serialization；
- throughput + label-producer 独立回归=`23 passed, 2 subtests passed`，compile/import/diff audit 通过；本步未
  冻结真实 adapter/config，未执行 GPU、policy forward、label、checkpoint 或 HF mutation。

### 2026-07-18（UTC 07-19）：formal label 物理 role firewall source core

- 新增 `code/causalcache/set_utility_label_partitions.py`：execution worker 可按 state 混合处理 role，但 publication
  必须写成 `labels/{train,tune,evaluation}/part-worker-XX.parquet`；同一 state 不能跨 execution worker；
- trainer inventory 只包含 train/tune shard identity，evaluation inventory 只能在绑定 model-seal SHA 后释放；
  trainer reader 不扫描 label root，因此不会先看到 evaluation path 或 payload；
- writer/validator 固定 absolute canonical root、逐层 no-follow dir-fd、`O_NOFOLLOW|O_EXCL` no-clobber write、
  exact three-role tree、stable inode read、固定 PyArrow schema 与 metadata-free postflight。cross-role row、跨 worker
  state、file/ancestor/role-dir symlink 均 fail closed；
- 本机独立 focused suite=`10 passed, 2 skipped`，skip 仅因本机未安装可选 PyArrow；首次 Hyper00
  PyArrow 24.0.0 复验发现默认 list-child 名 `item` 经 Parquet 2.6 readback 会变成 `element`，exact schema 因此
  fail closed。现已把两个 list child 显式冻结为 `element`；同一 runtime 的 schema equality、row equality 与
  metadata rejection exact nodes=`2 passed`；
- 本步没有读取 processor partial、没有 policy/model forward、label execution、GPU、checkpoint 或 HF mutation；
  正式 label 仍需等待 processor completed root、Git result 与 immutable HF revision。

### 2026-07-18（UTC 07-19）：processor publication Git SoT finalizer source（历史 source-freeze 里程碑）

- publication manager 新增 `finalize`：先复用现有 remote read-only `validate-only`，只有 receipt、private repo、
  no-overwrite commit boundary、annotated tag 与两份 retained fresh replay 全部复验后，才允许写 Git result；
- finalizer 要求 exact clean Git revision 与 committed pending summary/card，从一个新的 sibling
  `data/results/...` staging 原子生成 `summary.json` + `README.md`；final summary 内嵌完整 validated receipt，绑定
  receipt SHA/size/`0600` mode、source Git blob、immutable revision/tag，且 HF mutation count=`0`；
- 原 pending result、外置 receipt、commit/tag fresh replay 均保持原路径原字节，因此 publication manager 的
  `validate-only` 仍可独立重复。source path 拒绝 leaf/ancestor symlink；receipt 由单个 `O_NOFOLLOW` fd 同时完成
  mode check、稳定读取与 path/inode identity 复核；
- publication tests=`29 passed`，完整 processor/result/watcher suite=`151 passed`。本 source-freeze 步当时只闭合
  未来 publication 的 Git SoT 交接，没有访问 HF、没有创建 tag/revision，也不改变当时 formal 的 running 状态；
  后续实际 publication/finalization 终态见本页顶部最新里程碑。
- Hyper00 已准备 clean detached publication checkout `main@450a669` 与 persistent complete-history bundle
  `/data/tmp/causalcache-processor-publication-450a669.bundle`（SHA256=
  `d26bca2747b3ce3ef68475e9616964cbed1deba5d9c6b4dab3162108c4d99cd8`）。当前 checkout 只用于缩短后续交接；
  当时必须等 Git-safe result commit/push 后再用 incremental bundle 更新到 exact revision；该后续步骤现已全部
  完成，临时 HF token 也已删除，详见本页顶部最新里程碑。

### 2026-07-18（UTC 07-19）：train-only processor→label input firewall

- 新增 `code/causalcache/set_utility_label_inputs.py`：从 frozen processor roster 先推导全部 state/role，任何
  tune/evaluation allowlist 在打开 query record 前即拒绝；对未选记录只检查 tar member header/order 并跳过 payload，
  只有显式 train state 才进行 canonical JSON、image SHA 与 artifact provenance 校验；
- processor artifact 用一个 absolute `O_NOFOLLOW` regular fd 完成流式 SHA、seek、tar parse，并在结束时复核
  `dev/ino/mode/nlink/size/mtime/ctime` 与当前 path identity。symlink 和 hash 后同字节换 inode 均 fail closed；
- strict join 绑定 processor query、Freeze-B assignment、final candidate、request manifest、artifact/query witness，
  并分别保留 processor worker、未来 label execution worker 与 role partition。malformed UTF-8 tune/evaluation
  payload 仍可安全读取 allowlisted train state，证明没有发生非 train semantic decode；focused suite=`47 passed`；
- 本步只准备下一阶段 source core，不冻结真实 pilot config，不执行 policy/model forward、label、GPU 或 HF mutation；
  当时正式 pilot 仍必须等待 processor completed root、Git result 与 immutable HF revision，该 prerequisite 现已闭合。

### 2026-07-18（UTC 07-19）：processor v2 自动 recorder handoff

- 新增 `code/scripts/watch_set_utility_processor_freeze_v2_result.py`，把当前 CPU formal run 的结束边界机械连接到
  Git-safe recorder；watcher 本身不修改 formal root，也不执行 HF mutation；
- watcher 只认 canonical `supervisor.json` 的精确 4-field schema、formal/postflight 双零，以及
  `exit_code` / `postflight-exit-code` 的精确 `0\n`。任何 timeout、symlink、非零退出、字段/格式漂移或预先存在的
  result/staging 均 fail closed，且 recorder argv 永远不含 `--record-invalid`；
- recorder 必须来自独立 clean checkout，watcher 在 checkout 外运行并移除 `PYTHONPATH`，避免运行证据或环境污染
  source binding。processor/result/watcher focused suite 为 `145 passed`，`py_compile` 与 `git diff --check`
  通过；source commit `c2f6725626a67bf5b9f90135f953f998e7db9236` 已 push；
- Hyper00 已部署 persistent watcher
  `/data/logs/watch-set-utility-processor-freeze-v2-result-c2f6725.py`，SHA256=
  `26cba29f2dabf9f12756cf9c8d398ccf012754ce89a1ffe3987e6645266f33fd`；它在 recorder checkout
  `63734097983fa85f1470a52d1edd645b194433ab` 外运行，日志为
  `/data/logs/causalcache-processor-recorder-c2f6725/watcher.log`，timeout=`172800s`。当前 formal 科学状态仍是
  canonical postflight VALID、fresh recorder running；本 milestone 当时的 partial bytes 未被追认为结果，HF
  mutation 仍未执行。

### 2026-07-18（UTC 07-19）：processor v2 replacement source freeze

- `1c84dfe` 与 `dc90664` 两次 run 分别于 `2026-07-19T06:27:56Z` 和 `06:46:35Z` 主动终止，exit 均为
  `143`。两次都是 0 receipt、0 completed shard、0 candidate-part、0 policy output、0 HF mutation；原因是正式
  OCR 结束后必然遇到的 processor guard false positive 已被真实 smoke 提前证实：Transformers 5.6 只加载通用
  `transformers.models.auto.modeling_auto`，旧 guard 却拒绝任何 `modeling_*`；
- 两份 staging 分别为 6,110,870,181 bytes 与 14,201,155,380 bytes / 9 files，只作
  `LOCAL_FORENSIC_NOT_UPLOADABLE`。现有 resume 只接受 completed tar+receipt pair，会删除 `.tar.partial` 后重算，
  因而不能把旧 partial 复制到新 namespace 或追认为可恢复 substrate；
- replacement canonical config 为 9,290 bytes，SHA256=
  `e2c271e00749ca7643899c86fd216a635d007337630ba9a1a19b2314ff4afb70`。科学/输出合同、4 个 logical shards、
  2,400 queries、23-file tree 与 operation-budget 语义不变；execution 层固定每 shard 32 路 OCR、8 路独立
  AutoProcessor，总计 128/32 slots；
- processor child 会清除 ambient thread environment，再用 runtime API 在 AutoProcessor load 前设置并 getter
  验证 PyTorch intra-op=`28` / inter-op=`1`；completed-root postflight 要求四个 processor log 各有唯一
  canonical getter evidence。首个 AutoProcessor 前 modeling modules 必须为 0，load 后只
  allowlist 通用 `modeling_auto` registry，任何 architecture `modeling_*`、model forward/generate 继续 fail closed；
- Hyper00 真实 OCR 128-image smoke 的 8/16/32 路耗时为 `75.52/49.74/33.93s`，32 路相对 8 路为
  `2.23x`；processor 8-query smoke 的 1/4/8 路为 `85.66/28.11/26.93s`，最终 Torch 28/1 下 8 路为
  `25.71s`。串行/并发 canonical outputs 全等；
- 新增 immutable HF publication manager：只有 exact 23-file root 与 committed `PENDING_HF_UPLOAD` summary
  才能单次提交 25 files、创建无覆盖 annotated tag，并分别从 commit/tag fresh-download 25/25 逐 byte 验证。
  manager 支持 commit/tag/download/receipt failure 后的 exact absent→exact commit reconcile，未知 remote state
  fail closed；fresh replay 在 mutation 前检查，receipt 以 `0600` temp、fsync 与 no-overwrite atomic publish。
  该 source-freeze 时 processor prefix/tag 尚不存在。四进程联合 processor smoke 用 32 runtimes 处理 32 条真实 query，joint
  wall=`41.26s`、effective CPU=`30.79 cores`、aggregate peak RSS 约 48 GiB，四个 getter 全部为 `28/1`；
- clean formal run 已于 `2026-07-19T07:10:28Z` 从 pushed `main@e976b99` 在 Hyper00 启动。启动前 exact
  worktree、79 项 focused tests、Execution/source/snapshot validators 均通过；4×32 OCR 稳定窗口合计约
  `110.13` CPU cores、约 36.8 GiB RSS，error/traceback=0。producer 于 `08:03:11Z` atomic publish，canonical
  committed postflight 于 `08:45:41Z` 返回 VALID，formal/postflight 双零且 stderr=0；当前 fail-closed recorder
  正在做旧合同要求的 fresh validation。labels、checkpoint、matched-NLL、closed-loop 与 HF mutation 仍为 0。
  下一步等待 Git-safe recorder terminal，再进入 pending publication。完整合同与轻量 launch record 见
  [`set_utility_processor_freeze_execution_cf_v2_image_contract_repair.md`](set_utility_processor_freeze_execution_cf_v2_image_contract_repair.md)。

### 2026-07-18（UTC 07-19）：selected-image census v2 immutable HF publication

- 创建 private dataset `gavinlaw/causalcache-set-utility-new-development-mobile`；13-operation payload commit=
  `08e1b721760eb92ce8db9cb4d83ad4e01e957249`，包含原始 10-file root、artifact README 与 Git-safe summary；
- dataset card commit=`c1d19eb96d7fa7926f1eb9db3328dbff4e88eae0`，冻结 tag=
  `phase1-b2-image-format-census-v2-column-projection-repair`，tag resolution 已验证指向该 immutable revision；
- 从 tagged revision fresh-download 14 repo files 到独立本地路径；其中 10 个 formal files 总计 6,943,195
  bytes，与 Hyper00 原件逐 byte 相同，canonical inventory SHA256=
  `6c687bec18fd9bcb8f06dfd31c9a1685a6de976eed67b610de18fd3d7933bd4f`；
- Git dataset card 位于
  [`../data/cards/set_utility_selected_image_format_census_v2_column_projection_repair.md`](../data/cards/set_utility_selected_image_format_census_v2_column_projection_repair.md)，
  Git result summary 已从 `PENDING_HF_UPLOAD` 更新为 `PUBLISHED_HF_IMMUTABLE_VERIFIED`；
- 该 publication milestone 当时将下一步收敛为 versioned processor image-contract repair；后续已完成本页上节
  v2 source freeze。labels、training、matched-NLL、closed-loop 与 sealed test 继续 locked。

### 2026-07-18（UTC 07-19）：selected-image census v2 column-projection repair VALID

- clean pushed producer=`1a03b7eb8ea2c41c8fed5213fed1bcc9d73f846b`；Hyper00 CPU-only process 从
  `2026-07-19T04:02:14Z` 到 `04:08:52Z`，398 秒、exit code 0，4-worker line counts=
  `4700/4700/4693/4699`；
- exact `columns=["images"]`、batch schema=`["images"]` 与 row keys=`{"images"}` 全部运行时验证通过；
  OCR、AutoProcessor、model/policy、GPU、labels、training、matched-NLL、closed-loop、sealed test 与 HF mutation
  均为 0；
- committed completed-root postflight 返回
  `VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR`，重建 18,792 selector 精确顺序与
  全局唯一性、records/receipts、histograms、manifest/run identity、projection/source/predecessor hash chains；
  final root 为 10 files / 6,943,195 bytes，无 symlink，staging absent，postflight 前后 tree snapshot 一致；
- 正式 histogram：`PNG=18,792`；`RGBA=18,768`、`RGB=24`；opaque alpha=`18,768`、null=`24`；
  `EXIF=false=18,792`。non-selector records 与 v1 forensic root 逐项一致只作诊断，不作为有效性证据；
- Git-safe result 见
  [`../data/results/set_utility_selected_image_format_census_v2_column_projection_repair/`](../data/results/set_utility_selected_image_format_census_v2_column_projection_repair/)。
  artifact 后续已完成上节 immutable HF publication，并进一步完成 processor image-contract v2 source freeze；
  当前下一步是 formal processor-only run，不能重编码、补 alpha 或跳过 24 个 RGB records；
- final candidates、restoration labels、predictor training、matched-NLL、closed-loop 与 sealed test 仍为 0/locked。

### 2026-07-18（UTC 07-19）：selected-image census v2 column-projection repair source freeze

- v1 contract drift 的唯一允许后继已完成独立 protocol/config/runner/output/tag source freeze；config 10,292
  bytes，SHA256=`eb823bd794c555265107e5edd4ff9b0be60b9c907b476a313ce5e156779a549f`，当前状态为
  `SOURCE_FROZEN_FORMAL_RUN_PENDING`，尚无 formal output、有效 histogram 或 HF artifact；
- v2 保持原 1,200 trajectories / 18,792 observations / 527 selected shards / 4-worker denominator，只允许 exact
  `ParquetFile.iter_batches(batch_size=8, columns=["images"])`；在 `to_pylist()` 前必须断言 exact batch schema，
  每个 projected row 还必须只有 `images` key；
- regression 实际捕获 PyArrow projection 参数，而不再仅依赖 AST 禁止字段名；formal output 必须由
  committed completed-root postflight 独立重算 inventory、denominator、hash chains、projection witness 与全部
  forbidden-operation counts；
- source/contract validator 分别返回
  `VALID_CPU_ONLY_SELECTED_IMAGE_CENSUS_V2_COLUMN_PROJECTION_REPAIR_SOURCE` 与
  `VALID_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR_CONTRACT`；v2-specific tests=`23 passed`，
  v1+v2 focused=`37 passed`，全量 set-utility regression=`299 passed, 15 skipped, 24 subtests passed`；
- 本 source freeze commit/push 后授权从 clean `main` 启动正式 CPU-only run；runner manifest 只能进入
  `AWAITING_COMMITTED_POSTFLIGHT`，postflight VALID 前禁止 HF publication，也不能把 v1 forensic histogram
  当成 processor repair 输入；
- processor repair、final candidates、restoration labels、predictor training、matched-NLL、closed-loop 与 sealed
  test 全部继续 locked。完整 source-freeze contract 见
  [`set_utility_selected_image_format_census_v2_column_projection_repair.md`](set_utility_selected_image_format_census_v2_column_projection_repair.md)。

### 2026-07-18（UTC 07-19）：selected-image format census v1 attempt 合同失败

- clean pushed producer=`e636df1fa3890c0f889c99db227d6bb959d23383`；Hyper00 CPU-only process 从
  `2026-07-19T03:24:38Z` 到 `03:31:16Z` 以 code `0` 完成，4-worker line counts=
  `4700/4700/4693/4699`，并原子发布 10-file root；
- `2026-07-19T03:36:38Z` 的独立 contract audit 发现 `ParquetRowFactory` 调用
  `parquet.iter_batches(batch_size=8)` 时未传 `columns=["images"]`。因此 `batch.to_pylist()` 已物化完整 rows，
  包括 `messages`/`metadata`，违反 frozen `images_column_only_no_message_metadata_action_or_outcome_decode`；
- 正式终态为 `INVALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_COLUMN_PROJECTION_CONTRACT_DRIFT`。runner 只显式访问
  image field、没有调用 semantic parser，不能消除底层 full-row materialization 的授权违例；process exit 0 也
  不能覆盖 post-run contract failure；
- observed 18,768/24 mode histogram 与所有 hashes 只作 formal-ineligible forensic evidence，不能证明 full census
  闭合，也不能冻结 processor input repair；
- 10 files / 6,940,661 bytes、manifest SHA256=
  `282b86c62e3856f35da901c925f23d0baa54970211260e97f44a4aa6ee2b7180`、tree SHA256=
  `973b0f1b29059fde2f7e1001559b10a38629192e8e3db6f5774d41764ff627e3` 原样保留在 Hyper00，仅作 INVALID
  evidence；HF publication 明确禁止；
- OCR、AutoProcessor、model/policy forward、labels、training、matched-NLL、closed-loop 与 sealed test 仍为 0；
  但不能再声称 messages/metadata column access 为 0；
- 轻量 failure、完整 argv/provenance 与 replacement requirements 见
  [`../data/results/set_utility_selected_image_format_census_v1_attempt/`](../data/results/set_utility_selected_image_format_census_v1_attempt/)。
  下一步另立 versioned column-projection repair：exact `columns=["images"]`、runtime batch-schema assertion、
  mocked PyArrow projection regression、new config/output/tag identity，并重跑完整 18,792 denominator。

### 2026-07-18：selected-image format census v1 source freeze

- processor-freeze Execution-CF v1 因第 228 个 observation 是合法 `PNG/RGB`、旧 contract 只接受 opaque
  `PNG/RGBA` 而永久 fail closed；旧 output absent、`.incomplete`/logs/hashes 保留，不修改旧 bytes、不跳样本；
- 新 census 固定 Freeze-B v2 的 1,200 trajectories / 18,792 observations / 527 selected shards，按 whole-shard
  deterministic LPT 使用 4 个 CPU worker，observation load=`4700/4700/4693/4699`；
- config SHA256=`c0ecbf59dc6bd77881503e92b0e3eb8c011c2768d0721fa5a74c82c8fe173d10`，逐 byte 绑定 4 个 frozen
  inputs 与 23-file runtime import closure；validator 状态为
  `VALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_CONTRACT`；
- 针对合同审计，正式 worker 已改为 image-column-only extractor：pinned shard SHA、row count 与冻结 row
  index 负责 source identity，只读取 selected row 的 `images`，不调用 `inspect_candidate`、
  `load_selected_rows_once`、`build_selected_pilot`，也不解析 instruction/action/terminal outcome；
- 每个 observation 只写九字段 pseudonymous record；不含 raw path、instruction、action、state id、image bytes
  或 outcome。OCR、AutoProcessor、model/policy、torch/GPU、labels、training、closed-loop 与 HF mutation 全部禁止；
- atomic staging、record/receipt paired resume、hard-link no-clobber、全局 selector uniqueness 与 no-replace final
  publish 已实现。focused suite `30 passed`，py_compile 与 source-only validator 通过；
- 该 source-freeze milestone 当时为 `SOURCE_FROZEN_FORMAL_RUN_PENDING`；后续 v1 attempt 已完成进程但因上节
  column-projection contract drift 永久 INVALID。source 阶段本身不包含有效 output、HF revision 或 processor repair；
  完整协议见
  [`set_utility_selected_image_format_census_v1.md`](set_utility_selected_image_format_census_v1.md)。

### 2026-07-18：full-pool P-1 inventory 正式完成

- 当前数据路线不再以 16-shard / 13–64 小池子作为主入口。P-1 先对指定 revision 的
  610 个 `mobile/use/train` transport shards 只读 metadata，固定 path/size/LFS SHA256；再单独
  冻结 P0 Execution-A 进行 policy-blind semantic census；
- budget-agnostic predictor config SHA256=
  `9548159b219795b1c258c28f772f53351256e0d728b88dd409cb333bd2100fe4`；P-1 source config
  SHA256=`1b2b4374d1653bcf22444d8e708c71bc41ac87fa956243ddcb9bd963eeca7e96`；P0 source-only
  config SHA256=`7e65227e710009d3626bd0063d425c831dfc59e9a6bbe871b9d3e4d15e085e8b`；
- canonical consumed ledger 已从六个 byte-pinned 历史输入机械重建 107 个 identity：
  `legacy_train_only=58` 与 `forbidden_consumed=49`；manifest SHA256=
  `b6f44c603b99d2f954b981e01818cf0afa028ce3a410ed935203532a097bb4ad`。P0 source 已绑定该
  ledger；P0 source-only config 仍故意不绑定新的 P-1 manifest，因此不授权 download/row decode/census/output；
- split audit 已从 source identity firewall 扩展到 historical group firewall：forbidden consumed group
  不得以新 source ID 重新进入任何 role，legacy group 只能留在 effective train partition；P-1
  writer 返回的 manifest SHA 也已改为精确落盘 bytes SHA，避免末尾换行造成的错误绑定；
- 已实现 processor-only candidate freeze（排除 current-equivalent，recent `<=16`，超出 32k
  context 时逐个丢弃最旧 event）、`K=2/3/4` exact capped label schedule/producer、variable-`n`
  features、complete-table validator、group split firewall、joint search 与 trajectory-equal evaluator；
- Set Transformer 是 main，DeepSets 和 pairwise-additive 是 comparator。三者共享同一 feature/label/split；
  strict `UtilityModelConfig` 禁止 budget key，Set Transformer 必须显式冻结
  `num_heads/num_layers/dropout`，checkpoint 保存 model config 与 hash。CPU reference trainer 按 trajectory
  均匀训练，tune model selection 按 trajectory-equal objective；label batch validator 要求 Freeze-B
  显式传入 maximum reference-repeat KL，超阈 state fail closed；
- focused suite 为 `204 passed, 15 skipped, 24 subtests passed`；15 个 skip 仅因本机无
  PyTorch。全仓回归为 `1738 passed, 23 skipped, 38 failed, 644 subtests`；38 个失败均来自既有
  lifecycle 互斥测试、sandbox 下的 git worktree 操作和 full-suite import-order 问题，focused
  set-utility suite 无失败；
- 真实 P-1 已从 clean pushed `main@3a058b2` 完成：610/610 shards、88,186,663,372 bytes、
  610 unique paths/LFS SHA；manifest SHA256=
  `e892e7e8f226e9500d978147a9698ad206a70ad9c303ebd918350f9e10ae6c5e`，files-list SHA256=
  `e81e3ba6abe16f5da4d714c434e5e0879a54f747dab74bff531058ba47b978cd`。本路线的新 `D(S)`、
  predictor checkpoint 和 offline delta 仍不存在；
- P0 Execution-A 已绑定 P-1 manifest、consumed ledger、base parser 与 runner，config SHA256=
  `f01beae98432bae19d02f7811d94dc5fa569263b0d917188f2489e15edbf371f`；validator 返回
  `VALID_SET_UTILITY_FULL_POOL_CENSUS_V2_EXECUTION`，并确认 role/query/model/label/training/GPU/closed-loop/test
  counts 全为 0；
- P0 已从 clean pushed `main@0e16bcf` 在 Hyper00 完成：610 shards / 8,146 rows → 6,933
  unconsumed eligible trajectories / 6,928 groups；strata=`1,736/3,601/1,596`，decision range=6–54；
  107 个 consumed identity 全部观测并隔离；census manifest SHA256=
  `729d1e1046761177d53d0f320139331224c9f77f7add5097d04bce479566189b`；
- Freeze-B/A 已从 P0 机械固定 1,200 trajectory / 2,400 query：`train/tune/evaluation=1000/100/100`，
  三个 capacity stratum 各 400 条，source/trajectory/group overlap 全为 0；config SHA256=
  `df8c00bfddda7589e3c9b58cc36f5cbe305ad3fee9c6a8bc4272888a6c648440`，manifest SHA256=
  `144b0de1e66eff1624f6bd10fa6dbebd3d215c6d3d9965d9e9cd3dae295373e5`；后续审计确认它把
  `decision_count` 错作 terminal `decision_step_id`，实际选中倒数第二个 state，因此 v1 query plan 永久
  `INVALID_QUERY_PLAN_TERMINAL_OFF_BY_ONE`；
- Freeze-B v2 terminal-index repair 已在相同 P0/salt/quota 上从 corrected eligible universe 重新物化。
  terminal 固定为 `decision_count+1`，边界 count `6/10/18` 不再被错误排除；v2 仍为 1,200 trajectories /
  2,400 queries，v1/v2 source overlap=1,008，192 出/192 入。config/manifest SHA256 分别为
  `7d7dad8580939be67b56bb7ad5a9e06b771e6d0f25ffd3665da84d14466cbf75` /
  `915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30`；processor 前 exact B2
  schedule 为 171,730 rows，仍未授权 raw/processor/OCR/model/labels/training；
- 当前该 historical milestone 的直接下一步已被后续 processor v1 failure supersede；现行顺序是正式
  selected-image format census → versioned processor repair → candidate freeze/exact operation budget → train-only
  throughput pilot/Execution-B → exact-table production → Set Transformer/DeepSets/pairwise train/eval。完整交接见
  [`set_utility_freeze_b_v2_terminal_index_repair.md`](set_utility_freeze_b_v2_terminal_index_repair.md)、
  [`set_utility_predictor_v1.md`](set_utility_predictor_v1.md) 与
  [`set_utility_implementation_v1.md`](set_utility_implementation_v1.md)。

### 2026-07-18：repeated-selection closed-loop P0 冻结后暂停

- development-only validation-12 的 scientific contract、机械 roster、任意历史 selector/prompt 与五臂 paired
  evaluator 已完成并通过 72 个 P0 tests；与相邻 live-feature/OCR/GUI-Owl/evaluator suite 合跑为
  `127 passed, 28 subtests passed`；
- roster 固定 12 templates × 5 arms，按 short/medium/long 分层，每一步重新选择 memory；manifest SHA256
  `8e68e6eb1adde5627e83ce07cf4e8d7cef4843250a2e1b58baa247af57a07a42`；
- P0 从未访问 live environment、gate checkpoint、GPU 或 sealed split，closed-loop episode count 为 0；
- 路线调整后暂不实现/运行 live runner。协议与代码保留，后续只有 set-utility 离线开发通过新的门槛后才重新
  评估是否启用，见
  [`exploratory_closed_loop_validation12_v1.md`](exploratory_closed_loop_validation12_v1.md)。

### 2026-07-18：confirm-20 oracle-independent failure decomposition 完成

- Source-A=`d3451db74d7bdd415455b87a91718bbedfcd0884`，唯一单文件 Execution-B=
  `9f20e4b55c5e69ae1b1c26d58cd7c52bdd5e0af3`；config/runner SHA256 分别为
  `e522f5ca…cf020` / `88a98b42…b25d`。Hyper00 无 GPU container 只读重放已经消费的 confirm-20 完整
  `D(S)`；不产生 policy/restoration output，不训练 gate；
- `J` 逐字复用既有 budget-conditioned independent target：empty marginal 与其他 singleton-base conditional
  marginals 各占一半，strict-positive top-2，score 同分取较小 event id；
- exact/OCR/`J`/`I` raw utility sum=`0.856431/0.783268/0.765311/0.703385`。`J-OCR` raw
  mean=`-0.000898`、paired 90% interval=`[-0.006370,0.004145]`、support=`10/20`，正式得到
  `CASE_A_ORACLE_INDEPENDENT_LOSES_TO_OCR_RGB`；
- `J-I` raw mean=`+0.003096`、paired 90% interval=`[+0.000477,+0.006533]`，说明 distillation gap 明确存在；
  但修复 student 最多逼近仍低于 OCR 的 `J`，所以 projection 与 student 两层都有损失；
- private HF child report commit=`31aa5e22c08a3d56bc729fcbc88d790cce4249c0`，annotated tag object=
  `a7d6838d32e217162d99a41a7b51939bc2c5d450`。run/独立 validate 均完成 exact-three fresh replay，validate
  remote mutation=`0`、local write=`0`；
- 该分解不是新 GO，不能改变父 `NO_GO_INDEPENDENT_CONFIRM`，也不能解锁 closed-loop、matched-NLL、sealed
  test 或复用 confirm-20 做 v2 holdout。轻量结果见
  [`../data/results/independent_confirm20_failure_decomposition_v1/`](../data/results/independent_confirm20_failure_decomposition_v1/)，完整边界见
  [`independent_confirm_failure_decomposition_v1.md`](independent_confirm_failure_decomposition_v1.md)。

### 2026-07-18：independent confirm-20 continuation 有效 NO-GO

- restoration-only continuation Source-A=`8c0d3aeb22079c77fd66a8aa70861d5e80a141ac`，唯一 direct-child
  Execution-B=`68e71fd6397ee85f758fbb9e7e9332860ddf0466`；config SHA256 为
  `d54988fa05693a42574db2f9ec492d55405bcd2c2e902ccb79897f971c6fc7a9`；
- Hyper00 四张 H200 的 fresh topology envelope SHA256 为
  `0c25b5b052c0050238f0bee7bb47558f750b16774f02733acfb11782c8292aeb`，4 workers × 5 states 完成。formal
  operation counts 为 40 generations、340 teacher forwards、320 KL measurements、320 raw rows，retry/top-up/
  filter 均为 0；
- 20/20 reference coverage、20/20 memory-sensitive、retained baseline mass=`0.730356`、5/5 seed ratio 与 seed
  std checks 通过；但 ensemble/exact raw ratio=`0.821298 < 0.85`。independent raw utility=`0.703385`，低于
  recent=`0.724172`、policy-vision=`0.758562`、OCR/RGB=`0.783268`；相对三者 raw mean delta 分别为
  `-0.001039/-0.002759/-0.003994`；
- 有效 report 因而输出 `NO_GO_INDEPENDENT_CONFIRM`。paired closed-loop、matched-NLL 与 sealed AndroidWorld
  test 没有执行授权；不实施原计划 train-60 controller study；
- canonical private HF dataset 为
  `gavinlaw/causalcache-independent-confirm20-mobile@a0b408e58d629299be334a74ecbd0ec2fa2ed1fc`，tag
  `independent-confirm20-v1`；payload→report 保持 direct-child，annotated tag object 为
  `48921cafa00d7102d8b4c709d58061951f90e42c`，fresh replay 逐字节一致。轻量证据见
  [`../data/results/independent_confirm20_continuation_v1/`](../data/results/independent_confirm20_continuation_v1/)；
- durable `completion.json` 与 HF publication/replay 全部先于 CLI 最后的 stdout serialization。随后
  `json.dumps(MappingProxyType)` 抛出 `TypeError`；这是 post-terminal transport bug。只修 CLI 输出并加回归测试，
  不重跑科学实验、不改变 report 或 verdict；CLI regression file `5 passed`，post-B focused suite
  `78 passed, 2 deselected`。全仓 canonical run 为 1,474 passed / 8 skipped / 37 failed / 620 subtests；36 项
  failure 是已存在 Execution-B 后仍要求 B absent 的历史 Source-A lifecycle tests，1 项是 full-suite import-order
  CPU-only boundary，隔离重放通过；本次没有新 failure。

### 2026-07-18：independent confirm / closed-loop v1 Source-A 冻结

- 主方法固定为 formal-58 的原始 5-seed independent ensemble，不重训、不改 feature/loss/threshold；fresh-16
  只保留已发布 development aggregate，conditional v1 永久保持 `NO_V2_CONDITIONAL_RESCUE` negative ablation；
- machine-readable config SHA256 为
  `33e5c0f856a9074cd5460f0d6f7cdda2108b120d50b208234253ff3136d27b63`，绑定 41 个 Source-A paths 与
  12 个执行 modules。confirm 固定 20 条 decision-6 state、四个候选、`B=2`、完整 16-coalition table、
  raw-utility GO 与 20/20 reference coverage；normalized 两套口径只作 descriptive；
- 执行防火墙要求先封存 independent/三类 heuristic 的 label-blind payload HF commit，才允许 reference 与
  restoration access；report 必须是 payload commit 的 direct child。正式运行前还必须在同一 container、同一
  4×H200 allocation/model/snapshot 上完成 data-blind topology smoke，并在任何 confirm semantic decode、model/HF
  access 前验证 receipt 的 canonical bytes、显式 SHA、GPU UUID、两阶段顺序与 operation counts；
- AndroidWorld live adapter 已冻结 raw transition 到 exact 200d feature：oldest-to-newest、内部 age、排除唯一
  current-equivalent newest event、pinned OCR、canonical action、文本 delta 与 256×256 RGB MAD。paired
  closed-loop 的 ITT instance→template→partition estimator、template-cluster bootstrap，以及 matched-NLL 的
  state→origin→instance→template 聚合、template-level caliper 与 `z` statistic 都已有 fail-closed 纯函数实现；
- focused Source-A suite 为 122 passed。全 `code/tests` 为 1,441 passed / 8 skipped / 620 subtests，剩余 6 项
  分别是 5 个已完成历史 A/B 仍要求旧 runner-freeze 不存在的 lifecycle test，以及一个 full-suite import-order
  下故意拒绝 GPU runtime module 的 CPU-only boundary test；显式 deselect 这 6 个历史/顺序边界 test 后其余
  1,441 项全绿。它们不来自本次实现。在该 Source-A freeze 当时，confirm semantic decode、
  policy generation、restoration forward、HF destination mutation、closed-loop 与 sealed-test access 均为 0；
- 该 historical next step 随后完成了 Execution-B 与一次性 confirm；最终为 `NO_GO_INDEPENDENT_CONFIRM`，
  因此没有实现或运行 Aries emulator + Hyper policy 的 train-60 paired study。

原始 AAAI-27 paper target 曾包括 offline restoration attribution、fixed-budget independent gate、AndroidWorld
closed-loop frontier 与 matched-NLL mechanism test；当前 independent confirm NO-GO 后，后两项在本 v1 中未执行。
v2.1 full-45 因 exact canonical repeat agreement 只有 32/45，正式保持
`NO_GO_V2_1_FULL_45_SUBSTRATE`；bounded spatial audit 随后得到 eager-specific exact-stability recovery，并授权
全新的 v2.2-eager substrate。唯一 v2.2 fresh-45 attempt 已在双 H200 上取得 45/45 parse、45/45 exact repeat、
45/45 finite logits 与 45 个 memory-sensitive states，正式为 `PASS_V2_2_EAGER_FULL_45_SUBSTRATE`；deterministic
USTAR 与 private HF immutable revision 已完成 fresh-download 复核。restoration label v1 因 model snapshot
preflight 缺陷在 0 state / 0 forward 时永久封存为 `INVALID`；replacement v2 已完成 45/45 states，并闭合
420-row raw distance table、45 个 exact-subset oracle、435 条 deployment conditional-marginal labels 与 private
HF immutable artifact。当前已有 offline teacher labels、oracle 上界与 10 个 immutable learned gate checkpoint，
formal model seal 已闭合。fresh-16 parent v1 的 remote-tree failure 与 inventory-repair v1 的 claim-serialization
failure 均永久 `INVALID`；独立 claim-serialization repair 已以全新 A/B、namespace 和 private-HF identity 完成
exact 16/48/144/448 denominator、双 H200 49-batch/97-forward schedule、16-receipt durable chain、9+4 two-commit
publication 与独立 immutable replay。primary result 是有效 scientific `NO-GO`：conditional ensemble 的
normalized recovery / exact 为 `0.694229`，5 seeds 中 `0/5` 达到 `0.75`，seed std=`0.116838`；虽然相对最强
heuristic 的 mean normalized delta=`+0.326111` 且 90% bootstrap lower=`+0.049003`，positive support 只有
`9/16` trajectories。conditional 相对 independent 的 normalized delta=`-0.122394`，仅 `1/16` trajectories、
`1/5` seeds 为正，因此 set-conditioning primary 也 `NO-GO`。matched-NLL、closed-loop、旧 dev-5 与 confirm
仍未执行。随后完成的 read-only failure decomposition 得到 true-greedy / exact=`0.956636` normalized、
`0.975350` raw，说明 search 足够接近 exact；但 oracle `G-J` positive support 只有 `10/16 < 12/16`，冻结
route 为 `NO_V2_CONDITIONAL_RESCUE`。因此停止 set-conditioned main method，但保留并单独确认已有 positive
restoration signal 的 independent selector；旧 conditional post-GO 权限不被继承，新权限完全由上述 versioned
confirm/closed-loop contract 控制。
selector geometry 的 versioned reporting repair 已完成并支持
`set_conditioned_main_candidate + online_greedy_sufficient`；primary `n=4,B=2` OCR/RGB baseline v1 首次
Hyper00 attempt 在零 feature-score 阶段因 identity lexer 错误要求每行字段只出现一次而 `INVALID`。replacement
v2 只修 trajectory=`2` / OCR=`1` 的 exact occurrence profile，已在 Hyper00 完成 15-state formal aggregate、
逐 byte replay 与独立复算。它的 development mean recovery 为 `0.019571`、exact match 为 `0/5`，因此是已闭合
但不稳定的弱 similarity comparator；这不构成 learned gate 的负结论。policy-vision v1 formal attempt 又因
PyTorch UUID object type 在 0 feature 时 `INVALID`。UUID type-only v2 随后通过原失败点，但 pinned
`SizeDict` 不满足冻结的 `Mapping` interface check，再次于 0 feature fail closed。v3 已完成 formal GPU
artifact；首次 CPU byte-replay 的 evaluated-state projection bug 已由独立 versioned CPU audit exact-byte 闭合，
当前是 `VALID_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`，且 result commit 后的 clean-descendant
`validate` 已返回 `REVALIDATED`。
在这一 historical milestone 当时，confirm 和 AndroidWorld sealed test split 仍保持 locked。

48/16 label expansion 的 policy-blind derived artifact 与 192-state substrate 均已闭合：64 trajectories / 192
views / 384 images 绑定 private HF immutable revision `630363a6adb692d72774f16dd0653a50216313ff`；唯一双 H200
substrate 完成 192/192 states、185 memory-sensitive，并绑定 private HF immutable revision
`25ac19cf6ef98adc243d421cd0039ac104ddb539`。expanded exact-label config、coalition input、runner 与 raw
artifact reducer 已由 source A=`2c00c9118dc00cc1bda361325d24d79d8c14f8b6` 冻结，config SHA256 为
`65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5`；runner freeze 已作为单独的
execution B=`bd5cc78838c09a50214b1108fb18f62139c7419e` commit/push，committed validator 返回 GPU
authorization=true。唯一 v1 GPU attempt 随后完成 192/192 states 与全部固定 operation counts；内部 raw reducer
得到 `PASS_V2_2_EXPANSION_EXACT_LABELS_V1`，但 source-locked monitor 有 5/3849 个 sampling gaps 超过冻结
3 秒上限（max 3.883721 s），因此 post-worker validator 将全局 attempt 永久封为
`INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`。invalid evidence 已在独立 private HF immutable revision 闭合；
no-GPU scientific-repair child 也已完成正式 run 和完整只读 revalidation，local payload 为 `VALID +
REVALIDATED`，但原 producer 不重分类。repaired archive+sidecar 已发布到新的 private HF identity，tag-resolved
immutable commit 为 `7a6c254b8cec0dd3d8111dfc9c080de357e5cef3`，幂等 replay 与独立只读 postflight 均通过。
formal-58 label-data prerequisite 已满足，train-only repair cache 与 formal training 均完成 private-HF immutable
replay。Source-A=`e20f004…b9`、唯一 Execution-B=`bad28b7…a2f2` 已闭合 20 个 OOF trials、100 条 fold
tracks 与 10 个 final checkpoint。fresh-16 parent v1 与 inventory-repair v1 的 A/B、失败 evidence 均已封存。
独立 claim-serialization repair 的唯一代码语义变化是让 `LabelAccessClaim.claim` 返回 canonical-JSON deep
snapshot；Source-A=`f0dd53b…0ed1`、Execution-B=`ce523ff…a634`、完成态/immutable replay 与 private-HF tag
均已闭合。轻量结果见
[`../data/results/gate_v1_fresh16_claim_serialization_repair_v1/`](../data/results/gate_v1_fresh16_claim_serialization_repair_v1/)；
fresh-16 已消费，旧 dev-5 与 confirm 在该 historical milestone 当时尚未打开。
父协议与两层 repair 说明分别见
[`gate_v1_fresh16_evaluation.md`](gate_v1_fresh16_evaluation.md) 和
[`gate_v1_fresh16_inventory_repair.md`](gate_v1_fresh16_inventory_repair.md)、
[`gate_v1_fresh16_claim_serialization_repair.md`](gate_v1_fresh16_claim_serialization_repair.md)；旧 repair v1
namespace 不得续跑。

### 2026-07-18：fresh-16 failure-decomposition Source-A freeze

- 新 reporting-only child 将 parent primary 固定拆为 `E-C=(E-G)+(G-C)`；`E/G/J/C/I` 分别是 exact、
  true conditional greedy、oracle budget-conditioned independent、sealed conditional 与 sealed independent。
  `J` 精确复刻 independent raw target，不把 learned `I` 当 oracle teacher；
- 输入面只允许 parent report commit `3541fe1ea2c46e555c29cc53483e6f3b809f8f81` 上的 bundle manifest、
  48-row label table 与 48-row primary state records。raw trajectory、截图/OCR、feature、GUI-Owl、gate checkpoint、
  上游 raw labels、旧 dev-5、confirm、matched-NLL 与 closed-loop 均不可访问；
- config SHA256 为 `fa2cd3759150f838fce78b72a987d7a889ef23f5eec5ec41286ae69090104f1f`；
  Source-A validator 绑定 source blobs 与 transitive reducer dependencies，要求 runner-freeze B absent，且
  network/write/sealed semantic/model/HF mutation 全为 0；
- routing 固定为：`G/E` normalized/raw 均至少 `0.95` 且 n3/n4 raw 各至少 `0.90`；`G-J`
  normalized delta 至少 `0.02`、90% trajectory-bootstrap lower 大于 0、positive support 至少 `12/16`；同时
  `C/G` normalized 严格低于 `0.90` 或 raw 严格低于 `0.95`，才允许一次 conditional v2 rescue；
- focused contract/reducer/runner 为 50/50 passed；全仓为 1,312 passed、16 skipped、5 deselected、617
  subtests passed。5 个 deselect 均为已完成历史 A/B 中只在旧 Source-A 时要求 runner-freeze B absent 的 lifecycle
  tests；其余 suite 全绿；
- source 阶段没有正式 diagnostic result。开发期只读 byte replay 只用于验证 reducer/contract plumbing，不能作为
  canonical result；唯一正式结论必须在 clean pushed Execution-B 上发布 private-HF exact-three 并 immutable
  replay 后形成。fresh-16 已消费，后续不能再把它当 v2 holdout。

### 2026-07-18：fresh-16 failure decomposition 完成并停止 conditional rescue

- Source-A=`718a08b3d1ab78d5c2770e870cefabb2a8667f08`；唯一 direct-child
  Execution-B=`21b775020318f40941bdaa1d9fc9c8f99b1e5c40` 只新增 runner-freeze，SHA256
  `cc43d1cf609e5851543e5a0c3e256d9a6e82441d0bc9a55f1a5a1de8c0009a0c`。clean-pushed validator 绑定
  16-path Source-A inventory SHA256 `0d7eebdafe2600db2f6065f6f619e074318a34a68f1f660d3c10d7b1063cdec4`；
- Hyper00 新 CPU-only container `sglang-omni-jaxan-07181414` 未请求 GPU，容器内 GPU count 与
  `/dev/nvidia*` 均为 0。formal run 只 force-download parent exact-three 中的 3 个 sealed files，解码 48 label
  rows 与 48 primary state rows；raw image/trajectory、OCR、feature/model/checkpoint、policy/teacher forward、
  training、旧 dev-5、confirm、matched-NLL 与 closed-loop operation 全为 0；
- search gate 通过：`G/E` normalized/raw=`0.9566359002 / 0.9753504340`，`n=3/4` raw=`0.9365025844 /
  0.9880672525`。student gap material：`C/G` normalized/raw=`0.7256985651 / 0.9669258712`；
- oracle set-conditioning mean 与 bootstrap lower 虽为正（`G-J=+0.0816837545`，90% lower=`+0.0028158283`），
  positive trajectory support 只有 `10/16`，未达到冻结的 `12/16`。decision tree 因而精确失败于
  `oracle_set_headroom_pass`，输出 `NO_V2_CONDITIONAL_RESCUE`；不能只凭 mean positive 启动 v2；
- denominator-dominated=true、completion share=`0.7803981818`、seed failure=`systemic` 均只作解释，不能删
  small-denominator states 或翻转 parent v1。learned `C-I` normalized delta 仍为 `-0.1223941293`；
- private HF dataset
  [`gavinlaw/causalcache-gate-v1-fresh16-failure-decomposition-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-fresh16-failure-decomposition-mobile)
  exact-three commit=`9aa2540088c575f5c34dfb208e4f429fa53d355b`，annotated-tag object=
  `4914c5add7ea27b89b185e240ea2670abe116473`。formal run 为 `COMPLETED`、3 次预期 remote mutation；独立
  `validate` 为 `REVALIDATED`、remote mutation=0、local write=0，三文件逐 byte 相等；
- fresh-16 已消费且不能作为 v2 holdout。当前不执行 combined-21、matched-NLL、closed-loop 或 confirm；若未来
  研究 restoration-guided independent 模块，必须另立新问题、预注册 contract 与 untouched holdout。轻量结果见
  [`../data/results/gate_v1_fresh16_failure_decomposition_v1/`](../data/results/gate_v1_fresh16_failure_decomposition_v1/)。

### 2026-07-18：fresh-16 claim-repair primary 完成并判定 NO-GO

- Source-A=`f0dd53b9a0249f259a833b4d8ad3ff26096a0ed1`；机械生成的唯一 Execution-B=
  `ce523ff54ebfdc19a9c2bd49ad21548f0934a634`，是 A 的 direct single-parent child，唯一 diff 为 runner-freeze
  JSON。57-path source inventory 与 43-module loaded closure 均通过 clean-pushed validator；
- Hyper00 10 秒 preflight 选中 host GPU 0/1；container `sglang-omni-jaxan-07181130` 为 unprivileged `runc`、
  精确 2 张 H200、冻结 image digest。policy representative window 两张卡都多次达到 100% utilization；
- 第一次 launcher invocation 因 `fresh-download-parent` 尚未预建而在 runtime validation、GPU、fresh semantics、
  scientific state 与 HF mutation 之前 fail closed。保留 bootstrap log/hash 后预建独立 mode-0700 scratch root；
  新 state/artifact roots 仍 absent，随后完成唯一科学 state chain；
- formal run 与独立 immutable `validate` exit code 均为 0；ordinal `0..15` receipts、93-file /
  51,742,977-byte artifact、9-file payload commit `9f0c61b9…c908`、direct-child 4-file report commit
  `3541fe1e…8f81`、annotated tag object `34d5928d…c4339` 全部闭合。private HF tag
  `gate-v1-fresh16-claim-serialization-repair-v1` 已验证解析到 report commit，独立 replay remote mutation=0；
- selector 未 GO：conditional/exact normalized=`0.694229 < 0.80`，single-seed pass=`0/5 < 4/5`，seed
  std=`0.116838 > 0.08`，strongest-positive trajectory=`9/16 < 12/16`。同时 raw/exact=`0.943092`、相对最强
  heuristic mean delta=`+0.326111`、90% bootstrap lower=`+0.049003`，说明 restoration supervision 有 aggregate
  signal，但 frozen v1 student 不够稳定；
- set-conditioning 未 GO：conditional-independent normalized delta=`-0.122394`，bootstrap lower=`-0.363242`，
  positive support 仅 `1/16` trajectories 与 `1/5` seeds；只有接近零的 raw delta 为正，不能支持主张；
- 根据预注册 falsification boundary，停止当前 v1 AAAI main-method story并选择不执行 post-primary
  combined-21；matched-NLL、closed-loop 与 confirm 的 post-GO 权限保持 locked。下一步先冻结 read-only
  failure decomposition，区分 exact→true-greedy search gap 与
  true-greedy→student distillation gap；fresh-16 只能作为已消费 diagnostic split。

### 2026-07-18：fresh-16 claim-serialization repair Source-A freeze

- formal-58 training/model seal 已完成，本 repair 不重训 gate；
- 只修 `mappingproxy` public claim view 为 JSON-safe、non-aliasing deep snapshot；canonical claim payload 不变；
- full-15 derived inventory、4-file consumption、model/labels、16/48/144/448 geometry、evaluation/output/threshold
  全部继承 inventory-repair v1；
- 新 state/artifact/runtime receipt/HF identities 已分配，旧 roots 必须保留；未来 B 读取 token/构造 API 后、
  任何 fresh semantics/new-root/HF mutation 前必须先验证 HF owner/write role，再验证三个 repo/tag absence；
  private 404 只有在 owner/write-role check 通过后才能解释为 absence；
- config SHA256 为 `3979573be235d630ee2f46dc23be8747a843190c9b57ee81e1b3a17b4416d8c7`；57-path source
  inventory SHA256 为 `572992cf5ac75142cdf8f0e82bdfc2caad43ba37632c741eae277285db54d689`；
- focused regression 为 99 passed + 7 subtests passed（新增 repair 为 30 + 7）；全仓为 1263 passed、16 skipped、
  4 deselected、617 subtests passed。4 个 deselect 均为已完成历史 A/B 的 B-absence lifecycle tests；
- 该历史 milestone 当时仍是 source-only：B absent，fresh label/report/HF mutation/GO operation 全为 0；后续
  Execution-B 与完成态见上一节。

### 2026-07-17：正式 gate 数据扩展启动

- 明确旧 10 train / 5 development 只能支持 policy-free method shaping 与 trainer smoke，不能支撑 learned-gate
  泛化 claim；
- 从 frozen 111-trajectory eligible order 中机械排除 8 reference、15 legacy oracle 与 confirm structural prefix
  20，得到唯一 tail-64；前 48 固定为 train expansion，后 16 固定为 fresh development；
- 新增 64-trajectory split contract、materializer、committed validator 和 6 个 focused regression tests；
- 每条固定 steps 4/5/6，新增总量为 192 states、1,792 个完整 $D(S)$ rows、1,856 条 deployment conditional
  edges 与 1,984 次 planned teacher forward；
- Hyper00 持久盘的 16 个 source Parquet 已现场逐文件重算 SHA256，全部匹配 Git source manifest；旧 derived HF
  revision 与新 64 条交集为 0，因此下一步必须生成独立 expansion derived artifact；
- 当前仍为 source-only：新 policy output、restoration label、gate checkpoint、matched-NLL、closed-loop 与 confirm
  access 全为 0。

### 2026-07-17：label-expansion policy-blind derived artifact 闭合

- Hyper00 CPU-only builder 先对 16 个 source Parquet 重算 hash 并重载冻结的 64 rows，再物化 64 trajectories、
  320 shared events、192 decision views、384 images/OCR；
- exact-six artifact tree SHA256 为 `9394b369e2b741e6aacf9ece4fc5dae3e6337b25a7e65402307e4e7862b94abc`，
  总大小 347,902,778 bytes；
- generation、post-write、standalone pre-upload 与 immutable fresh-download 的 OCR aggregate 均为
  `f3423ece941224706406d4bc7e1f1eac7f2f6512616a0368c0cce37a235ec2ff`，每次 384 records；
- private HF `restoration-v2-label-expansion-v1.0.0` 固定到
  `630363a6adb692d72774f16dd0653a50216313ff`；旧 OCR、derived 与 screening tags 仍解析到原 revisions；
- completion 只证明 policy-blind data bytes 与 provenance；policy forward、restoration output、gate training、
  matched-NLL、closed-loop 与 confirm access 仍全部为 0。

### 2026-07-17：192-state substrate source-only freeze

- 固定 64 trajectories / 192 states，双 H200 even/odd 各 96 states；同一 state 的 generation、teacher 与 KL
  只能在同一 worker/device 完成，禁止 stealing、retry、resume、top-up 或 replacement；
- substrate reference schedule 固定为 384 generation、576 teacher forward 与 384 KL；1,792 subset distances、
  1,856 deployment edges 与 1,984 label teacher forwards 只记录 workload，本 contract 不授权执行；
- decision-view loader 强制 canonical source/state/step identity 与严格 history slice；request manifest 必须从实际
  included events 重算 step IDs 和 payload SHA；
- processor-byte canary 对 128 个含 excluded events 的 states 执行 192 次逐 event mutation，同时对 192 个 states
  验证合法 history action 会改变 bytes；只泄漏 step-4 future event 5 的 serializer 会 fail closed；
- source lock 扩展到 63 个 direct/transitive files，覆盖 v2.2 eager prompt/runtime/parser/teacher/KL、model snapshot、
  parent configs/artifacts 与 expansion derived loader；current/future expert target read、semantic consumption 与
  backfill 均固定为 0；
- focused 13/13、全仓 839 tests 通过（12 skip）。本 source-only validator 明确不授权 policy/GPU；下一步必须先
  从 clean pushed source commit 物化 canonical config，再单独冻结实际 runner 与 processor serializer binding；
- source freeze 已推到 `main@6bf3f8c85e4f3be2ba40f1a64c3e2e05542b977d`；随后 clean-main materializer
  生成 `causalcache_restoration_v2_2_expansion_substrate_v1.json`，SHA256
  `42144f33e2473c787b3648c0ace18b22902fa3376615732aef8fd3aa5780a6e5`。validator 重建 96/96 worker
  inventory 与 384/576/384 counts，同时返回 `policy_or_gpu_execution_authorized_by_this_validator=false`。

### 2026-07-17：192-state substrate runner source 闭合

- 实现双进程 `cuda:0/cuda:1` parity runner；每个 state 的 2 generation / 3 teacher / 2 KL 保持同卡完成，
  root/sibling ledger、state marker、runtime、canary、terminal 与 aggregate 均禁止 retry、resume 和 top-up；
- processor canary 使用真实 `apply_chat_template` 与 `_encode_exact_batch`，绑定 rendered prompt、input IDs、
  attention mask、pixel values、image grid 的 bytes/shape/dtype；全局 barrier 要求 192 次 included mutation、
  128 个 excluded states 和 192 次 excluded mutation全部通过后才允许第一个 generation marker；
- runner freeze 固定 parent 63 + canonical base config 1 + reserved execution source 3，共 67 个文件，并分离
  runner source commit A 与包含 freeze 的 execution commit B；formal run 要求 clean `main == origin/main == B`
  且 A 是 B 的 ancestor；
- raw-artifact reducer 从原始 state/kernel/runtime/ledger 重新计算固定 192-state gate，不信任 aggregate headline；
  deterministic USTAR、fresh immutable HF validation、失败 high-water 与 bounded crash window 均已实现；
- 同一 frozen runner 内新增 `monitor-sidecar`：只观察容器内两张可见 GPU，ready、JSONL samples、带 run hash
  的 stop request 与 terminal summary 形成可归档握手；summary GPU UUID 集合必须与两个 worker runtime UUID
  精确相等，monitor 证据不进入 scientific gate；
- expansion 联合 focused tests 32/32 通过；包含 monitor/worker GPU UUID cross-binding 回归后的完整本机 suite
  为 858 tests PASS（12 skip），`compileall` 与 diff check 也在 commit A 前执行。本里程碑仍未执行 policy/GPU，也未生成
  restoration labels、gate checkpoint、matched-NLL、closed-loop 或 confirm output。

### 2026-07-17：192-state label-expansion substrate 与 immutable artifact 闭合

- runner source commit A=`1a3833d6951c768ce1bdd5f976d1044c291d002e`；从 clean pushed A 确定性物化
  67-file freeze（SHA256 `91a768202f653f4e2ca2960adf27a6c6c4f3a288d79c44345a865a1d53010040`），
  execution commit B=`642feb28b4f7ce4e7bf9f7791f7fb0f6919c1839` 只新增 freeze。正式 validator 在 Hyper00
  clean checkout 重建全部 source blobs 并授权唯一双 H200 attempt；
- 全局 processor canary 在任何 generation 前通过 192 included-history mutations、128 excluded states 与
  192 excluded-event mutations。even/odd workers 各完成固定 96 states，192/192 completed、0 failed，
  retry/top-up/forbidden operations 全为 0；run-contract SHA256 为
  `b467113130da74d943a00c2f11d7cad242204e475b7b806e76f666e670a1ed6a`；
- 实际 384 generation、576 teacher forward、384 KL measurement 与计划精确相等；parse coverage、finite-logit
  coverage、exact repeat canonical-action agreement 均为 1.0，mean repeat KL=0，185/192 states 的
  reference-vs-summary KL 超过冻结 epsilon，正式 outcome 为 `PASS_V2_2_LABEL_EXPANSION_SUBSTRATE_V1`；
- source-locked monitor 从 preclaim 前覆盖模型加载、processor-only canary、state 间隙与 forward，共 2,368
  samples、803 个低于 90% 的 incident；进入 state kernel 后外部 10 秒连续窗口多数为 90%--100%。raw evidence
  原样保留全部 incident，不做事后过滤；
- 406-file deterministic USTAR SHA256 为
  `4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d`，tree SHA256 为
  `56f291053121ccf813698a48beb6269b1fa1d096b7e974f6eb9424f55bc45543`。private HF dataset
  `gavinlaw/causalcache-restoration-v2-2-label-expansion-substrate-mobile` 的 tag
  `v2.2-label-expansion-substrate-v1` 绑定 immutable revision
  `25ac19cf6ef98adc243d421cd0039ac104ddb539`；fresh-download 与 source archive 逐 byte 相同，并由 raw
  reducer 再次返回 PASS；
- 轻量 Git 结果位于 `data/results/restoration_v2_2_label_expansion_substrate_v1/`。本步只解锁 expansion
  exact-label source：当前仍无 1,792-row $D(S)$ table、1,856 conditional edges、formal gate、matched-NLL、
  closed-loop 或 confirm output。

### 2026-07-17：expansion exact-label source A 准备完成

- 科学 config
  `code/configs/causalcache_restoration_v2_2_expansion_labels_v1.json` 固定 64 trajectories / 192 states、
  steps 4/5/6 的 $n=2/3/4$ candidates、$B=2$，SHA256 为
  `65f7fa1d35a0b1fdd4fa09fe09120e858252406a3b850d85d9415ab34d6feed5`；
- label run 固定生成 1,792 条完整 power-set raw $D(S)$，并由 policy-free reducer 独立重算 1,856 条
  deployment edges、3,072 条 full-hypercube edges、1,984 条 pair interactions、576 条 exact-permutation
  attributions 与 192 个 primary exact-subset oracles；raw $D(S)$ 是唯一 canonical truth，negative marginals
  和 non-monotonicity 不得 clamp；
- operation schedule 固定 1,984 teacher forwards 与 1,792 GPU KL measurements；generation、expert read、
  gate、matched-NLL、closed-loop、confirm、retry 与 top-up 全部为 0。full-vocabulary logits/probabilities 和 KL
  intermediate 保持 GPU-only，每次只允许返回最终 distance scalar；
- even/odd 双 H200 workers 各 96 states，同一 state 的 reference、repeat 与全部 coalitions 不得拆卡；runner
  durable claim、sibling high-water、monitor handshake、raw validator、deterministic USTAR 与 fresh immutable HF
  validation 已实现；
- 唯一 attempt 固定为 `restoration-v2-2-expansion-exact-labels-v1`，canonical output / ledger / archive 位于
  `/data/experiments/causalcache/`，planned private HF target 是
  `gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-mobile`、tag
  `v2.2-expansion-exact-labels-v1`；
- 本里程碑仍是 source-only。source A commit/push 后必须从 clean A 确定性物化 runner freeze，再以单独的 B
  commit/push 闭合；只有 clean `main == origin/main == B` 上的 committed validator 才能授权唯一 GPU attempt。
  当前 expansion restoration label、gate checkpoint、matched-NLL、closed-loop 与 confirm output 均为 0。

### 2026-07-17：expansion exact-label runner freeze B 闭合

- clean pushed source A=`2c00c9118dc00cc1bda361325d24d79d8c14f8b6` 上的 source validator 重建
  64 trajectories / 192 states、1,792 raw distances、1,856 deployment edges、1,984 teacher forwards 与
  even/odd 96/96 worker schedule，并明确返回 GPU authorization=false；
- 从 A 原子物化 canonical runner freeze，SHA256 为
  `c0447acda3f09bccc65461ed08092fa6b166370721767cf5f35eb19cc59583d6`，随后只把该 freeze 作为
  execution B=`bd5cc78838c09a50214b1108fb18f62139c7419e` commit/push；
- committed validator 证明 A 是 B 的 ancestor、B 等于 canonical `origin/main`、freeze bytes 已提交且
  76-file source inventory 与 A 的 Git blobs 完全一致，状态为
  `VALID_COMMITTED_PUSHED_EXPANSION_EXACT_LABEL_RUNNER_FREEZE`，GPU authorization=true；
- 本里程碑仍没有创建 global attempt ledger、output root 或 policy output，也没有占用 GPU。下一步只允许在
  Hyper00 完成 host/GPU/disk/container 与 10 秒 idle preflight 后启动唯一双 H200 formal run。

### 2026-07-17：expansion exact-label v1 full-forward monitor-cadence INVALID

- Hyper00 在 clean execution head=`bccaab394ccbade6c9abf6281ab4d6e03820a362` 上完成唯一双 H200 attempt；
  even/odd workers 各完成 96/96 states，retry/top-up/forbidden operations 全为 0，1,984 teacher forwards、
  1,792 KL 与 1,792 scalar host transfers 精确命中；
- raw body 含 192 states、1,792 条 $D(S)$、1,856 deployment edges、3,072 full edges、1,984 interactions、
  576 attributions 与 192 exact oracles；repeat KL max/mean 均为 0，internal aggregate 为
  `PASS_V2_2_EXPANSION_EXACT_LABELS_V1`，derived payload SHA256 为
  `9d6481ad2204f5a2a24731b3424448f8ea410a0cee03f544fe1568b67febd8fd`；
- source-locked monitor 从两个 worker 启动前覆盖到终止后，共 3,850 个连续 samples、index 0--3849；冻结
  max-gap 为 3.0 秒，实际 5/3849 intervals 超限，max=3.883721 s、p99=2.292645 s。post-worker execution-
  evidence validator 因而正确 fail closed，外部 ledger 永久为 `INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`；
- 外部 INVALID ledger SHA256 为 `77d3ad318a9c57e78a15edac0bb5273ade9ceeb95858cc6b33c731a6ee66a120`，它声明的
  prior completed ledger SHA256 与 raw root 内 snapshot 的
  `c107f4b77d6dabb4c0123028ee0e5e911751c47f00a938fb108e55af6bfc4b01` 一致；原 root 共 402 files /
  3,579,535 bytes，继续保留在 Hyper00 staging；
- 原 v1 不得 retry/resume、调阈值或追认为 formal PASS；canonical PASS archive/HF repo/tag 均未创建。轻量
  failure binding 位于 `data/results/restoration_v2_2_expansion_exact_labels_v1_attempt/`；
- 下一步先冻结独立 CPU-only child validation repair：保持 producer bytes 与 original INVALID tombstone 不变，
  独立重算 raw labels、operation counts 与 external inputs，并把 cadence failure 永久保留为 provenance。repair
  immutable fresh replay 之前，formal-58 gate、fresh-16、matched-NLL、closed-loop 与 confirm 全部 locked。

### 2026-07-17：invalid-forensic source-only P0 冻结

- 新增独立 protocol `causalcache_restoration_v2_2_expansion_exact_labels_invalid_forensic_v1`，frozen config
  SHA256 为 `5290a51a250e31be4fcf0a970c77ef31c08c92c5892edd19a12ecb14d9d5a6a2`；它只保全
  original `INVALID` bytes，不调用旧 formal PASS packager；
- archive layout 将 402-file producer root 放入 `attempt_root/**`，将 external INVALID global ledger 与两个
  worker high-water ledgers放入 `terminal_external_ledgers/**`，再加 manifest 后固定为 406 regular members；
- validator 强制 external INVALID ledger 的 `claimed_ledger_sha256` 指向 root 内 completed snapshot，并要求
  worker root/sibling/external 三份 ledger逐 byte 相等；
- final execution log 被验证为 162-byte pre-failure prefix 加唯一 130-byte invalidation line；execution evidence
  只绑定 prefix，不能错误要求 append 后 final log hash 相同；
- strict deterministic USTAR、`O_NOFOLLOW + fstat`、pre/post source recollection 与 hard-link no-replace 已由
  10 个 focused tests覆盖；与原 expansion artifact 联合 41/41 tests 通过；
- 本 P0 明确 `formal_label_loader_eligible=false`、`hf_publish_authorized=false`，没有读取真实 Hyper output，
  没有 package/upload。下一步先冻结 crash-recoverable private-HF publication contract，再从 clean pushed source
  执行只读 package 与 fresh replay。

### 2026-07-17：真实 invalid-forensic local archive 闭合

- 从 clean pushed `main@bb73bfee899f436ad292afb5d2d603d706001a5f` 在 Hyper00 只读收集真实 v1 bytes；
  collect、atomic publish 前与 publish/readback 后三次 source snapshot 完全一致；
- 402 个 producer-root files、3 个 external ledgers 与 1 个 manifest 组成 406-member deterministic USTAR，
  archive SHA256 为 `8e205d73196604c0d8d342f9415755b090b96e83555ca55c386b12b8427ca489`，size
  3,880,960 bytes，tree inventory SHA256 为
  `bc481dd77e26764dad3458e91a3e0247ade6049af7adb1744672aa6f2fb437d1`；
- strict reader 从 archive 全量重建 USTAR 并逐 byte 相等；producer status 仍为 `INVALID`，manifest 仍
  `formal_label_loader_eligible=false`，model/GPU/network operations 均为 0；
- 当前 archive 只在 Hyper00 staging，尚不是 reusable SoT。下一步冻结 P1 private-HF publication，完成
  no-overwrite upload/tag、immutable fresh download 与 byte replay；结果见
  `data/results/restoration_v2_2_expansion_exact_labels_invalid_forensic_v1/`。

### 2026-07-17：独立 expansion math audit 闭合

- 新增仅依赖 Python stdlib 的 raw-$D(S)$ audit，不 import production reducer；独立重算 deployment/full edges、
  pair interactions、exact-permutation Shapley、$B=2$ at-most-budget exact oracle、符号/非单调统计与 algebra
  residual；
- comparator 对 reducer 的 math state/summary/count projection 逐字段 exact compare，并对 audit payload 自身做
  canonical hash；1e-15 的 injected marginal drift 会 fail closed；
- 5 个 focused tests 与原 expansion artifact suite 联合 36/36 通过；其中 synthetic fixed denominator 为完整
  192 states。该 audit 将作为 CPU validation repair 的第二套数学实现，不改变原 v1 attempt 的 INVALID 状态。

### 2026-07-17：gate v1 trainer/evaluator source 闭合

- 实现 label-blind signed-hash feature、330/200 维 conditional/independent MLP、层级加权 SmoothL1 + ranking、
  CPU FP32 full-batch AdamW、固定 2 LR × 5 seeds train-only OOF、final refit 与 deployment selector；
- formal roster 机械锁定 source-major step 4/5/6、canonical state/candidate geometry 和冻结五折 digest；
- 新增 frozen ensemble provenance，绑定 gate config、formal-58 feature/label、完整 OOF selection、10 个 checkpoint
  artifact/model-state digest；fresh-16 与 combined-21 必须复用同一 ensemble identity；
- formal evaluation 不接受裸 scorer 或 bootstrap override；10,000 trajectory bootstrap、seed 271828、90% type-7
  interval 与所有 GO threshold 仍由 preregistration 固定；
- 26 个 focused tests 通过，本机仅 PyTorch optimizer smoke 跳过；Hyper00 CPU-only source overlay 上的 deterministic
  two-step smoke 通过，覆盖 coalition interaction、负 target、checkpoint memory roundtrip、conditional rescore 与
  independent one-shot；paper metric、持久 checkpoint、formal label/dev/confirm access 均为 0；
- formal-58 训练仍被 expansion exact-label immutable artifact 阻断，本里程碑不构成 learned-gate 效果证据。

## 已完成里程碑

### 2026-07-14：论文骨架

- 使用 AAAI-27 官方 author kit 建立 anonymous submission LaTeX；
- 编译并逐页检查 3 页论文骨架；
- 明确 Shapley-style attribution 不是 estimator novelty；
- 将 claim 收窄为 decision-time behavioral restoration，并由 closed-loop 验证长期意义；
- 引入 budget-conditioned value、validated teacher、policy-visible context budget 和可为空的选择。

### 2026-07-14：实验契约 v0.1

- 冻结 archive / low-fidelity index / policy-visible context 三层接口；
- 冻结五字段低保真事件 schema 与 executable action canonicalization；
- 冻结 reference validation、near-budget coalition、稳定性指标和 matched-NLL 协议；
- 增加机器可读配置、fixture、验证 CLI 与单元测试。

### 2026-07-14：Synthetic estimator validation

- 实现 shared antithetic permutation 与 variable-cost maximal near-budget coalition；
- 实现 exact permutation enumeration、standard error、positive-value knapsack、Spearman 与 top-budget Jaccard；
- 在 8-event 非线性 synthetic frozen behavior 上验证两个负 restoration-gain event 不会被强制选择；
- 5 seeds 下，$K=4$ 的 mean top-budget Jaccard 为 0.90，$K\ge 8$ 为 1.00；mean standard error 从 2.99（$K=4$）下降到 1.19（$K=32$）；
- exact marginal-score selection 的实际 utility 仅为 global subset optimum 的 85.9%，确认 interaction 会破坏 attribution 的可加性，后续真实实验必须报告 reconstruction error 并保留 budget-aware set loss。

以上结果只验证实现与指标链路，不构成论文效果证据。

### 2026-07-14：GUIOdyssey deterministic pilot artifact

- 从 `cua-lite/GUIOdyssey` 固定 revision 的单个 Parquet shard 提取 upstream GUIOdyssey 成功轨迹 `0054832199799795`；
- 实现 tap、long press、type、swipe、system button 的 executable canonicalization，以及 action 与 terminal signal 分离；
- 生成 10 screenshots、9 events、9 decisions 的 deterministic tar shard，连续两次构建 SHA256 一致；
- 明确拒绝失败轨迹，且 policy-visible manifest 不包含 expert inline reasoning；
- 上传到私有 Hugging Face dataset；当前 canonical revision 为 `1de9c34ff029d4c01665cdaca74436ae24bff276`；
- 该 artifact 只验证真实数据接口，不构成 restoration 或任务成功率证据。

### 2026-07-14：Qwen3-VL real-policy forward smoke

- 固定 `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` 的 14 个加载文件，并校验四个 weight shard 的 LFS SHA256；
- 实现统一的 summary-only、mixed-fidelity、full-history prompt contract 与 JSON executable action parser；
- 在 GUIOdyssey trajectory `0054832199799795` 的 decision step 4 上，三种 fidelity 输入都生成与 recorded action 完全匹配的 `type_text("Cryptocurrency Market")`；
- 输入长度分别为 501、865、1,593 tokens，单张 A6000 peak allocated memory 为 17.71、17.85、18.10 GB；
- 结果见 `data/results/qwen_policy_smoke/`。这只验证 policy forward 链路，不能说明 restoration 有收益。

### 2026-07-14：Qwen3-VL full-history coverage（negative result）

- 在同一成功 trajectory 的全部 9 个 decisions 上运行 full-history deterministic generation；
- schema v0.3 输入下，9 个输出中 8 个满足 executable JSON schema；decision step 4 的 `type_text` 和 step 8 的 `swipe` 与 recorded action 匹配；
- 在 schema v0.2 executable equivalence 下，tap 为 0/7、swipe 为 1/1、type_text 为 1/1，总 coverage 为 2/9（22.2%）；
- 1 个状态输出 unsupported `click` alias，另外 6 个 tap 的 coordinate bin 不匹配；
- 按预注册 validation contract，拒绝 Qwen3-VL-8B-Instruct 作为主 frozen teacher，不放宽标准；
- 结果见 `data/results/qwen_policy_coverage/`。这不是对 CausalCache 机制的 falsification，而是 backbone selection 的负结果。

### 2026-07-14：Executable equivalence contract v0.2

- 将不同 policy grammar 的 swipe/start-end 与 scroll/direction 统一为 viewport content direction；
- 例如手指从屏幕底部向顶部滑动统一为 `scroll:down`；
- 重建 deterministic pilot、更新私有 HF revision，并在相同规则下重跑 Qwen3-VL coverage；
- 该修改在 UI-TARS coverage 前完成，避免为新 candidate 事后放宽验证。

### 2026-07-14：High-fidelity action contract v0.3

- 发现 archived raw tool call 已保存，但 high-fidelity policy prompt 错误地只暴露 coarse canonical target；
- high-fidelity event 现显式包含并暴露原始 action arguments，low-fidelity event 仍只暴露 coordinate bin 或 scroll direction；
- 重建 deterministic pilot、更新私有 HF revision，并最后一次重跑 Qwen3-VL consistency coverage；
- Qwen3-VL coverage 仍为 2/9，因此 rejection 决策不变。

### 2026-07-14：UI-TARS-1.5-7B coverage（negative result）

- 固定并逐文件校验 `ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`；
- 实现 native mobile prompt、Qwen2.5-VL resized-coordinate normalization 和统一 executable parser；
- 在相同 9-decision pilot、相同 full-history validation contract 下得到 9/9 parsed、4/9 executable match；
- tap 为 2/7、swipe 为 1/1、type_text 为 1/1，action-type gate 通过，但 44.4% overall coverage 低于预注册的 50%；
- 不因一个相邻 coordinate-bin miss 事后放宽 equivalence，拒绝该 candidate 进入 attribution pilot；
- 结果见 `data/results/ui_tars_policy_coverage/`。这仍是 backbone selection 的负结果，不是 CausalCache 方法效果。

### 2026-07-14：OpenCUA-7B 接口审计

- 固定 `xlangai/OpenCUA-7B@a2efb7d2b104d477a4a2666a357e79550a28aafc` 与 38 个必要文件；
- 确认 remote `forward()` 返回 token logits，满足后续 teacher-forced component distance 接口；
- 确认官方 PyAutoGUI grammar 可映射到统一 `ExecutableAction`，absolute coordinate 需按 smart-resized image 归一化；
- 记录自定义 1D RoPE、tokenizer/chat template、remote code 与长于官方默认 image history 的复现风险；
- 实现共享 resized-coordinate conversion、OpenCUA prompt、PyAutoGUI parser、pinned runtime 与显式 logits probe，22 个单元测试通过；
- 接口审计通过，只允许在相同预注册 gate 下继续，不据此认定它适合作为主 teacher。
- 第一次 load smoke 在容器预装的 Transformers 5.6.0 上因 remote `tie_weights()` 签名不兼容而失败；依据上游源码固定独立 `transformers==4.53.0` runtime，不对模型代码做运行时补丁。
- 第二次 import smoke 发现 system-site `kernels 0.14.1` 与 Transformers 4.53.0 所需的 Hugging Face Hub 冲突；在同一 venv 内固定兼容的 `kernels==0.11.7`，不修改全局容器包。
- 第三次 load smoke 发现 adapter 错用 Transformers 5.x 的 `dtype=` 参数；固定 4.53.0 runtime 已改为对应的 `torch_dtype=`，其余实验接口不变。
- 第四次 smoke 成功加载 28/28 权重 shard，但 processor 要求 system message 使用 typed content list；消息容器已修正且新增回归测试，prompt 文本不变。
- 第五次 smoke 成功得到 finite logits，并验证 1/3/7-image generation；发现 summary/full 各输出两个 PyAutoGUI call，parser 已收紧为每个 decision 必须恰好一个 executable call，拒绝静默截取。
- canonical smoke 的 summary-only logits shape 为 `[1, 526, 152064]` 且全部 finite；只恢复 event 2 时唯一 action 与 recorded `type_text("Cryptocurrency Market")` 匹配；结果见 `data/results/open_cua_policy_smoke/`，不作为 attribution 效果证据。
- 实现 OpenCUA 9-decision coverage runner，逐步记录 image count、input tokens、raw output、single-action parse、显存和 latency，并直接读取预注册 gate 配置。

### 2026-07-14：OpenCUA-7B coverage（negative result）

- 在相同 9-decision full-history gate 下得到 7/9 parsed、1/9 executable match；
- tap 为 1/7、swipe 为 0/1、type_text 为 0/1，overall gate 与 action-type gate 均失败；
- 3--19 image full-history 输入全部运行完成，最长 5,172 tokens、峰值 allocated GPU memory 18.71 GB，排除 OOM 或短 context 作为主要失败原因；
- 失败集中在 desktop-oriented grounding、multi-action output 和 mobile trajectory action mismatch；
- 拒绝该 candidate 进入 attribution pilot，结果见 `data/results/open_cua_policy_coverage/`。

### 2026-07-14：ShowUI-2B 接口审计与 revision 冻结

- 官方 model card 的 phone navigation action space 明确定义 `INPUT`、`SWIPE`、`TAP`、`ANSWER` 与 `ENTER`，因此不是只能输出点击坐标的 grounding-only 接口；
- 固定 `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`，并记录 11 个 runtime 文件的 size 与 SHA256；
- 固定使用原生 phone prompt、单 dictionary output 与 `[0, 1]` 相对坐标；
- 第一次 smoke 的三种 fidelity 都生成正确 `INPUT` 文本但返回 `position=None`；在 gate 前明确按统一 executor 只消费文本参数，位置非空时才额外校验；
- 修正后的 smoke 返回 `[1, 695, 151936]` finite logits，且 1/3/7-image 三种输入均生成唯一、正确的 `type_text('cryptocurrency market')`；
- 峰值 allocated GPU memory 不超过 4.45 GiB，确认单张 A6000 可运行；
- 下一步在完全相同的预注册 gate 上评估 9 个 full-history decisions。

### 2026-07-14：ShowUI-2B coverage（negative result）

- 在相同 9-decision full-history gate 下得到 9/9 parsed、2/9 executable match；
- tap 为 1/7、swipe 为 0/1、type_text 为 1/1，overall gate 与 action-type gate 均失败；
- 3--19 image 输入全部运行完成，最长 5,305 tokens、峰值 allocated GPU memory 5.01 GiB；
- 后半段多次过早生成 `ANSWER('task complete')`，说明失败来自 behavior coverage 而非 runtime；
- 拒绝该 candidate 进入 attribution pilot，结果见 `data/results/showui_policy_coverage/`；
- 四个预登记 candidate 全部未过门槛，停止在当前 pilot 上继续枚举 backbone，转向 benchmark-native policy/evaluation stack 设计。

### 2026-07-14：AndroidWorld benchmark-native stack 冻结

- 固定 `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc` 作为新的 primary policy candidate；
- 官方 AndroidWorld adapter 报告 69.0% success，并原生使用最近 5 张截图与更早 action text，接口与 mixed-fidelity memory 问题直接对齐；
- 固定 canonical AndroidWorld 与 MobileAgent adapter revisions、模型 14 个 runtime files、SHA256、task hash partition 和 validation gate；
- Aries 已确认 x86_64、Docker 27.2.1 与 `/dev/kvm` 可用；
- 详细执行顺序见 `docs/androidworld_stack.md`。当前只完成 stack preregistration，尚未把 candidate 标记为 accepted。

### 2026-07-14：GUI-Owl native logits/history smoke

- 14 个 pinned snapshot files 已在 Aries 持久盘完成 SHA256 校验；
- 补齐 AndroidWorld `open`、`answer`、`key`、`recents` executable action types，并实现 pinned `mobile_use` prompt/parser；
- 单图与 native 5-image history 均返回 finite logits，并各生成唯一合法 click；
- 5-image 输入为 2,365 tokens，峰值 allocated GPU memory 17.07 GiB；
- 两条输出都与 fixture tap 匹配只视为 incidental，不计入 coverage；
- 结果见 `data/results/gui_owl_native_smoke/`。下一步启动 AndroidWorld emulator/reward smoke。

### 2026-07-14：AndroidWorld container compatibility fix

- pinned MobileAgent Dockerfile 因 `openjdk:18-jdk-slim` 不再可解析而无法原样构建；
- 新增严格、可测试的构建前处理，仅替换为 `eclipse-temurin:17-jdk-jammy`；
- upstream `setup.py` 的未声明 `pkg_resources` 在 uv 隔离构建中引发第二次失败，
  按构建器建议严格切换为 `--no-build-isolation`；
- 源码编译的 Python 3.11 环境不能复用 Ubuntu Python 3.10 的 wheel，因此固定 uv
  `0.11.28`，并预装 `wheel==0.45.1` 与 upstream 已锁定的 `grpcio-tools==1.71.0`；
- 保留 upstream checkout 与 pinned revision 不变，未修改 task、reward、agent 或 prompt。

### 2026-07-14：AndroidWorld environment/reward smoke

- 在 Aries KVM 上构建并启动 Pixel 6 / API 33 / Google APIs x86_64 emulator，读取到 116 个
  AndroidWorld task types；
- 使用 validation seed `271828` 和单组合 suite 初始化 `SystemWifiTurnOn[0]`；
- 4 个状态变更全部通过 `/execute_action` 完成，截图 payload SHA256 发生变化；
- `/task/score` 从 0.0 变为 1.0，随后 `/task/tear_down` 成功，正式 smoke 耗时 44.189 秒；
- Contacts 首次 setup 有权限文案 mismatch warning，后续 validation 必须单独标记 setup failure；
- 结果见 `data/results/androidworld_environment_smoke/`。下一步先提交冻结 task partition manifest，
  之后才启动 GUI-Owl validation rollout。

### 2026-07-14：AndroidWorld task partition 冻结

- 从 live pinned registry 读取并排序 116 个 task types，registry SHA256 为
  `185ae2019706693bd32ecc25ffd0c8f87be87331cae6f7d7e31c91674c962b89`；
- 按预注册 SHA256 bucket rule 得到 train 60 templates / 180 instances、validation 31 / 62、
  test 25 / 75；
- 不对不均匀 split 做事后 rebalance，final test 在 validation gate 通过前保持 sealed；
- manifest 与重建测试见 `code/configs/androidworld_task_partition.json` 和
  `docs/androidworld_task_partition.md`。

### 2026-07-14：AndroidWorld validation execution plan

- 只实例化 validation split，未生成 final-test 动态参数；
- 固定 62 个 goal、template、complexity、home reset flag 与官方 dynamic step budget，instance
  records SHA256 为 `8204e7f832f1d70becd51f299977f0e4a322a080bbfab64010345c48f901b0e8`；
- 62 个实例全部从 home screen 开始，step budget 最小 10、最大 60；
- 发现 upstream wheel 漏装 `task_evals` 子包，按官方 server 相同条件从 Docker `/` 源码根运行，
  未修改 benchmark package；
- 计划见 `code/configs/androidworld_validation_plan.json`，下一步只跑单个 validation instance 的
  GUI-Owl closed-loop smoke。

### 2026-07-14：GUI-Owl 单实例 AndroidWorld closed-loop smoke

- 首次 `SystemWifiTurnOn[0]` 尝试在 policy generation 前暴露 upstream a11y reset failure；保留
  官方 `initialize_task → agent.reset` 顺序，没有为通过 smoke 跳过 reset；
- `ClockStopWatchPausedVerify[0]` 首条模型动作暴露 MobileAgent HTTP server 的 JSONAction schema
  mismatch：旧 schema 拒绝四坐标 swipe；新增 fail-closed build-context patch，与 pinned GUI-Owl
  converter 的 `new_json_action` 对齐；
- 修复 image `sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486`
  已通过独立四坐标 swipe transport smoke；
- 最终 frozen-policy episode 从 reward 0 开始，4/4 outputs parsed，3 个 click 与 1 个
  `status/task_complete` 全部执行成功，policy 明确 done，最终 reward 与官方 success 均为 1；
- 单实例耗时 78.675 秒，最多 4 张可见截图、2,040 input tokens、16.90 GiB peak allocated GPU
  memory；完整 trace 见 `data/results/gui_owl_androidworld_validation_smoke/`；
- 该结果只允许进入完整 62-instance validation rollout，不接受 GUI-Owl 为主 teacher，也不构成
  CausalCache 方法效果证据。

### 2026-07-14：完整 validation 首次启动与 action-alias 修复

- 首次完整 rollout 在 4 个独立 emulator worker、单 A6000、单模型串行 generation 下启动；
- 前 16 个已 checkpoint episode 中，2 个 Clock 实例 official success，另外 14 个在首步被本项目
  bridge 错误拒绝；这些首步都是结构合法的 `action=open_app`，错误不是 policy parse failure；
- pinned MobileAgent converter 同时接受 `open` 与 `open_app` 并映射到 AndroidWorld
  `action_type=open_app`，HTTP executor 也已在环境 smoke 中执行同一 action type；
- rollout 被立即中止，16 个 checkpoint 标记为 implementation-invalid 并从正式 gate 排除；诊断摘要见
  `data/results/gui_owl_androidworld_validation_attempt1/`；
- bridge 已补齐 `open_app` alias 与双 alias regression test。正式 rollout 必须从空 checkpoint 目录
  重启，不能 `--resume` 这次无效尝试。

### 2026-07-14：完整 validation 第二次启动与 native-resolution 审计

- 修复 `open_app` 后从空目录启动 4-worker validation，在 45 个 checkpoint 时得到 12 个
  official success、7 个 infrastructure failure、1 个 parse failure；即使余下 17 个全部成功，
  上界也只有 29/62；
- 在按协议 early-stop 后复核 pinned adapter，发现 runner 强制 256 visual tokens/图，而上游先做
  1080×2400→1092×2408 resize，再由 GUI-Owl processor 产生 grid `[1,150,68]`；merge size 2
  对应 2,550 effective visual tokens/图；
- 同一 pinned processor 对原图和上游 resize 图均返回 `[1,150,68]`，因此 model-default resolution
  是可机器复现的唯一修正，不是基于 success 调参；
- 第二次运行整体标记为 configuration-invalid，不用于拒绝 GUI-Owl，也不改变 validation plan、
  gate threshold、prompt、action equivalence 或模型权重；诊断见
  `data/results/gui_owl_androidworld_validation_attempt2/`；
- runtime 现在显式区分 model-default 与 fixed-token preprocessing，并记录实际 `image_grid_thw`
  和 effective visual token count；
- model-default 1/5-image smoke 随后通过：GUIOdyssey fixture 每图 grid `[1,116,138]`、4,002
  visual tokens，5 图共 20,010 visual tokens / 21,185 input tokens；last-token logits finite，
  generation 是唯一合法 click，5-image generation 峰值 22.77 GiB；
- 该 fixture 的横向截图 token 数高于 AndroidWorld；1080×2400 AndroidWorld processor replay
  仍固定为 `[1,150,68]` / 2,550 tokens。结果见 `data/results/gui_owl_native_resolution_smoke/`。

### 2026-07-14：GUI-Owl 正式 native-resolution validation 判负

- 从空结果目录启动 4-worker、单 A6000、model-default resolution 的 frozen validation；
- 在 47/62 个原子 checkpoint 时得到 15 个 official success、23 个正常终止失败、9 个
  infrastructure exception；496/496 actions parsed；
- 所有 generation 均记录 `[1,150,68]` grid 与每图 2,550 effective visual tokens，关闭了前次
  256-token configuration-invalid 问题；
- 固定 62 条分母下至少需要 31 个成功；剩余 15 条全成功的上界也只有 30/62，因此按预注册规则
  early-stop，并停止 GPU runner 与 utilization monitor；
- 不报告受 worker 完成顺序影响的 15/47 为 benchmark success，只报告固定分母下界 15/62 与
  上界 30/62；test partition 保持 sealed；
- GUI-Owl 正式拒绝为 validated teacher，不升级 AndroidWorld attribution contract，不在该 stack
  上生成 restoration labels 或训练 gate；结果见 `data/results/gui_owl_androidworld_validation/`；
- 47 条 reusable traces 已聚合为单个 gzip JSONL shard 并上传到 private Hugging Face dataset
  `gavinlaw/causalcache-androidworld-validation-mobile@v0.1.0`
  (`3fcca45fffe9842c9fcebbf5c6c27c9540bb1515`)；Git 只保留 README、summary 和 manifest，不重复
  提交逐 episode 小文件。

### 2026-07-14：仓库 SoT 与跨芯片执行结构

- 顶层固定为 `README.md` 总索引、`paper/` LaTeX、`code/` 可执行逻辑、`data/` 小数据、`docs/`
  交接记录；原 package/scripts/tests/configs/requirements 与轻量 results 已机械迁移并保留 Git history；
- `pyproject.toml` 从 `code/` 发现 `causalcache` 与 `scripts` packages；Makefile、tests、active configs、
  README/docs 链接同步更新，并新增 layout regression test；
- 新增 `AGENTS.md`、`code/README.md`、`data/README.md` 与 `docs/execution.md`，固定每步
  verify→HF→docs→commit→push main 的完成条件；
- 实测 Hyper01 为 x86_64、8×H200 且 KVM 可用，但 Docker root `/var/lib/docker` 位于只剩约
  7.7G 的根盘，尚无 AndroidWorld image；因此 H200 先承担 policy/offline 工作，已验证的 Aries
  stack 继续承担 closed-loop MVP；
- Mac `~/hf_key.txt` 只允许通过 stdin 用于单次 HF API 调用，不进入 argv、环境变量、远端持久盘、
  Git 或日志；
- 迁移后 51 个单测、contract validation、synthetic deterministic replay、fresh editable install、30 个
  JSON、全部相对 Markdown links 与 AAAI LaTeX 构建均通过；synthetic summary 的陈旧
  `contract_version` metadata 从 `0.1.0` 对齐到实际 config `0.3.0`，其数值结果不变；
- 本次只改变 repository/execution contract，不改任何历史 experiment semantics 或结果数字。

### 2026-07-14：Replacement teacher v1 预注册

- 本轮唯一 candidate 冻结为
  `mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060`；官方报告
  AndroidWorld 71.6%，但该数字不作为本项目 gate 结果；
- 新 checkpoint 与已接入的 Instruct 版本共享全部 10 个非权重 runtime files，四个 BF16 weight
  shard 的 SHA256 单独固定在 `code/configs/gui_owl_1_5_8b_think_snapshot.json`；
- prompt、native 5-image history、action grammar、model-default visual preprocessing、deterministic
  generation、62-instance validation plan、95% parse gate 与 50% success gate 全部保持不变；
- Hyper01 只承担 standalone logits/parser smoke；Aries 继续承担已验证的 closed-loop stack，未预注册
  跨主机 policy/environment topology；
- 在任何 candidate inference 前发现并修正 smoke 协议错误：原 decision step 4 只会产生
  4-image history，现固定为 step 6 以真正覆盖 5-image 上限；同时显式固定
  `max_new_tokens=256`；
- `UI-Voyager` 未选为主 teacher，因为官方推理只暴露当前截图，加入历史截图会改变其已报告策略接口；
- 首次 smoke 后的当时状态为 `smoke_format_adaptation_pending`，不是 accepted teacher，也不是
  CausalCache 效果证据。

### 2026-07-14：GUI-Owl Think 首次 strict-parser smoke

- Hyper01 单 H200 在 Git `e40a0c780c96dda1d43ac5ae1469ccca86d9887d` 运行，model/data revision、
  Docker digest、完整 argv 与 runtime 已写入
  `data/results/gui_owl_1_5_8b_think_smoke_strict/run_manifest.json`；
- single-image 与 5-image history 的 last-token logits 均 finite，实际 image count 为 1/5，峰值
  显存约 19.15/24.47 GB；
- 两个 raw output 都是一个闭合 `<think>...</think>` prefix，后接一个原生
  `Action + mobile_use <tool_call>`；旧 strict parser 按设计拒绝，parse coverage 0/2；
- 该次运行状态为 `invalid`，不是 candidate rejection；预注册只允许在任何 validation 前
  做一次与 action correctness 无关的 fail-closed format adaptation，并先提交测试。

### 2026-07-14：GUI-Owl Think format-only parser 适配

- 只允许 raw output 开头最多一个小写、闭合的 `<think>...</think>` block；剔除后仍
  full-match 单行 `Action:` 和唯一 `mobile_use` tool call；
- 未闭合、多 block、非前缀位置、suffix 或额外文本全部 fail closed；历史 action description
  与当前 executable parser 复用同一 boundary helper；
- 两个已记录 smoke raw outputs 与人工 malformed cases 均有回归测试；未查看或使用
  AndroidWorld success，prompt、action mapping、equivalence 和 threshold 均未修改；
- 该里程碑结束时状态为 `smoke_rerun_pending`，必须先 push 该 parser commit 才能原参数重跑。

### 2026-07-14：GUI-Owl Think interface smoke 通过

- Hyper01 单 H200 在 Git `72c59c4d5f2f9c4f7eb42758d6606b1736e6121d` 以原 model/data/fixture/
  generation 参数重跑，完整 provenance 见
  `data/results/gui_owl_1_5_8b_think_smoke/run_manifest.json`；
- single-image 和真实 5-image history 的 finite logits 与 parse 均为 2/2，通过预注册
  interface gate；`executable_match` 0/2 仍只作 diagnostic；
- 与首次运行比较，single-image raw output 逐字相同；5-image 只有 Action description 中
  一个句点的引号内/外位置不同，thinking、tool call 与 canonical action 相同；
- 该里程碑结束时状态为 `androidworld_validation_pending`，不是 accepted teacher；只允许进入冻结的
  62-instance validation plan。

### 2026-07-14：AndroidWorld automatic early-stop orchestration

- full runner 新增显式 `--early-stop-when-success-is-mathematically-impossible`，不再依赖
  外部人工观察后终止 GPU process；
- 只在 episode JSON 原子写盘后计算固定分母 success 上界；上界严格低于 gate 时设置
  stop event，已在途 worker 仍完成 score/tear-down；
- early-stopped summary、CLI 输出与 full-run summary 均可正常写入；`--resume` 增加 plan index、
  instance、filename 与完整 run contract 校验，非 resume 运行拒绝非空 output directory；
- 每条 episode 新增 Git/model/runtime/processor/generation/plan/server identity；CLI 的 full Git SHA 必须
  等于 clean checkout HEAD，防止 Think run 静默复用 Instruct checkpoint；
- 62 条、50% gate 的边界测试固定：47 checkpoints/16 successes 不能停，47/15 必须停。

### 2026-07-14：GUI-Owl Think Aries 正式运行 preflight

- 在 Aries `/mnt/data6/jiaxuanluo/causalcache` 准备独立 clean checkout、venv、model cache 与空实验目录；
  preflight checkout 为 Git `aaa19a9efc72b3a48cb05375f19649a8f9e3f210`，正式 runner 启动前会同步
  本里程碑对应的最新 pushed `main` 并把 full SHA 写入每条 episode contract；
- model snapshot 为 `mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060`，
  14/14 pinned files 已核验；runtime 为 Python 3.12.3、PyTorch 2.11.0+cu130、CUDA 13.0、
  Transformers 5.6.0；
- policy container `sglang-omni-jaxan-07141905` 使用 image
  `sha256:81b5df11b32ad8460be270a67066196cb7c6d4fb92cb5d05a44fb06d1ec88d21`，只暴露 Aries
  physical GPU 1（A6000 UUID `GPU-7bf06053-364d-92a4-fdd0-b04b2dd66291`，容器内 `cuda:0`）；
- 首次带 `--privileged` 的空 policy container 会看到全部 8 张 GPU，未加载模型即删除重建；正式
  container 去掉该权限后 `nvidia-smi` 与 PyTorch 均只看到一张卡；
- 四个 pinned AndroidWorld executors 的 `5000–5003/health` 全部通过，server image digest 保持
  `sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486`；Think validation
  output directory 在启动前不存在；
- 为避免提前观察后重复 validation 样本，不再另跑 `ClockStopWatchPausedVerify[0]`。完整 runner 的首个
  原子 checkpoint 同时作为 infrastructure canary，结果不得用于修改冻结协议。

### 2026-07-14：AndroidWorld validation deterministic packaging

- 新增 `scripts.package_androidworld_validation`，对完整或数学确定 early-stop summary、冻结 plan、
  episode instance/index/filename/run contract 与聚合计数做 fail-closed 复核；
- raw episode 按 `plan_index` 写成一个 canonical UTF-8、`mtime=0` 的 deterministic gzip JSONL shard，
  避免把 resumable checkpoints 作为大量小文件上传；
- HF stable repo 内按 `data/<policy-slug>/` 与 `runs/<policy-slug>/` 隔离 policy artifact；payload manifest
  记录目标 repo/tag 和内容 SHA256，但不记录尚未产生的 HF OID，避免 revision 自引用；
- 相同输入双重打包 byte-identical、early-stop 决定性与错误 contract 拒绝均已覆盖；当前全量 65 个
  tests 与 contract validation 通过。该工具在正式 run Git commit 之后实现，只用于事后 artifact
  packaging，不改变已运行的 policy、plan 或 gate。

### 2026-07-14：GUI-Owl Think AndroidWorld validation 判负

- 在 clean Git `36526c58997be5409f65c5976fb35e17aa007ad7`、Aries physical GPU 1、四个 pinned
  AndroidWorld executors 上运行冻结的 62-instance plan；Python 3.12.3、PyTorch 2.11.0+cu130、
  CUDA 13.0、Transformers 5.6.0、BF16、model-default visual resolution、最多 5 images、
  `do_sample=false`、`max_new_tokens=256`；
- 40 个原子 checkpoint 时已有 8 个 success，剩余 22 条即使全部成功也只有 30/62，触发自动停止；
  两个在途 worker 完成 score/tear-down 后，最终 summary 为 42 checkpoints、9 successes、20
  unobserved，上界 29/62（46.77%），低于 31/62 gate；
- 513 model steps 中 512 parsed（99.81%），parse gate 通过；outcomes 为 9 official success、23
  terminal failure、1 parse failure、9 infrastructure failure；9 exceptions 包括 6 个 HTTP 500、2 个
  live-instance mismatch、1 个 nonzero initial reward；
- wall time 6384.83 s；GPU monitor 的 86 个 10 秒 window-average utilization mean/median 为
  84.7%/87%，30 个窗口至少 90%，forward peak 反复为 100%；较低窗口来自 emulator round-trip，未改变
  单 GPU 与冻结 worker topology；
- 相同输入独立打包两次 byte-identical；42 条 raw traces 上传 private HF dataset
  `gavinlaw/causalcache-androidworld-validation-mobile@v0.2.0`
  (`0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`)，上传后 5 个文件 SHA、解压记录数与 plan index
  均重新核验；旧 `v0.1.0` tag 未改写；
- candidate 状态改为 `rejected_by_androidworld_validation_gate`。按预注册规则停止 replacement round，
  test split 保持 sealed，不升级 teacher contract、不生成 restoration labels、不训练 gate。

## 当前 artifact 状态

### 2026-07-14：go/no-go existence diagnostic 预注册

- 在任何 restoration forward 前冻结 `exploratory_oracle_diagnostic_v1`：Qwen exact revision、旧
  GUIOdyssey artifact、decision step 4/8、预算 512/1024、full-vocabulary
  teacher-forced action-path mean KL、全部可行 coalition、固定 baselines 与四分支判据；
- 明确这两个 states 是在 2/9 coverage 后观察到的 matched subset，只能产生当前 representation/policy
  stack 的工程结论或硬负信号，禁止训练 gate、替换 primary teacher gate 或输出论文级 `GO`；
- paper-level go/no-go 仍要求未观察 restoration 的独立多轨迹 manifest，且必须先通过未修改的 50%
  full-history executable-match reference gate；旧 single-trajectory rejection 保留披露；
- compute 改用 Hyper00（Hyper01 有用户任务）：只读审计时 Hyper00 8×H200 均为空、`/data01` 与
  `/data02` 空间充足；正式 GPU forward 前仍须重新运行 10 秒 idle-cleanup preflight，并显式绑定至多
  一张即时空闲 GPU。

### 2026-07-14：pre-forward visual-cost correction

- artifact 的 10 张截图均为 2208×1840；在未运行任何 restoration forward 的前提下，用 pinned
  Transformers 5.6.0 `smart_resize` 静态核验 256-token resize target，实际尺寸为 476×392；
- 按 patch 14、merge 2 核算，每张图是 238 effective visual tokens，每个 restored event 的前后两张图
  成本为 476。原预注册中把 processor target 直接写成 effective cost 512，现已显式纠正；
- 预算 cap 保持 512/1024 不变，因此可行 coalition 仍分别为最多 1/2 个事件，states、distance、baselines、
  threshold 与 forward 数量均未改变。正式 runner 还会逐 coalition 对 `image_grid_thw` 做 fail-closed 复核。

### 2026-07-14：teacher-forced action-path KL runtime

- Qwen runtime 新增 canonical action 无 special-token 编码、严格 causal shift 的
  `prompt + action[:-1]` teacher forcing，以及 full-vocabulary float32 log-prob 输出；
- 每次 forward fail-closed 检查单 batch、logits shape、vocabulary boundary 与 finite 值，并记录 prompt/
  forced token 数、image grid、effective visual tokens、latency 和 peak GPU memory；
- KL helper 同时返回 per-token、sum 和 mean，拒绝 materially negative 或非 finite 输入；纯 CPU 测试覆盖
  shift、special-token rejection、self KL 与已知 Bernoulli KL，尚未产生 GPU 实验结果。

### 2026-07-14：go/no-go deterministic reducer

- 新增纯 CPU reducer：首个 action JSON 的 sorted compact canonicalization、16×16×16 joint RGB
  histogram cosine，以及完整 coalition table 上的 recent、similarity、uniform maximal-random 与 global
  oracle；
- oracle ties 固定按 distance、cost、event ids 排序；global oracle 可选择空 memory，负 restoration 不会
  被强制计为收益；normalized recovery 使用预注册的 `max(D(empty), epsilon)` 分母；
- outcome reducer 要求输入 states 与 config 完全一致，并保证 positive evidence 来自同一个
  memory-sensitive state；非法、重复或非 finite 输入统一输出 `INVALID`。

### 2026-07-14：go/no-go fail-closed runner

- runner 已闭合 full-history executable validation、canonical action tokenization、full/repeat reference、
  36 个 exhaustive mixed-fidelity coalition forwards、实际 visual-token 复核、deterministic baselines、exact
  与 sampled restoration selectors、selected-coalition generation 和最终 outcome reducer；
- attribution 内部把 manifest 的 1-based step ids 显式映射到 estimator 的 contiguous 0-based ids，再映射
  回 result；$K\in\{4,8,16\}$、5 seeds 只读取同一 KL cache，报告 standard error、Spearman、Jaccard 与
  exact-selector utility ratio；
- CLI 要求 clean full Git SHA、dataset SHA、model snapshot repo/revision 与 container image digest，记录完整
  argv、host/GPU、Python/PyTorch/Transformers、时间、latency 与 peak memory；成功不落 full logits，失败原子
  写入一个 `failure.json`；90 项 CPU tests 与 contract validation 通过，尚未启动正式 GPU forward。

### 2026-07-14：go/no-go attempt 1 implementation-invalid

- Hyper00 preflight 重新确认 GPU 0/1 空闲，formal run 只绑定 physical GPU 0；clean Git `527711b`、
  pinned dataset SHA、model snapshot 与 container image digest 全部通过 runner contract；
- decision step 4 full-history generation 通过 executable match，但首个 action-path reference forward
  在 Qwen3-VL 3D RoPE 前 fail closed：追加 12 个 action prefix tokens 后 attention mask 为 2023，未同步
  扩展的 `mm_token_type_ids` 仍为 2011；
- 本次生成 0 个 restoration distances，`valid_for_diagnostic=false`，不解释为方法结果；轻量失败摘要见
  `data/results/go_no_go_diagnostic_v1_attempt1/`；
- 修复同步扩展 `attention_mask=1` 与新文本的 `mm_token_type_ids=0`，未知 aligned tensor 继续拒绝；
  states、预算、distance、baselines 与 threshold 不变，更新 clean main 后从新目录重跑。

### 2026-07-14：go/no-go diagnostic v1 `INCONCLUSIVE_POSITIVE`

- 修复后的 clean Git `2715f31` 在 Hyper00 physical GPU 0 完成 38 个 reference/coalition forwards、2 个
  repeat probes 和 selected-memory generations；两状态 repeat KL=0，全部实际 visual-token accounting 与
  full-history executable validation 通过；
- step 8 summary-only action-path KL 为 0.109727；512-token cap 下 oracle `[1]` recovery 0.526，对比
  recent/similarity `[7]` 0.332、random 0.368；1024 cap 下 oracle/restoration `[1,7]` recovery 0.842，
  对比 recent/similarity `[6,7]` 0.487、random 0.581；
- step 8 的非 recent event 1 exact gain 为正且远高于 epsilon；$K=16$ 五 seeds 全部选择 `[1,7]`，
  min Spearman 0.964、Jaccard/utility ratio 1.0，满足 selection-biased diagnostic 的 positive rule；
- step 4 也 memory-sensitive，但 absolute KL 仅 0.000349；1024 oracle 相对 recent 的 normalized gain 只有
  0.007。全部 selected memories 生成的 action 仍 executable-match，因此没有 action recovery 或 success
  结论；
- canonical compact result 与全部 coalition distances 已写入
  `data/results/go_no_go_diagnostic_v1/`。这两个 states 来自已观察的 matched subset，明确禁止升级为 paper
  `GO`、训练 gate 或改写 rejected teacher decision；
- wall 170.30 s 中记录的 teacher-forced GPU forward 仅 4.09 s；monitor 三个 active 10 秒窗口均为 0%，
  瓶颈是 CPU full-vocabulary KL。独立扩展前必须完成 GPU-side KL/batching，否则 attribution 成本 gate
  不通过。

### 2026-07-14：独立 UI-TARS reference gate 预注册

- 在读取新的 GUIOdyssey source rows 或运行新 policy output 前，冻结
  `code/configs/independent_reference_gate_v1.json`；source pool 为 exact transport revision 的前 16 个
  mobile/use train shards，旧 trajectory `0054832199799795` 显式排除；
- eligibility 只读取 source structure，限制 successful mobile、最后一步 terminal、单 executable action、
  4--12 decisions、安全唯一 source ID、合法图片与原 parser 可表示 action domain；不允许开放式
  `except ValueError: skip`；
- trajectory 用 fixed salt + NUL + source ID 的 SHA256 全序排列；reference/oracle 分别取满足
  8 trajectories/48 decisions/3 apps 与 15 trajectories/60 decisions/3 apps 的最短不重叠前缀，policy
  output 后禁止 top-up；
- UI-TARS revision、256 visual-token target、256 generation cap、10x10 equivalence 与原 interface file
  SHA 全部锁定。reference 仍用原 50% overall + tap/swipe/type_text 每类至少一 match，不新增结果驱动
  threshold；
- Hyper00 正式运行前必须先在旧 9-decision artifact 复现 A6000 的 9/9 parsed、4/9 match boolean vector；
  anchor 不一致则不查看 Hyper00 独立 outputs，转 Aries 执行未修改协议。
- Hyper00 已从 immutable transport revision 下载冻结的 16 个 source shards，共 2,252,923,738 bytes；
  在任何 row decoding 前逐文件计算 size/SHA256 并写入
  `data/manifests/independent_reference_gate_v1_source_files.json`。后续 builder 必须逐项验证，不能接受
  revision 相同但 local bytes 不一致的输入。
- Hyper00 随后用 exact UI-TARS snapshot 在旧 9-decision artifact 运行 behavioral anchor；9/9 parsed、4/9
  match 及逐 step vector 与 Aries A6000 完全相同，`HARDWARE_ANCHOR_PASSED`，因此无需因芯片差异转回
  Aries。compact result 位于 `data/results/ui_tars_hyper00_hardware_anchor/`；
- anchor peak 17.83 GB，generation latency 合计 30.77 s。monitor 两个 active windows 平均只有 21%/18%，
  原因是单样本 variable-history preprocessing + 短 generation；reference gate 保持单卡监督执行，正式
  coalition-scale oracle 前必须先做 batching/concurrency 与 GPU-side KL。
- 新增 v0.4 multi-trajectory builder：逐文件 hash、命名 eligibility exclusion、NFKC app normalization、
  salted shortest-prefix split、image collision/path 检查和 byte-deterministic tar；旧 v0.3 API 不变；
- 新增 formal reference runner：模型加载前验证 clean Git、frozen interface、base 50% gate、HF artifact
  revision/tar/manifest SHA、完整 split denominator、UI-TARS snapshot 实际文件 SHA 与 H200 anchor；合法
  scientific failure 输出 `NO_GO_CURRENT_REFERENCE_STACK`，实现/契约异常原子写 `failure.json`；
- full suite 从 91 增至 106 tests，另通过 contract validation、py_compile 与 whitespace check；
- 在 clean Git `7bd1851` 上从 212 rows 得到 111 eligible trajectories；命名 exclusions 为 length-above 81、
  length-below 1、旧 source 1、invalid executable action 6、terminal failure 12；
- 两个独立 output dirs 均生成同一 manifest SHA
  `3870900442dd0c8f037c9127e0c61c53c9d08c3735164c3c57c857f2c65eb949` 与 163,061,760-byte tar SHA
  `b16bd9c631c3e0ad2576715db6aad72be82ca9dda71612576c5f8b8ab52b0fe2`；230 images、tar 231 members；
- reference 最短前缀为 8 trajectories/75 decisions/14 app labels（tap 58、swipe 2、type_text 9、home 6）；
  disjoint oracle 为 15/132/23（tap 93、swipe 6、type_text 20、home 13）；
- artifact 已上传 private HF
  `gavinlaw/causalcache-guiodyssey-independent-mobile@v0.1.0`
  (`84c9f5a335e9612ccb4bd566f977574f359b2485`)；从 immutable OID force-download manifest/tar 后 size/SHA
  与 pre-upload bytes 完全一致。此时尚未运行任何 independent policy output；下一步为 formal reference
  gate。

### 2026-07-14：独立 UI-TARS reference gate 判负

- 在 clean Git `585fd2aa8061f069008d985552ddaec4bbfd5246`、完整 artifact/model/interface/hardware-anchor
  fail-closed validation 后，于 Hyper00 physical GPU 1 运行冻结的 8 trajectories / 75 decisions reference；
- 69/75 outputs parsed，27/75 executable match（36.0%）；预注册 threshold 至少需要 38/75；tap 21/58、
  swipe 0/2、type_text 5/9，因此 overall 与 required-action 两个 gate 均失败；
- 6 个 parse failure 都是 prompt 允许但 parser 未实现的 `open_app`。这是 v1 action-contract mismatch；即使
  事后把 6 个全部乐观计为正确也只有 33/75（44.0%），且 swipe 仍为 0/2，所以 no-go 对它稳健；
- 逐 decision 复核确认 75 个 `(source_id, decision_step)` 唯一、source list 等于冻结 reference split、旧
  anchor source 已排除、75/75 match flags 可从 raw record 重算；正式输出为 `summary.json`，没有
  `failure.json` 或 incomplete denominator；
- 按 protocol 输出 `NO_GO_CURRENT_REFERENCE_STACK` 并停止；disjoint 15-trajectory / 132-decision oracle
  split 未运行，不生成 restoration labels、不训练 gate、不改 v1 prompt/parser/equivalence/threshold；
- raw per-decision summary 与 monitor 上传 private HF
  `gavinlaw/causalcache-guiodyssey-independent-mobile@reference-gate-v1`
  (`b3e1245c6c6a1723fe2ca3a861148008df39df46`)；四个文件从 immutable revision 强制重下载并核验
  SHA256，轻量结论见 `data/results/independent_reference_gate_v1/`；
- 单 H200 wall 296.75 s、generation latency 合计 185.26 s、peak allocated 18.20 GB；10 个 active monitor
  windows 的平均 window utilization 20.4%、sample max 83%。逐 decision preprocessing/短 generation
  仍需 batching，但科学 no-go 已使本路线不进入 coalition-scale oracle；
- GUIOdyssey source 是 train shards，不能仅凭当前 provenance 排除 UI-TARS 训练数据重叠。任何 v2 必须
  在新 untouched split 前先预注册一致的 `open_app` action contract；这不会回改 v1 结论。

### 2026-07-15：Restoration v2 scientific contract 冻结

- 保留 v1 `NO_GO_CURRENT_REFERENCE_STACK`，不修改原 prompt、parser、threshold 或 negative result；v2
  是 stable self-behavior 的新 estimand，不是降低 expert top-1 gate；
- primary substrate 固定为
  `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`，reference admission
  只检查 restricted action parse、finite logits 和两次 canonical action 一致；expert alignment 只报告，
  不得筛选状态；
- strong low-fidelity event 固定八字段且出现在每个 memory；high fidelity 只增加一张 post-action image，
  不增加 before image、raw action arguments 或额外 action text；
- confirm 固定 decision step 6：events 1--4 为 visual candidates，event 5 的 post-state 等于 current，因而
  summary-only 且不能选择；reference 为四张历史 post-state 加 current，primary capacity `B=2`，允许
  positive-value selector abstain，另做 exact-two ablation；
- teacher forcing 固定 native assistant prefix + nonsemantic Action carrier，并只在 deterministic
  `<tool_call>...</tool_call>` token span 上计算 full-vocabulary mean KL；不复用 full-history 自由文本
  description，避免向 coalition 泄漏 target；
- 旧 15 条 oracle trajectories 的 raw artifact 已读取/打包但无 policy/restoration output，按原顺序拆为
  10 条 label-train、5 条 development；confirm 继续 v1 frozen hash order，从 exact 8+15 后取 20 条
  structurally eligible trajectory，每条一个 step-6 state，禁止 output-driven top-up/filter；
- substrate gate 固定至少 20 states、parse 0.99、finite 1.0、repeat agreement 1.0、memory-sensitive 至少 8；
  confirm 固定 20 states、`K=16`、oracle recovery 0.30、strongest-baseline gain 0.10 + paired 90% bootstrap、
  Spearman 0.80、Jaccard 0.75、utility ratio 0.90；
- v2 action inventory 删除 `key`/`Menu`，接受 `tap`/`open_app` alias，坐标固定 `[0,999]` 且映射进有效像素；
  exhaustive round-trip fixture 100% 通过前禁止 policy output；
- recent/random/OCR+RGB/policy-vision baseline 公式、tie-break 与 source hashes 必须在 confirm 前冻结；
  Spearman/Jaccard/utility ratio 只在 memory-sensitive states 上取 deterministic median，并锁定 ties 和
  degenerate denominator；
- scientific config SHA256 为
  `9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc`；config validator 与回归测试已加入，
  完整说明见 `docs/restoration_v2.md`。这一步没有运行 GPU 或产生新 artifact。

### 2026-07-15：Restoration v2 CPU interface 冻结

- 新增独立 `gui_owl_v2` action/prompt adapter，未修改历史 v1：prompt inventory 与 parser/bridge 精确闭合
  九个 canonical actions，只接受 `tap/open_app` 两个 alias，拒绝 `key/Menu/time/terminate:failure`；
- strict parser 拒绝 missing/extra/wrong-type 参数、duplicate JSON keys、非 finite JSON、thinking 与额外
  prose；canonical target 固定 nonsemantic Action carrier 和 NFKC compact tool call；
- `[0,999]` 到 pixel 使用 integer half-up 等价公式，6 个 extents 的全部 6,000 个 scalar checks 均满足
  endpoints、monotonic 与严格不越界；
- strong LF 八字段 serializer 固定 key order、compact UTF-8 newline、ordered screen-text arrays、32-token
  cap、foreground/executor provenance、screen-change bins，并补齐 swipe direction/displacement 等实现细节；
- post-state-only builder 在 image load 前校验 trajectory/event/decision identity、summary bytes/SHA 与
  current-equivalent event，并对实际 image bytes 重算 SHA 后才 decode；steps 4/5/6 共穷举 28 个
  coalitions（step 6 为 16 个），每个 state 内所有 text blocks byte-identical，image count 始终为
  `1+|S|`，before image、raw source tool call 与额外 action description 从未进入 prompt；
- action fixture 为 14 valid / 23 invalid，CPU parse/canonicalize/reparse/AndroidWorld-payload 全部通过；
  source/fixture/spec 由 `data/manifests/restoration_v2_interfaces.json` 逐文件 hash；
- 真实 pinned AndroidWorld `new_json_action.JSONAction` constructor 和 device-side executor dispatch 尚未
  执行，manifest 分别标为 `pending`；这一步未加载 GUI-Owl、未生成 policy output，也没有新的 HF
  artifact。完整语义见
  `docs/restoration_v2_interfaces.md`。

### 2026-07-15：Pinned AndroidWorld constructor preflight

- 在 Aries 创建 exact CausalCache commit `a9e2afa` 与 MobileAgent revision `11cea575...` 的两个 clean
  detached checkout，没有复用历史 dirty 开发目录；
- 14 个冻结合法 payload 全部通过真实 `android_world.agents.new_json_action.JSONAction(**payload)`；
  module SHA256 为 `14ca00cabf3d5b83e4d55cb683a09a4beccbbc658e21039ca5cf8cef3f543e3f`；
- 结果绑定 interface manifest SHA256 `02744d82...`、runtime image repo digest `6a8f60af...`、完整 argv、
  host/container 与起止时间，见 `data/results/restoration_v2_constructor_preflight/`；
- 该结果不证明 device-side executor dispatch；未加载 policy、未使用 GPU、未生成任何 v2 policy output。

### 2026-07-15：Executor-dispatch formal runner 冻结

- 新增 Aries host-side Docker inspection，绑定 runtime/server container ID、actual image ID、5003→5000
  port mapping、persistent mounts、live health 与四个 executor source hashes；
- formal runner 从每个 native output 重新执行 v2 parser 与 bridge，不直接信 fixture payload；14 个 case
  保持单 worker、固定顺序、每 case 独立 reset、每 case 一次请求且零 retry；
- valid response 必须 exact HTTP 200 JSON echo；另对缺坐标 click 先从 pinned source 记录 constructor-acceptance
  witness，再要求相同 payload 在 live actuation 必须 HTTP 500，并要求失败后 health 与 cleanup reset 正常；
- 独立 reducer 从 raw records 重算 14-case denominator、action-type counts、response binding 和 verdict；
- pre/post inspection 对同一 unique attempt 做时间夹持和 container/source identity 比较；compact
  action/reset/health response 保留 raw UTF-8 body，reducer 重算 bytes/SHA/JSON；大体积 screenshot 只记录
  frozen runner 计算的 digest 与 shape；所有 attempt files exclusive-create，失败不能被同路径成功重跑覆盖；
- 当前仅完成代码与 tests，必须先 commit/push，再在 Aries 正式运行。本步骤没有 policy/GPU output。

### 2026-07-15：Executor-dispatch formal run 通过

- 从已推送 commit `b6e57c2` 的新 clean detached checkout 运行 formal attempt
  `rv2-20260715T101814Z-53016a40`，未复用旧开发目录；
- pre/post inspection 锁定相同 Aries runtime/server container、image、5003→5000 port、mount 与四个 live
  source hashes；dispatch 总耗时 53.39 秒，首帧为 1080x2400；
- frozen parser/bridge 重新生成的 14 个 cases 全部返回 exact HTTP 200 success echo；11 个 AndroidWorld
  action types 的 raw denominator 与 counts 由 reducer 重算；
- 缺坐标 click 的 pinned constructor-acceptance witness 通过，相同 payload 在 live actuation 返回 HTTP 500，
  随后 health 与 cleanup reset 正常；
- canonical offline verdict 为 `PASSED_EXECUTOR_DISPATCH`，summary SHA256
  `61956a457fb15a8fdfd35baf1a9f07c48ab45910a94829d537c9a591b2ffcc39`；四件套见
  `data/results/restoration_v2_executor_dispatch/`；
- 第 4 项 action dependency 已闭合。本次未加载 policy、未使用 GPU、未生成 v2 policy/restoration output。

### 2026-07-15：Restoration-v2 selection/exposure materializer

- 确认 parent HF manifest 虽记录完整 111-pool hash，但 `trajectories` 只含原 8+15 条，不能从 parent tar
  猜出 confirm；正式选择必须重扫 pinned 16 个 Parquet；
- 新增 fail-closed CPU pipeline：先验证 2.25 GB source files，再重建 212 rows / 111 eligible pool，要求
  count、canonical pool SHA、exclusion counts 与原 reference/oracle 顺序全部精确复现；
- confirm 算法固定为排除 exact 8+15、保留 `decision_count>=5`、取首 20 后才检查 app diversity，禁止
  为 diversity top-up；`decision_count>=5` 恰好保证 decision step 6 存在；
- selection manifest 保存完整 pool records、8/15/20 disjoint proof、train/dev/confirm 的 30/15/20 state
  IDs，以及每个 state 的 current、全部已存在 candidate events、紧邻 current-equivalence event 与 action hashes；
- exposure 使用 append-only events 与 reducer，术语明确为 confirm `policy-output untouched`，而不是
  `raw unseen`；已有 v1 UI-TARS output 只覆盖 reference 8 条；
- synthetic/mutation tests 覆盖 step-6 语义、fixed-prefix/no-top-up、pool mutation、overlap 和 exposure
  digest。当前只完成实现；必须从已推送 commit 在 Hyper00 正式生成两个 manifest 后，dependencies 2/3
  才能标为 passed。本步骤未生成 policy/restoration output。

### 2026-07-15：Selection formal attempt 1 被 validator 拒绝

- 从已推送 `main@826b45f` 的 clean detached Hyper00 worktree 重建出 212 rows / 111 eligible pool，builder
  产出 fixed-prefix 20 IDs、30/15/20 state counts；未加载 policy、未使用 GPU；
- 独立 validator 随后拒绝 exposure：canonical JSON 使用 `sort_keys=True`，落盘再读回后 role-object key
  order 改变，而 validator 错把 dict insertion order 当作 expert-access event 的语义；
- 这是 serialization/validator contract bug，不是 selection scientific failure；attempt 路径
  `/data/tmp/restoration-v2-selection-826b45f` 保留且不会覆盖，但不能作为 canonical artifact；
- 修复将 role inventory 改为显式 frozen tuple，并新增“serialize→parse→validate”回归测试。必须先
  commit/push 修复，再从新 commit/new output path 重跑 formal materialization。

### 2026-07-15：Selection formal attempt 2 通过，后因 metadata 不全被 supersede

- 从 pushed `main@ed0706ecb72d9a828f452f7308859f95f9554c55` 的同一 clean detached Hyper00
  worktree，在全新 output path 重跑；source 212 rows、111 eligible count、pool SHA `84d685...`、五类
  exclusion counts 与 frozen 8/15 role order 全部复现；
- 排除 exact 23 条后剩 88 条，其中 84 条满足 `decision_count>=5`；confirm 是 structural order 首 20，
  对应 global indices 23--29、31--43，index 30 因结构条件被跳过，不存在 result-dependent filter；
- confirm 覆盖 29 个 normalized app labels，远高于 frozen minimum 3；8/15/20 union 为 43 且两两不交；
- exact state denominator 为 label-train 30、development 15、confirm 20；65 states 的 current/candidate/
  current-equivalence image 与 validated-action hashes 均已冻结；
- selection SHA256 `13197eedc413717a3f190aa53453c6f34b1db82c57f0b47945d552f38bec74f3`，
  exposure SHA256 `0b1a4dfdb9f23be1a7456f78801f26e0b0800b7f32f011af913257b6919535ae`；
- 独立 validator 通过；第二个空目录的全量重建与第一次 passing build 两文件 byte-identical；canonical
  ID 与科学选择结果有效；但轻量 summary 未保存完整 argv、起止时间与 dtype 声明，因此不作为
  最终 canonical runtime record，也不事后伪造缺失 metadata。

### 2026-07-15：Selection canonical formal run 通过

- 从 pushed `main@30879c09e896a61929c66935f98c21e6c3fc7ff5` 的全新 clean detached Hyper00
  worktree 重跑，formal run 为 `2026-07-15T11:11:29.634935126Z`--`11:11:36.857590859Z`；
- 完整 argv、validator argv、所有 input/config/source hashes、container digest、Python/pyarrow、
  `dtype=not_applicable`、null seed 与 negative output declarations 已记录在 result summary；
- canonical selection SHA256 `292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b`，
  exposure SHA256 `bc1224826e0642c2374f6cfbebd676389d8c17ce6bf8a789f08903740cf97a95`；
- 第二次 rebuild 为 `11:11:52.565730649Z`--`11:12:00.086714732Z`，字节级一致，且两次均通过
  独立 validator；dependencies 2/3 正式闭合。本步骤未加载 policy、未使用 GPU、未生成任何
  v2 policy/restoration output。

### 2026-07-15：OCR/image implementation identity 冻结

- 确认 Hyper00/Aries 都没有现成 OCR runtime 或 weights；选择 CPU-only `RapidOCR==3.8.4` +
  `onnxruntime==1.24.4`，显式 PP-OCRv5 mobile detector + English recognizer，关闭 orientation cls，
  intra/inter-op threads 均为 1；
- Hyper00 persistent venv 已安装 full exact lock 并 `pip check` 通过；det/rec 从 RapidOCR pinned
  ModelScope v3.8.0 URLs 下载，inactive classifier 从 hash-pinned wheel 提取，三个 SHA 与 upstream
  manifest 一致；
- 新增 fail-closed backend config、exact runtime lock、lazy optional-dependency adapter、canonical node/record
  schema、256x256 Pillow bilinear implementation、config/golden validator 与 mutation tests；
- Git golden 已冻结 2x2 RGB 和两行 English text PNG bytes 及 prepared hashes；OCR expected output
  仍必须从已 push commit 在 Hyper00 两个独立进程生成；
- 本里程碑未使用 GPU，未加载 GUI-Owl，未产生 policy/restoration output。HF model
  upload、immutable re-download、source manifest 和 real-screen golden 尚 pending，因此 dependency 5 未闭合。
- 从 pushed `main@0e1dfa1` 的首次 Hyper00 synthetic inspection 完成真实 OCR forward，识别得到
  `Causal Cache / Step 42`，但 serialized package-source evidence 以 basename 为 key，两个不同目录的
  `main.py` 发生覆盖；该 run 已登记为 `INVALID_EVIDENCE_SCHEMA_PACKAGE_PATH_COLLISION`，不回写 expected，
  修复后必须从新的 pushed commit 重跑两个独立进程。
- 修复后从 pushed `main@09e4f6d` 在 Hyper00 两个独立进程运行 `inspect-golden`，UTC brackets 为
  `11:47:47.140669493Z--11:47:48.656088201Z` 与
  `11:47:57.685531727Z--11:47:59.202436027Z`；两份 canonical JSON byte-identical，SHA256 均为
  `6cda74bb4f33708795842c947dac53e380e9f0d4e3debcbf9bb38334a645af44`。expected fields 已回写 fixture，
  但仍必须从包含 expected 的新 pushed commit 跑 `validate-golden` 才能标记 synthetic golden passed。
- pushed `main@b82c3d8` 的两次 `validate-golden` 均通过且输出 byte-identical（SHA256 `e1fee087...5245`），
  但审计发现 fixture 顶层 status 仍含 mutable pending lifecycle。该 status 已改为永久内容描述
  `synthetic_expected_inspection_frozen`；因此 b82c3d8 validation 作为 superseded passing attempt 保留，
  必须再从最终 static-status fixture commit 重跑后才给 synthetic golden 最终 verdict。
- 从 pushed `main@820fa54` 对最终 static fixture 两次运行 `validate-golden`，UTC brackets 为
  `11:53:34.926118232Z--11:53:36.505118659Z` 与
  `11:53:53.276034846Z--11:53:54.879217749Z`；两次均为 `PASSED_OCR_GOLDEN_VALIDATION`，validation JSON
  byte-identical，SHA256 `3f4fde7c...49a6`，最终 fixture SHA256 `8c81feb3...bca6`。synthetic golden passed；
  在该 run 时 HF immutable revision、6-image real-screen golden 和 final manifest 仍 pending。

### 2026-07-15：OCR model artifact immutable-verified

- 已创建 private HF model `gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en`，上传 model card、artifact
  manifest 与 det/rec/inactive-cls 三份 ONNX；tag `v1.0.0` 解析到 full immutable revision
  `0dbc766a73ee88d10d52285d434dbfec58617835`；
- 从该 full revision fresh re-download 6 个文件并逐文件复算 size/SHA256，6/6 与上传前 manifest 一致；
  verified at `2026-07-15T12:01:12Z`，HF CLI `1.23.0`；
- 新增 `data/manifests/restoration_v2_ocr_backend.json` 与离线 `artifact-source` validator，绑定 8 个 Git
  source、HF repo/revision/file inventory、fresh re-download 与 synthetic summary；正式 outcome 为
  `PASSED_OCR_ARTIFACT_SOURCE_VALIDATION`；
- 本步骤不把 model cache、token 或 ONNX 放进 Git；Hyper00 staging 现在只是可重建 cache。未加载 GUI-Owl、
  未生成 policy/restoration output；dependency 5 仍只因 6-image real-screen golden 与完成态 manifest pending。

### 2026-07-15：Real-screen golden source/materializer 冻结

- 在读取任何 real-screen OCR/policy/restoration output 前冻结 17-file source contract；当前 SHA256 为
  `374a38c997a1ee9a715a8cf6ce9b7ca26edc1cf56f503c2d42a97436afac16c5`，source-only validator outcome 为
  `PASSED_REAL_SCREEN_SOURCE_VALIDATION`；
- 从已冻结 selection witnesses 复算 45 screening states / 180 occurrences，其中 candidate post-state 135、
  current 45；按 SHA 去重得到 75 unique images（55 portrait / 20 landscape / 0 square）；confirm 97 unique
  images 与 eligible pool 的 SHA overlap 为 0；
- policy-blind 排序已固定 portrait/landscape 各 3 张 exact SHA/path/dimensions；同 SHA 的全部 occurrence paths
  会逐路径验 bytes，canonical representative 是最小 member path，不因当前数据碰巧一 SHA 一 path 而改变契约；
- 新增 raw Parquet reload、deterministic 5-file HF payload materializer 与独立 validator。validator 重建完整
  75-image pool、重放 6 次 OCR、逐字节比较 canonical USTAR/JSONL/manifest，并输出覆盖 `.gitattributes`、
  `README.md` 和三个 payload 文件的 `artifact_tree_sha256`；canonical USTAR trailer 与 HF extra-file
  mutation 均有回归覆盖；
- 本里程碑只冻结 source/runner，不是 real-screen OCR 结果。下一步先 push clean `main`，再在 Hyper00
  CPU runtime 独立构建两次、验证 5-file tree byte-identical、上传 private HF dataset、按 immutable revision
  fresh re-download 并重放 validator。完成前 dependency 5 仍 pending，且没有 v2 policy/restoration output。

### 2026-07-15：Real-screen OCR golden 与 dependency 5 通过

- 从 pushed `main@dcc6e217b4885cef5f745d987a1ec74e57109717` 的 clean Hyper00 checkout，用 CPU-only
  RapidOCR runtime 两次独立 materialize exact 6-image payload；UTC brackets 为
  `12:56:51.799980862Z--12:57:21.514769385Z` 与 `12:58:23.052897194Z--12:58:53.066616371Z`；
- 两个输出目录的 `.gitattributes`、README、raw-image USTAR、OCR JSONL 与 manifest 5/5 byte-identical，
  complete tree SHA256 `605d6396b0cde84697ff3f2407a3630d7ccdb822f55c2bdae56be82e942a7e25`；
  两次独立 validator 都从 2.25 GB raw Parquet 重建 75-image pool 并重放 6 次 OCR，结果为
  `PASSED_REAL_SCREEN_ARTIFACT_VALIDATION`；
- exact tree 已上传 private HF dataset
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@ocr-real-screen-golden-v1.0.0`，tag 解析到 immutable
  revision `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`；从全新本机 cache 下载后 5/5 hashes 一致，送回
  Hyper00 第三次 raw-source + OCR replay 仍通过；
- 完成态 `data/manifests/restoration_v2_ocr_backend.json` 绑定 14 个 Git source、HF model/dataset 两个
  immutable revisions、11 个远端文件 identity 与
  `data/results/restoration_v2_ocr_backend/real_screen_summary.json`。离线 outcome 仍为
  `PASSED_OCR_ARTIFACT_SOURCE_VALIDATION`，但现报告 `dependency_5_closed=true`；
- confirm images、GUI-Owl policy 与 restoration output 均未使用或生成。OCR dependency 5 已 passed；完整
  derived artifact、baseline implementation/source hashes 与 execution config 仍 pending。

### 2026-07-15：Deterministic baseline 纯公式实现

- 新增 `code/causalcache/restoration_v2_baselines.py`，实现 summary-only、recent、uniform-random exact
  expectation、OCR+RGB similarity 与 frozen-policy vision similarity；所有输入均按四候选 exact coverage
  fail closed；
- random 不采样，也不使用 confirmatory attribution seed；它按 lexicographic 顺序枚举六个 2-of-4 subsets，
  对 runner 提供的六个 normalized recovery 求解析均值；
- OCR+RGB 使用 frozen OCR normalization 后的 token set Jaccard 与 256x256 RGB 16x16x16 joint histogram
  cosine 各 0.5；policy-vision 对 spatial-merger 后的 visual token rows 做 mean-pool、L2-normalize 与 cosine；
- 11 个定向 tests 与 16 个 scientific-contract tests 通过。本步骤未加载 policy、未生成 policy/restoration
  output；policy-vision extractor identity 和完成态 source-hash manifest 仍 pending，因此 dependency 6 尚未
  标为 passed。

### 2026-07-15：Policy-vision extractor 与 dependency 6 通过

- 新增唯一 policy-vision baseline 入口 `code/causalcache/policy/gui_owl_v2_vision.py`；旧 generic
  Python-double vision entry 已删除，避免 runner 在两套数值语义间误选；
- extractor 只调用 Qwen3-VL final main merger `pooler_output`，按逐图 `t*h*w/4` 边界在 accelerator 上做
  BF16→FP32 mean/L2/cosine；明确排除 pre-merger `last_hidden_state` 和 layer 8/16/24 DeepStack outputs；
- 正式调用前必须逐文件验证 GUI-Owl 14-file snapshot（17,545,907,171 bytes）、repo/revision、exact local
  inventory、Transformers 5.6.0 与三份 source SHA。Aries verifier 对实际 snapshot 全部通过；Hyper00/Aries
  source SHA 一致；
- Aries 真实 PyTorch runtime 的 5 个 fake-model Torch tests 全部通过；测试不加载 GUI-Owl，不调用 language
  model、LM head 或 generation，也未读取 confirm score；
- `data/manifests/restoration_v2_baselines.json` 绑定 11 个 Git source 与全部公式/identity，离线 validator
  outcome 为 `PASSED_BASELINE_SOURCE_VALIDATION`；dependency 6 已 passed。

### 2026-07-15：完整 derived builder/validator source 就绪

- 新增 policy-blind full builder、independent validator 与 15 项定向 schema/artifact tests；正式 inventory 固定为
  35 trajectories、175 events、65 states、210 张 `observation-000..005` 与 210 条 uncapped OCR records；
- builder 不再接受裸外部 OCR JSONL，而是验证 exact raw source 后，用 pinned RapidOCR/ONNX Runtime/wheel/model
  直接生成记录；builder post-write 和 standalone validator 都重跑 210 OCR 并逐条 exact compare；
- raw GUIOdyssey tool call 必须与 archived canonical executed action 的 type/target/text/case 全字段一致；
  low-fidelity 保存 OCR multiset delta、screen-change、discard counts、exact serialization/SHA，high-fidelity 只保存
  post-state image pointer；
- formal 路径同时绑定 clean `origin/main` Git revision、v2 contract HF repo、v1 source dataset、selection/exposure、
  OCR model/wheel/package-source/runtime/recognizer identity，并对 nonformal counts、dirty checkout、identity drift、
  action drift、OCR replay drift与 extra files fail closed；
- 本机完整 212 tests 通过（9 项因可选 Pillow/PyTorch runtime 跳过）；Aries pinned Pillow 相关回归 66/66
  通过。独立审计发现的裸 OCR、counts、action、provenance 与 replay blockers 均已修复；尚未运行正式全量
  build，因此 dependency 1 仍 pending，且仍无 policy/restoration output。

### 2026-07-15：完整 derived artifact 与 dependency 1 通过

- exact pushed builder commit `1a01f2323647d092cab67f0531ecb877a4a255de` 在 Hyper00 CPU-only runtime
  完成两次正式 build：UTC `14:26:37.779--14:47:57.486Z` 与
  `14:51:24.170--15:12:43.238Z`；两次均为 35 trajectories / 175 events / 65 states / 210 images /
  210 full OCR records；
- 两次 exact 6-file artifact bytes 完全一致，tree SHA256 为
  `475e6cf2e8ae8d621317896fd6fd14b0cbd8f5cf6feb71da10687b7281d96a6e`，OCR aggregate 为
  `1e04ddbdd2fd6e5fc50206c436f512908b269fc95566476c799a075eb9af7010`；
- artifact 已上传 private HF dataset
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@restoration-v2-derived-v1.0.0`，tag 解析到 immutable
  `89f136abaff797e14fe758a198996e51032a10a6`。repo `main` 当前为 9 files；旧
  `ocr-real-screen-golden-v1.0.0` 仍固定到 `9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`；
- 第一次 `hf download` CLI preflight 同时传入 `--cache-dir` 与 `--local-dir`，在下载前被参数检查拒绝，
  零文件落盘，记为 superseded preflight failure。随后使用新的空目录只投影 exact 6 files，fresh immutable
  download 成功；
- Hyper00 在 UTC `15:20:56.587--15:31:30.534Z` 对该 immutable projection 完成第三次 210-record OCR
  replay，tree/OCR aggregate 与两次 build 完全一致。三次均为 `policy_loaded=false`、
  `policy_output_generated=false`、`restoration_output_generated=false`；dependency 1 已闭合，详细轻量证据见
  `data/results/restoration_v2_derived_artifact/`。

### 2026-07-15：GPU-resident KL 与 deterministic microbatch 源码前置

- 新增 `causalcache.restoration_v2_gpu_kl`：正式 API 只接受同一 CUDA device 上的
  `[B,T,V]` tensor，reference 固定为 normalized FP32 log-probabilities，candidate 为 BF16
  logits 或 normalized FP32 log-probabilities；FP32 `log_softmax`、full-vocabulary KL 和 token mean
  都不离开 GPU；
- cached `[1,T,V]` reference 通过 zero-stride `.expand(B,-1,-1)` 复用，正式返回值是
  GPU 上的 FP32 `[B]`；独立 float64 CPU oracle 只用于审计，等价容差为
  `atol=1e-6, rtol=1e-5`；
- 新增 `causalcache.restoration_v2_batching`：只允许显式 microbatch size 2，按 exact
  `(image_count, sequence_length)` 升序分组，组内按唯一 `input_index` 排序，并固定
  `automatic_oom_fallback=false`；
- Mac 全量 233 tests passed，10 个 optional PyTorch/Pillow runtime tests skipped。本步没有加载
  policy、不使用 GPU，也没有生成 v2 policy/restoration output。Hyper00 CUDA audit、实际
  runtime 与 execution config 仍 pending，因此 dependency 8 未通过。

### 2026-07-15：GUI-Owl v2 runtime 与 CUDA audit runner 源码前置

- 新增 `causalcache.policy.gui_owl_v2_runtime`：完整验证 pinned model snapshot 与 Transformers source，
  固定单卡 BF16、每图 target 2560 effective visual tokens 并记录 actual grid/tokens、native assistant
  prefix/fixed carrier/tool-call action span；batch-1/2 teacher forcing 必须共享同一 decision state 的同一个
  canonical reference action，并只返回 GPU BF16 logits；native generation 固定 deterministic kwargs 并
  strict parse；
- GPU KL 删除 validation path 的四次 `.item()`：finite/normalization/nonnegative/final-finite predicates
  留在 device，invalid example 映射为最终 `NaN` distance；primitive audit 固定
  `validation_scalar_host_reads=0`、`full_tensor_host_transfers=0`；
- 新增 `scripts.audit_restoration_v2_gpu_compute`：从 clean exact Git checkout 绑定显式
  host/GPU/container/runtime identity，并审计 batch-1/float64 CPU、batch-2/two-batch-1、
  logits/log-probs、zero-stride shared reference、invalid-to-NaN 和 single-state microbatch 2/no-OOM；
- Mac 全量 256 tests passed，10 个 optional runtime tests skipped。本步骤未使用 GPU、未实例化或加载
  GUI-Owl，也未生成 v2 policy/restoration output。Hyper00 formal CUDA audit、execution config 与
  readiness validator 仍 pending，因此 dependency 8 未通过。

### 2026-07-15：Hyper00 formal GPU compute audit

- 从已推送 clean detached `main@a0f06f495a78bee43112651040fca6c3d31701ac` 在 Hyper00 单张 H200
  执行 policy-blind synthetic audit；preflight 复核 host/GPU/disk/container，两轮 10 秒采样均为 0%，
  physical GPU 0 被显式选中，container 内只可见 `cuda:0`；
- batch-1 GPU 对两个独立 float64 CPU oracle 的绝对误差为 `6.48e-9`、`3.05e-8`；batch-2 对两次
  batch-1 与 logits 对 pre-normalized log-probs 的误差均为 `0.0`；zero-stride reference storage 共享、
  invalid-to-NaN、validation scalar host reads 0、full-tensor host transfers 0 均通过；
- formal summary SHA256 为 `0b0adbd0...8134`。新增完全独立的 offline validator：不 import audit runner、
  GPU KL 或 batching，而从 run commit Git blobs 重新计算 source hashes，并复核 standard JSON、argv、
  UTC、host/container/GPU identity、数值 equivalence、NaN 与 planner；validation outcome passed；
- 加入 validator 与 negative tests 后，Mac 全量 262 tests passed，10 个 optional runtime tests skipped；
- 本 run `policy_loaded=false`、`policy_output_generated=false`、`restoration_output_generated=false`。
  它只闭合 compute primitive audit，不是 model runtime pass；dependency 8 与 screening 仍 locked。

## Artifact 状态

- Git 代码、配置、论文与轻量测试 fixture：本仓库 `main`；
- GUIOdyssey pilot：私有 Hugging Face dataset `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- GUIOdyssey independent gate：私有 Hugging Face dataset `gavinlaw/causalcache-guiodyssey-independent-mobile@v0.1.0` (`84c9f5a335e9612ccb4bd566f977574f359b2485`)；schema v0.4，immutable re-download verified；
- Rejected policy candidate：上游 Hugging Face model `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；共享机器副本只是可重建 cache；
- Rejected GUI-tuned policy candidate：上游 Hugging Face model `ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`；Aries 副本只是可重建 cache；
- Rejected computer-use policy candidate：上游 Hugging Face model `xlangai/OpenCUA-7B@a2efb7d2b104d477a4a2666a357e79550a28aafc`；Aries snapshot 与 venv 只是可重建 cache；
- Rejected GUI navigation policy candidate：上游 Hugging Face model `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`；Aries snapshot 只是可重建 cache；
- GUI-Owl-1.5-8B-Instruct：上游 Hugging Face model `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；v1 native validation 上界 30/62、未通过 50% success gate；v2 只将同一 checkpoint 用作 stable self-behavior substrate；
- Rejected replacement policy：上游 Hugging Face model `mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060`；native validation 上界 29/62，未通过 50% gate；
- AndroidWorld native validation traces：私有 Hugging Face dataset `gavinlaw/causalcache-androidworld-validation-mobile@v0.2.0` (`0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`)；旧 Instruct artifact 保持在 `v0.1.0`；
- restoration v2 executor evidence：Git `data/results/restoration_v2_executor_dispatch/`，formal verdict
  `PASSED_EXECUTOR_DISPATCH`，14/14 cases，summary SHA256 `61956a45...`；
- restoration v2 exact selection/exposure：Git `data/manifests/restoration_v2_selection.json` 与
  `data/manifests/restoration_v2_exposure.json`，SHA256 分别为 `292c7e52...` / `bc122482...`；20 confirm
  trajectories、45 screening states，formal verdict `PASSED_PREOUTPUT_SELECTION_VALIDATION`；
- independent candidate dataset 已冻结；reference raw record 位于 private HF
  `@reference-gate-v1` (`b3e1245c6c6a1723fe2ca3a861148008df39df46`)；reference 判负，oracle records
  按协议未生成。旧 oracle raw trajectories/images/expert actions 已被 builder 读取和打包，不是 raw unseen；
- restoration v2 derived dataset 的 canonical destination 是 private HF
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`；完整 artifact 已固定在
  `restoration-v2-derived-v1.0.0@89f136abaff797e14fe758a198996e51032a10a6`，旧 real-screen OCR golden
  保持在 `ocr-real-screen-golden-v1.0.0@9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`；
- restoration v2 OCR 三模型的 canonical artifact 是 private HF model
  `gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en@v1.0.0`
  (`0dbc766a73ee88d10d52285d434dbfec58617835`)；6/6 files 已从 immutable revision fresh re-download
  验 hash，Hyper00 `/data/artifacts/causalcache-ocr-ppocrv5-mobile-v1` 只是可重建 cache；
- selection-biased go/no-go compact result：Git `data/results/go_no_go_diagnostic_v1/`；raw 209 KiB debug
  summary 只含可丢弃的 per-token/runtime 展开，canonical distances 与结论已压缩进 Git；
- gate v1 formal selector checkpoint：尚未生成；Source-A 已冻结 private Hugging Face model destination
  `gavinlaw/causalcache-gate-v1-formal58-selector-mobile` 和 tag `gate-v1-formal58-train-v1`，但
  repo/tag/revision/artifact 均已验证为 absent；
- 当前没有仅存在共享机器或本地磁盘上的正式实验 output；Taurus/Aries/Hyper 目录只作为 Git/HF artifact
  的 staging/cache。

## 八项 pre-output dependencies 状态

1. derived artifact immutable HF revision/file hashes：passed，
   `restoration-v2-derived-v1.0.0@89f136abaff797e14fe758a198996e51032a10a6`，tree SHA256
   `475e6cf2...a6e`；
2. exact confirm trajectory/state IDs：passed，selection SHA256 `292c7e52...`；
3. exposure ledger：passed，ledger SHA256 `bc122482...`；
4. restricted prompt/parser/bridge/executor fixture：passed；CPU prompt/parser/bridge、真实 pinned `JSONAction`
   constructor 与 device-side executor dispatch 均有独立 evidence；
5. pinned accessibility/OCR identity：implementation/config/weights SHA、synthetic golden 与 HF immutable
   model revision、6-image real-screen golden、HF dataset immutable re-download 与完成态 manifest passed；
6. baseline specification/source hashes：passed，见 `data/manifests/restoration_v2_baselines.json`；
7. v2 interface source hashes：passed，见 `data/manifests/restoration_v2_interfaces.json`；
8. 引用 scientific-config SHA 的 execution config：passed。GPU KL/microbatch/runtime/
   audit source、Hyper00 formal compute audit 与 real processor audit 均已通过；confirm-safe screening loader、
   readiness validator 与固定 45-state production runner source 也已实现；完成态 config/source-hash
   inventory、readiness manifest 与 clean-Git authorization 全部通过。

GPU-side scalar KL、GUI-Owl runtime 与 coalition microbatch 已 implementation-ready，formal CUDA audit 已
通过。新增 runner 在 readiness 8/8 前不 import policy runtime；readiness 后也先对固定 45 states 的
reference/summary 共 90 个 prompt 做 processor-only shape sweep，任何 context overflow 在首个 forward 前
输出 `INVALID_BEFORE_POLICY_FORWARD`。每个 state 在首次 policy call 前写 exclusive attempt marker；崩溃后
存在 marker 而无 terminal record 时禁止 resume 重跑。合法 scientific failures 只包括 parse、repeat-action
mismatch 与 non-finite distance；contract/runtime error 不得伪装成 `NO_GO_V2_SUBSTRATE`。

### 2026-07-15：Restoration v2 screening execution source 完成

- 新增只读 derived loader，先运行 public artifact validator，再逐 byte 复核 canonical JSONL/USTAR、selection
  witnesses 与 90 个 screening image SHA；API 只暴露 10 条 label-train + 5 条 development、45 states，confirm
  state/image 无法寻址；
- 新增真实 `AutoProcessor` audit：完整 pinned model snapshot/Transformers source hash、deterministic portrait/
  landscape PNG、assistant-prefix/carrier/tool boundary、1-image、5-image 与 nested batch-2 均 fail closed；它只
  调用 `AutoProcessor.from_pretrained`，权重文件只做 SHA 读取，不实例化为 tensors，不 forward/generate；
- 新增 readiness validator：dependency 8 必须同时包含历史 GPU audit/独立 validation、真实 processor summary、
  14 个固定 source roles、clean pushed `main` 与 ancestor Git blobs；公开状态只可能是
  `SCREENING_ALLOWED + CONFIRM_LOCKED`，不能借 dummy `real_processor_*` 文件绕过；
- 新增 45-state runner：每 state 两次 deterministic full-history generation、两次独立 reference teacher
  forward、一次 summary teacher forward；reference FP32 log-probs 与 full-vocabulary KL 留在 GPU，只读取两个
  final distance scalars；global `max(1e-4,10*mean_repeat_kl)` 决定 memory sensitivity；
- parse failure 保留 raw generated text/metadata；OOM、prompt/shape/teacher/kernel error 是 fatal invalid，不能
  自动 fallback 或污染科学 gate；完整本地验证为 309 tests passed；当前最小本机环境另有 10 个
  optional-dependency skips（8 个 Pillow、2 个 PyTorch/GPU），所有既有
  contract/interface/executor/selection/OCR/baseline validators 通过；
- 本里程碑仍是 source-only：没有运行 GUI-Owl policy、没有生成 v2 policy/restoration output、没有创建新的
  HF artifact，dependency 8 仍 pending。

### 2026-07-15：第一次 real processor audit 按设计 fail closed

- clean pushed commit `6535e71d07b841e381f131f471d74467d6bef649` 在 Hyper00 容器
  `6263d8cd...c3ac21` 运行 processor-only audit；完整 17 GB model snapshot SHA 已先通过，随后在任何
  processor case、model weight materialization、forward/generate 或 policy/restoration output 前，以
  `actual processor min/max target or merge size drifted` 终止；
- 根因不是 scientific config 或 pixel target 改变：Transformers `5.6.0` 的 pinned
  `Qwen2VLImageProcessor` 将显式 `min_pixels=max_pixels=2621440` 归一化到
  `image_processor.size.shortest_edge/longest_edge`，并不保留 direct `min_pixels/max_pixels` attributes；
- 审计器现改为严格验证该真实 `SizeDict` 表示、四个 non-edge fields 为 `None`、merge size 2，并把 class/
  module/representation 全部写入 evidence；readiness validator 同步要求这些 exact fields。该失败是
  pre-output superseded attempt，dependency 8 仍 pending，可在新 pushed commit 上安全重跑 processor audit。

### 2026-07-15：real processor audit 正式通过

- clean pushed commit `82442da8193063b59e7b538d321406e26401d393` 在同一 Hyper00 container 完成 formal
  audit；summary SHA256 `7d5ac1bd13ba5def46dfb2ca419d59bb0da1ff970f186e9fd092d4006d8b43b8`，UTC
  bracket `17:38:30.280432Z--17:38:51.584295Z`；
- 完整 14-file / 17,545,907,171-byte model snapshot 与 pinned Transformers source 复核通过；唯一 pretrained
  loader 是 `AutoProcessor.from_pretrained`，model weights 未 materialize 为 tensors，forward/generate、
  policy output、restoration output 均为 false；
- 真实 processor 为 `Qwen3VLProcessor + Qwen2Tokenizer + Qwen2VLImageProcessor`；portrait/landscape grids
  为 `[1,152,68]` / `[1,68,152]`，对齐后每图 2,584 effective visual tokens；1-image/5-image sequence 为
  2,943/13,286，nested batch-2 sequence 为 2,946，无 padding；
- readiness validator 现对实际 tokenizer 3/8/15 token boundaries、全部 grid/sample/tensor inventory、PyTorch/
  Pillow runtime 与 class/module 做 exact equality，不再只检查“内部自洽”；processor 子项已闭合，但
  dependency 8 仍等待 execution config、完整 source inventory 与 readiness manifest。

### 2026-07-15：execution config 完成

- 新增 `code/configs/restoration_v2_execution_hyper00_v1.json`，SHA256
  `f2b6521ed8b1d65d4b6170c94c5c4cf8e46d135e352bdb52b21bf4e8f64173a5`；
- exact 8-dependency evidence inventory、14 个 source roles/current hashes、scientific config SHA、derived HF
  identity、GUI-Owl snapshot、Hyper00/H200/container runtime、GPU KL dtype/host-transfer contract 与
  microbatch=2/no-fallback 全部通过 `_validate_execution_config`；
- canonical policy 额外绑定 processor summary SHA 与真实 2,584-token geometry、1/5-image/nested batch-2
  sequence lengths 和 3/8/15 teacher boundaries，防止把 2,560 构造 target 误报成实测 accounting；
- config 仍声明 `CONFIRM_LOCKED` 且不授权 inference；dependency 8 的最终公开状态仍等待后续 clean pushed
  commit 中的 readiness manifest 与 Git ancestry/source-blob validation。

### 2026-07-15：readiness manifest 物化，等待 clean authorization

- 新增 `data/manifests/restoration_v2_readiness.json`，绑定 config SHA `f2b6521e...73a5`、implementation
  commit `a2aeb7f1930d40cb569c8e2adfc0dc39e950d131` 与同一 14-source inventory；
- manifest 结构、8/8 count、config/source hashes、canonical remote/main、pre-output declarations 与
  `SCREENING_ALLOWED + CONFIRM_LOCKED` 均已离线通过；confirm role 仍明确 forbidden；
- 当前工作树包含未提交 manifest，因此正式 validator 按设计尚不能通过 clean-Git gate。下一步先
  commit/push，再运行 CLI；在该 CLI 返回前仍不允许 policy import/output。

### 2026-07-15：dependency 8 正式闭合

- manifest commit `429c4584ba7c18eee0b96741b6c1514bd4d4d7ec` push 后，正式
  `scripts.validate_restoration_v2_readiness` 在 clean `HEAD == origin/main` 上返回
  `SCREENING_ALLOWED + CONFIRM_LOCKED`，8/8 dependencies passed；
- summary SHA256 `20b9e81050e56622cfe9bfe4dd1343f33513ae7f37fe25716d13b86ea8115964`；Git binding
  同时记录 implementation commit `a2aeb7f...d131` 与 current/origin commit `429c458...d7ec`；
- validator 未 import policy，policy/restoration output 均为 false。dependency 8 正式闭合，但授权范围仅为
  10 条 label-train + 5 条 development 的固定 45-state screening；`v2_confirm_primary` 继续 locked。

### 2026-07-15：screening preflight 触发 GPU-2 runtime re-anchor

- 10 秒 preflight 发现旧 config 锁定的 physical GPU 0 正被其他任务占用 112+ GiB 显存、86%--87%
  utilization；未共享该 GPU，未启动 screening；
- preflight 选择空闲 physical GPU 2。首次新容器因 `--privileged` 使 PyTorch 看见 8 卡，在任何审计/policy
  load 前删除；完成态 non-privileged 容器 `69f2b174...194df` 严格只见 `cuda:0`；
- 新 H200 UUID `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81` 上 formal GPU audit、独立 validator 与 real
  processor audit 全部通过，SHA256 分别为 `dce79769...9dac`、`ad53e186...2a6`、`69bb8ddb...d119`；
- 全程没有 policy/restoration output。旧 GPU-0 readiness 保留为历史 pass；planned GPU-2 screening 暂停，
  直到 canonical evidence、execution config 与 readiness manifest 重新绑定并正式授权。

### 2026-07-15：GPU-2 canonical transition 已锁定，等待 re-sign

- 将 re-anchor 的 GPU summary、independent validation、real processor summary 升级为 canonical evidence；
  SHA256 分别为 `dce79769...9dac`、`ad53e186...2a6`、`69bb8ddb...d119`，三份均与保留在
  `restoration_v2_runtime_reanchor/` 的原始副本 byte-identical；
- execution config 改绑新 container `69f2b174...194df`、GPU UUID
  `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81` 与新 evidence，14-source inventory 同步更新；完成态 config
  SHA256 为 `819cb973...91ca0`；
- production runner 新增 pre-import live runtime guard：Torch 与 `nvidia-smi` UUID 必须直接一致，并逐字段核对
  visible GPU count、driver、compute capability、SM count、Python、PyTorch/CUDA/cuDNN、Transformers；GPU 或
  software drift 在 artifact/model load 前 fail closed；
- readiness manifest 暂时设为 `SCREENING_LOCKED_RUNTIME_REANCHOR_PENDING_RESIGN` 且 implementation commit 为
  `null`，避免 GPU-0 历史 authorization 被误用于 GPU-2；focused GPU/readiness/runner tests 41/41、全量
  tests 314/314 通过，另有 10 个 optional-dependency skips；formal readiness CLI 按预期以 locked-state error
  非零退出；
- 本里程碑仍未 import policy runtime、未加载 model、未生成 policy/restoration output。下一 commit 只负责把
  manifest 绑定本 transition 的 clean pushed commit，不再修改 execution config 或 14-source files。

### 2026-07-15：GPU-2 readiness manifest 已 re-sign，等待 formal CLI

- transition commit `14faaa44cf1b2044b1f1bcb3c9dcfce36eb452aa` 已 push 到 canonical `main`；该 commit
  包含 exact GPU-2 config、14-source inventory、canonical evidence 与 live runtime guard；
- readiness manifest 恢复 `SCREENING_ALLOWED` schema，`implementation_git_commit` 精确绑定 `14faaa4...452aa`，
  config SHA 仍为 `819cb973...91ca0`，confirm 仍为 `CONFIRM_LOCKED`；
- 本步骤只修改 manifest、tests 与 docs，没有修改 execution config 或 14-source files，也没有 policy/model
  import/output。manifest commit/push 后必须在 clean `HEAD == origin/main` 运行 formal CLI；在此之前操作上仍
  locked。

### 2026-07-15：GPU-2 dependency 8 正式重新闭合

- re-sign commit `caa4f376026d13acd21db1f88b893bc92dd482b1` push 后，formal readiness CLI 在 clean
  `HEAD == origin/main` 返回 8/8、`SCREENING_ALLOWED + CONFIRM_LOCKED`；
- Git binding 记录 implementation commit `14faaa4...452aa` 与 validation/current/origin commit
  `caa4f37...82b1`，execution config SHA 仍为 `819cb973...91ca0`；
- GPU-2 canonical readiness summary SHA256 为 `36bd183f...aa02`；GPU-0 首次 summary 以
  `gpu0-summary.json`/`20b9e810...5964` 归档，不再授权当前 runtime；
- validator 再次明确 `policy_imported_by_validator=false`、policy/restoration output 均为 false。现在只允许
  固定 45-state development screening，confirm 继续 locked。

### 2026-07-15：第一次 fixed 45-state substrate screening 合法判负

- 从 clean pushed `main@a0001cbc4d4c3e2584ae3bbcff217ce664e5e064` 在 Hyper00 non-privileged container
  `69f2b174...194df`、单张 physical GPU 2（H200，`GPU-e19275bf...66b81`）运行；live runtime guard、derived
  artifact validation 与全部 90 个 processor-only prompt shapes 先通过，sequence length 范围
  3,245--14,499，未超过固定 context；
- 45 个固定 state（30 label-train + 15 development）各生成一次，共 1,943 tokens；45/45 都在
  `reference_generation_1` 触发 `GUIOwlV2GenerationParseError`，因此正式 aggregate 为 strict parse 0/45、
  `NO_GO_V2_SUBSTRATE`。没有 OOM、resume、state retry、top-up、sample mutation 或 confirm access；
- parse 之后未进入任何 teacher forward，故 finite-logit coverage、repeat agreement 与 memory-sensitive count
  的 0 值都是未测量下游量，不是 logits/non-sensitivity 证据；KL measurement 与 restoration label 均为 0；
- raw run 的 45 native outputs、45 state records、45 attempt markers、run manifest 与 log 已打成单个
  deterministic tar，上传 private HF
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@restoration-v2-substrate-screening-v1.0.0`
  (`c073e143b935a79befd8ab1fd7123796792efad8`) 并 fresh re-download 验 hash；Git 只保存轻量 summary 与
  artifact manifest；
- post-hoc format inventory 发现 45/45 都有 exact `Action:` + `<tool_call>` prefix，43/45 条 output 的
  单动作 first balanced JSON 可以 canonicalize，但这只是宽松 diagnostic；在拒绝多 action、截断、第二 JSON 与额外 observation 后，保守的 envelope-only
  normalization 上界只有 40/45，低于 frozen 0.99 gate（45 states 必须 45/45）。它不 retroactively 改写
  本次 0/45；confirm 继续 locked。

### 2026-07-15：immutable parser compatibility replay source 已实现

- 新增 versioned、format-only compatibility parser；原 strict parser 总是先运行，仅允许历史 trace 中明确定义的
  `Action:` envelope、四种 wrapper 与五种安全 suffix。它不补 `}`、不取多动作的第一个、不接受第二 JSON 或
  observation，并继续复用 frozen action canonicalizer 与 AndroidWorld bridge；
- 新增 immutable tar replay 与 clean-main CLI：绑定 archive SHA、exact 96-member order、run contract、source
  run commit、HF immutable revision、原 aggregate 和 45 个 no-retry failure records；archive 从已哈希 bytes
  解析，避免 path reopen TOCTOU；输出只保留 raw-output SHA，不复制 raw text；
- 独立 synthetic 45-state archive 覆盖预期 `0 strict / 43 single-action canonical-first / 40 conservative`
  计数、gate-pass outcome derivation，以及 SHA、member-order、official-result drift、dirty/remote Git、committed
  blob、exclusive write 和 non-finite JSON 的 fail-closed 路径；
- pre-registered golden manifest 进一步固定 45-record canonical classification SHA 和五个 rejection 的
  state/output hashes。40 个 accepted output 中仅 25 个是 clean EOF；其余 15 个分别丢弃重复 opener 8、extra
  brace 4、extra brace + opener 3，model-emitted canonical closer 为 0，因而必须称为 suffix recovery 而不是
  native well-formed parse；
- 本里程碑只冻结 auditor source，不把本地直接调用得到的计数当作正式结果。必须先 commit/push，再从 clean
  `main` 对 HF immutable archive 运行 formal CLI，随后才允许把 replay summary 写入 Git；全量 tests 为
  327 passed、10 optional-dependency skips。

### 2026-07-15：immutable parser compatibility replay 正式判负

- source commit `fc3adf13d48bb016014f7efa62bd27c8a4d12f49` push 后，从 clean `main` 运行 formal
  CLI；审计前后均确认 `HEAD == origin/main == GitHub remote main`，实际 import 的四个 source 与 committed
  blobs 完全一致；
- 已提交 golden contract、HF immutable revision、archive SHA、run contract/source commit、96-member tar
  inventory、原 0/45 aggregate、45-record classification SHA 与五个 rejection identities 全部通过；
- 正式结果为 strict 0/45、single-action canonical-first 43/45、conservative recovery 40/45；后者低于
  frozen 0.99 gate 对应的 required 45/45，因此输出 `NO_GO_ADAPTER_ONLY`；
- 40 个 accepted 中 clean EOF 25、suffix recovery 15，model-emitted canonical closer 0；该证据排除
  “只换 parser 就能继续 v2”的路线，但没有否定 restoration hypothesis；
- replay 没有 raw-text Git exposure、model import/load、forward/generation、retry、top-up 或 confirm access。
  完整 formal result SHA256 为 `78272ee5...a6e0`，见
  `data/results/restoration_v2_parser_compatibility/`；结果回归加入 committed-result/golden binding 后，全量
  tests 为 328 passed、10 optional-dependency skips。

### 2026-07-15：Restoration v2.1 official-tool interface 与 pilot contract 已冻结

- v2 的 `NO_GO_V2_SUBSTRATE` 和 parser replay 的 `NO_GO_ADAPTER_ONLY` 保持不变；v2.1 明确是新的
  policy-interface reset，不是对旧 raw output 的 relabel 或 compatibility adapter；
- 新增独立 official `tools=` schema、two-message prompt、single-line whole-output parser 和 AndroidWorld
  bridge round-trip。assistant 不再输出 `Action:` carrier；alias、CRLF、多行 JSON、duplicate key、非 finite
  number、非 NFKC text、truncated JSON、第二 call/JSON、observation 与任何前后文本均 fail closed；
- 新 runtime 固定 pinned chat-template file/text SHA、assistant/opener/closer/EOS/pad token IDs；generation
  使用 closer token `151658` 作为唯一 EOS并 suppression 标准 EOS，greedy batch-1、256-token cap、
  `skip_special_tokens=false`。没有 model closer 就按 truncation 失败，不做 host completion；
- teacher distance span 从 opener 到 model closer inclusive，且 reference/candidate teacher logits 对 generation
  forbidden EOS columns 应用相同的 GPU-resident、BF16 finite-min mask，避免 generation 与 KL 的策略支持
  不一致；
- machine-readable contract
  `code/configs/causalcache_restoration_v2_1_pilot.json` SHA256 为
  `9d51a2ed5d6cc382f297c1b8af3100d784090f72800d637136b88982763fdbf7`。它固定 processor-only
  45-state × 2-fidelity 共 90 prompts，随后只对 15 个 `v2_development` states 各生成一次；
- pilot 通过条件是 15/15 exact parse、15/15 model-emitted closer、15/15 AndroidWorld bridge，且所有
  truncation/extra-output/retry/top-up 计数为 0。pilot 明确禁止 teacher、KL、restoration、expert、label-train
  policy output，也不会向 decoder 暴露任何 confirm state/prompt/image；pass 也只授权 unchanged-interface
  full-45 substrate，不授权 confirm generation；
- 本里程碑是 source-only：validator 返回 processor preflight pending 且不授权 generation；尚未加载 v2.1
  model、forward、generate 或产生 policy/restoration output。下一步必须先从本 source commit push 后运行
  Hyper00 processor-only preflight。

### 2026-07-15：Restoration v2.1 formal audit/pilot runner source 就绪

- 90-prompt real `AutoProcessor` audit 会从 45 个 screening states 构造 full-history/summary-only 两种输入，
  检查 official tools 注入、tensor/shape、image-count distribution、token boundary 与 `seq_len + 256 <= 32768`；
  独立 reducer 从逐 prompt records 重算结论，不信任 runner 自报 aggregate；正式 evidence 还记录完整 argv、
  Hyper00/container/image、Python/platform/package、start/end/duration，并明确 CPU-only、无 GPU operation、seed
  not applicable；
- official Jinja/tojson correction 已在任何新 output 前冻结；teacher targets 与 official assistant
  `tool_calls` JSON 逐字节一致（canonical insertion order、原始 Unicode UTF-8），解码后的 text 仍做 strict NFKC；
  interface source SHA256 为 `90cbefc851bed105de6ea0c8f719aae6313a479ca4589e4160fc4ec3e8de3964`；
- fixed-15 runner 只按冻结顺序处理 `v2_development` states；每个 state 在 generation 前写 attempt marker，
  只允许一次 full-history generation，parse failure 是科学 `NO_GO`，OOM/runtime/contract failure 是 invalid，
  interrupted attempt 不得 hidden retry；formal attempt ID 与 persistent output path 已在 pre-output contract
  固定，alternate output path 不能绕过 marker；
- formal runner 在任何 policy runtime import/model construction 前先 durable claim global ledger、root 与
  run manifest。constructor/OOM 写 0/0 canonical `INVALID`；ledger-only、root-partial 或 marker-only crash 在
  `--resume` 时只能封存为 `INVALID`，不能继续 generation。成功构造后 exclusive 写完整 35-key
  `runtime_identity.json`，并在每个 attempt/generation 前复核；
- raw evidence packager 已在 output 前冻结：normalized deterministic USTAR 同时包含 canonical output 与 sibling
  ledger；PASS/NO_GO 重算 exact fixed-15 gate，INVALID 复核 strict partial inventory，并从 source commit 重放
  config、selection、policy/source blobs 与 processor artifact binding。raw archive 进入 private HF，Git 只保存
  immutable revision/hash/compact reduction；
- confirm 语义已精确化：loader 必须验证并 hash 包含 confirm bytes 的完整 artifact，但 decoder/processor
  暴露的 confirm state、prompt、image 都为 0；`confirm_processor_prompt_count=0`，confirm generation
  count=0；
- 该里程碑仍是 source-only。正式 Hyper00 processor audit 与 fixed-15 pilot 均未运行，不报告 parse、closer、
  bridge 或 generation 结果，也不运行 teacher/KL/restoration；旧 `NO_GO_V2_SUBSTRATE` 与
  `NO_GO_ADAPTER_ONLY` 不变；
- 与路线评审的映射未改变：stable self-behavior reference、post-state-only intervention 和 strong
  low-fidelity summary 均已在 v2 冻结，v2.1 只关闭 official-tool interface 风险；
- 输出前验证为 v2.1 63/63、全仓 391 passed（10 个 optional-dependency skips）；全部既有 contract/interface/
  executor/selection/OCR/baseline validators 与 AAAI LaTeX build 通过。processor audit 与 pilot 尚未运行。

### 2026-07-15：v2.1 processor preflight 首次启动在 preprocessing 前 fail closed

- source `main@f689025929e608dcdd10d35b910cef0b2f2424e1` 已 clean push，并在 Hyper00/container
  `69f2b174...94df` 完成 host/disk/path preflight；
- 第一次 CPU-only CLI 把 `--derived-artifact-root` 错传为 repo 内层 `derived/restoration-v2-v1`。artifact loader
  在调用 `AutoProcessor` 前发现缺少根层 `.gitattributes`/`README.md`，因此 fail closed；
- `formal-result.json` 不存在，model weights、policy forward/generation、restoration 与 confirm processor input 均未
  发生。这不是 scientific result，也不消耗 fixed-15 policy attempt；
- 正确 root 是 `/data/tmp/causalcache-restoration-v2-derived-1a01f23-hf-redownload`。先修正并 push 正式 argv，
  再从新的 clean `main` 重启 processor-only audit。

### 2026-07-15：v2.1 90-prompt processor preflight 正式通过

- 修正文档 argv 后，从 clean pushed `main@d0205afd789cda602fbf8964505d9de9a1b1fe53` 在 Hyper00/container
  `69f2b174...94df` 运行真实 `AutoProcessor`；45 states × full-history/summary-only 共 90 prompts 全部完成，
  official tools 注入 90/90，image-count distribution 为 1-image 45、3/4/5-image 各 15；
- input token 长度 3,759--15,013，context overflow 为 0，最长输入加固定 256 generation budget 后为 15,269，
  低于 32,768。processor class、prompt、shape 与 teacher-golden records 均有独立 SHA256 reduction；
- 运行耗时 59.589 秒，明确记录 `gpu_operations_executed=false`、policy model 未实例化、weights 未
  materialize 为 tensors、forward/generate 未执行、confirm processor prompt 为 0。因此本步骤没有产生任何
  v2.1 policy output，也没有运行 teacher、KL、
  restoration 或访问 confirm input；
- 7,609,803-byte raw JSON（SHA256 `5349ffc6b91bf93ed26d25104fe6c907f31ccc497007a5c0ae6ed5ddf5c84991`）
  已上传 private HF dataset `gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile`，tag
  `v2.1-processor-preflight-v1` 固定到 immutable revision
  `85576161b7cb8bbae14e46a482c42b5be5bf1d7e`；fresh immutable download 的 byte hash/size 与 API tag binding
  均已复核；
- Git 只保存 `data/results/restoration_v2_1_processor_preflight/` 的轻量 manifest/说明。该 PASS 只关闭
  processor gate；fixed-15 pilot 尚未运行，parse/closer/bridge 结果仍未知。

### 2026-07-15：v2.1 fixed-15 interface pilot 正式通过

- processor artifact 的 reuse-mode validator 先从 fresh immutable HF download 在 clean
  `main@70724bdb82484cd5645d174765724bd1f83b5ace` 重新得到 90 prompts PASS；随后 GPU preflight 确认
  Hyper00 全部 8 张 H200 无 compute process，canonical root/ledger 不存在；
- 唯一 formal attempt 使用 container-local `cuda:0`（映射 host physical GPU 2，UUID
  `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81`）、BF16、greedy batch-1、256-token cap；15 个冻结
  `v2_development` states 各 generation 一次，84.279 秒完成；
- 正式 outcome 为 `PASS_V2_1_INTERFACE_PILOT`：strict whole-output parse 15/15、model-emitted closer
  15/15、AndroidWorld bridge 15/15；generation 15，retry/top-up/truncation/extra output 均为 0；
- teacher forward、KL measurement、restoration label、expert action read、label-train/confirm generation 均为 0，
  没有 sample mutation。该 PASS 只证明 native official-tool interface 可用于下一阶段，不证明 stable
  reference、memory sensitivity 或 CausalCache 方法有效；
- utilization monitor 在 model load/processor 与 batch-1 generation 交错时观测到低于 90% 的窗口；已立即核查
  为单卡、固定 batch-1 protocol，不能再减少 GPU 数且不得为 pilot 改 batching。run 在 84.279 秒内完成；这不是
  throughput efficiency 结果，full-45 source 需继续记录 latency/utilization；
- canonical output 与 sibling ledger 已封装为 34-file deterministic USTAR，133,120 bytes，SHA256
  `f71d5fd575dde48ae8b3e50a19dd2fbecfa02d7d5ae6087f909a47dfd7032064`。private HF tag
  `v2.1-interface-pilot-v1` 解析到 immutable revision
  `bdff8ca71f150afd80d6291b4ecec76cbf9e7432`；fresh immutable download 的 byte hash/size 已复核；
- Git 只保存 `data/results/restoration_v2_1_interface_pilot/` 的 34-file inventory/binding 与说明。manifest
  push 后，artifact validator 已在 clean `main@64581118eb3375a03234ef49e5ce1f7bc41d354e` 从 fresh archive
  返回 `VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT`、`archive_hash_verified=true` 与同一 PASS；
  committed binding 已闭合。

### 2026-07-15：v2.1 full-45 substrate source contract 已冻结

- 新增独立 child protocol `causalcache_restoration_v2_1_full_45_substrate`；machine-readable contract
  `code/configs/causalcache_restoration_v2_1_full_45.json` 的 SHA256 为
  `0924dd66fab9440bed66585765e9b1f5ab6fb80fbdf65a1492efba7ae81e116b`，并绑定 fixed-15 PASS、
  90-prompt processor PASS、parent v2 scientific config 与 exact 30+15 state projection；
- exact projection SHA256 为 `65e7085f01bc26e425bab7f5148c6729bc8c6d0a62802ebc5e8dfeaed6439249`。
  每 state 固定两次 full-history generation；仅当两次均 strict parse、model-emitted closer、bridge 成功且
  canonical action 一致时，才执行两次 reference 和一次 summary-only teacher forward，再计算 repeat 与
  summary-reference GPU KL；
- 全 attempt 的 planned/maximum schedule 同为 generation 90、teacher 135、KL 90。parse/mismatch/non-finite
  仍留在 45 分母；runtime/contract/OOM/bridge invariant 进入 `INVALID`。gate 要求 45/45 parse、45/45 finite、
  45/45 repeat agreement，并有至少 8 个 states 满足 summary KL 大于
  `max(1e-4, 10 * mean(repeat KL))`；
- canonical attempt/root/sibling ledger/archive 与 private HF destination 均和 fixed-15 完全独立。resume 只跳过
  已有 terminal prefix；root 外 ledger 以 atomic replace + file/directory fsync 维护 attempted/completed high-water
  journal。marker 无 terminal、删除 marker+terminal pair 或删除整个目录都会使 attempt `INVALID`，不能重生成或
  换目录重跑；teacher suppression 也精确绑定 BF16 minimum，不接受任意负 finite 近似；
- source inventory 覆盖 full-45 source 及其递归本地 import closure；正式 runner 在 policy runtime import 前
  验证 clean pushed `main`、Git blobs、derived artifact，并从 private HF immutable revisions 的 fresh download
  复核 fixed-15 archive 与 processor JSON；
- raw evidence manager 将 canonical root 与 sibling ledger 打成 deterministic USTAR，并从逐 state records
  重算 denominator、operation counts、gate 与 outcome。正式 raw archive 只进入 private HF，Git 只保存轻量
  manifest 与 immutable binding；archive reader 会按同一 USTAR serializer 重建并要求原始 bytes 完全一致，
  因而拒绝 GNU tar、trailing bytes 与非 canonical header；
- 本里程碑是 source-only，validator 明确返回 policy/GPU 未授权；尚未运行 full-45 generation、teacher、KL、
  restoration 或 confirm。full-45 focused tests 33/33、全仓 424 tests（10 skips）、全部既有 validators 与
  AAAI LaTeX build 已通过。source commit/push 与 clean descendant validation 完成后，才允许唯一 formal
  attempt。

### 2026-07-15：v2.1 full-45 substrate 正式判为 NO-GO

- 唯一 `restoration-v2-1-full-45-substrate-v1` attempt 从 clean pushed source
  `7a5b6d5710fe4d054936b5aa474648149f725edb` 在 Hyper00 单张 H200 上完成；运行时间为
  `2026-07-15T22:59:20.590509Z`--`23:04:48.502942Z`，327.912 秒；
- 90/90 generations 均 strict parse、有 model-emitted closer 并通过 AndroidWorld bridge；但只有 32/45
  states 的两次 exact canonical action 完全一致，低于冻结的 45/45 gate。13 个 mismatch 全部保持 action
  type：12 个 `click→click`、1 个 `swipe→swipe`；label-train/development 分布为 10/3，step 4/5/6
  分布为 3/5/5；
- 32 个 agreement states 产生 96 次 teacher forward 与 64 次 GPU KL。repeat KL 全为 0，epsilon 固定为
  `1e-4`；32/32 summary-reference KL 超过 epsilon，min/median/mean/max 为
  0.000783/0.031472/0.044022/0.196197，因此 interface 和 history sensitivity signal 都存在；
- frozen reducer 仍必须输出 `NO_GO_V2_1_FULL_45_SUBSTRATE`，因为 repeat agreement 与相应 finite-logit
  coverage 均只有 32/45。retry、top-up、sample mutation、expert read、restoration coalition、baseline、gate
  training 和 confirm work 全为 0；不能把 coordinate jitter 事后 relabel 成 PASS；
- utilization monitor 观测到 model load/CPU processor 与 batch-1 generation/teacher 交错导致平均窗口低于
  90%，但 active GPU 峰值为 99--100%，仅使用一张 GPU，且 frozen batch-1 protocol 不允许为本次结果修改
  batching；
- canonical 94-file deterministic USTAR 为 962,560 bytes，SHA256
  `8cd53d6e56d5ad509da2af91d73d9e83b4db989ffc26aa18e1bca84e4c4f4fa4`。private HF tag
  `v2.1-full-45-substrate-v1` 固定到 immutable revision
  `814506ef1450838d4bc6ed3d89fe53e0773d92fb`；fresh immutable download 已逐 byte 复核，Git compact result
  位于 `data/results/restoration_v2_1_full_45_substrate/`；
- manifest push 后，Hyper00 clean `main@554c51e417d702a6bc759b1f592979a4c38c5283` 从 fresh immutable archive
  运行 committed-binding validator，返回 `VALID_RESTORATION_V2_1_FULL_45_ARTIFACT`、
  `archive_hash_verified=true`、94 files 与 tree SHA256 `6c5e515a...19281`。source commit、当前 descendant、
  archive、manifest、reducer outcome 与 HF revision 的完整 SoT chain 已闭合。

### 2026-07-15：Interaction-aware gate ablation 设计记录

- 合作方指出 synthetic 已观测到 non-additive event interaction，而当前独立 event score 会把
  coalition-conditioned marginal 压缩成 context-independent average value；该问题被确认为真实的
  subset-objective information bottleneck，不是 restoration attribution 本身假设 event 独立，也不是当前
  independent student 违反已经冻结的 averaged-$G$ teacher contract；
- 新增 `ablations/interaction_aware_gate.md`，记录 set-conditioned iterative gate：每选入一个 event 后，基于
  更新后的 coalition embedding 与剩余 budget 给候选重新打分；同时保留 independent gate 与 exact subset
  oracle 作为 ablation 两端；
- 文档明确三项限制：当前 near-budget label distribution 不覆盖 iterative gate 的全部前缀 cardinalities；标准
  permutation edge 不自动提供同一 coalition 下的 conditional-ranking pairs；pure complementarity 也可能让
  greedy 从空集停止。因此 set conditioning 只能声称缩小 interaction-induced oracle gap，不能声称解决任意
  interaction；
- 该里程碑仅为 Git source-only proposal，没有修改 frozen v2/v2.1 contract、代码或论文，没有运行 policy/GPU，
  没有读取 confirm，也没有新增 HF artifact。真实 ablation 仍被 v2.1 substrate NO-GO 阻断。

### 2026-07-15：Subset-search v1 source/config/runner 冻结

- 合作方进一步指出“conditional marginal 包含 interaction”并不保证最终 greedy subset 全局最优；该问题被拆成
  value-source 与 search-algorithm 两个轴，避免把既有 phase-0 的 0.859 误称为 greedy gap。前者是 exact
  averaged marginal 经 exact additive knapsack 后相对真实 subset objective 的 projection gap；只有 true
  conditional greedy 相对 exact subset 的差才是 search regret；
- 新增 deterministic black-box search module：exact subset、raw/density conditional greedy、从 raw-greedy
  seed 出发的 2x2 bounded exchange、允许 non-positive prefix 的 true-utility beam-$2/4/8$。所有方法允许空集，
  统一 utility/cost/cardinality/lexicographic tie-break，并记录全部 unique set evaluations、evaluated candidates、
  sequential rounds；exchange 成本包含 seed 与未采用 neighbors；
- 冻结 `code/configs/subset_search_ablation_v1.json`，SHA256
  `161518c3e951829528b63b9146878236d918d4d620b4a6acbecc53f7c4db921f`。输入包含五类 controlled
  functions、既有 phase-0 mixed synthetic、$n=8/12/16/24,b=3$ scale sweep，以及旧 v1 两个
  selection-biased development states 的完整 singleton/pair table；
- 旧 real table replay 明确使用 `distance_mean`、每 event 476 visual tokens、budget 512/1024；它不运行新
  policy forward，只验证 exact/greedy/exchange/beam 的 cached-table consistency。state sensitivity threshold
  `1e-4` 仅单列为 abstention sensitivity，canonical search stopping threshold 固定为 0，不能混成 interaction
  search gap；
- true-$U$ greedy/exchange/beam 在真实部署都需要 frozen-policy rerun，因此只称 offline optimizer diagnostic；
  learned marginal head 没有 direct set-utility/path contract 时不能直接用于 beam/removal。exact 只保证冻结
  single-step restoration utility 最优，不代表 terminal success 最优；
- focused tests 16/16、全仓 440 tests（10 个 optional-dependency skips）、既有 contract/interface/executor/
  selection/OCR/baseline validators 与 AAAI LaTeX build 全部通过。该里程碑只冻结 source/config/runner，尚未
  生成 formal result，没有运行 GPU、policy、gate、closed-loop 或 confirm；v2.1 继续保持
  `NO_GO_V2_1_FULL_45_SUBSTRATE`。

### 2026-07-15：Subset-search v1 首次 CPU attempt 在 artifact validation 前 fail closed

- source milestone `d5cd389f6a7b2720cc1daaa3373cefa52480cf7b` push 后，从 clean canonical main 运行
  frozen CLI；14 个 scenarios 在 CPU 上 51.668 秒完成，外部/untouched data、policy、GPU、confirm access 与
  learned-gate/closed-loop evaluation 均为 0；
- phase-0 exact averaged-marginal knapsack / exact subset 为 0.858594，而 true conditional greedy 为 1.0，
  正式确认前者应命名 objective-projection gap，不是 greedy search gap；
- controlled complement trap 中 raw greedy/beam-2 ratio 0.5，2x2 exchange/beam-4/8 为 1.0；mixed
  non-monotone 中 raw greedy 0.666667、exchange/beam-4/8 为 1.0；variable-cost 中 raw 0.833333、density
  1.0；
- $n=8/12/16/24,b=3$ exact queries 为 93/299/697/2325；raw greedy queries 22/34/46/70，ratio
  1.0/0.95098/0.95098/0.970588；beam-4 queries 51/88/125/197，在该 deterministic generator 上均 exact；
  2x2 exchange 虽均 exact，但 $n=24$ 使用 1,185 queries，不能简单视为便宜修复；
- 旧 v1 selection-biased cached table 的四个 state/budget cases 中，true greedy、exchange、beam 与 exact
  coalition 全部相同，greedy ratio 均为 1.0。step 8 / 1024 有 2 个 positive、19 个 negative pair
  interactions 但没有 greedy trap；真实 table 只验证 implementation consistency，不支持 stronger search 的
  paper-level收益；
- 单列 threshold sensitivity 发现把旧 `1e-4` state floor 当 search threshold 会令 step 4 / 1024 从 exact
  `[1,3]` 提前停在 `[3]`，ratio 0.944896；该差距明确归入 abstention，不归入 search regret；
- runner 输出后、任何 result commit 前执行独立 scientific replay；所有数值相同，但 in-memory trace coalition
  是 tuple，JSON load 后为 list，严格 payload equality 因类型不一致触发 `AssertionError`。这是 artifact
  serialization contract bug，不是搜索数值不稳定；按 fail-closed 边界，首次 result 目录已删除，未 commit/push；
- source 已在 `_trace` boundary 显式输出 list，并加入完整 JSON round-trip equality regression。下一步先
  commit/push 该 fix，再从新的 clean source canonical rerun；不能沿用首次 output 或放宽 validator。上述数值在
  canonical rerun 前只算 provisional diagnostics。

### 2026-07-15：Subset-search v1 canonical CPU rerun 与 pre-commit replay 通过

- JSON serialization fix 已作为 `main@7569ce2ac1e63f565be4e0d4dcc9626285aa355c` push；全仓 440 tests
  （10 skips）通过后，从该 clean canonical source 重新运行 frozen CLI；
- canonical run 于 `2026-07-15T23:58:44.535445Z`--`23:59:38.142234Z` 在 CPU 上完成，wall time
  53.606 秒，14 个 scenarios 的 scientific payload SHA256 为
  `26846d509d421dcb49f2d1554598893a65829c6a25610ef399ce69b383274edf`；policy/GPU operation 均为 0，
  未访问外部/untouched data 或 confirm；
- 写出后、任何 result commit 前再次从 config/input 构造完整 payload；`json.loads(json.dumps(payload)) ==
  payload`、与 summary scientific payload 的完整 equality 和 canonical SHA256 三项全部通过，输出
  `PRECOMMIT_SCIENTIFIC_REPLAY_MATCH`；
- canonical result 逐项复现首次 provisional 数值：phase-0 projection/true-greedy ratio 0.858594/1.0；
  complementary trap raw/beam-2 0.5、exchange/beam-4 1.0；mixed raw 0.666667、exchange/beam-4 1.0；
  variable-cost raw/density 0.833333/1.0；
- scale sweep exact queries 93/299/697/2325；raw greedy queries 22/34/46/70、ratio
  1.0/0.95098/0.95098/0.970588；beam-4 queries 51/88/125/197 且在本 generator 全部 exact；exchange
  到 $n=24$ 需要 1,185 queries；
- 旧 cached real table 的四个 cases 仍全部 greedy=exact；step 8 / 1024 的 2 positive + 19 negative pair
  interactions 没有形成 greedy trap。正式结论仍是保留 set-conditioned greedy 主线，stronger search 只作
  offline diagnostic；
- 轻量 summary/report 位于 `data/results/subset_search_ablation_v1/`。下一步 commit/push result，再从 clean
  descendant main 运行正式 committed validator。

### 2026-07-15：Subset-search v1 committed result validation 闭合

- canonical summary/report 已作为 `main@45bcf7e3ab984e2f4c977af46a27312b1ed8ba0a` commit/push；该
  result commit 只包含轻量 Git artifacts，没有新增 reusable dataset/model，因此无需创建 HF repo；
- 从 clean `main@45bcf7e` 调用独立 validator。它重新构造 14-scenario 完整 scientific payload，并同时校验
  source commit blobs、当前 source/config/input hashes、JSON round-trip identity 与 committed summary blob；
- validator 返回 `VALID_SUBSET_SEARCH_ABLATION_V1`，source commit 为
  `7569ce2ac1e63f565be4e0d4dcc9626285aa355c`，scientific payload SHA256 为
  `26846d509d421dcb49f2d1554598893a65829c6a25610ef399ce69b383274edf`；
- formal result 因而从 source、运行、pre-commit replay、result commit 到 clean-descendant committed replay
  全链路闭合。v2.1 outcome 仍为 `NO_GO_V2_1_FULL_45_SUBSTRATE`，没有 policy/GPU/confirm operation；
- 方法决策保持：set-conditioned greedy 是线上主线，exact 是小规模离线 ceiling，beam/local 只作可替换的
  true-utility offline diagnostics。旧 real table 中 greedy/exact 比为 1，尚无证据为线上 stronger search
  增加复杂度。

### 2026-07-15：Spatial reference audit v1 source-only 合约冻结中

- subset-search 支线保持闭合，主线回到 reference substrate；本 audit 只使用 v2.1 已暴露的 13 个
  `CANONICAL_ACTION_MISMATCH`（10 label-train、3 development；12 click、1 swipe），v2.1 的
  `NO_GO_V2_1_FULL_45_SUBSTRATE` 不重算、不改写；
- parent raw archive、13-state canonical projection、actual action/token hashes、derived artifact、selection、OCR、
  model snapshot、source inventory 和 append-only child exposure ledger 均进入 fail-closed config；旧
  `restoration_v2_exposure.json@bc122482...` 保持 byte-immutable，confirm 20 仍为 policy-output untouched；
- 严格 CUDA deterministic GEMM 需要项目规则禁止的 `CUBLAS_WORKSPACE_CONFIG`，因此 profile 诚实冻结为 BF16
  auto、BF16 eager-control 与 4-state FP32 eager-control，不声称数学 deterministic。eager-control 固定 seed、
  eager attention、cuDNN deterministic/benchmark off、TF32 off 和最高 float32 matmul precision；
- formal runtime 进一步在 durable attempt claim 前 exact 锁定 image digest
  `sha256:6a8f60af...d349acfa`、Python `3.12.3`、PyTorch `2.11.0+cu130`、CUDA `13.0`、cuDNN
  `91900`、Transformers `5.6.0`、driver `570.172.08`，并要求固定 behavior-changing environment-variable
  名单全部 absent；auto observed attention 必须 non-eager，eager profiles 的 non-null observation 必须全为 eager；
- operation ceiling 仍为 generation 60、teacher forward 120。每 state 的 teacher diagnostic 改为 2 次相同
  shape shared-prefix forward，从同一 vector 比较两个竞争 token；另对两条父 action 各做 1 次 full-branch
  diagnostic。margin 明确不是旧 generation-time score，也不进入 pass/fail；
- canonical root 与 root 外 sibling ledger 已冻结；profile/state 都必须在 forward 前 durable claim，alternate
  output、删除、retry、top-up 和跳过 profile 顺序全部禁止。raw root 计划上传 private HF dataset
  `gavinlaw/causalcache-spatial-reference-audit-mobile@spatial-reference-audit-v1`，当前 revision 仍 pending；
- raw packager 已在任何 audit forward 前冻结：它把 canonical root 与 sibling ledger 封装为单个按 path 排序、
  metadata 归一化的 deterministic USTAR，拒绝 symlink/non-regular member，exclusive-create 后重新读回并要求
  canonical-byte identity；本地 archive 固定为
  `/data/experiments/causalcache/spatial-reference-audit-v1.tar`，当前尚未生成；
- 未来 exact restoration 的 subset 计数已纠正：step-4/5/6 为 4/7/11，label-train 220 rows、development
  110 rows，总计 330；它只是新 substrate gate 通过后的 prospective table，不是当前 policy-forward 数，也未运行；
- 当前仍是 source-only：没有新 audit GPU profile、semantic result、restoration、gate training 或 confirm policy
  input/output。完成 source tests、commit/push 和 clean descendant validation 后，才允许 Hyper01 单张 H200
  preflight 与唯一 audit attempt。
- 聚合判定修正为三分支：eager unstable 才直接进入 semantic；eager 13/13 且 auto 非 13/13 才能称
  eager-specific recovery；若两个 BF16 profile 都是 13/13，只报告本次 numerical audit inconclusive，不归因于
  eager。FP32 与 margin 始终不决定分支。

### 2026-07-15：Spatial reference audit v1 formal raw 与 validation-repair 前置

- 从 clean pushed `main@c093bd8f92ab97427acb427bd2d66fb6b20b556a` 在 Hyper01 non-privileged
  container `7ca0845bbaa5...ff4b`、单张 `NVIDIA H200` 上完成唯一 attempt；三个 profile 与 sibling ledger
  都已 durable terminal，未发生 retry、resume、alternate path 或 output deletion；
- raw metrics 为 BF16 auto 7/13 exact generated-token stable、BF16 eager-control 13/13、FP32 eager-control
  4/4。operation counts 精确为 generation 60、teacher forward 120，confirm/restoration/gate 均为 0；FP32
  仍只作描述性 probe；
- 原 formal wrapper 在三个 profile 之后、summary 写入之前由 independent validator fail closed，exception 为
  `ValueError: generation per-image effective visual token count drifted`，exit code 为 1；raw root、ledger、formal
  log/exit 与三个 terminal 的 pre-repair SHA 已冻结，不允许重跑 profile；
- 只读诊断确认旧 validator 有两个纯离线 contract bug：它把 target `2560` 错当成所有 realized grid 的固定
  token 数，而实际合法 realized set 为 2516/2560/2584；它还把 teacher aligned inventory 错写为
  `attention_mask + input_ids`，frozen runtime 的实际定义是 `attention_mask + mm_token_type_ids`；
- 180/180 shape nodes 的 grid accounting 正确；60/60 generation 的 grid、effective visual tokens、text tokens 与
  prompt tokens 和已验证 v2.1 parent raw 的同 state witness exact。仅在内存中窄修上述两项、且不传
  `--summary-output` 的 dry-run 已让旧 validator 其余全链通过，没有第三个 blocker，candidate decision 为
  `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；
- 新增 source-only validation-repair child，旧 33-file audit source/config/validator/exposure ledger 全部
  byte-immutable。repair 先绑定 pre-repair tree、ledger、terminal、formal log/exit 和 parent archive，再用 dynamic
  grid accounting + parent-shape witness 运行其余旧 validator；全部读取验证完成后才 exclusive-create 一个
  provenance-complete `summary.json`，不加载 processor/model，不调用 GPU/policy，也不修改已有 raw；
- 完整说明见 `docs/spatial_reference_audit_v1_validation_repair.md`。repair source commit/push、远端 clean-main
  execution、旧 deterministic USTAR、HF immutable fresh-download 与 Git result 尚未闭合，因此 candidate decision
  暂不提升为正式 artifact verdict。

### 2026-07-15：Spatial reference audit v1 artifact 闭合

- validation-repair source 在 clean pushed `main@a2528d7e95e73c25639568650f63abce58e4e491` 冻结，full
  CPU suite 为 482 passed / 11 optional-dependency skips；唯一 Hyper01 offline repair exit 0，只新增 canonical
  `summary.json`，未加载 processor/model，也没有新增 generation、teacher、confirm、restoration 或 gate operation；
- repaired validator 复用旧 validator 全链，并把旧 config + 33-file source inventory 逐 blob 绑定回 raw commit
  `c093bd8f...`。60/60 generation shape、120/120 teacher shape 与 parent witness 通过；observed per-image effective
  visual tokens 为 2516/2560/2584，但只作 evidence inventory，不作 acceptance whitelist；
- 正式 decision 为 `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`：BF16 eager-control 13/13 exact stable，BF16
  auto 7/13；FP32 4/4 仍是 descriptive probe，不控制结论。原 `NO_GO_V2_1_FULL_45_SUBSTRATE` 永久不变；
- deterministic USTAR 含 72 members，1,873,920 bytes，SHA256
  `d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc`，tree inventory SHA256
  `6423c13f4b7067f5a2139442bab0be11a1ef5da9f880f51a1131768afa963b82`；
- private HF dataset `gavinlaw/causalcache-spatial-reference-audit-mobile` 的 canonical immutable revision 为
  `d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`，tag `spatial-reference-audit-v1` exact 解析到该 revision。
  从全新 cache 强制下载后，README/manifest/raw 三文件 allowlist、archive hash/size、canonical USTAR rebuild、
  72-member tree、summary SHA 与 scientific payload hash 均 exact；
- Git 只回写 `data/results/spatial_reference_audit_v1/` 的 compact summary/artifact binding。下一步不是
  restoration，而是先冻结一个全新 `v2.2-eager` source 与 fresh 45-state attempt；任何失败都不回写本 audit。

### 2026-07-15：Restoration v2.2-eager fresh-45 source-only freeze

- parent spatial artifact 固定为 `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`（BF16 auto 7/13、BF16
  eager-control 13/13）；它只授权新的 eager substrate，不改写
  `NO_GO_V2_1_FULL_45_SUBSTRATE`；
- machine-readable contract 为 `code/configs/causalcache_restoration_v2_2_eager.json`，完整说明为
  `docs/restoration_v2_2_eager.md`。formal freeze 已从 clean pushed
  `main@b3a6303d69b1145fbf195e0bd18b9b3065a6f213` 建立；config SHA256 为
  `f473bb8a1657072235dd73bf78a93aff27b438d7baca65b7ef6096cf985effa7`，50-file formal inventory SHA256 为
  `bb6351e4dd5b4470ed1add86dbcbaa714a56063ba8ecf07268b6f13f630f9e54`；
- scientific contract 只改变 runtime：BF16 eager、seed 0、TF32 off、cuDNN deterministic on/benchmark off、
  float32 matmul `highest`；明确不声称 strict CUDA determinism。official-tool interface、prompt/parser、teacher
  target、45-state projection、gate 与 per-state schedule 均继承 v2.1；container/Python/PyTorch/CUDA/cuDNN/
  Transformers/driver/H200 stack 则 exact 绑定回 spatial audit，避免引入第二个 runtime 变量；
- formal attempt 必须 fresh 运行全部 45 states，旧 v2.1 raw、state records、aggregate、terminal 与 ledger
  均不可复用或计数。全 attempt generation/teacher/KL 计划与硬上限仍为 90/135/90；
- worker topology 固定为同机同容器两张 H200：even worker 23 states，odd worker 22 states，不允许 state
  stealing。coordinator 必须在任一 runtime import/model construction 前，以独立于 v2.1 的 global sibling
  ledger 对整个 attempt 做 exclusive durable claim；任一 worker failure 或 parity inventory drift 都使 attempt
  `INVALID`；
- 两个 runtime identity 在首个 marker/generation 前经过 coordinator barrier；logical device 固定为 0/1，
  preflight 选择的 physical/NVML index 可为任意两个不同非负编号，并由 distinct UUID/PCI 交叉证明；
- v2.2 lifecycle 不支持 resume；任一 worker/coordinator/主机中断都会把唯一 attempt 终止为 `INVALID`，已有
  terminal prefix 也不能由新 invocation 跳过并补齐。global claim 会预绑定两个 worker 的 root 外 sibling
  high-water ledger，每个 state 前后都必须 durable 更新；stale/missing root mirror、orphan/missing state files
  只按 sibling truth 形成 forensic inventory，policy-free `seal-interrupted` 可封存但绝不 retry；
- confirm state/prompt/image/decoder/generation/teacher、restoration coalition/label、baseline 与 gate
  construction/training/selection 的计数全部固定为 0。planned private HF dataset 为
  `gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile`，当前没有 raw archive、immutable revision
  或 result verdict；
- 本里程碑没有执行 GPU policy/model forward、没有加载 processor/model、没有产生 v2.2 policy output。source-only validator 通过
  也不自动授权正式 run；执行仍需 clean pushed main、双 GPU preflight、fresh parent evidence 和独立 global
  claim。v2.2 定向 54 tests 与全仓 538 tests 均通过（全仓 11 skipped）；planned HF repo 仍为空，raw archive、
  immutable revision 与 Git compact result 均尚未产生。
- Hyper00 CPU-only pre-claim authorization 先后按预期拒绝了错误 derived root、HF local-dir 自动生成的 cache
  metadata，以及 spatial repaired raw schema 的错误读取。前两项通过 exact six-file clean materialization 解决；
  第三项暴露 runner 错把 canonical `profile_summaries[*].metrics` 当成 compact Git `profiles[*]`，已在
  `main@59b2138` 修复并加入 fail-closed schema 回归测试。随后还发现 v2.2 自有 pretty-JSON loader 错拒绝 hash
  正确的历史 processor evidence；`main@b3a6303` 改为复用 v2.1 strict JSON parent contract，并验证 duplicate key
  仍 fail closed。所有检查均发生在 global claim/runtime import 前，canonical output、ledger、generation、teacher
  与 KL count 仍为 0。

### 2026-07-16：Restoration v2.2-eager fresh-45 substrate 与 immutable artifact 闭合

- 从 execution source `main@8ae07519f14ac3635f292ee93a7b6d624507427e` 在 Hyper00 同机同容器的两张
  H200 上完成唯一 fresh-45 attempt；even/odd worker 分别固定处理 23/22 states，未发生 retry、top-up、resume、
  state stealing 或 alternate output；
- 45/45 states 完成并 parse，45/45 exact repeat canonical action agreement，45/45 finite logits，45 个
  memory-sensitive states，finite repeat KL mean 为 0.0；正式结论为
  `PASS_V2_2_EAGER_FULL_45_SUBSTRATE`；
- generation/teacher/KL 精确为 90/135/90，全部 gate checks 为 true；restoration coalition/label、baseline、
  gate training/selection 与 confirm operation 均为 0，因此本结果只通过 stable reference substrate gate，不是
  CausalCache 方法效果证据；
- run window 为 `2026-07-16T06:05:09.464587Z`--`06:15:19.889274Z`，wall time 610.425 秒；双卡 steady-state
  monitor windows 为 92%--100% average utilization；
- canonical evidence 已封装为 102-file deterministic USTAR，SHA256
  `b22827e6e2d8d33b03686fc177dc8f9c55c5133470fbe40f9fb3e33cce809fb5`，size 1,269,760 bytes，tree
  SHA256 `e7f0047b20c27bac1f4d61d7c0bff2e63c7760469cf9ccaa188dca948b7a4359`；
- private HF dataset `gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile` 的 tag
  `v2.2-eager-full-45-substrate-v1` 绑定 immutable revision
  `3577099d505b8c652d764f41269df911128ec767`；不同路径 fresh download 与 source archive 逐 byte 相同；
- Git compact result 位于 `data/results/restoration_v2_2_eager_full_45_substrate/`。eager 已通过，因此不进入
  executable/UI-element equivalence 补救分支；restoration 与 confirm 必须由新的独立 source contract 授权。

### 2026-07-16：Restoration v2.2 exact-label source freeze

- label source 已在 clean pushed `main@3942d687d03bf63ea683fe8ad906a161eb10dc27` 冻结；machine-readable
  contract SHA256 为 `56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87`，29-file
  source inventory SHA256 为 `8d0ecf6b0df75f9fa7753631b8fca5efd98c71a7953007e684393e6e8fe52d9b`；
- protocol 固定 45 states、420 条 canonical raw `D(S)`、45 个 primary exact-subset oracle、435 条
  deployment conditional-marginal labels；operation schedule 固定为 465 teacher forwards、420 GPU KL
  measurements 与 0 generation；
- 正式 topology 固定为同一 Hyper host/container 的两张 H200，even/odd worker 分别处理 23/22 states，
  microbatch 1；canonical output、global ledger、archive 与 planned HF repo/tag 已写入 contract/执行文档；
- fail-closed contract、input projection、parent binding、per-state/aggregate operation accounting、worker failure
  sealing、deterministic USTAR 与 fresh immutable artifact validation 均已实现；完整回归 575 tests OK
  （skipped 11），`make paper` 通过；
- 本里程碑只冻结并验证 source。正式 GPU attempt 未运行，raw label rows、exact oracle output、HF repo/tag/
  immutable revision 与 Git compact result 均未产生，也没有 local staging；confirm、gate、matched-NLL 和
  closed-loop 全部未运行且协议禁止混入本步。

### 2026-07-16：Restoration v2.2 exact-label v1 zero-forward invalid attempt

- formal v1 在 Hyper00 新双 H200 container 中通过 Git/contract/parent/derived authorization，并于
  `2026-07-16T16:39:43.874890Z` durable claim；run-contract SHA256 为
  `4dd470cbde3184decb0f38133dc9347b528304f44b13762c881540aefa4ccebf`；
- 两个 worker 随后都发现指定 GUI-Owl immutable snapshot directory 不存在。global ledger 于
  `2026-07-16T16:40:14.012087Z` 以 `LABEL_ATTEMPT_INVALID` 终止，attempted/completed states 为 0/0，两个
  worker 的 state marker 与 state record inventory 均为空；
- 因失败发生在 runtime/model load，teacher forward、KL、raw `D(S)`、exact oracle、conditional marginal、
  confirm、gate、matched-NLL 和 closed-loop 均为 0；这不是 restoration scientific signal 的 falsification；
- 根因是旧 authorization 只检查 `--model-dir` 为 absolute path，snapshot existence 检查晚于 global durable
  claim；同时人工 shell smoke 未 fail-fast，在失败的 `test -d` 后仍打印 success；
- v1 root/ledger 永久保留且 retry/resume=false。Git compact failure binding 位于
  `data/results/restoration_v2_2_eager_labels_v1_attempt/`，没有可上传的 label dataset，HF repo/tag 未创建；
- repair 必须是全新 attempt identity，把 model snapshot existence/revision/inventory validation 前移到 claim 之前，
  并经过新 source freeze 后才可运行。

### 2026-07-16：Restoration v2.2 exact-label v2 pre-claim repair source freeze

- replacement contract `causalcache_restoration_v2_2_labels_v2_repair.json` 的 SHA256 为
  `7fc96883b905a5f026c18e65c3c7e377972559c775ac7ec95030ea1f04c2f94c`；v1 config/SHA 与 invalid evidence
  保持不变，v2 source parent 绑定 zero-forward failure SoT commit `bf9e377ceb9c656af8ae84c223ad23d730136cb1`；
- 新 attempt/revision 为 `restoration-v2-2-eager-labels-v2` / `v2_preclaim_repair`，output、ledger、archive、
  PASS/INVALID outcome、HF tag/path 与 v1 完全分离；scientific estimand、45 states、420/435 label denominator、
  465 teacher forwards、23/22 parity 和 microbatch 1 不变；
- canonical model projection 固定为 `/data/artifacts/models/GUI-Owl-1.5-8B-Instruct`。global durable claim 前先
  完整验证 14 files / 17,545,907,171 bytes、`.snapshot.json` revision、每文件 SHA 与 Transformers source；
  missing/wrong-basename/symlink/partial snapshot 不得创建 root 或 ledger；
- runner、spawn worker rebuild、state/worker/global envelope、aggregate、archive prefix 和 artifact manifest 均按
  profile 验证。targeted 21/21、full suite 580 PASS（skipped 11），`compileall` 与 `git diff --check` 通过；
- 本里程碑仍是 source-only：v2 formal GPU attempt、raw labels、HF repo/revision、gate、matched-NLL、closed-loop 与
  confirm 均未运行。

### 2026-07-16：Restoration v2.2 exact labels 与 immutable artifact 闭合

- 从 source/packaging commit `5ae40d4aed4eb20b931216776b379bc6ae55629d` 在 Hyper00 同机同容器的两张
  H200 上完成唯一 replacement attempt；run-contract SHA256 为
  `b78ca1e70652c7eef68efc3472adf78cec4510e507a0c00cfcee2c2cf05285d9`。45/45 states `PASS`，even/odd
  workers 分别完成 23/23 与 22/22，retry、top-up 和 generation 均为 0；
- formal payload 包含 420 条 canonical $D(S)$、435 条 deployment conditional-marginal labels、45 个
  exact-subset oracle 与 465 个 pair-interaction rows。exact oracle 的 mean normalized recovery 为 overall
  `0.8735119444`、train `0.9028414853`、development `0.8148528628`；44/45 states 获得正 oracle utility，
  oracle cardinality 0/1/2 的 state 数为 1/3/41；
- conditional marginals 的严格正/负计数为 360/75，22/45 states 至少包含一条负 marginal；pair interactions
  的严格正/负计数为 188/277；raw utility 的严格正/零/负计数为 338/45/37。interaction 与 non-monotonicity
  支持后续 set-conditioned gate 设计，但不证明 distilled gate、matched-NLL 或 closed-loop 效果；
- 101-file deterministic USTAR 的 SHA256 为
  `99120d5444d31962d9f4254c3e40bc5f06d3e4d3d90a74b1749e7ccd7aefb29e`，tree SHA256 为
  `c5104594b0741810f3d49d007a63a74f16ee4236dd137d1dea92b9373065c45f`。private HF dataset
  `gavinlaw/causalcache-restoration-labels-mobile` 的 tag `v2.2-eager-train-dev-exact-v2` 已绑定 immutable
  revision `8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`；fresh immutable download 重新通过 raw reducer 并与
  source archive 逐 byte 相同；
- Git compact result 位于 `data/results/restoration_v2_2_eager_labels_v2/`。当前只闭合 offline oracle/labels；
  gate checkpoint、matched-NLL pairs、closed-loop episodes 和 confirm artifact 均不存在，confirm 保持 locked。

### 2026-07-16：Restoration v2.2 selector-geometry source freeze

- 新增 machine-readable contract
  `code/configs/causalcache_restoration_v2_2_selector_geometry.json`，SHA256 为
  `8022dcdec272916b7975d696a3ce6b54022c7414cd348a715c55b0d3d694dad5`；它绑定 exact-label raw
  `8f6baae5...`、derived dataset `89f136ab...` 与既有 baseline formula identity；
- 主切片固定为 decision step 6 / `n=4,B=2`，step 5 / `n=3,B=2` 为 secondary，step 4 /
  `n=2,B=2` 明确只作无压缩 ceiling；budget curve 按 n 分开报告 `B=0..n`；
- selector 固定为 exact at-most-B、true conditional greedy、budget-conditioned independent、full-path Shapley
  independent、recent 和 analytic exact-cardinality random，并增加 exact-cardinality 与三类 forced-fill
  non-monotonicity sensitivity；actual $U(S)$ 是唯一评价值，score sum 不能替代；
- 统计单位固定为 trajectory：train/development 分别整块 bootstrap，overall 按 split 分层，10,000 次、seed
  271828、90% percentile interval。primary 分开报告 search gap 与 objective-projection gap；interaction strength
  使用 `mean(abs(I))/D(empty)`，只从 train 按 n 取 tertiles，再应用 development；
- 内部 method-shaping 规则已在读取 selector aggregate 前固定：set-conditioning green 需要 development primary
  gap 至少 0.05、至少 4/5 trajectories 同方向且 bootstrap lower bound 大于 0；gap 不超过 0.01 时优先
  independent；search 则分 green/yellow/strong-diagnosis 三档。它们都不是 paper success 或 confirm unlock gate；
- 本里程碑只实现 contract、policy-free reducer primitives、统计与 tests。新 policy/generation/teacher/KL、gate
  training、matched-NLL、closed-loop、confirm/test 和 policy-vision feature forward 均为 0；正式 table aggregate
  只能从本 source push 后生成。

### 2026-07-16：Selector-geometry reporting-completeness audit 与 v2 repair source

- v1 table replay 从 clean pushed `main@7a1a8cc28a9a8f6a745f372406fe751b9bf4ff53` 生成 180 条
  state-budget records；两套独立 raw-table reducer 对 primary 225 个 state-level fields、15 个 aggregates 与
  selected coalitions 的 mismatch 均为 0，第三套审计也复算了 budget grid、bootstrap 和 shaping；
- contract-completeness audit 发现 v1 output 不完整：冻结 contract 要求联合报告
  `role × n × B × interaction-strength × has-negative-marginal`，旧 reducer 只给 primary 的两个边际 gap 表；
  analytic exact-cardinality random 也缺少已知 cardinality、exact-match probability 与 expected Jaccard；
- 该问题不改变 selector coalition、utility/recovery、bootstrap 或 method-shaping，但按预先冻结的 bug 边界不能
  静默修改 v1。旧 result 未提交，临时 bytes 不属于 source of truth；
- 新增 versioned repair config
  `code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json`，SHA256 为
  `2d312f54559f67aafe7efec2656d23171000e8b0f41d2d3c923f6a3c8b43be4c`。它 exact-bind parent v1 config/source，
  固定 train/development 共 144 个 joint Cartesian cells（含统一 empty schema、每 cell 全 10 selectors）与
  analytic-random 的 `k=B` / expected match / expected Jaccard；
- v2 reducer 同时重建 legacy payload 并逐字段验证 selector/utility/recovery/bootstrap/shaping 不变；独立 contract
  validator 从 180 rows 重算 train-only per-`n` tertiles、development assignment、random expectations 与完整
  cell key inventory。source-only targeted regression 为 43/43 PASS，全量回归为 636 tests PASS（skipped 11），
  `compileall`、`git diff --check` 与 AAAI LaTeX 编译通过；正式 v2 result 仍未生成。

### 2026-07-16：Selector-geometry v2 repaired formal result

- repair source 已 commit/push 为 `main@9a4eca5a53c2a9a3340c6274b9fa5ff9012a5a64`。一次手写错误 full SHA 的
  invocation 被 clean-Git gate 在读取/写入前拒绝，canonical output 仍不存在；随后 exact `git rev-parse HEAD`
  invocation 才生成唯一 canonical result；
- canonical result 位于 `data/results/restoration_v2_2_selector_geometry_v2_repair/`，status 为
  `COMPLETED_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR`，pre-commit byte replay 返回
  `VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR`；complete scientific payload SHA256 为
  `cc505443a7efdc68c8eeca754f24c9143cabf72a090f024fde05011e783cbb21`，180-row JSONL SHA256 为
  `b3f67714bb5667ceda945a3cb953b8987108aef607d5819a617272d48450cb03`；
- result 与 artifact invariant test 已 commit/push 为
  `d0f25d812869d5fc7b58284a25abe3aa8049b0aa`；clean descendant `main` 再从 immutable raw labels 重建三份
  files 并逐 byte 比较，返回同一 `VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR`；
- v2 显式物化 144 个 joint cells，其中 83 nonempty、61 empty；每 cell 都有 10 selectors，random 在所有
  `B=0..n` 上报告 exact cardinality、analytic match probability 与 expected Jaccard。selector、utility、
  recovery、bootstrap 与 shaping 相对 v1 invariance validator 全部通过；
- primary `n=4,B=2` 的 train/development/overall normalized recovery：exact 与 true greedy 均为
  `0.877376 / 0.925342 / 0.893364`，budget-conditioned independent 为
  `0.835977 / 0.766303 / 0.812753`，full-path Shapley independent 为
  `0.815367 / 0.884128 / 0.838287`，recent 为 `0.554136 / 0.019666 / 0.375979`，analytic random 为
  `0.430032 / 0.139471 / 0.333178`；
- exact 与 true greedy 在 primary 15/15 states 相同，development search gap 为 0；budget-conditioned
  independent development objective-projection gap 为 `0.159039`，5/5 同方向，90% interval
  `[0.007836, 0.355867]`。冻结 decision 是
  `set_conditioned_main_candidate + online_greedy_sufficient`，不打开 confirm，也不是 paper success gate；
- caveat 保持不变：development 只有 5 条 trajectories；normalized gap 受一个小 `D(empty)` state 放大；
  full-path Shapley / forced-fill independent 把 development gap 缩到 `0.041214`。因此当前只决定 gate
  architecture 候选，不声称 independent selector 普遍失败；
- 一套不 import v2 reducer/contract 的独立审计对 180 rows、144 cells、45 个 interaction assignments、全部
  analytic-random endpoints 与 pre/post legacy projection 重算，`mismatch_count=0`；preserved selector
  projection SHA256 为 `d99066c850ce836f5ac851783568455499bfbb1b60f202a9d9bbcf935eb263bb`，bootstrap
  projection SHA256 为 `9d17311f1b2291a4a58da9a64013f2487a329630749dfc8757c1d3026b66a61c`；
- 第二套 stdlib artifact audit 还独立重建 complete payload hash、三组 10k bootstrap 与 83×10 个 nonempty
  cell-method summaries，全部一致。独立 scientific helper 本身不从 records 重聚合每个 cell 数值或顶层 wrapper，
  但 formal `validate` 会从 immutable raw labels 重建三文件并逐 byte 比较；该 coverage boundary 已记录，当前
  artifact 未发现自相矛盾。

### 2026-07-16：OCR/RGB baseline v1 source-only freeze

- 新增 machine-readable contract
  `code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline.json`，SHA256 为
  `08f57505d71e603d81d6915ef27cf008d0fe097a56d70a05f8b3519211e2e6f9`；当前没有 formal aggregate 或
  OCR/RGB result 数值；
- reducer 会验证完整 45-state / 420-row immutable label archive，只消费 15-state / 240-row primary
  `n=4,B=2` slice；固定 10 train + 5 development trajectories、75 张 unique images 与 60 个 candidate scores；
- archived OCR-token Jaccard 与 deterministic 256×256 RGB 16³ joint-histogram cosine 各占 0.5，固定选满 top-2；
  同时报告允许少选的 exact at-most-2 oracle 与 cardinality-matched exact-2 oracle；
- paired statistics 以 trajectory 为单位，固定 10,000 次、seed 271828、90% percentile interval；development
  五条 paired deltas 逐 trajectory 保留；
- 完整 derived files 会 hash/inventory，并对 nonselected identity 做 opaque scan；只有 allowlisted
  train/development records 做 semantic parse。confirm prompt/image/OCR 不解析、不提取、不评分，confirm/test
  feature access 为 0；
- GPU、OCR inference/model load、policy/policy-vision forward、teacher/KL、gate、matched-NLL 与 closed-loop
  全为 0。正式工作只允许 Hyper00 CPU reduction，当前 formal run 仍 pending；完整协议见
  `docs/restoration_v2_2_ocr_rgb_baseline.md`。

### 2026-07-16：OCR/RGB baseline v1 zero-score invalid attempt

- clean pushed `main@aa5898f25e2e7d647363fe701ac90134bf744a5c` 与 immutable input preflight 通过后，formal
  reducer 在 derived trajectories 第 0 行 identity scan fail closed；
- v1 lexer 错误地要求整行 `source_id` 恰好出现一次；immutable schema 在 top-level 和 nested selection
  metadata 合法地保存两个完全相同的值。这是 execution-contract bug，不是 scientific estimand 失败；
- byte-only forensic 确认 35/35 trajectory lines 各有两个相同 `source_id`，210/210 OCR lines 各有一个
  `image_member_path`；没有 semantic parse record content；
- selected trajectory/OCR semantic parse、image extract/decode、OCR/RGB/combined score、distance lookup 与 formal
  state row 均为 0；GPU/policy/gate/confirm/test operation 也全部为 0；
- canonical output 与 staging directory 均不存在；v1 identity 不得重跑。compact evidence 见
  `data/results/restoration_v2_2_ocr_rgb_baseline_v1_attempt/`；
- replacement 必须使用新 protocol/config/output identity，只修 exact per-file occurrence 和 equal-value validation，
  scientific inputs、feature、selection、statistics 与 operation ceiling 不变。

### 2026-07-16：OCR/RGB baseline v2 identity-repair source-only freeze

- 新 contract
  `code/configs/causalcache_restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.json` 的 SHA256 为
  `d68cb032ef3c56c330d57329507d409b20f878b7510bd09d88e2eff1f3f898f3`，绑定 parent source/config、v1 failure
  commit 与 exact-two failure files；
- repair 只把 trajectory identity occurrence 固定为每行 2 次且值相同；OCR 仍为每行 1 次。0/1/3 次、escaped
  identity 或同一行值不一致均 fail closed；nonselected record 仍不做 semantic parse；
- shared scanner 默认仍为 v1 的每行一次，v2 runner 才显式传 `(2,1)` 和新 protocol ID；synthetic regression
  证明两版本输出除 protocol identity 外一致，避免 silent same-identity retry；
- shared reducer parent/repaired bytes 与 parent→formal-source exact changed-path allowlist 均由 contract/runner
  fail closed；formal source 还必须是 failure commit descendant，source-only validator 要求 v2 output/staging absent；
- parent scientific contract、immutable inputs、feature、selection、statistics、operation ceiling 与 confirm/test lock
  全部不变。v1 没有 scientific payload，因此不声称 v1/v2 values 相同；
- formal v2 output 仍未生成。下一步从包含本 source freeze 的 clean pushed `main` 在 Hyper00 CPU runtime 执行，
  再做 byte replay、独立 state/aggregate audit 和 Git 回写；完整协议见
  `docs/restoration_v2_2_ocr_rgb_baseline_v2_identity_repair.md`。

### 2026-07-16：OCR/RGB baseline v2 formal comparator 闭合

- clean pushed `main@a9bede85ab8bd10623c5755b944b3c26865c6485` 在 Hyper00 container
  `sglang-omni-jaxan-07170035` 的 Python 3.12.3 / Pillow 12.2.0 CPU runtime 完成唯一 canonical aggregate；
  15 states、60 candidate comparisons、75 unique images，GPU/OCR inference/policy/gate/confirm/test operation 全为 0；
- 同一 source、immutable raw-label archive 和 exact-six derived projection 的 `validate` mode 逐 byte 重建
  exact-three files。scientific payload SHA256 为
  `5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2`，state JSONL SHA256 为
  `07e29538232620bbea3bb12d1c0ce2649ee505285f53daf006dea3f670ba2d0a`；
- train/development/overall mean normalized recovery 为 `0.771189/0.019571/0.520650`，exact coalition match 为
  `3/10、0/5、3/15`。overall 相对 budget-conditioned independent 的 difference 为 `-0.292103`，90% interval
  `[-0.625159,-0.046921]`；相对 recent/random 的 intervals 均跨 0；
- development trajectory `0141544666483837` 是唯一负 recovery：OCR/RGB 选 `[3,4]`，
  `D(empty)=0.000616046`、`D(selected)=0.002243123`、recovery `-2.641163`；exact `[1,2]` recovery
  `0.923640`。raw witness、hash/path、coalition lookup 与符号均一致，因此保留为 non-monotone + small-denominator
  failure，不删除、clamp 或改 denominator；
- 新 artifact regression 不依赖 formal reducer 聚合结果，锁定 exact-three hashes、canonical 15-row JSONL、
  scientific payload、HF/Git provenance、完整 operation dict、headline/bootstrap、异常点和 v1 zero-score failure
  boundary；原 source-only test 改为 result-stage aware；
- 另一套不 import 项目 reducer 的 raw-to-result 审计得到 44/44 checks、0 mismatch；60 scores、15 complete rows、
  comparator、aggregate、development deltas 与 7×10,000 bootstrap 的 max absolute diff 都是 0，并独立复原
  scientific payload hash。audit/remote/lexical projection SHA256 分别为 `9e842494...12df`、
  `3bf18032...07e2`、`70ba220d...b2c9`；
- exact-three result 与 artifact regression 已 commit/push 为
  `main@7e59591573cb31f178dfd07422cc2e3c8aeff573`；Hyper00 从该 clean descendant checkout 重建三份
  files，返回 `VALID_RESTORATION_V2_2_OCR_RGB_BASELINE_V2_IDENTITY_REPAIR`，worktree 保持 clean；
- 本结果的结论是 OCR/RGB 已闭合但 development 不稳的弱 non-learned comparator。没有新增 HF artifact；
  policy-vision、gate checkpoint、matched-NLL、closed-loop 与 confirm output 仍不存在。

### 2026-07-16：policy-vision feature-only source freeze

- 新增独立 machine-readable contract
  `code/configs/causalcache_restoration_v2_2_policy_vision_baseline.json`，SHA256 为
  `a2319f8ea52d53fa01487cdbcbfef20b86ac81dcfab1c8a36ce4583b0b503023`；formal feature/result 尚未生成；
- 准确 comparator 名称固定为 frozen-policy-backbone vision similarity。events 1--4 post-state 与 event-5/current
  只走 GUI-Owl 27 层 vision tower 和 final main spatial merger `pooler_output`，逐图 BF16 token rows 转 FP32
  mean/L2 后做 cosine top-2；不输入 goal、text、OCR、action/history summary；
- 新 feature-only runtime 使用 direct `AutoImageProcessor` exact-two tensor keys，禁止 tokenizer/chat template；
  H200/BF16/eager/TF32-off/eval/frozen/single-device、GPU UUID/PCI/nvidia-smi、Pillow/Transformers/model snapshot
  均 fail closed。top model、language model、LM head 与 generation 路径安装 poison guard，正式 operation 为 0；
- identity 直接绑定已验证 OCR/RGB exact-three 的 15 states/75 images，但不复用其 feature/score/selection。
  feature worker 的参数类型不含 restoration labels 或 geometry utility；只有 60 个 canonical cosine 和 coalition
  合并完成后，CPU reducer 才解析 immutable 45-state/420-row raw labels 做 15 次 selected-coalition distance
  lookup；GPU 前只做 raw archive path/size/SHA256 byte-identity verification；
- 双 H200、无 DDP，按原 primary state index parity 固定 8/7 shards。15 个 canonical batches 各只预处理一次，
  复用同一 CUDA tensors 做 canonical + same-device replay；odd worker 另跑 ordinal-0 cross-device sentinel。
  总 schedule 为 16 processor batches、80 image assignments、31 vision forwards、124 cosine scalar transfers；
  replay tolerance 预注册为 absolute `1e-6` 且 ranking/coalition 必须 exact match；
- processor-only、zero-model-forward 输入检查固定 75 图 grid histogram、767,020 raw patches 与 191,755 merged
  tokens；这些只作为输入 identity，不是 policy-vision scientific output；
- 输出仅允许 Git exact-three lightweight files，不保存 4096-d embeddings。统计将包含 train/dev/overall
  recovery、small-denominator sensitivity、exact/exact-B geometry、对 restoration-aware/recent/random/OCR-RGB
  的 paired bootstrap、cosine saturation/margin 与预注册 OCR 异常点；负 recovery 不删、不 clamp；
- confirm/test、gate、matched-NLL 与 closed-loop 仍为 0。source-only validation 和正式执行边界见
  `docs/restoration_v2_2_policy_vision_baseline.md`；source push 后才允许 formal GPU run。正式入口为
  `code/scripts/run_restoration_v2_2_policy_vision_baseline.py` 的 `run` 模式，显式区分 Hyper SSH alias、宿主
  hostname、容器内 hostname 与 Docker object name，并读取宿主 `docker inspect`/`nvidia-smi` 生成的 evidence
  JSON；结果 commit 后只用其 `validate` 模式做 CPU reconstruction。

### 2026-07-16：policy-vision v1 zero-feature invalid attempt

- 唯一 v1 invocation 从 clean pushed `main@c0937056e94d110cd67e593288f9e0c3a3b24809` 在 Hyper00 GPU 0/1
  启动，10 秒 preflight 中两卡均为 0% 且容器内只暴露这两张 H200；
- runtime 在 `torch.cuda.get_device_properties(device).uuid` 上观察到 `torch._C._CUuuid`，而 v1 canonical
  parser 只允许 `str/bytes`，因此在 model snapshot full hash、processor/model load 或 feature forward 前抛出
  `ValueError`；
- canonical output 与 hidden staging 在失败前后均不存在；processor batch、vision forward、cosine、selection、
  semantic label load、gate、matched-NLL、closed-loop 与 confirm/test 均为 0；
- v1 不重试。轻量 evidence 位于
  `data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/`；下一步先冻结只接受 pinned
  `torch._C._CUuuid` 后继续原 UUID/PCI/`nvidia-smi` cross-check 的 versioned replacement，科学契约不变。

### 2026-07-16：policy-vision GPU UUID type-only v2 source freeze

- 新 repair contract 为
  `code/configs/causalcache_restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair.json`，SHA256
  `23733169ef5ba60a84f4447080ef12e1893858aa375ff50d8d4e3595699e774e`；它绑定 parent v1 source、
  `a809a908` failure exact bytes、zero-feature 计数与 v1 canonical/staging 必须继续不存在；
- semantic repair 只有一项：在显式 v2 profile 下要求
  `type(observed_uuid) is torch._C._CUuuid`，再执行 `str(value)`，随后复用原 UUID format、expected UUID、
  `nvidia-smi` exact-one-row 与 PCI/index checks；同 module/name 的伪造类、普通 `str/bytes`、未知 profile 均拒绝；
- v1 默认 profile 与旧 runtime identity 保持不变。v2 runtime profile、protocol、run/valid status 和 canonical
  output 全部使用新 identity；15 states、75 images、双 H200 8/7 worker、31 feature forwards、replay tolerance、
  feature/statistics contract 均继承 parent；
- 共享 runner 的 v1 `run` 已永久 tombstone；v2 在 model/feature access 前对固定 persistent ledger 做
  `O_EXCL` durable claim。formal source commit 还必须相对 `a809a908` exact 匹配冻结的 17-path diff，不能用
  未声明的实现改动进入 H200；
- committed-result `validate` 从 recorded feature rows、immutable labels 与 witness 重建 exact-three bytes，不
  重新执行 vision features；独立审计的边界是 reducer/math，而不是第二次模型 forward；
- source validator 与 focused regression 已通过；这只授权一次新的 clean/pushed-main formal run，不是
  comparator recovery，也不解锁 gate、confirm、matched-NLL 或 closed-loop。

### 2026-07-16：policy-vision v2 zero-feature SizeDict-interface invalid attempt

- 唯一 v2 attempt 从 clean pushed `main@fe7640395d3b6aea2e5e3a8cc34a49efc5ba2d2f` 在 Hyper00 GPU 0/1
  启动；durable ledger 在 worker/model/feature 前成功 `O_EXCL` claim，UUID exact-type、expected UUID、
  `nvidia-smi`/PCI 与全部 source/input/host preflight 均通过；
- `Qwen2VLImageProcessor.size` 的实际类型是 `transformers.image_utils.SizeDict`。其 `dict(size)` 精确等于
  冻结的 `longest_edge=shortest_edge=2621440`，但 `isinstance(size, Mapping)` 为 false，因此 v2 在 policy model
  load、processor batch 或 feature forward 前按冻结 source fail closed；
- formal policy model load、feature、cosine、selection、semantic label load、result、gate、matched-NLL、
  closed-loop 与 confirm/test 均为 0；canonical output/staging 不存在。failure record 记录三次 post-failure
  processor-only diagnostic，只检查 type/repr/dict/Mapping，不加载 policy model 或运行图像 batch；该次数、
  traceback 与 failure-observed 时间属于本次执行记录，不提升为独立耐久日志证据；
- v2 ledger SHA256 为 `492e7c7a...4502`，停止的 container `sglang-omni-jaxan-07170602` 与 ledger 均保留。
  同一 protocol 不重跑；轻量 evidence 位于
  `data/results/restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/`。

### 2026-07-16：policy-vision exact SizeDict-interface v3 source freeze

- v3 overlay config SHA256 为 `794474d8bc60463ba10fdd772691461f5910ca5e5501f7cccff4c53542b84b7f`，
  绑定 v2 config、`main@b1d0755` failure commit 与两份 failure files exact bytes；formal source diff
  也必须以 `b1d0755` 为唯一 baseline。v1/v2 canonical output/staging 与新的 v3 output/staging 必须同时不存在；
- runtime 只新增 `(UUID-v2, SizeDict-v3)` profile tuple：要求当前 loaded
  `transformers.image_utils.SizeDict` exact type、精确 module/name、非 `Mapping`、四个 non-edge fields 为 `None`，
  且 `dict(size)` 恰为冻结的两个 edge keys/values；subclass、同名伪造类、generic iterable、extra/wrong key 都
  在 model load 前 fail closed；
- v1/v2 profile、runtime identity 与 metadata schema 保持不变。v2 runner 在 contract/input/model access 前永久
  tombstone；v3 使用全新 protocol、canonical output 与 fixed mode-0600 `O_EXCL` ledger，失败同样不能 resume 或
  same-protocol retry；
- 本里程碑没有运行 v3 model/feature、没有 comparator recovery，也不解锁 gate、matched-NLL、closed-loop 或
  confirm/test。下一步只能从包含本 source freeze 的 clean pushed `main` 做新的 Hyper00 GPU preflight 和唯一
  formal attempt。

### 2026-07-16：policy-vision v3 formal artifact 与 CPU replay failure

- 唯一 v3 attempt 从 clean pushed `main@a935a3cf5efb1fa7a952ca6f45a5994609b367e9` 在 Hyper00 新容器
  `sglang-omni-jaxan-07170635` 的 GPU 0/1 启动；config SHA256 为 `794474d8...b84b7f`，host evidence SHA256
  为 `85e9b58c...af462`，mode-0600 ledger SHA256 为 `6aabf8dc...891ef`；
- formal GPU schedule 完成 15 states、75 unique images、31 vision feature forwards。same-device 15-state 与
  cross-device sentinel 的 maximum absolute score difference 都是 0；policy/LM/LM-head/generation、gate、
  matched-NLL、closed-loop 与 confirm/test operation 均为 0；
- raw restoration-label archive bytes 在 GPU 前只做 immutable SHA 验证；selected coalition 完成后才对 labels 做
  semantic parse/join，feature workers 从未接收 label semantics；
- exact-three artifact 的 README/state-scores/summary SHA256 为 `70ce9496...ed1f`、`8c597726...58b9`、
  `5ed21d6c...1c0e`，scientific payload SHA256 为 `811e59c7...f48`。overall/train/development mean normalized
  recovery 为 `0.404691 / 0.535228 / 0.143615`，exact match 为 `3/15 / 3/10 / 0/5`；在 CPU replay repair 前
  这些只描述 pending artifact；
- 首次 CPU `validate` 在 `validate_feature_worker_provenance` 抛出
  `policy-vision row-to-worker provenance drifted`。根因是 reconstructor 保留 evaluated row 的七键 state，
  而 feature provenance 明确要求四键 `index/role/trajectory_id/state_id`；worker/device/GPU 和 scientific values
  未观察到漂移；
- bounded CPU diagnostic 只做上述四键 projection，其余 rows、summary、labels、witness 不变，三份 artifact
  全部逐 byte exact match。该诊断不替代 versioned repair；GPU ledger 已 claim，formal artifact bytes 与停止的
  container 必须保留，GPU 不重跑。failure binding 位于
  `data/results/restoration_v2_2_policy_vision_baseline_v3_cpu_validation_attempt/`。
- 独立 stdlib-only 数值审计复算了 15 states / 60 candidates 的 top-2、utility、recovery、exact match、所有
  comparator delta、development paired rows、bootstrap summaries 与 scientific payload，未发现 blocker。overall
  mean recovery 受一个小 baseline-distance 负 outlier 明显影响，ratio-of-sums 为 `0.703501`，后续与
  `0.404691` mean 并列解释。`exact_coalition_overlap_with_*` 表示 policy-vision selection 与对应 comparator
  selection 完全相同，不得误读为 comparator 与 exact-subset oracle 相同。

### 2026-07-16：policy-vision v3 validation-repair v1 source freeze

- 新协议只修 evaluated row 七键 state 到 feature provenance 四键 state 的反投影；旧
  `_feature_record_from_evaluated_row`、producer runner/config、exact-three artifact、failure evidence 与 GPU ledger
  均不修改；
- producer source=`a935a3cf5efb1fa7a952ca6f45a5994609b367e9`，artifact commit=
  `597f050297342d9f29eb383985c2014f5782b2bb`。repair contract SHA256 是
  `64f63ab7563c227423c15ef82d5fd74d8be11579248908c7ec2880137e5d6ddf`，formal source diff 从 artifact commit
  起精确锁定 13 paths，并记录 producer 与 validation 两套 Python closure；
- source tests 复现 legacy provenance failure，验证 exact 7-key→4-key projection、producer/failure/artifact Git
  binding、tamper/extra/symlink rejection、mock exact-three replay、无 GPU/model CLI、`0600` O_EXCL ledger、atomic
  sibling staging 与 repair-ledger readback；completion seal 在 publish 前另以 `0600` O_EXCL 创建，锁定 runtime
  identity、finished time 与两份 final output 的 exact size/SHA，并有 hostname/cwd/finished/output tamper tests；
- formal `run` 必须位于不暴露任何 `/dev/nvidia*` 的新 CPU-only container。它从 immutable raw labels 重建三份
  producer files，要求逐 byte 相同；新 result 只允许 `README.md/summary.json`，新 ledger 固定在
  `/data/experiments/causalcache/restoration-v2-2-policy-vision-v3-validation-repair-v1-attempt.json`，completion seal 固定在
  `/data/experiments/causalcache/restoration-v2-2-policy-vision-v3-validation-repair-v1-completion-seal.json`。source PASS 不把
  pending artifact 提前升级为 `VALID`。

### 2026-07-16：policy-vision v3 validation-repair v1 formal VALID

- 唯一 CPU-only audit 从 clean pushed `main@dbb45637cf79c3573bbbc6051b8b480e3f76d69d` 执行；validation
  Python closure 固定为 156 paths / `e8daf215ca51b4c4d9e5f8a82399ec70d7d68700351968cace3386e235160a1d`。
  运行位于 Hyper00 `node-radixark-16-0000` 的 container `sglang-omni-jaxan-07170735`（ID
  `02df32fa30f768217854fa12365ae81d64a7b6817f32ec36b0f9fe29bc439261`），image 为
  `hongccc/sglang-omni:dev@sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
  Docker DeviceRequests 为空；
- runtime 为 Python 3.12.3、container hostname `02df32fa30f7`、
  `Linux-6.8.0-1043-aiext-x86_64-with-glibc2.39`，工作目录是
  `/data/experiments/causalcache/policy-vision-v3-validation-repair-v1/repo/code`；开始/结束时间分别为
  `2026-07-16T23:53:34.740660+00:00` / `2026-07-16T23:53:36.344206+00:00`。runner 只看到 CPU，
  NVIDIA device nodes、`nvidia-smi`、CUDA runtime import 与 forbidden module import 均为 0；
- sibling audit 返回 `VALID_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`。两份 exact output 是
  README 836 bytes / `3492dbae9a13d3d1d70e7aacf2cd7b405d1f9cc5956af980c519c8fb3ceee7e9` 与 summary
  10461 bytes / `f7a5ff63a754d06d7b61dcc46516ee2ed22e0a6a3b92b8b0be8a68869cf362b1`；summary 的
  validation payload SHA256 为 `47c6122d151d03980cfffe984e73f61cdc287e6bc543eb09e4db45e6a8064e6d`；
- exact-three producer files 3/3 逐 byte 相同，feature/candidate denominator 为 15/60；所有 GPU/model/image/
  policy/teacher/KL/restoration/gate/matched-NLL/closed-loop/confirm/test operation 都为 0。唯一语义变化仍是 evaluated
  state exact 7-key→feature state exact 4-key，所有 non-state reconstruction fields 不变；
- mode-0600 attempt ledger 为 2918 bytes / `6b65bef9d6edb73ee4275e89923bfd0e6123fb275fccb2b5dda9e57245f7b5d3`，
  completion seal 为 771 bytes / `837c52f403dbb9f21bf968ff5a0ba10199d06fe36ee49431787eff5154c6a52b`。
  新 artifact regression 锁定 exact-two、canonical JSON、source/closure、replay denominator、zero-op 与 repair scope；
  frozen config 的 13-path formal source inventory 未修改；
- committed-descendant revalidation 已在独立 validation-repo 的 clean
  `main@174801112c58d831249fd54f4f8bc9af01524b44` 完成；runner 仍使用 source
  `dbb45637cf79c3573bbbc6051b8b480e3f76d69d`，返回
  `REVALIDATED_RESTORATION_V2_2_POLICY_VISION_V3_VALIDATION_REPAIR_V1`。postflight exact-two mode 0644 / hashes、
  attempt ledger 与 completion seal mode 0600 / hashes 均与 formal run 相同，且没有 GPU operation；
  validation 未修改 artifact、config、code 或 test。container 已停止并保留，exit code 137。

## 下一步

Subset-search v1 与 spatial audit artifact 均已闭合，不继续为 optimizer 本身追加算法，也不得重跑任何 audit
profile。v2.2-eager fresh-45 substrate 与 immutable artifact 已正式 PASS，不进入 semantic-key /
canonical-representative 补救分支；formal label v1 已 zero-forward fail closed，不能重跑。replacement v2 的 45-state
offline oracle/labels、private HF immutable artifact、selector-geometry v2 repaired result 与 OCR/RGB v2
comparator 已闭合；OCR/RGB v1 zero-score `INVALID` 历史保持不变。policy-vision v1 唯一 invocation 在
zero-feature UUID type probe 阶段 `INVALID`；type-only v2 跨过 UUID 后又在 zero-feature SizeDict interface check
阶段 `INVALID`，且 durable ledger 已阻止重跑。exact loaded SizeDict v3 GPU artifact 与 CPU state-projection repair
现已 formal `VALID` 且从 clean descendant 完成 `REVALIDATED`，不得重跑。独立数值审计已经完成且无 blocker；
policy-vision comparator lifecycle 与 validation chain 均已闭合。gate v1 formal-58 training 与 fresh-16 primary
现均已闭合；fresh-16 的 selector 与 set-conditioning gates 都是 `NO-GO`，旧 5 条 combined compatibility、
matched-NLL、closed-loop 与 confirm 不因该结果自动打开。

48/16 structural selection manifest、pre-output exposure ledger、policy-blind derived artifact、192-state substrate、
exact labels、ledger-neutral repair 与 repaired-label private-HF publication 均已闭合。formal-58 cache v1 的唯一
execution 仍永久 pre-semantic fail-closed；其 one-leaf transport-repair 已从 Source-A/Execution-B 完成 Hyper00
no-GPU cache build、exact-three private-HF publication、fresh immutable replay 与只读 revalidation。formal-58
cache 已被唯一 formal training B 消费；OOF/final fit、10-checkpoint private-HF publication 与只读 replay 均闭合。
claim-serialization repair 已生成 fresh labels、primary report、HF tag 与独立 immutable replay；完整 machine-readable
结果见 `data/results/gate_v1_fresh16_claim_serialization_repair_v1/`。只读 failure decomposition 也已完成、发布并
revalidate：search 通过且 student gap material，但 oracle set-conditioning positive support=`10/16` 未达到冻结
`12/16`，route=`NO_V2_CONDITIONAL_RESCUE`。因此不再为当前 conditional student 追加架构、搜索或 ablation，
也不执行 combined-21、matched-NLL、closed-loop 或 confirm。下一项是整理论文/项目复现材料，并明确把当前结果
作为 go/no-go negative evidence；若未来转向 restoration-guided independent 前置模块，必须另立问题、contract 与
untouched holdout，不能把已消费 fresh-16 重新包装为验证集。

### 2026-07-16：gate v1 preregistration source freeze

- machine-readable contract 为 `code/configs/causalcache_gate_v1_preregistration.json`，SHA256
  `37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b`；
- formal train / combined development / primary fresh development 固定为 58/21/16 trajectories、
  1682/609/464 conditional edges；confirm 20 条仍 sealed；
- conditional 模型为 330→64→64→1、25,409 params，independent 为 200→88→88→1、25,609 params；后者只做
  one-shot projected target，前者每轮读取 selected-set sum 并重打分；
- normalization、small-`D(empty)`、负 gain、分层权重、SmoothL1+ranking、CPU FP32 AdamW、五个 seed、
  train-only 5-fold OOF、LR/epoch tie rules 与 fresh-16 GO thresholds 已全部静态 hash；
- focused tests 6/6 通过，validator 返回 `VALID_CAUSALCACHE_GATE_V1_PREREGISTRATION`，conditional 与
  independent 的所有 roster weight sums 均精确为 `1/1`；
- 本里程碑没有 expansion policy output、gate training/checkpoint、development semantic access、matched-NLL、
  closed-loop 或 confirm operation。

### 2026-07-16：label expansion pre-output exposure source freeze

- 新账本绑定 split config、parent selection、16-Parquet source manifest、历史 pre-output exposure 与 spatial-audit
  prior-output exposure；structural manifest 落盘时再动态绑定其 SHA 与 generator revision；
- expansion-64 分别与 reference-8、old-train-10、old-development-5、prior-output-union-23、sealed-confirm-20 与
  all-reserved-union-43 做六个机械交集，预期全部为空；
- 输出只保留 counts/digests，不发 source IDs、instruction/action、image/OCR identity 或 policy/restoration value；
- source tests 6/6 通过，strict JSON、source drift、overlap、top-up mutation 与 overwrite 均 fail closed；
- 当前只冻结 source；canonical exposure manifest 尚待 structural manifest 从 clean pushed main 物化后生成。

### 2026-07-16：label expansion policy-blind derived source freeze

- 新 builder 固定 64 trajectories、320 shared events、192 decision views、384 images 与 384 OCR records，输出
  root metadata + 三个 payload 的 exact-six tree；
- artifact 不含 policy/restoration output；共享 events 只作为多-state storage，consumer 必须按每个 decision 的
  `history_event_step_ids` 切片，per-decision current expert action 不进入 view；
- formal preflight 在 OCR engine 初始化前重哈希原 16 个 Parquet、reload exact 64 rows，并验证 48/16 order、
  transport identity、mixed image extension/magic 与 observation 000--005 inventory；
- post-write 与 standalone validator 强制 full 384-record OCR replay、payload count/schema、content witness、
  exact-six、symlink/special-entry、generator/input bytes 与 legacy/confirm overlap fail closed；
- independent review 的 manifest-SHA、self-consistent reorder、共享 action claim、hardcoded PNG、payload count 与
  supplied-manifest provenance 问题均已修复；focused tests 14/14 通过；
- 当前只有 source，没有本地 canonical artifact、HF upload/immutable revision、policy output 或 restoration label。

### 2026-07-17：48/16 structural manifest materialized

- 从 clean pushed `main@d2a4705117e46cb25e0e35f0ee6b1a7fe1f94ef0` 运行唯一 policy-blind materializer；
- canonical manifest SHA256 为 `4aec4deffc6405c3d06ca3001d082e4fbd85ee44f55785708a0cee573edcd169`，
  committed validator 重建得到同一 bytes/summary；
- 固定 48 train +16 fresh development、192 states、1792 个完整 subset-distance slots、1856 条 deployment
  conditional edges 与 1984 次 planned teacher forwards；
- `policy_output_generated=false`、`restoration_output_generated=false`、`structural_manifest_only=true`；下一步先
  绑定 exposure ledger，仍不授权 policy/gate/confirm。

### 2026-07-17：label expansion exposure ledger materialized

- 从 clean pushed `main@50a3f736cd83ad76ab6073a685e2f9e52e485290` 绑定已提交 structural manifest；
- canonical ledger SHA256 为 `e5f7e7b5ffeb71ac117b5cb4a470eb31648978a734edb9a521125d3bda29a323`，
  source validator 与 standalone validator 返回同一 summary；
- expansion-64 对 reference-8、old-train-10、old-development-5、prior-output-union-23、sealed-confirm-20、
  all-reserved-union-43 的交集均为 0；IDs 与 semantic content 未发出；
- structural manifest SHA `4aec4def...cd169` 与 generator `d2a4705...4ef0` 已动态绑定；selection/top-up/output
  access declarations保持全零；
- 本账本只关闭 pre-output identity/exposure 边界，仍不授权 expansion policy/restoration output、gate training、
  development evaluation 或 confirm access。

### 2026-07-17：invalid expansion-label forensic P1 publication source freeze

- P0 deterministic USTAR 保持不变：406 members、3,880,960 bytes、archive SHA256
  `8e205d73196604c0d8d342f9415755b090b96e83555ca55c386b12b8427ca489`、tree inventory SHA256
  `bc481dd77e26764dad3458e91a3e0247ade6049af7adb1744672aa6f2fb437d1`；原 v1 attempt 仍永久
  `INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`，formal loader eligibility 仍为 false；
- P1 config SHA256 为 `affb54cf44a656394c983cebe5d004d3e1ff216537c43da942dc031fa6baaccb`，固定 private
  dataset repo、archive/sidecar path、no-overwrite tag 与三态 crash recovery。claim 在任何网络 mutation 前以
  mode-0600 原子 no-replace 写入；已有 completion 的 replay 不允许重建已删除的 remote；
- archive 与 sidecar 必须在 frozen-title 的同一个 commit 首次出现；恢复时同时核对 immutable head、direct
  predecessor 两 path 均缺失、tag target 与 immutable revision。split-commit、partial/mismatch、response-loss、
  tag drift、intermediate symlink 与 relative fresh root 均 fail closed；
- P1 专项 22/22、全部 expansion-label wildcard 79/79 通过，config canonical JSON、compileall 与 diff-check
  通过。本里程碑没有读取真实 token、没有访问 Hugging Face、没有上传，也没有解锁 CPU repair、formal gate、
  matched-NLL、closed-loop 或 confirm；下一步从 clean pushed `main` 保存外部 `git ls-remote` receipt 后执行
  唯一 P1 publication/fresh replay。

### 2026-07-17：invalid-forensic P1真实 invocation 在 tag interface 处 fail closed

- clean pushed `main@3941d82a8ad83daaec55726a799d4d200735428d` 完成离线 prepare：archive SHA
  `8e205d73...a489`、sidecar SHA `c67914d0...b561`；claim 为 mode 0600、1,383 bytes、SHA
  `0a07692b...a5de`；
- private pair commit `5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416` 已创建，direct predecessor
  `1e1bb828d196779e4fc855890922c4865fcd6458` 只有 `.gitattributes`；两目标 path 同 commit 首次出现，
  immutable readback 与 P0 strict USTAR reader 已通过；
- tag 已存在且 `dataset_info(tag).sha` 解析到 pair commit；但 `list_repo_refs().target_commit` 返回 annotated-tag
  object SHA `ca652858c3d59eab066eb7b31399690a65ea5f44`。v1 parser 要求该值等于 resolved commit，故抛出 exact
  `ValueError` 并在 completion 前停止；
- completion seal 不存在，final tag/immutable pre-post fresh attestation 未完成。remote pair/tag 原样保留，
  不重跑同一 P1 source、不重复 commit/tag、不改写原 expansion attempt。下一步冻结 read-only versioned
  tag-resolution repair；其 immutable fresh replay 闭合前，CPU scientific repair 与 formal gate 继续 locked。

### 2026-07-17：read-only annotated-tag resolution child source freeze

- child config SHA256 为 `f257dcdebb218533a080739e8a5067cc44fe35fd533ad57808db5ce789cfd956`，
  分别冻结 main=`5efe1ae8...b416`、annotated-tag object=`ca652858...5f44`、tag-resolved commit=
  `5efe1ae8...b416` 与 immutable revision=`5efe1ae8...b416`，不再错误要求 object SHA 等于 commit SHA；
- parent P1 source `3941d82...428d`、failure evidence `dcc37e5...47a`、config `affb54cf...accb`、mode-0600
  claim `0a07692b...a5de`、原 completion 必须缺失，以及 archive/sidecar/pair/predecessor exact bytes/inventory
  全部静态绑定；
- remote 只允许 `dataset_info/list_repo_refs/list_repo_commits/list_repo_files` 与 force-download，mutation call
  count 固定为 0。四身份与 exact pair provenance 在 fresh immutable download/P0 strict readback 前后均需一致；
  child 使用独立 mode-0600 原子 claim/completion，已有 completion 的重放仍做完整 read-only replay；
- child 专项 22/22、P0+P1+child 54/54 通过，独立审计无 blocker。本步骤没有触网、没有读 token、没有修改
  remote 或原 P1 state，也未解锁 formal labels；下一步只允许从 clean pushed `main` 执行真实 read-only child。

### 2026-07-17：invalid-forensic immutable transport read-only 闭环

- 从 clean pushed `main@d61dff5857bee7815da0c40709d0e7032fb9dc86` 在 Hyper00 执行正式 read-only child，
  status 为 `COMPLETED_READ_ONLY_INVALID_FORENSIC_TAG_RESOLUTION_V1`；随后按同一 contract 做第二次幂等
  replay，claim/completion bytes 与 SHA 均未变化；
- child claim 为 mode 0600、3,205 bytes、SHA `3dd268f7...3a15`；completion 为 mode 0600、3,971 bytes、
  SHA `c0f70c53...2195`。原 P1 completion 在两次 replay 前后均不存在；
- private HF annotated-tag object `ca652858...5f44` 与 tag-resolved / immutable pair commit
  `5efe1ae8...b416` 被分别验证；main、tag object、tag-resolved 与 immutable identity 在 fresh download 前后
  完全相同，pair/predecessor provenance 也无漂移；
- archive `8e205d73...a489`、sidecar `c67914d0...b561`、3,880,960-byte size、406 members 与 tree
  `bc481dd...437d1` 均匹配，P0 strict readback 通过，remote mutation call count 为 0；
- private HF dataset 现为 invalid-evidence reusable transport 的 canonical source，原 Hyper00 archive 只是
  staging copy。该结论保持 `formal_label_loader_eligible=false`、`gate_training_unlocked=false`、producer
  永久 `INVALID`；下一步冻结独立 CPU scientific repair core/runner，再发布新 identity 的 repaired labels。

### 2026-07-17：ledger-neutral scientific-repair core source freeze

- 新 core 只接受 P0 strict reader 的 `InvalidForensicEvidence` 与 explicit `InvalidForensicContract`，重跑完整 P0
  validator 并 exact compare dataclass/manifest/inventory/tree；formal tier 在 external replay 前额外绑定 frozen P0
  config `5290a51...d5a6a2` 与 tree `bc481dd...437d1`；
- completed-to-invalid ledger chain、三份 worker ledger、append-only invalidation suffix 与原 exact cadence failure
  都作为不可删除的 provenance。旧 3 秒公式仍完整重算并报告失败，新 acceptance 只使用 monitor lifecycle 的
  identity、索引、单调时间与首尾覆盖，不使用数值 gap threshold；
- formal external replay 独立冻结 parent/derived/OCR provenance，校验 production module/callable/source/code
  identity，并直接调用 import-time captured production replay；dynamic executor `__call__` monkeypatch、同名 callable、
  mutable parent/derived globals 与 synthetic P0 均在执行前 fail closed；
- core SHA256 为 `6b1c022758844284a0c8364a33aa12457db1fa08dba9e5567c8fe3a48cfa744c`，47,044
  bytes。focused 19/19、expansion wildcard 144/144、`py_compile` 与 diff-check 全部通过，独立 final re-audit 无
  blocker；
- Hyper00 read-only path exploration 在显式禁用 GPU、阻断 model-framework imports 时完成 production replay smoke：
  192 states / 1,792 coalition witnesses，约 276 秒。它不是 formal result；下一步仍需 source-freeze 新 no-GPU
  runner/config/state machine，并从 immutable HF invalid archive 的新空目录执行正式 child。

### 2026-07-17：no-GPU scientific-repair runner source freeze

- runner contract SHA256 为 `4f4202944674192090cf3df19b864c96e7082628ab25ca8287affa32944e64c2`；
  runner/config/CLI/test bytes 分别为 99,238 / 10,974 / 6,963 / 39,190，代码 SHA 分别为
  `f3e86d90...2205` / `4f420294...64c2` / `8d0f6434...97b6` / `d05c6fbe...7786`；
- formal source 必须 clean-pushed 且 local HEAD、`origin/main` 与实时 `git ls-remote` 相同；runtime 加载的全部
  `causalcache.*` / `scripts.*` 模块逐个绑定 Git tracked bytes，允许执行中 lazy-import closure 增长，但每次
  revalidation 都重新校验完整闭包；
- formal runtime 固定为 Hyper00 新建 unprivileged no-GPU container：Docker `DeviceRequests=[]`、无 explicit
  device、`NVIDIA_VISIBLE_DEVICES=void`、空 `CUDA_VISIBLE_DEVICES`、无 `/dev/nvidia*`。direct bootstrap 在
  package import 前安装 fail-closed framework import guard；PIL 是唯一允许的 image decode library；
- private HF invalid evidence 必须从 immutable commit `5efe1ae8...b416` force-download 到新空目录，并在 durable
  claim 前完成 identity、sidecar 与 P0 strict readback。正式状态顺序固定为 fresh read→claim→full core
  recompute→strict four-member USTAR readback→completion-last；crash-recovery 和 no-replace 规则已覆盖；
- artifact formal reader 从 192 raw states / 1,792 rows 重跑 projection validation、existing reducer 与独立 math
  audit，精确校验 1,856 deployment edges、3,072 full edges、1,984 interactions、576 attributions 和 192 exact
  oracles；不信任 stored summary，negative values 保留；
- focused 29/29、全部 expansion wildcard 173/173、direct-script source-only validation、`py_compile` 与
  `git diff --check` 通过。当前只得到 `VALID_SOURCE_ONLY_NO_GPU_SCIENTIFIC_REPAIR_RUNNER_V1`，没有 formal
  execution、network read、GPU/model operation、repaired HF artifact、gate training、matched-NLL、closed-loop 或
  confirm access；下一步在 clean pushed source 上执行正式 no-GPU child。

### 2026-07-17：no-GPU scientific-repair formal VALID + REVALIDATED

- 唯一 formal run 从 clean pushed `main@b14f489fe55b51a83917b57e6a54fb73d268342a` 在 Hyper00 的新
  unprivileged no-GPU container 执行；launch receipt SHA256 为 `75931689...a2926`，DeviceRequests/explicit
  devices/NVIDIA nodes 均为空，framework import guard 生效；
- `run` 返回 `VALID_REPAIRED_EXPANSION_EXACT_LABEL_SCIENTIFIC_PAYLOAD_V1`；claim SHA256
  `f2a19d6f...70f5`、completion SHA256 `6a477535...4905`，completion 在 strict output readback 后最后写入；
- 4-member deterministic USTAR 为 3,020,800 bytes / `1a9fdcc0...0e01`，tree `09bb681b...f44b`。完整
  `validate` 再次 fresh-download/recompute，返回 `REVALIDATED_REPAIRED_EXPANSION_EXACT_LABEL_SCIENTIFIC_PAYLOAD_V1`，
  且 claim/output/completion 全部 `created=false`、bytes 不变；
- 64 trajectories / 192 states、1,792 raw distances、1,856 deployment edges、3,072 full edges、1,984
  interactions、576 attributions 与 192 exact oracles 全部重算一致；独立 math audit PASS，external state
  projection 192/192 byte-identical；
- 原 cadence negative control 保留 5/3,849 >3 秒、max 3.883721 秒；structural lifecycle rule PASS 且不使用
  numeric gap threshold。原 producer 仍永久 `INVALID`，P1 completion 仍缺失；
- repair runtime 的 GPU/model/policy/teacher/KL/gate/matched-NLL/closed-loop/confirm/test/remote-mutation operation
  count 全部为 0。轻量结果位于
  `data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_v1/`；local artifact 还不是 reusable SoT，
  README/summary 分别为 2,483 bytes / `dd9e7858...dd84` 与 11,691 bytes / `4b5aaa84...0f59`。下一步冻结
  独立 HF publication/fresh-replay contract，gate 继续 locked。

### 2026-07-17：repaired-label private-HF publication source freeze

- publication config SHA256 为
  `547b291022e2430fe6af5158d9350132f405c4b6019198488386dd7fb50be0aa`，唯一 destination 是 private dataset
  `gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile` 与 tag
  `v2.2-expansion-exact-labels-scientific-repair-v1`；本 source-freeze 没有创建或读取该 destination；
- producer binding 固定 local archive `1a9fdcc0...0e01` / 3,020,800 bytes / four-member tree
  `09bb681b...f44b`、mode-0600 claim `f2a19d6f...70f5`、mode-0600 completion
  `6a477535...4905`、repair source `b14f489f...342a` 与 lightweight result `534bd7ae...89e7`；
- remote state machine 只接受 `EMPTY`、`BYTE_IDENTICAL_UNTAGGED`、`TAGGED_BYTE_IDENTICAL`。mode-0600
  remote-base receipt 在 content mutation 前绑定 base main、完整 reachable history 与 recursive tree/blob
  inventory；pair history 必须只多一个 commit，去掉两 target 后既有 blob identity 必须完全不变；
  partial/mismatch/split/interposed/extra-path、existing-blob modification、overwrite、CAS conflict 和 tag conflict
  均 fail closed；
- annotated-tag object 与 tag-resolved commit 分开绑定。Hyper00 的 `huggingface_hub==1.16.1` API signature 已
  核对，并在既有 private annotated tag 上实测 refs target 与 resolved commit 确实不同；publication completion
  同时记录两者，要求 resolved commit 等于 immutable exact-pair commit；
- create-repo/commit/tag response-loss 只能通过 readback 恢复。publication claim 与 remote-base receipt 均为
  durable mode-0600 no-replace 且保持 gate locked；immutable force-download、exact pair provenance 与 strict
  repaired transport readback 全部通过后，保留的 completion stage 才 hard-link 为 final seal。stage/final 必须
  same inode，final link 是最后 namespace mutation；completed replay 不允许重建已删除的 remote；
- focused fake-HF tests 25/25、scientific-repair wildcard 73/73、`py_compile` 与 `git diff --check` 通过。为核对
  当前 Hub annotated-tag 语义，仅用 token 对既有 invalid-forensic private repo 做了只读 object/resolved 查询；
  没有访问或修改 repaired destination，没有 gate fit、matched-NLL、closed-loop 或 confirm access。
  下一步先 commit/push source，再从独立 clean checkout 执行真实 publication 与幂等 immutable replay。

### 2026-07-17：repaired-label private-HF publication 与独立 postflight 闭合

- 从 clean pushed `main@e47c50665c24fed5d9886665233962f85c1210e0` 在 Hyper00 CPU-only container 执行唯一
  publication；没有 GPU/model/policy/gate/matched-NLL/closed-loop/confirm operation；
- 新 private dataset
  `gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile` 的 tag
  `v2.2-expansion-exact-labels-scientific-repair-v1` 分别绑定 annotated-tag object
  `6a907ba2a3dce07c9f0810a84a8755c389da4a57` 与 tag-resolved immutable pair commit
  `7a6c254b8cec0dd3d8111dfc9c080de357e5cef3`；
- 3,020,800-byte 4-member USTAR 的 SHA256/LFS oid 均为 `1a9fdcc0...0e01`；sidecar SHA256 为
  `9d86c755...c4ee5`。base recursive blob inventory 与 pair non-target inventory 完全相同，证明 pair commit
  只新增两个冻结目标；
- completion mode 0600、7,948 bytes、SHA256 `995b3ee2...a2ff`，retained stage 与 final 为同一 inode。
  第二次 invocation 返回 byte-identical completion 且不执行 remote mutation；
- 新空目录的独立 force-download postflight 返回
  `PASS_INDEPENDENT_READ_ONLY_REPAIRED_PUBLICATION_POSTFLIGHT_V1`，重新验证 private visibility、tag/main/
  immutable identity、exact pair、remote blob/LFS 与 strict 4-member archive；
- 该里程碑令 `formal_label_loader_eligible=true`、`gate_training_unlocked=true`，但只清除 formal-58
  label-data prerequisite。原 producer 和 invalid-forensic payload 不重分类；gate 未训练，fresh-16、matched-NLL、
  closed-loop 与 confirm 仍未解锁。轻量结果位于
  `data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/`。

### 2026-07-17：gate v1 formal-58 train-only cache Source-A freeze

- 新 source-only config 为 `code/configs/causalcache_gate_v1_formal_cache_v1.json`，SHA256
  `1d7527e8a7bede8aaab8a21f7757f786674530238ae99fe3ce5b196cbce67261`；固定 legacy train-10 + expansion
  train-48、58 trajectories / 174 states / 522 candidate features / 1,624 raw distances / 1,682 conditional
  targets / 522 independent targets；
- selective core 把调用边界拆成 feature-only build、label-only build 与 join-only audit。feature completion
  durable 落盘后才能创建 label-access claim；claim 前 label transport download 与 semantic decode 均为 0，claim
  后也只 decode 30 个 legacy train states、144 个 expansion train states 与 290 条 selected OCR records；
  development/confirm semantic decode 为 0；
- feature/label cache 是两个独立 deterministic USTAR，float 使用 big-endian binary64 lowercase hex。cache rows
  不保存 role/split/raw text；label cache 不含 q64/h64/g8，feature cache 不含 D(S)。repaired label 必须联合绑定
  immutable archive、sidecar 与 mode-0600 publication completion；
- Source-A/Execution-B 分离：A 包含 config/contract/core/runner/CLI/tests，B 只能新增机械生成的
  `causalcache_gate_v1_formal_cache_runner_v1.json`，且必须是 A 的 direct single-parent child。正式 execution 还要求
  HEAD、origin/main 与 live `git ls-remote` 一致，并固定 Hyper00 no-GPU CPython/runtime/env；
- gate wildcard focused suite 共运行 61 tests，结果 OK（1 个本机缺 PyTorch 的 optional skip）；显式绑定仓库
  `code/` source path 的全量 suite 共运行 1,089 tests，结果 OK（12 个 optional runtime skips）。source-only
  validator 保持 network/write/GPU/model/training/semantic access 全零。当前未生成 runner freeze B、cache、HF
  destination/tag/completion，也没有 gate fit、development metric、matched-NLL、closed-loop 或 confirm access。
  下一步只允许从 clean pushed A 机械生成并单独 push B。

### 2026-07-17：formal-58 cache v1 pre-semantic transport failure

- Source-A=`990f015b02e0dc89dedaab3853fab9f07fb0884d` 已 push；从 clean A 机械生成的 runner freeze SHA256
  `9e941eaff5cc44a44784d22a7568a692d460aea9b07cfd580bf362940158f106` 作为唯一 diff 的 direct-child
  Execution-B=`079c0952017a9e2936f5741bbe15345255b0e481` 单独 commit/push；
- Hyper00 container `sglang-omni-jaxan-07172332` 使用 CPython 3.12.3/x86_64、`runc`、无 GPU DeviceRequests/
  device nodes 与冻结 thread env。live private `origin/main` 与 B 一致；
- run 创建 mode-0600 global claim 后，在 `_verify_downloads()` 发现 `expansion_feature_trajectories` transport
  mismatch：config 错误值为 `00fe93e9...353a6d`，immutable bytes、producer preupload/fresh-download/payload
  witnesses 与 Hyper00 `sha256sum` 均为 `fe93e9de...353a6d`。两者都是 64 位合法 SHA；错误值误插前导 `00`
  并漏掉真实 digest 中连续出现的第二个 `fe` byte；
- claim 为 7,062 bytes / SHA256 `a9372c7a...54b1b`，旧 namespace 的 feature completion、label-access claim、
  label completion、remote receipt、staged/final completion 与两个 cache 全部不存在。失败发生在任何
  trajectory/OCR/label semantic decode 之前，因此 semantic decode、GPU/model、training、HF mutation、matched-NLL、
  closed-loop 与 confirm operation 均为 0；read-only postcheck 证明 destination repo 仍不存在；
- 旧 claim 不删除、不覆盖、不续跑。下一步必须用独立 transport-repair overlay 绑定旧 claim 与 Git-pinned
  producer completion，只允许修复该一个 transport SHA leaf，并更换全部 local state/cache/HF tag/target namespace，
  再重新执行 Source-A → machine-generated Execution-B。轻量证据位于
  `data/results/gate_v1_formal58_cache_v1_attempt/`。

### 2026-07-17：formal-58 transport-repair Source-A 冻结（历史 source freeze）

- repair 以 parent cache contract 的独立 overlay 记录，不修改旧 v1 config、旧 A/B、旧 failure summary 或旧
  claim。唯一允许变化是 `expansion_feature_trajectories` 的
  `derived/restoration-v2-label-expansion-v1/trajectories-00000-of-00001.jsonl` transport SHA：原错误 binding
  `00fe93e9...353a6d` 改为实际 bytes `fe93e9de...353a6d`，size 固定为 1,245,673 bytes；data bytes 与 semantic
  contract 均不变；
- 正确 digest 必须同时复核 Git 固定 producer artifact 的 fresh-download、preupload、payload-manifest 三个 file
  record。所有其他 input leaves 归一化回 parent 后必须完全相同；
- repair `run`/`validate` 在 token、Hub client、download 或新 claim 之前，先验证旧 global claim 仍为 mode 0600 /
  7,062 bytes / `a9372c7a...54b1b`，旧 B=`079c095...0e481`、failure summary 与旧 cache/successor-state absence
  均未漂移。因此 repair 不能把 v1 fail-closed path 当作可 resume 的状态机；
- repair 使用新的 local claim/cache 前缀，以及独立 private HF repo
  `gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile`、tag
  `gate-v1-formal58-cache-transport-repair-v1` 与 exact-three target paths；
- 这是执行前的 historical source-freeze record；随后发生的唯一 B、formal run 与 read-only revalidation 见下一条。

### 2026-07-17：formal-58 transport-repair cache 完成并 revalidated

- repair Source-A=`4f8c01b026167d6e9429716a082f3abd7c0c1bc9` 的 config
  `code/configs/causalcache_gate_v1_formal_cache_transport_repair_v1.json` SHA256 为
  `aaf82fd5e994588bc22f0f745139e349219863eee5fa8cb4cc93c4a9e48e123a`；唯一 direct-child
  Execution-B=`f96c197c0fd31bfd299b5ab6e9e1416f6183bc6d` 只新增 runner freeze，SHA256 为
  `14141d0d790c4cf4d6c12b3bc336e1e6d2ebd26fde5f4be40b58627bd86a2b34`；
- Hyper00 unprivileged `runc` no-GPU formal `run` 返回
  `VALID_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`。新 claim 前复核 retained v1 claim、六个旧 successor 的缺失、两份
  旧 cache 的缺失及 producer 三重 witness；旧 v1 invocation 仍为永久 `INVALID`，没有被删除、覆盖或续跑；
- run 物化并 strict-readback 58 trajectories、174 feature/label states、522 candidate features、1,624 raw
  distances、1,682 conditional targets、522 independent targets 与 174-state join audit。feature/label USTAR SHA256
  分别为 `81fded50c4450700220742d3e0be9a5585d1bc51086150515b463bbdf4b4df8e`（993,280 bytes）与
  `4f9ef172aaa94c3ea8ce53aa43336c9fcb7800c24e181e1462d8239e31053cee`（163,840 bytes）；
- canonical reusable artifacts 是 private HF dataset
  [`gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile)，tag
  `gate-v1-formal58-cache-transport-repair-v1` → immutable commit
  `a61b31bf2e69be00f94469f4a2f2d6b336fcc386`，annotated-tag object
  `c603397b9b1b1c3ad472f49125f827b8a096997d`；
- 同一 committed B 的只读 `validate` 返回
  `REVALIDATED_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`，没有创建 claim/cache/state，remote mutation count 为 0。
  completion 令 `formal58_cache_loader_eligible=true` 与 `formal58_training_input_eligible=true`，但运行中
  training example、optimizer step、OOF metric、checkpoint、model load/forward、matched-NLL、closed-loop、development
  与 confirm operation 都为 0。结果记录在
  `data/results/gate_v1_formal58_cache_transport_repair_v1/`；下一步仍需要单独的 train-only gate source/execution
  contract，不能提前打开任何 downstream evaluation。

### 2026-07-17：gate v1 formal-train Source-A 冻结

- machine-readable config 为 `code/configs/causalcache_gate_v1_formal_train_v1.json`，SHA256
  `bff920266b3f691618005b239c8a0aaa99369b45c0612ae628eec1a8f7eebf2f`。Source-A 精确绑定 repair feature cache
  `81fded50c4450700220742d3e0be9a5585d1bc51086150515b463bbdf4b4df8e`、label cache
  `4f9ef172aaa94c3ea8ce53aa43336c9fcb7800c24e181e1462d8239e31053cee`、bundle manifest
  `15c8bf56ddad4f6f278599c32aaadcd813e0e016db0d523b4892eb47f8d9d144`、HF tag
  `gate-v1-formal58-cache-transport-repair-v1` 与 resolved immutable commit
  `a61b31bf2e69be00f94469f4a2f2d6b336fcc386`；
- formal-58 join audit SHA256 为
  `551e70b7e99f7a761f9c50d2adae3be72b0933044982a015a65e93372e41d77b`，gate v1 preregistration
  SHA256 为 `37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b`；任一上游 hash、
  roster、fold assignment 或 training contract 漂移都 fail closed；
- access firewall 只允许 formal-58 semantic decode。Source-A 本身 semantic decode 为 0；未来 B 训练也必须保持
  fresh-16 semantic decode=0、legacy dev-5 semantic decode=0、confirm-20 access=0、matched-NLL=0 与
  closed-loop=0；
- OOF/final-fit 输出 schema 已冻结：conditional 和 independent 各 `2 LR × 5 seed` 的完整
  OOF grid，每 trial 五个 fold model，合计 100 条 fold-training track；每 family 用选定 LR 和每
  seed selected epoch 在全部 58 trajectories 上 final refit，最终计划 5 conditional + 5 independent
  checkpoints；
- 预期 completion 必须保存完整 OOF grids、selected LR/epochs、selection SHA、十个 canonical
  model-state/checkpoint artifact SHA、family training reports、ensemble manifests、run manifest、operation counts 与
  cache/source/runtime provenance。这些都是未来输出，Source-A 里不存在；
- private HF model destination 已冻结为 `gavinlaw/causalcache-gate-v1-formal58-selector-mobile`，tag
  `gate-v1-formal58-train-v1`；在 Source-A 当时 repo/tag/revision/artifact 均已验证 absent；
- Source-A validator 只做 source/config/hash 与 absence validation，明确返回 `training_executed=false` 与
  `execution_authorized=false`。该历史 source-freeze 时 Execution-B 尚未生成，optimizer/OOF/checkpoint/model upload/fresh-16/旧
  dev-5/confirm/matched-NLL/closed-loop 均未执行；source-only replay 验证 17 个 Source-A paths，network/HF/
  write/torch/semantic-decode 计数全为 0。显式绑定仓库 `code/` 的全量 suite 为 1,138 tests，结果 OK
  （13 个本机 optional runtime skips）；Hyper00 pinned no-GPU container 的 54 个 formal-train/pipeline tests
  全部通过且无 skip，包含真实 PyTorch/safetensors checkpoint canonical replay。host `docker inspect` 也确认
  `DeviceRequests=null`、unprivileged `runc`、container hostname/image identity 与唯一 `/data02/jaxan:/data`
  bind。该历史 milestone 当时的下一步只能从 clean pushed Source-A 机械生成唯一 runner-freeze B，不得修改
  trainer；实际 B/result 见下一条。

### 2026-07-18：gate v1 formal-58 training publication + replay 闭合

- Source-A=`e20f004ab79afaba4a04a108e70779d49087b2b9` 已机械生成唯一 direct-child
  Execution-B=`bad28b74c421ccf6be1ab2f4407f7bad3414a2f2`；B 只新增 runner freeze，SHA256 为
  `a89bb081c6a98e4da7da178e8930357773e7ee23cea211600cffea4500ba7532`；
- Hyper00 的 pinned unprivileged `runc` container 以 no-GPU、CPU FP32、单线程执行 formal run。58
  trajectories / 174 feature states / 174 label states 完整 join；fresh-16、legacy dev-5、confirm、matched-NLL
  与 closed-loop operation 全为 0；
- conditional / independent 各完成 `2 LR × 5 seed × 5 folds`。五 seed mean OOF
  raw-utility/oracle ratio 分别为 `0.8925353801368878` / `0.9063764691683989`，两者均选 LR `3e-4`；
  final epochs 分别为 `[62,59,98,4,13]` / `[60,51,6,56,54]`。OOF 只作 train-only LR/epoch selection，
  不是 generalization metric；
- 10 个 safetensors、2 个 full OOF reports 与 4 个 manifests 已发布到 private HF model
  [`gavinlaw/causalcache-gate-v1-formal58-selector-mobile`](https://huggingface.co/gavinlaw/causalcache-gate-v1-formal58-selector-mobile)。payload
  commit=`a6c9e7f6dab6bc27794438b5b66da07fd59b2889`，manifest commit=`23f6786075c7bff91f93fd7e8a878e070efb72a9`，
  annotated-tag object=`fa85e74685d5ab509e60b469c6c8cae61efab4d6`；
- 正式 `run` 返回 `VALID_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1`，完成态只读 replay 返回
  `REVALIDATED_GATE_V1_FORMAL58_TRAIN_PUBLICATION_V1`，逐个重放全部 checkpoint 且 remote mutation count 为
  0。completion/staging 是 mode-0600 同 inode hard-link；完整轻量记录见
  `data/results/gate_v1_formal58_train_v1/`；
- `gate_trained=true` 只关闭 train-only model seal，不是 selector/set-conditioning GO。现有 heuristic runner
  硬编码 `n=4,B=2`，而 fresh-16 含 `n=2/3/4`；因此下一阶段必须在任何 fresh semantic access 前冻结
  variable-`n` recent/OCR-RGB/policy-vision 语义与 n=4 compatibility test，并新增只解析 fresh rows 的 selective
  loader。fresh primary 封存/replay 后，才能另立 combined-21 contract 打开旧 dev-5。

### 2026-07-17：fresh-16 primary evaluation Source-A 冻结

- 新增 machine-readable Source-A contract
  `code/configs/causalcache_gate_v1_fresh16_evaluation_v1.json`，精确绑定 preregistration、formal model
  completion、10 个 checkpoint、derived input、repaired-label input 与 GUI-Owl snapshot；formal model seal 不可
  因 fresh evaluation 修改或重训；
- 冻结 fresh roster digest 与 16 trajectories / 48 states / 144 candidate feature occurrences / 448 distances /
  464 deployment conditional edges。trajectory selective reader 只允许 rows `[48,64)`，label selective reader
  只允许 rows `[144,192)`；generic full semantic reader 被拒绝；
- 新增 `n=2/3/4,B=2` 的 dynamic recent、OCR/RGB 与 policy-vision natural extension，并要求 `n=4` 与旧
  comparator score/order/selection 完全兼容；conditional learned selector 保持 coalition-conditioned rescore，
  independent 保持 one-shot；
- 冻结未来双 H200 policy phase 为 2 workers × 24 states、49 processor batches、97 vision feature forwards 与
  292 cosine scalar transfers。这里只是 operation contract；本 Source-A 没有 model load、forward 或 GPU job；
- label firewall 要求 feature/heuristic/learned selections 全部完成 local durable seal 后，才创建 label-access
  claim并读取 48 label states。旧 dev-5、confirm、matched-NLL、closed-loop 在整个 primary execution 中保持
  0；
- publication 固定为 9-target payload commit 后接 direct-child 4-target report commit，annotated tag 只指向
  report commit，13 files 必须 immutable fresh-download byte replay。primary report 完成、hash 与 replay 之前，
  不得打开旧 dev-5；
- label-firewall 进一步封存 conditional/independent ensemble 与全部五 seed decisions；conditional artifact 保存每轮
  candidate score vector、add/stop 与 coalition transition。label 解封后只允许 pure sealed-decision evaluator；
  report commit 前先做 local 13-file replay，远端 replay 再加载 10 checkpoints 并用刚加载的 formal provenance
  byte-rebuild learned/report/state artifacts；
- config SHA256 为 `c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e`；source-only
  validator 返回 37-path inventory，network/write/torch import/fresh semantic decode/model/report 全部为 0，且
  `evaluation_executed=false`、`execution_authorized=false`。focused fresh16/pipeline/formal suite 为 101 passed、
  2 skipped、62 subtests；全 `code/tests` 的 1197 passed/16 skipped 后仅有两个历史 formal Source-A absence test
  因 repo 已合法存在 formal Execution-B 而失败，与本 fresh diff 无关；
- 当前尚未读取任何真实 fresh-16 trajectory/OCR/image/label，也没有 GO verdict。下一步是 commit/push
  Source-A；随后从 clean A 机械生成唯一 runner-freeze B 并单独 commit/push，最后才在两张 H200 上运行。完整说明见
  `docs/gate_v1_fresh16_evaluation.md`。

### 2026-07-18：fresh-16 v1 pre-semantic inventory failure

- Source-A=`97694eff052ecbdc5f12f58b6f9ee10f4dd616ab`，机械 direct-child
  Execution-B=`a8bb27ccf9b8820c1d8487f63883f813e6845645` 均已 push；config SHA256 为
  `c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e`；
- Hyper00 container `sglang-omni-jaxan-07181130` 是 unprivileged `runc`、精确 2 张 H200、冻结 image digest。
  GUI-Owl verified projection 已重放 14 files / 17,545,907,171 bytes，logical CUDA UUID 一一匹配；
- formal attempt 在 derived input download/semantic decode 前，由 exact-tree validator 返回
  `immutable input repo file inventory drifted`。immutable revision 实际有 15 paths；v1 只声明 4 个消费 paths
  并默认允许 `.gitattributes`/`README.md`，漏绑 9 个同 repo 历史 artifact paths；
- 旧 state namespace 只保留 `runtime_receipts` 与 `global_claim`。fresh trajectory/OCR/image/label decode、label
  claim、checkpoint load、policy worker、vision forward、report、HF mutation、旧 dev-5、confirm、matched-NLL 与
  closed-loop operation 全为 0；planned HF destination/tag 仍不存在；
- 旧 v1 不删除、不覆盖、不续跑，也不解释为 selector NO-GO。轻量证据在
  `data/results/gate_v1_fresh16_evaluation_v1_attempt/`。下一步另立 full-remote-inventory repair A/B，绑定旧 failure
  receipts 与 successor absence，切换新 state/artifact/runtime-receipt namespace后才能再次授权 fresh access。

### 2026-07-18：fresh-16 full-inventory repair Source-A

- 新增 operational overlay
  `code/configs/causalcache_gate_v1_fresh16_inventory_repair_v1.json`，SHA256
  `5ba1b2d433c01defa0faae92a163dc9a9a916d967218abdc7607bd3724829a44`。它绑定父 A/B、failure evidence commit、
  旧两条 receipt、14 个 successor absence、空 artifact directory、run log/start/exit 与旧 HF destination absence；
- immutable derived revision 的 exact tree 冻结为 15 paths，path-list SHA256 为
  `f881b67d028dfe7ce147df0611213a2b9147b2a1432563180e68ebfaee47a202`：4 consumed + 2 base + 9
  historical auxiliary。remote metadata preflight 验证完整树，但 download/byte check/semantic input 仍只处理原
  4 个 consumed files；
- effective-delta proof 保证 `evaluation_contract`、`formal_model_input`、`label_input` 与 `output_contract` 完全不变；
  roster、gate、checkpoint、model、labels、thresholds、bootstrap、9+4 output targets 与双 H200 schedule 均未修改；
- repair 使用新 execution namespace `gate-v1-fresh16-evaluation-inventory-repair-v1`、新 artifact/runtime-receipt
  paths，以及 planned private HF dataset `gavinlaw/causalcache-gate-v1-fresh16-inventory-repair-mobile` / tag
  `gate-v1-fresh16-inventory-repair-v1`。当前该 destination 尚未创建；
- source-only validator 返回 `VALID_SOURCE_ONLY_GATE_V1_FRESH16_INVENTORY_REPAIR_V1`、47-path inventory，
  B absent，所有 network/write/torch/fresh semantic/model/report/HF mutation counts 为 0，且 execution
  authorization=false。focused repair/parent-runner regression 为 73 passed、8 subtests；全仓回归为 1,231
  passed、16 skipped，另有 3 个只在历史 Source-A 阶段要求旧 formal/fresh runner-freeze B 不存在的 lifecycle
  tests，因这些 B 已合法封存而预期失败；将这 3 项显式 deselect 后其余 suite 全绿。当前没有新的 semantic
  result 或 GO verdict；
- run fail-closed 顺序进一步固定为 clean pushed B → retained v1 evidence → token/HF read → parent delegate；新
  roots 创建前还会验证 full-15 non-label metadata、14-file/17,545,907,171-byte GUI-Owl projection、精确
  `cuda:0/cuda:1` H200 UUID 映射与 roots absence，并将这些摘要写入第一条 durable runtime receipt；
- 本 milestone commit/push 即 repair A；随后从 clean pushed A 重放 source validator，机械生成唯一
  `code/configs/causalcache_gate_v1_fresh16_inventory_repair_runner_v1.json` 并作为 direct-child B 单独
  commit/push；随后才能在 Hyper00 新 namespace 上执行唯一双 H200 run 与 immutable validate。完整交接见
  `docs/gate_v1_fresh16_inventory_repair.md`。

### 2026-07-18：fresh-16 inventory-repair v1 pre-label claim-serialization failure

- repair Source-A=`6fb3e868e293bce191ce30a5c6a15ecc544c4591`、唯一 direct-child Execution-B=
  `c22734ffc85935882f57ddb081c9194d6dae92d0` 均已 commit/push；config SHA256 为
  `5ba1b2d433c01defa0faae92a163dc9a9a916d967218abdc7607bd3724829a44`；
- Hyper00 attempt 于 `2026-07-18T04:27:04Z` 启动并 exit `1`。full-15 remote inventory、4 consumed-file
  transport、16 trajectory / 80 OCR semantic decode、80 selected images、48 feature states、144 candidates、
  10 checkpoint loads、双 H200 2 workers / 97 vision forwards 与 label-blind heuristic seal 均已完成；
- `pretty_json_bytes(claim.claim)` 不能序列化 `mappingproxy`，因此执行精确停在 ordinal `0..7` 之后、
  `label-access-claim` 写入之前。fresh label semantic/access claim/report/HF mutation、旧 dev-5、confirm、
  matched-NLL 与 closed-loop 均为 0，不能报告 GO/NO-GO；
- mode-0600 ordered receipts、独立 heuristic seal、88-file / 51,508,707-byte artifact tree、run log/start/exit 与
  Docker receipt 已逐字节冻结；失败执行器报告的 artifact canonical inventory SHA256 为
  `5443db6df16234c29b32501bc9f766670d428141fb02c4a665be936e1ca2587c`。旧 v1 与 repair planned HF repos
  均不存在，remote mutation 为 0；
- 轻量证据见 `data/results/gate_v1_fresh16_inventory_repair_v1_attempt/`。旧 state/artifact/runtime-receipt
  namespace 不删除、不覆盖、不续跑；claim-serialization repair Source-A 已冻结旧 bytes/successor absence 的
  expected bindings、未来 B read-only revalidation 与新的 local/HF identities。下一步是 commit/push A 后从 clean
  A 机械生成唯一 B。

### 2026-07-18：independent confirm-20 v1 在 restoration 前 execution INVALID

- independent 主线 Source-A=`e1cc8b3a0b8d06433674fe32652078a47b256f10` 与唯一 direct-child
  Execution-B=`f1e91964096dfa699c9fa930922bc2fe64c17bce` 已 push；contract SHA256 为
  `33e5c0f856a9074cd5460f0d6f7cdda2108b120d50b208234253ff3136d27b63`；
- Hyper00 四 H200 data-blind topology smoke 通过，receipt SHA256 为
  `8e06034e7b485a00fe288dc824eabd538cea02dee876027d54ac8b2432e22f13`。formal attempt 完成 20-state
  feature/independent/dynamic-recent/OCR-RGB/policy-vision label-blind payload，并发布 private HF payload commit
  `6d0cd95997186293e01c65276f3c082c11a9f52d`；8 files fresh-download byte replay 全过；
- restoration worker 采用 POSIX `fork` 传递不可伪造的 typed payload receipt，但 formal parent 已提前初始化
  CUDA；四个 worker 在构造 runtime 时触发 `Cannot re-initialize CUDA in forked subprocess`。失败发生在第一条
  reference generation/teacher forward 之前；reference/restoration/exact oracle/report/closed-loop/matched-NLL/
  sealed-test access 全为 0；
- 该 attempt 永久 `INVALID_INDEPENDENT_CONFIRM20_V1_CUDA_FORK_BOUNDARY`，不是 selector NO-GO；同一 identity
  不重试，远端 payload 不覆盖。轻量证据见 `data/results/independent_confirm20_v1_attempt/`；
- 下一步另立 restoration-only continuation Source-A/B：只采用并 fresh replay 已有 exact payload，修复
  CUDA-clean parent 边界，再运行原冻结 restoration/report。阈值、数据、模型、seed、selection 与 GO 判据全部
  不变；有效 confirm GO 前 paired closed-loop 保持 locked。

### 2026-07-18：independent confirm-20 restoration continuation Source-A

- 新增 restoration-only continuation contract，config SHA256 为
  `d54988fa05693a42574db2f9ec492d55405bcd2c2e902ccb79897f971c6fc7a9`；它精确绑定父 A/B、旧 failure/summary、
  topology receipt、private HF base→payload 两提交历史、8 个 payload file bytes 与 report/tag absence；
- continuation 只读采用现有 payload commit `6d0cd95997186293e01c65276f3c082c11a9f52d`。fresh replay 同时检查 remote
  tree/history、逐文件 SHA/size、semantic seal 与操作前后 remote snapshot；禁止 create repo/commit/tag、payload
  republish、formal-58 checkpoint load、independent scorer 与 policy-vision；
- 已定位的 CUDA/fork 错误通过三层边界修复：formal coordinator 要求 fresh exec；在任何 topology/HF/model work
  前安装 tripwire 阻断 parent `is_available()`、`device_count()` 与 `_lazy_init()`；完整 model snapshot verification
  在独立 `spawn` child 完成。原 restoration worker 仍采用 `fork` 传递 opaque typed receipt，child 通过 at-fork
  handler 恢复 CUDA 入口；创建 context/Queue/Process 紧前再验证 active tripwire、`is_initialized()` 与 bad-fork；
- fresh topology smoke 在 formal CLI 外运行；Execution-B commit、runner nonce、model/snapshot 与 device/physical
  UUID allocation 共同生成 challenge，spawn/fork 两 phase 各 4 个实际 worker response 必须 8/8 覆盖并由 formal
  CLI 重算。该公开摘要只作软件审计绑定，不声称硬件 attestation；旧 v1 inner receipt SHA 另行显式拒绝，不能
  通过手工包装或重序列化复用。continuation output identity 在 formal coordinator 读取/验证 envelope 以及 HF/model/semantic access 前
  以 `attempt.json` 独占，所有后续 failure 都 durable 落盘；
  report commit/tag/replay 的 partial failure 会只读 reconcile remote 状态，记录精确 mutation count 或保守下界，
  不回滚、不重试、不再硬编码 mutation=0；
- runner 只允许执行 deterministic parent/payload replay、原冻结 restoration schedule、固定 evaluator、report
  direct-child publication、annotated tag 与 immutable replay。`runtime_continuation_count=1`、scientific retry/top-up/
  filter count 均为 0，旧 v1 仍永久为 execution `INVALID`；
- 112 项 continuation/parent focused regression 已通过；30-path source-only validator 返回
  `VALID_SOURCE_ONLY_INDEPENDENT_CONFIRM_CONTINUATION_V1`，所有 payload/HF/model/GPU/restoration access count 为
  0。全仓回归为 1,474 passed、8 skipped；36 failures 均为已封存历史 Source-A tests 要求既有 B 不存在，或
  既有 subprocess `PYTHONPATH` / CPU-only import-order 隔离假设；后两项独立按其正确隔离环境重放均通过。
  Execution-B、new restoration output 与 report/tag 在该 Source-A freeze 时均不存在；
- 该历史下一步随后已完成：Execution-B=`68e71fd…0466`，有效 continuation report 为
  `NO_GO_INDEPENDENT_CONFIRM`。终态与停止决定见本文顶部同日结果段；paired closed-loop、matched-NLL 与
  sealed test 均未执行。

## 2026-07-18：Processor-only Execution-CF source freeze

- 在 Freeze-B v2 terminal repair 后冻结 processor-only candidate-freeze contract；config SHA256=
  `66fd93c64669be734f82830af7d623e9cb97263f90613a5809eb665078b2fba7`。
- contract byte-bind Freeze-B v2、P0 census、610-shard inventory、base parser、GUI-Owl snapshot、OCR config/
  completion，以及 runner/import closure 的 26 个 source files；hash chain 与 live bytes 均 fail closed。
- 1,200 trajectories / 2,400 queries 映射到 527 selected shards / 18,792 observations；whole-shard LPT 的
  四 worker loads 为 `4700/4700/4693/4699`，不会把同一 Parquet shard 拆给多个 worker。
- 新增 prefix-safe external artifact：每个 query 只含自己的 history prefix、validated OCR records 和
  candidate/current images；不保存 raw row、target action、terminal outcome、policy output、KL 或 utility。
- 新增独立 OCR/AutoProcessor runtimes、strict image contract、AutoProcessor-only AST/runtime guards、
  newest-suffix drop、`n>=4` floor、exact subset/model-operation schedule、atomic no-overwrite 与 completed-worker
  resume。
- processor focused + label schedule/producer + Freeze-B v2 regression 共 `75 passed`；无 GPU、policy forward、
  label generation、training、evaluation outcome access 或 HF mutation。
- 下一步：从 clean pushed `main` 在 Hyper00 CPU-only container 运行正式 4-worker OCR + AutoProcessor freeze；
  output 先落 `/data` persistent staging，随后回写 Git-safe manifest 与 `PENDING_HF_UPLOAD`。

## 2026-07-19：Processor-only formal run 启动

- 唯一正式 run 已从 clean detached `main@b3472bfbd0a26541a8c6f2e207b88741e256f59a` 于
  `2026-07-19 02:03:17 UTC` 在 Hyper00 启动；Execution-CF config SHA256 仍为
  `66fd93c64669be734f82830af7d623e9cb97263f90613a5809eb665078b2fba7`，没有改 threshold、候选规则、
  OCR normalization、worker count 或 denominator。
- container `sglang-omni-jaxan-07181624` 的 Docker `DeviceRequests=null`；本步是 CPU-only，GPU、
  policy/vision forward、restoration label、training、matched-NLL、closed-loop 与 HF mutation 全为 0。
- output root 固定为
  `/data/artifacts/causalcache-set-utility-processor-freeze-v2-b3472bf`；当前只存在 sibling
  `.causalcache-set-utility-processor-freeze-v2-b3472bf.incomplete`，状态为
  `RUNNING_INCOMPLETE_NOT_UPLOADABLE`，不能视为完成 artifact。
- 启动窗口四个 OCR worker 全部存活、error/traceback 为 0；`2026-07-19 02:08 UTC` partial tar
  约 `244 MB`，可完整读取 `30/2400` 个 query records。该数字只证明活性，不改变正式 2,400-state
  denominator。
- 目标 publication 已固定为 private dataset
  `gavinlaw/causalcache-set-utility-new-development-mobile:phase1-b2-processor-freeze-v1`。只有 atomic
  rename、全量只读 postflight 和 Git-safe summary 回写完成后，才会进入 `PENDING_HF_UPLOAD`；当前
  Execution-CF 明确禁止上传。

## 2026-07-19：Processor-only v1 formal attempt fail closed

- worker 0 在第 228 个 selected observation 的 pre-OCR validation 发现 source 为合法 `PNG/RGB`、
  `2208x1840`、无 EXIF，而 v1 contract 只接受 `PNG/RGBA` opaque。primary failure 时间为
  `02:14:20 UTC`，outcome=`INVALID_DERIVED_ARTIFACT_BEFORE_POLICY_OUTPUT`。
- 因 primary failure 已使 atomic publication 不可能，worker 1--3 收到 SIGTERM 以避免继续数小时无效
  CPU 工作；outer process 于 `02:16:22 UTC` 以 code `1` 退出。此清理不是 primary invalidity 的原因。
- output root 未创建；0 receipt、0 completed artifact shard。外置 `.incomplete` 保留 8 files /
  `666,629,393` bytes，path/size/file-SHA tree digest=
  `9190b1b140de9b507b4b396694443a5e1293848ae9b02b1309ef81a53bc57e84`。
- 正式终态为 `INVALID_PROCESSOR_FREEZE_EXECUTION_CF_V1_IMAGE_CONTRACT_DRIFT`。AutoProcessor、
  policy/vision forward、final candidates、restoration labels、training、matched-NLL、closed-loop、
  HF mutation 和 threshold/denominator change 全为 0；原计划 HF tag 未发布。
- Git-safe failure evidence 位于
  `data/results/set_utility_processor_freeze_execution_cf_v1_attempt/`。下一步先做全部 18,792 selected
  images 的只读 format census，再另立 versioned repair；不得覆盖原 `.incomplete`、跳样本或静默放宽
  contract。

## 2026-07-19：Completed-root read-only postflight

- 新增独立 completed-root validator；从 frozen roster 重建 shard descriptors、candidate parts/final schedule、
  exact operation budget、run identity、真实 OCR format tally 和 Git-safe manifest。
- validator 对 tar 做第二次流式 canonical audit，拒绝 semantic-equivalent byte rewrite、错误成员顺序、
  trailing member、partial/extra file、symlink、candidate/receipt/run-identity tamper。
- postflight + processor artifacts/freeze/contract focused suite 为 `40 passed`，CLI help 与 py_compile 通过。
- 当前实现明确绑定 v1 Execution-CF contract；v1 attempt 没有 completed root，仍为 INVALID。未来 v2
  repair 必须显式适配该 validator，不能把它当作自动授权。

## 2026-07-19：Token-level Set Utility Predictor v2 partial pilot

- 主方法已从 64 维手工 feature 升级为 frozen GUI-Owl full visual/text token sequences、learned latent
  resampler 与 query-conditioned set predictor；DeepSets 只替换 set aggregator。
- 与 dense label rollout 并行冻结 745 train/tune states（646/99 states，67/7 trajectories），evaluation 未加载；
  去重 cache 含 1,043 images、1,052 texts、22,782,959,321 bytes。
- 8-state overfit objective 下降 35.5%；DeepSets-d256、SetTransformer-d256、SetTransformer-d512 的 best tune
  objective 为 `0.3564/0.5753/0.3747`，三者均较首轮下降。
- 该结果只通过 optimization/data pipeline diagnostic。tune trajectory 数仍少，DeepSets 暂时最好，raw MAE
  未一致改善；未运行 evaluation baseline、at-most-B selector、matched-NLL 或 closed-loop。
- Git-safe summary 位于 `data/results/set_utility_token_predictor_v2_partial/`。dataset artifacts 已发布到
  `gavinlaw/causalcache-set-utility-new-development-mobile@e1240bde`，checkpoints 已发布到
  `gavinlaw/causalcache-set-utility-predictors-mobile@a55666c1`；同名 immutable tag 为
  `set-utility-token-v2-partial-a73cc18`。Hyper00 23.58GB root 现只作 verified local cache。

## 2026-07-20：Long-history oracle ceiling diagnostic v1 source freeze

- 动机:四轮 NO-GO 的失败集中在 long/very-long slice,但 exact/true-greedy oracle 只在 `n<=8` 测过;
  长历史 headroom 未知,直接决定 v2 decision-aware distillation 是否值得以 Long+ 为目标。
- 冻结 250 个 train long/very-long states(220 long + 30 very_long,158 trajectories,cap=3,
  `SHA256(salt:state_id)` 确定性采样,`n_t` 均值 22.6)。wave 1 = empty/full anchors + 全部 singleton +
  recent B1--B4 + 每预算 1 个 random subset(7,654 coalitions);wave 2--4 = true-greedy 前缀的全部
  one-event expansion + additive top-k(总计约 2.4 万 coalitions,约为 enrichment run 的 45%)。
- 复用 variable-history labels runner、scientific config(480 context-fit)、execution config 与 source
  artifact,全部以 SHA256 绑定进 [`causalcache_set_utility_long_oracle_v1.json`](../code/configs/causalcache_set_utility_long_oracle_v1.json)。
- 预注册判定:`oracle_greedy_minus_recent_macro` 的 95% CI `lower>0.05` → headroom confirmed;
  `upper<0.03` → headroom insufficient,general-B long-horizon 主张收缩;否则扩样。合同见
  [`docs/set_utility_long_history_oracle_v1.md`](set_utility_long_history_oracle_v1.md)。
- 实现:`set_utility_long_oracle.py` + 两个 CLI;`test_set_utility_long_oracle.py` 8 passed;真实
  manifest 上 wave-1/wave-2 物化冒烟通过(7,654 / 5,904 coalitions)。train 标签显式允许复用为 v2
  candidate-complete marginal supervision;tune truth、evaluation、policy replay、closed-loop 保持锁定。

## 2026-07-20:Long-oracle v1 wave-1 labels 启动

- wave-1 schedule 从 clean `912e5d1` 重新物化,content SHA256=`2b1752f9...a92d9`(与冒烟一致),已部署到
  Hyper00/Hyper01 `/data02/jaxan/runs/causalcache-long-oracle-w1-schedules-912e5d1`;source snapshot 在
  `/data02/jaxan/worktrees/causalcache-912e5d1`(`git archive` 部署)。
- preflight 空闲卡:Hyper00 4/5/6/7、Hyper01 6/7(decision distillation v2 labels 正占用 Hyper00 0--3 与
  Hyper01 2--5,互不冲突)。partition_count=6:Hyper00 GPU 4--7 跑 partitions 0--3、Hyper01 GPU 6--7 跑
  partitions 4--5,每卡 3 state lanes,共 18 lanes。
- 复用 `run_set_utility_variable_history_labels.py` + 480 context-fit scientific config + labels execution
  config;output root `/data02/jaxan/runs/causalcache-long-oracle-w1-labels-912e5d1`;容器
  `sglang-omni-jaxan-07201525`(Hyper00)与同型 Hyper01 容器,unprivileged。
- 顺带修复 `run-gpu-cluster-job` launcher 的多卡 `--gpus device=a,b,c` CSV 引号 bug(此前会被解析成
  `Count+DeviceIDs` 冲突,单卡不受影响)。
- wave 1 完成后:拉回两端 `states/*.json` 合并 → 物化 wave 2(true-greedy 扩张)→ 同 allocation 重启;
  wave 1+2 即可出 B1/B2 headroom 初判。

## 2026-07-20:Long-oracle wave-1 完成,B1 headroom 初判为强阳性

- wave-1 labels 250/250 states 全部 `COMPLETED`、0 skip(Hyper00 164 + Hyper01 86)。中途两次执行修正,
  均不改科学 denominator:(a) Hyper00 GPU 5 被另一用户容器新落的 68GB 常驻进程挤压,单 lane OOM;
  (b) Hyper01 双卡在 3 lanes 下 141/143.8GB 触顶。两台容器改为低密度重启(普通卡 2 lanes、共享 GPU 5
  单 lane,共 11 lanes),state/microbatch 断点保留全部已完成工作。
- **wave-1 单事件初判(非正式 verdict,正式判定需 wave 2--4 的 macro)**:158 trajectories 上
  trajectory-equal B1 recovery,oracle(best singleton)=`0.5279` vs recent=`0.0325`,paired delta=
  `+0.4954`,trajectory bootstrap 95% CI=`[0.3917, 0.6234]`;long bin oracle/recent=`0.5285/0.0482`,
  very_long=`0.4369/-0.4981`(只恢复最近单事件在超长历史上为负收益)。best singleton 落在 recent-4
  之外的比例=`58.4%`,平均 age fraction=`0.363`。长历史 B1 oracle 与小历史 exact(`0.5304`)几乎一致,
  restoration 信号在长历史上没有衰减;差距完全在 selector。
- wave-2(true-greedy step-2 全扩张,5,904 coalitions/250 states)已物化并在两台以相同低密度布局启动;
  materializer 补绑 `schedule_shards_sha256` 进 manifest(此前 manifest content hash 只覆盖计数元数据)。

## 2026-07-20:Long-oracle wave-2 完成,B2 headroom 同样为强阳性

- wave-2(true-greedy step-2 全扩张)250/250 states `COMPLETED`、0 skip、跨 wave reference action 漂移 0。
  执行修正:wave-2 启动间隙 Hyper00 GPU 4 被其他用户 119.6GB 任务占用,partition-0 双 lane 在模型加载即
  OOM;GPU 6/7 也各新增 ~44GB 外部进程。h00 重启为避开 GPU 4 的 4 单 lane 布局后跑完;科学 denominator
  不变。
- **wave 1+2 联合初判(158 trajectories,trajectory-equal)**:B2 true-greedy oracle=`0.6611` vs recent=
  `0.2827`,paired delta=`+0.3784`,95% CI=`[0.3072, 0.4691]`;B1 维持 `+0.4954 [0.3917, 0.6234]`。
  长历史 B2 oracle(0.661)与小历史 exact B2(0.700)同量级,restoration headroom 在长历史上完全成立。
- **additive top-2 只有 `0.5035`,显著低于条件贪心 `0.6611`**:与小历史上 J-additive 恢复 90% exact B2 的
  结论不同,长历史上纯 singleton-additive 选择显著不足,条件边际结构不可省略。这直接支持 decision
  distillation v2 的 conditional/listwise 目标设计,并否定"只蒸馏 singleton marginal"的简化路线。
- wave-3(5,833 coalitions,greedy step-3 扩张 + additive top-3)已在 Hyper00 恢复的 4×2 lanes 与
  Hyper01 2×2 lanes 上启动。

## 2026-07-21:Long-history oracle diagnostic v1 完成,HEADROOM_CONFIRMED

- wave-3/wave-4(greedy step-3/4 全扩张 + additive top-3/4,5,833 + 5,641 coalitions)均 250/250
  `COMPLETED`、0 skip、无执行事故;四 wave 合计 25,032 个 coalition 标签,跨 wave reference 漂移 0。
- **正式判定(预注册)**:trajectory-equal B1--B4 macro,oracle_greedy=`0.6777` vs recent=`0.2104`;
  paired delta=`+0.4674`,95% CI=`[0.3591, 0.6304]`,预注册阈值 `lower>0.05` 以 ~7 倍裕量满足,
  verdict=`HEADROOM_CONFIRMED`。长历史 oracle B4=`0.7851` 与小历史 exact `0.8248` 同量级。
- 结构性发现:(a) recent 在长历史上 macro 与 random 相当(`0.210` vs `0.223`),very_long 上为负
  (`-0.418`);(b) additive top-B 在 B2--B4 平台化(macro `0.530`),条件贪心显著更高——长历史
  selector 必须建模条件边际;(c) best singleton 58.4% 在 recent-4 之外,平均 age fraction `0.363`。
- 结论直接支持 decision distillation v2 以 long/very-long 为主攻:headroom 不是瓶颈,蒸馏才是。
  25,032 个 train 标签允许并入 v2 candidate-complete supervision。
- artifact:reducer summary 入库 `data/results/set_utility_long_oracle_v1/`;完整 payload 已存
  Hyper00 `/data02/jaxan/artifacts/causalcache-long-oracle-v1-179b0d8`(SHA `9238f63d...0c8b34`),
  `PENDING_HF_UPLOAD`(会话内自动上传被权限分类器拦截;目标 repo/path/tag 已写入 result README)。

## 2026-07-21:Long-oracle v1 artifact 发布

- 完整 payload 已发布到 private HF dataset revision `8d5a5021d8e69999ed944574bc8e243f386c2288`,
  immutable tag=`set-utility-long-oracle-v1-179b0d8`;远端 readback `payload.tar.gz` SHA256 与本地一致
  (`9238f63d...0c8b34`)。Hyper00 `/data02/jaxan/artifacts/causalcache-long-oracle-v1-179b0d8` 保留为
  mirror,`PENDING_HF_UPLOAD` 状态解除。
