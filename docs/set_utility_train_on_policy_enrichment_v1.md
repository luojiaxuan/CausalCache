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
DeepSets 与 Set Transformer 使用相同 enriched snapshot、conditional-marginal loss、seed 和 contextual entity
representation 并行重训；只替换 set aggregator，随后按真实 tune restoration recovery 选择 winner。

重训后只接受以下判定：Set Transformer 在相同固定 tune truth 上 primary macro 高于 recent，且 paired
trajectory bootstrap 不显示稳定退化；否则继续 `NO_GO`，不进入 policy experiments。

## Train labels 与 enriched snapshot 结果

- Hyper00/Hyper01 各使用 4×H200、每卡 3 resumable state lanes；两端分别完成 `1,044/1,044` 与
  `1,088/1,088` states，24/24 worker receipts 为 completed，0 skipped/OOM/retry；
- 共生成冻结 schedule 的 `52,744` coalition distances；consolidated raw root 为 Hyper00
  `/data02/jaxan/runs/causalcache-contextual-train-on-policy-labels-enrichment-v1-5714b52`，约 19.5MB；
- broad table 与 targeted table 有 14,762 个重复 rows，最大绝对差 `1.1921e-7`，通过 `1e-6` tolerance；
  新增 37,982 个 rows；
- enriched snapshot 为 Hyper00
  `/data02/jaxan/artifacts/causalcache-contextual-inputs-enriched-v1-b32efb1`，content SHA256=
  `2711ab55cbd5f86fd4f221d2cb3a4cbc30a7fbba7b2bf0fcfb9ffbb3409d407f`；parent SHA256=
  `af18388e86406a7d3921e6f3d8e02c9b18cdeaacca3250f29f2f4fed102139c1`；
- 1,063 个 tune states 的 canonical payload SHA256 保持
  `04ebf46ad96a53e65645ed8ea4808f2d862b909fc471d7cca03197490ff0df22`，未进入训练标签生成；
- schedule、raw labels、train selections 与 enriched input 已发布到 private HF dataset revision
  [`9b436c9c`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/9b436c9c8ac645d20f2aa86ba0d519b14f5d6934/artifacts/set-utility-train-enrichment-v1-2711ab55)，
  immutable tag=`set-utility-train-enrichment-v1-2711ab55`；
- DeepSets/Set Transformer 已在 Hyper00 GPU 0/1 并行启动，run root 为
  `/data02/jaxan/runs/causalcache-contextual-training-enriched-v1-b32efb1`。

首次并行 attempt 中 DeepSets 正常完成 epoch 1；Set Transformer 在首个 backward 触发 PyTorch cuDNN
`mha_graph.execute` runtime error。该 attempt 不作为模型负结果。修复只在 committed training config 中关闭
cuDNN SDP backend，保留其他 SDP backend、模型、seed、labels 与 optimizer 不变；repair 另立输出目录，原失败
日志保留。
