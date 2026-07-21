# Decision distillation v2

状态：`LABELS_RUNNING`。本目录只记录轻量结果；raw traces、schedule、labels 与 checkpoints 保存在 persistent
storage，完成后发布到 private Hugging Face。

## 已完成

- contract/config Git revision：`87bea18`；evaluation access=`false`；
- 旧 enriched DeepSets 与 Set Transformer 分别在自己的训练 config 下完成 10,658-state train beam-4 trace；
- 两模型共享 input content SHA=`2711ab55...09d407f`、contextual cache SHA=`44405c2c...597604` 与
  search config SHA=`e0a25aba...de6a1`；
- target sampler 固定选中 1,066 states / 486 trajectories，long/very-long/medium/short=
  `693/53/213/107`，每 trajectory 最多 3 states；
- candidate-complete schedule 共 365,043 个去重 coalitions，content SHA=
  `d5e508c298558f5f042e57991a783a8ffd019c161a38c2f926984eebfa10187a`。

## Persistent artifacts

- Set Transformer trace：Hyper00
  `/data02/jaxan/runs/causalcache-decision-v2-beam-traces-v1-fcf3fa0`；
- DeepSets trace：Hyper00
  `/data02/jaxan/runs/causalcache-decision-v2-beam-traces-deep-v1-fcf3fa0`；
- schedule：Hyper00
  `/data02/jaxan/runs/causalcache-decision-v2-schedules-v1-87bea18`；
- intended HF dataset repo：`gavinlaw/causalcache-set-utility-variable-history-mobile`；status=
  `PENDING_HF_UPLOAD`。
- label allocation：[`causalcache_set_utility_decision_v2_labels_workers_v1.json`](../../../code/configs/causalcache_set_utility_decision_v2_labels_workers_v1.json)，
  Hyper00 GPU 0--3 + Hyper01 GPU 2--5，8 个 partitions，每卡一个 resumable worker；2026-07-20
  22:18 UTC 启动，output root=`/data02/jaxan/runs/causalcache-decision-v2-labels-v1-87bea18`。
- 单 lane 启动采样的 GPU 间歇平均约 70%--80%；不改变 GPU 数或 scientific identity，按
  [`workers v2`](../../../code/configs/causalcache_set_utility_decision_v2_labels_workers_v2.json) 切为每卡 2 个
  deterministic state lanes，复用已完成 state/microbatch。
- 2026-07-21 01:27 UTC 只读进度：Hyper00/Hyper01 分别 274/301，共 575/1,066 states（53.9%）与
  13,930/25,915 microbatches（53.8%）；全部 575 terminals 都是
  `COMPLETED_VARIABLE_HISTORY_LABEL_STATE`。两台容器均运行中，过去一小时合计新增 223 states，按短窗
  吞吐估计剩余约 2.2 小时；这只是动态 ETA，不是完成声明。

## 交接后训练调整

现有 365,043-coalition runner 继续原地断点执行，不改 schedule 或 label identity。完成后将其与 immutable
long-oracle 的 250 states / 25,032 rows 合并：两批监督重叠 94 states，预期 union 为 1,222 个
decision-supervised states、Long+ 902 states；重复 `D(S)` 按 `1e-6` tolerance 去重。

重训仍比较 DeepSets 与 Set Transformer，但 conditional listwise 和 decision regret 升为主损失，普通
regression/ranking 降为校准项；trainer 在启动前强制检查至少 1,100 个 decision-supervised states 和 800 个
Long+ states。完整配置、合并入口与顺序见
[`docs/set_utility_decision_distillation_v2_long_oracle_training.md`](../../../docs/set_utility_decision_distillation_v2_long_oracle_training.md)。
Hyper00 已对四个 immutable long-oracle waves 做真实 merge rehearsal：新增 18,894 rows、去重 6,138 rows，
duplicate max delta=`2.98e-8`，tune byte-identity 通过；正式 merge 仍等待当前 v2 labels 完成。

只有 fixed-tune B1--B4、macro CI 与 long-history gate 全部通过，才允许访问 untouched evaluation；本调整不
解锁 policy replay、closed-loop 或 matched-NLL。
