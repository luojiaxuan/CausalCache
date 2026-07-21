# Decision distillation v2

状态：`DDP_TRAINING_SOURCE_READY`。本目录只记录轻量结果；raw traces、schedule、labels 与 checkpoints 保存在 persistent
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
- 最终 inventory：Hyper00/Hyper01 分别 543/523 states，合计 1,066/1,066；25,915/25,915
  microbatches、16/16 worker receipts 全部 completed，两台容器 exit 0。Hyper01 states 已 byte-identical
  staging 到 Hyper00，523-file digest=`85232921...fa8e3`。

## 交接后训练调整

现有 365,043-coalition runner 继续原地断点执行，不改 schedule 或 label identity。完成后将其与 immutable
long-oracle 的 250 states / 25,032 rows 合并：两批监督重叠 94 states，预期 union 为 1,222 个
decision-supervised states、Long+ 902 states；重复 `D(S)` 按 `1e-6` tolerance 去重。

重训仍比较 DeepSets 与 Set Transformer，但 conditional listwise 和 decision regret 升为主损失，普通
regression/ranking 降为校准项；trainer 在启动前强制检查至少 1,100 个 decision-supervised states 和 800 个
Long+ states。完整配置、合并入口与顺序见
[`docs/set_utility_decision_distillation_v2_long_oracle_training.md`](../../../docs/set_utility_decision_distillation_v2_long_oracle_training.md)。
Hyper00 已对四个 immutable long-oracle waves 做真实 merge rehearsal：新增 18,894 rows、去重 6,138 rows，
duplicate max delta=`2.98e-8`，tune byte-identity 通过；该演练在 v2 labels 完成前执行，现已进入全量 merge。

首次全量 merge 的数据与 coverage 校验通过，但 manifest 只保留 direct parent，而 hidden cache 绑定的是再上一层
contextual input；trainer 因而会正确 fail closed。该 root 标记为 superseded staging，不用于训练。当前 source
repair 将完整 `ancestor_content_sha256s` 写入 manifest，并让 trainer/selector 只接受显式 lineage 中的 binding；
修复后另立 versioned output root，不能原地改写首次 merge。

lineage-repaired v2 training input 已物化到 Hyper00
`/data02/jaxan/artifacts/causalcache-decision-v2-long-oracle-training-inputs-v2-f7f6b14`：content=
`3d011990970a9eff1828c667854936c0f9b56d6f702bc635e54dfe9fcead36ce`，states=
`9d9ef9917eb9da635e044845ddcff71924447eafb3b5b6091d13d42f5be10956`，226MB；cache ancestor
`af18388e...f102139c1` 命中并通过 validator。最终 census 为 5,550 decision-supervised states / 67,322
complete expansion groups，Long+ 902 states，超过 1,100/800 的冻结启动门槛；status=`PENDING_HF_UPLOAD`。

## Distributed training repair

首次双单卡 launch 中，DeepSets 正常进入训练，Set Transformer 在 batch=8 forward 峰值占用
139.40/139.81GB，申请额外 650MB 时 OOM；该 container 随后按用户要求停止并迁移，两个 output roots 都没有
summary/checkpoint，不能作为结果或续跑入口。失败不改变 label/input/config identity。

正式 repair 使用
[`causalcache_set_utility_decision_distillation_v2_long_oracle_ddp_v1.json`](../../../code/configs/causalcache_set_utility_decision_distillation_v2_long_oracle_ddp_v1.json)：

- 每模型 4 ranks，per-device batch=2、global batch=8，不改变 LR、loss、seed、epoch 或 effective batch；
- 每 rank 在独立 H200 上预载完整 frozen cache；DDP 只切 training rows，不把 119GB cache 跨卡拼接；
- global order 补齐到 8 的倍数后按 global batch 分片，padding 数写入 summary；
- conditional listwise/regret 使用跨 rank 的 active-decision denominator，避免局部 batch 无条件监督时稀释主损失；
- rank 0 用原 evaluation batch=8 跑完整 fixed tune、保存 checkpoint，再广播同一 early-stop decision；
- Set Transformer 固定 Hyper00 4×H200，DeepSets 固定 Hyper01 4×H200。Hyper01 cache 已由 10.0.32.x
  内网补全：3,572/3,572 files、119,134,024,064 bytes、manifest SHA256=`88db0d2a...58ec4`。

只有 fixed-tune B1--B4、macro CI 与 long-history gate 全部通过，才允许访问 untouched evaluation；本调整不
解锁 policy replay、closed-loop 或 matched-NLL。
