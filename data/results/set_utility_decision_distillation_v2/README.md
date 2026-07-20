# Decision distillation v2

状态：`LABELS_PENDING`。本目录只记录轻量结果；raw traces、schedule、labels 与 checkpoints 保存在 persistent
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
  Hyper00 GPU 0--3 + Hyper01 GPU 2--5，8 个 partitions，每卡一个 resumable worker。

## 下一步

Hyper00/Hyper01 各最多 4×H200 并行生成 train-only restoration labels；合并 enriched snapshot 后，DeepSets
和 Set Transformer 使用相同 v2 loss 训练。只有 fixed-tune B1--B4、macro CI 与 long-history gate 全部通过，
才允许访问 untouched evaluation。
