# Decision distillation v2 + long-oracle 训练执行单

状态：`SOURCE_READY_LABELS_RUNNING`。本文只调整 v2 标签完成后的训练输入与目标；不改写已冻结的
365,043-coalition v2 schedule，不访问 evaluation，也不授权 policy replay、closed-loop 或 matched-NLL。

## 为什么调整

Long-history oracle 已在 250 个 train long/very-long states 上确认 headroom：oracle-greedy macro=
`0.6777`，recent=`0.2104`，paired delta=`+0.4674 [0.3591, 0.6304]`。但 additive top-B 在
B2--B4 平台化，说明仅拟合 singleton/scalar utility 会丢掉冗余与互补关系。因此本轮重训把
`S -> S∪{j}` 的 conditional listwise 与 decision regret 作为主目标；普通 regression/ranking 只负责校准。

## 冻结输入

- base contextual input content SHA256=`2711ab55cbd5f86fd4f221d2cb3a4cbc30a7fbba7b2bf0fcfb9ffbb3409d407f`；
- contextual hidden cache content SHA256=`44405c2cf96f468e6d0f087e8249bdb504c7efd150673e9bda836d2f52597604`；
- decision-v2 schedule：1,066 states / 365,043 coalitions，content SHA256=
  `d5e508c298558f5f042e57991a783a8ffd019c161a38c2f926984eebfa10187a`；
- long-oracle：250 states / 25,032 rows，和 decision-v2 重叠 94 states；两者 union 预期为
  1,222 decision-supervised states，其中 long+ 902 states；
- long-oracle canonical artifact：HF dataset
  [`gavinlaw/causalcache-set-utility-variable-history-mobile@8d5a5021`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/8d5a5021d8e69999ed944574bc8e243f386c2288/artifacts/set-utility-long-oracle-v1-179b0d8)，
  tag=`set-utility-long-oracle-v1-179b0d8`，payload SHA256=
  `9238f63dca61d7b18377a5df03448c2b15484522cb0e4a60cb32ffa9170c8b34`；Hyper00 mirror=
  `/data02/jaxan/artifacts/causalcache-long-oracle-v1-179b0d8`。

合并时以 `(state_id, coalition_event_step_ids)` 为 key；重复 `D(S)` 的绝对差必须不超过 `1e-6`。
tune rows 必须 byte-identical，evaluation access 必须保持 `false`。若 v2 runner 出现合法 skip，上述预期计数
可下降，但训练启动前必须至少有 1,100 个 decision-supervised states、其中 800 个 long+ states；否则 fail
closed，不通过降低门槛启动训练。

## 版本化实现

- 配置：
  [`causalcache_set_utility_decision_distillation_v2_long_oracle_training_v1.json`](../code/configs/causalcache_set_utility_decision_distillation_v2_long_oracle_training_v1.json)；
- 合并模块：
  [`set_utility_contextual_multisource_enrichment.py`](../code/causalcache/set_utility_contextual_multisource_enrichment.py)；
- CLI：
  [`materialize_set_utility_contextual_multisource_inputs.py`](../code/scripts/materialize_set_utility_contextual_multisource_inputs.py)；
- trainer 会在模型加载前统计 complete conditional-expansion groups，并执行上述 coverage gate。
- nested enrichment manifest 必须保存 `ancestor_content_sha256s`；frozen contextual cache 的
  `input_content_sha256` 只能命中当前 content、direct parent 或该显式 ancestor 列表，不能用关闭 binding
  validator 的方式复用 cache。

`main@905cbbf` 已在 Hyper00 的 `hongccc/sglang-omni:dev` CPU container 对 immutable long-oracle 四个
wave 做真实只读合并演练：250/250 states/source 全部通过，新增 18,894 rows、去重 6,138 rows、重复最大
绝对差 `2.98e-8 < 1e-6`，tune payload firewall 通过；带 PyTorch focused tests 为 10/10 passed。该演练输出
位于 disposable container `/tmp`，不是正式训练 artifact。

训练损失权重固定为：

| 项 | 权重 | 角色 |
|---|---:|---|
| raw regression | 0.25 | 距离尺度校准 |
| normalized regression | 0.25 | state 内归一化校准 |
| within-state ranking | 0.25 | 普通排序辅助 |
| conditional marginal | 1.0 | 条件增益回归 |
| conditional listwise | 2.0 | 主决策目标 |
| decision regret | 2.0 | 主部署目标 |

listwise/regret 只在存在完整 expansion group 的 rows/states 上归一化，不能被没有 conditional labels 的
scalar rows 稀释。

## 标签完成后的唯一执行顺序

1. 等现有 Hyper00/Hyper01 v2 workers 自然结束；检查 1,066-state inventory、skip/error、schedule/runtime
   binding。先把 Hyper01 terminals 复制到 Hyper00 的 versioned staging path，不移动或改写原始 root。
2. 用 multisource CLI 合并 decision-v2 和 long-oracle wave 1--4。四个 wave 的 Hyper00/Hyper01 terminal
   roots 分别作为同一 source 的多个 `--label-root` 输入；忽略 macOS AppleDouble `._*` 文件。
3. 校验 union inventory、重复距离 tolerance、tune byte identity、contextual cache ancestry 和
   decision-supervision census；把合并 snapshot 发布到现有 private HF dataset 的新 immutable revision/tag。
4. 运行两个共享输入、seed、训练轮数与搜索合同的模型：DeepSets 和 Set Transformer。架构都保留每个 event
   的 16 个 contextual entity latents；差异只在 set aggregator。按 GPU preflight 结果并行启动，每台 Hyper
   最多 4 GPUs。
5. 在冻结的 1,063-state tune truth 上先做训练侧诊断：B1--B4/macro/Long+ recovery、age-bin singleton
   ranking、best-singleton-outside-recent4 recall、oracle gap 与 latency。诊断不能改 checkpoint 或 gate。
6. 只按原 fixed-tune machine gate 判定：候选模型 B1--B4 每点均高于 recent、macro paired bootstrap 下界
   大于 0、Long+ 点估计高于 recent，三条缺一不可。
7. `GO` 才密封 checkpoint 并访问 untouched evaluation；`NO_GO` 则停止，不启动 policy replay、closed-loop
   或 matched-NLL，也不追加第二轮 DAgger。

## 当前边界

这次调整只利用已经独立生成的 train-only long-oracle labels 改善蒸馏合同。它不推翻 v1 或 enrichment-v1
NO-GO，不把 oracle diagnostic 当 learned-selector 结果，也不以 tune loss 下降替代真实 subset utility gate。
