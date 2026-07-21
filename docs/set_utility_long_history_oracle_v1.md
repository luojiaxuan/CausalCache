# Long-history oracle ceiling diagnostic v1

状态:`COMPLETED / HEADROOM_CONFIRMED`。250/250 states、四 wave 全部 0 skip;
`oracle_greedy - recent` macro=`+0.4674`,95% CI=`[0.3591, 0.6304]`,以约 7 倍裕量满足预注册
`lower_95 > 0.05`。完整结果见
[`data/results/set_utility_long_oracle_v1/`](../data/results/set_utility_long_oracle_v1/README.md)。

## 动机

连续四轮 NO-GO(heldout v1、contextual 25%、contextual full v4、train enrichment v1)中,learned selector
的失败集中在 long/very-long slice(recent `0.4008` vs learned `0.330--0.354`),而 exact/true-greedy oracle
诊断只在 `n<=8` 小历史上做过(B1 oracle `0.5304` vs recent `0.1486`)。**长历史上 restoration oracle 相对
recent 的 headroom 从未被测量。** 若 headroom 小,再好的蒸馏也无法在 Long+ 上翻盘,general-B
long-horizon 主张需要收缩;若 headroom 大,v2 decision-aware distillation 的投入才有明确目标。本诊断
在任何 v2 训练投入前先回答这个问题。

## 设计

- **总体**:frozen inventory 中 `role=train` 且 `history_bin ∈ {long, very_long}` 的 states
  (可用 2,337 long / 103 very_long)。不触碰 tune/evaluation,不影响任何已冻结合同。
- **采样**:very_long 优先,`SHA256(salt:state_id)` 确定性排序,`very_long=30`、`long=220`,每条
  trajectory 最多 3 states。实际选出 250 states / 158 trajectories,`n_t` 均值 22.6、范围 17--43。
- **Wave 式 true-greedy 标注**(复用既有 `run_set_utility_variable_history_labels.py` runner、
  scientific/execution config 与 source artifact,估计总计约 2.4 万 coalition 标签,约为 enrichment run 的 45%):
  1. wave 1:empty anchor + **全部 singleton**(candidate-complete)+ recent B1--B4 prefix + 每预算 1 个
     确定性 random subset + full anchor;
  2. wave 2:true 最优 singleton `S1*` 的全部 one-event expansion(`n-1` 个);
  3. wave 3:`S2*` 的全部 expansion + additive top-3(wave-1 singleton 距离最小的 3 个事件);
  4. wave 4:`S3*` 的全部 expansion + additive top-4。
- 每个 wave 独立 output root,reference action 每 wave 重算;跨 wave reference 漂移或重复 coalition
  距离漂移超过 `1e-6` 的 state 按 state 剔除并记录,不改总设计。

## 指标与判定

主指标:trajectory-equal B1--B4 normalized recovery,方法为 `oracle_greedy`(at-most-B:前缀族取最优)、
`recent`、`additive`(top-B singleton)、`random`;另报 long/very_long 分 bin、best-singleton 是否落在
recent-4 之外的比例、best-singleton age 分布。`oracle_greedy - recent` macro 用 trajectory-clustered
paired bootstrap(10,000 次,seed 20260721)。

预注册判定(见 [config](../code/configs/causalcache_set_utility_long_oracle_v1.json)):

- `lower_95 > 0.05` → **headroom confirmed**:长历史信号充足,v2 decision-aware distillation 以 Long+
  为主目标继续;
- `upper_95 < 0.03` → **headroom insufficient**:长历史 oracle 本身赢不了 recent 多少,general-B
  long-horizon selector 主张必须收缩/重构,蒸馏迭代不再以 Long+ 翻盘为目标;
- 其余 → `INCONCLUSIVE_EXTEND_SAMPLE`,先扩样再判。

## Firewall 与复用

- 只标 train states;tune truth、untouched evaluation、policy replay、closed-loop 全部保持锁定;
- 本诊断产生的 train singleton/expansion 标签显式允许复用为后续 v2 candidate-complete marginal
  supervision(`labels_reusable_for_training=true`),这是与 tune-side truth 的关键差别;
- 不修改任何已冻结合同;`NO_GO_TRAIN_ON_POLICY_ENRICHMENT_V1` 的结论不受本诊断影响。

## 实现

- 模块:[`set_utility_long_oracle.py`](../code/causalcache/set_utility_long_oracle.py)
  (确定性采样、wave coalition 构造、true-greedy 选择、reducer);
- schedule CLI:[`materialize_set_utility_long_oracle_schedules.py`](../code/scripts/materialize_set_utility_long_oracle_schedules.py);
- reducer CLI:[`evaluate_set_utility_long_oracle.py`](../code/scripts/evaluate_set_utility_long_oracle.py);
- config:[`causalcache_set_utility_long_oracle_v1.json`](../code/configs/causalcache_set_utility_long_oracle_v1.json),
  绑定 assignment manifest、scientific config(480-token context-fit)与 labels execution config 的 SHA256;
- tests:`code/tests/test_set_utility_long_oracle.py`(8 passed);wave-1/wave-2 物化已在真实 manifest 上
  冒烟(7,654 / 5,904 coalitions)。

## 执行计划

Hyper00/Hyper01 各 4×H200、每卡 3 state lanes(partition-count=8,Hyper00 取 0--3,Hyper01 取 4--7),
复用 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-source-v1-a7213db` source 与既有
GUI-Owl snapshot。每个 wave 完成后把两端 `states/*.json` 拉回合并,物化下一 wave 并分发。wave 1+2 完成
即可先出 B1/B2 headroom 初判,wave 3+4 补齐 macro。
