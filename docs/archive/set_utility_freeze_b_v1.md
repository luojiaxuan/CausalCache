# Set Utility Freeze-B v1

> **历史 INVALID：** 后续审计确认本版本把 `decision_count` 直接当作 terminal
> `decision_step_id`，实际选择了倒数第二个 state。v1 bytes 保留为失败证据，但 query states 不得用于
> processor、labels 或 training。版本化修复见
> [`set_utility_freeze_b_v2_terminal_index_repair.md`](set_utility_freeze_b_v2_terminal_index_repair.md)。

## 当前结论

Freeze-B/A 已经完成并固定 roster、query plan、feature schema、training grid、artifact destination 与计算上限；
它仍不授权 label generation 或 training。配置 SHA256=
`df8c00bfddda7589e3c9b58cc36f5cbe305ad3fee9c6a8bc4272888a6c648440`，manifest SHA256=
`144b0de1e66eff1624f6bd10fa6dbebd3d215c6d3d9965d9e9cd3dae295373e5`。

## Roster

| Role | 6–9 | 10–17 | 18+ | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 334 | 333 | 333 | 1,000 |
| tune | 33 | 34 | 33 | 100 |
| one-shot evaluation | 33 | 33 | 34 | 100 |
| total | 400 | 400 | 400 | 1,200 |

相同 `instruction_app_group_sha256` 只取一个 deterministic representative；新 role 之间 source、trajectory、
group 均为零 overlap。历史 58 条 `legacy_train_only` 和 49 条 `forbidden_consumed` 的 source/group firewall
继续生效。P0 中只有 5 条 duplicate-group nonrepresentative；没有新的 candidate 命中历史 consumed group。

每条 trajectory 固定两个 query：其 stratum anchor（decision 6/10/18）和 terminal decision。为保证两者不同，
roster 机械排除 decision count 正好为 6/10/18 的记录。2,400 个 query 的初始 candidate histogram 已写入
manifest；当前 candidate ids 只是 recent-16 pre-processor plan，不能当作最终 label universe。

## Feature 与训练冻结

主 schema 是 `q64/h75/pair8` 轻量特征加 frozen GUI-Owl final-main spatial-merger mean-pooled、float32
L2-normalized 4096-d visual embedding。query dimension=4,160，event dimension=4,171，pair dimension=8；
Set Transformer、DeepSets、Pairwise 使用相同 feature bytes。OCR/RGB 与 recency 仍从轻量前缀机械恢复，
budget、source/group identity 和 utility 不进入 feature。

训练固定三 seed `17/29/43`。三个 family 共享 hidden `256/512` 和 learning rate `1e-4/3e-4`；Set
Transformer 另比较 2/3 layers，固定 8 heads、dropout 0.1。tune 负责 grid selection；one-shot evaluation
不得用于选择或重开。

Reusable feature/label dataset 预留 private HF
`gavinlaw/causalcache-set-utility-new-development-mobile:phase1-b2-v1`；model 预留
`gavinlaw/causalcache-set-utility-predictors-mobile:phase1-b2-v1`。两个 repo/tag 当前都尚未创建或绑定 revision，
不能写成已上传。

## 下一执行边界

当前 328,800 raw rows / 338,400 model operations 是 `n=16` 全状态最坏上限。下一步独立 contract 只允许：

1. 读取 roster 指定的 raw rows；
2. 生成 low-fidelity/OCR substrate 并执行 processor-only recent-16 suffix fit；
3. 固定最终 candidate ids、exact label schedule 与四 worker shards；
4. 从 train-only 的 12 个预注册 state 做不写 KL/utility 的 throughput pilot。

该完成态 push 前后均不允许 restoration forward、optimizer step、one-shot evaluation label access、closed-loop
或 sealed test。
