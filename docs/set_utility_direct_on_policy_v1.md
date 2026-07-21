# Direct marginal on-policy coverage v1

状态：`RUNNING_PER_EPOCH_TRUE_RECOVERY_SELECTION`。这是用户在 direct-v3 fixed-tune NO-GO 后显式授权的新假设，
不追认或改写旧 gate。旧 fixed-tune 已消费，后续只作 development；1,046-state untouched evaluation 继续密封，
直到新 selector 和 fallback policy 在 train-holdout/development 上冻结。

## 动机

Stage-B 最终训练只消费 5,550 个带 complete expansion group 的 train states，其中 5,015 用于优化、535
trajectory-disjoint states 用于 checkpoint selection。完整 train pool 有 10,658 states；另有 5,108 states
完全没有 `|S|<=3` complete expansion group。现有 365k coalition rows 高度相关，最终只有 62,332 个可用于
direct conditional decision 的完整 group；current direct selector 在 train 的 top-1 为 `0.613`、train-holdout
只有 `0.298`，并在 B2 出现明显 rollout error accumulation。因此本轮扩充的单位不是随机 subset 数，而是部署
路径真正访问的 `(state,S,{S∪j})` candidate-complete decision group。

## 数据收集

冻结配置为
[`causalcache_set_utility_direct_on_policy_v1.json`](../code/configs/causalcache_set_utility_direct_on_policy_v1.json)。
在全部 10,658 train states 上运行同一 frozen direct-v3 checkpoint，并对 5,108 个当前零 complete-group
states 收集三条 nested at-most-`B` 路径：

- `direct`：当前 conditional-marginal selector；
- `recent`：按时间最近优先；
- `hybrid`：direct top event 相对 newest remaining event 的预测优势至少为 `0.001` 时 override，否则 fallback
  到 recent；当最佳候选不高于 `-0.001` 时 STOP。

三者只用于 train-only data collection，不是论文结论。对每条路径的 `|S|=0,1,2,3` bases，补齐所有
`S∪{j}`；已存在的 `D(S)` 按 coalition identity 复用，只调度缺失 rows。schedule 固定为 256 logical shards，
label runner 仍以 coalition microbatch 原子断点恢复。只补每个 state 的 empty→singleton 最低需要 24,787 个
新 coalitions；direct/recent/hybrid 路径会在此基础上增加部署对齐的 B2--B4 conditional groups，最终数量由
selection 后的 missing-only schedule manifest 冻结。

实际 rollout 已覆盖 10,658/10,658 train states，11/11 workers completed，0 duplicate、0 missing。
最终 missing-only schedule 包含 5,108 states / 359,189 coalitions，medium/long/very-long=
3,578/1,499/31；schedule-time projection 的现有/resulting complete groups=62,332/120,172。后续对 merged
bytes 的正式 census 修正 resulting 为 96,759；原 projection 把已有 groups 的 states 上未实际调度的 desired
coalitions 计入了结果。schedule content SHA256=
`981a329c...ec506`（含 5,108 个 zero-cost full anchors），轻量结果见
[`data/results/set_utility_direct_on_policy_v1/`](../data/results/set_utility_direct_on_policy_v1/README.md)。

## Label 与训练输入完成

正式 rollout 完成 5,108/5,108 states、33,175/33,175 microbatches，得到 359,189 个 sampled
coalition distances 和 5,108 个 `D(C)=0` anchors；0 skip、0 non-finite、0 duplicate rows、0 bad logs。
terminal content SHA256=`689b3ad3cdf0fafdd3e3a40979aa86109eae40f467c6ade1551e7bf530c67a0e`。

合并后的 formal training input 位于 Hyper00
`/data02/jaxan/runs/causalcache-direct-on-policy-training-input-v1-9e94593`，content SHA256=
`6c9243a2a2846180f717ea59e3692a3005ae153b8d206f7ad1ec4a8bf1a1bfad`。它增加 359,189 rows；
5,108 个重复 full anchors 在 `1e-6` 规则下 exact equal，最大差异为 0；tune bytes 未改变。
formal optimizer inventory content SHA256=`b7452b1bad141d96407bf80c62b1d4778461af260f9837852cd4d8e480646109`，
覆盖 900 trajectories、9,287 states 与 84,441 candidate-complete groups；checkpoint denominator 仍是
冻结的 256 held-out states。

epoch 1 已在两条 fresh 6-rank 训练线上完成。Set Transformer / structured DeepSets 分别产生 5,718 / 6,216
个缺失 coalition 请求，union 后为 198 states / 7,681 teacher forwards。合并 schedule content SHA256=
`956672017b533f70f9c1357d25fa0636df15febdd18248004326d87adf884f4c`。该 union 已完成并封存为
manifest content SHA256=`a6ba7c267052de75e3771915e8de1730c967f0202d632a73eb170d207e9efa36`：
198/198 states、7,879 rows（含 198 anchors）、0 skip/error。epoch 1 真实 Set Transformer / structured
DeepSets B1--B4 macro recovery=`0.38508/0.39251`，Long+=`0.36343/0.37929`；两者只在结算后进入 epoch 2。
Set epoch 2 的 1,665 个新 forward 已封存为 manifest content SHA256=
`b7225a0091faacad366ffe7e6c38bf1f07ae68673c3facfc457767ed8cc0c7fe`；macro/Long+ 提升至
`0.39491/0.38618`，超过 minimum delta，成为新 best。Set epoch 3 macro=`0.00610`，未替换 best、
stale=`1`；DeepSets epoch 2 全 STOP、macro=`0`，同样未替换 epoch 1。两模型后续实际查询分别独立补标，
避免因无新 query 的 epoch 造成不必要的跨模型同步等待。

实现入口：

- selector：
  [`run_set_utility_direct_marginal_tune_selectors.py`](../code/scripts/run_set_utility_direct_marginal_tune_selectors.py)
  的 `--role train --collection-config ...`；
- missing-only schedule：
  [`materialize_set_utility_direct_on_policy_schedules.py`](../code/scripts/materialize_set_utility_direct_on_policy_schedules.py)；
- pure coalition logic：
  [`set_utility_direct_on_policy.py`](../code/causalcache/set_utility_direct_on_policy.py)。

## 训练修复

新训练必须同时比较：

1. 小型 structured pairwise/DeepSets direct-marginal head；
2. 当前 Set Transformer direct-marginal head。

两者共享 frozen contextual GUI-Owl tokens。epoch sampler 按 base cardinality、history bin 与 teacher STOP
分层；all-negative groups 需要单独控制最大候选相对 STOP 的 calibration。每个 epoch 都保存 checkpoint，不再用
单一 scalar holdout regret 选择；正式 checkpoint selection 是 trajectory-disjoint train-holdout 上真实
B1--B4 macro recovery。训练/选择不读取 evaluation。

train-heldout 已按实际 structured input（不是较早的 10,680-state assignment）冻结：10,658 train states / 1,000
trajectories 被切为 optimizer 9,287 states / 900 trajectories 与 heldout 1,371 states / 100 trajectories；checkpoint
denominator 从 heldout 中固定 256 states，short/medium/long/very-long 各 64。22 个只在旧 assignment、但不在
本轮 frozen input 的 states 以 `ABSENT_FROM_FROZEN_STRUCTURED_TRAINING_INPUT` 明示排除。每个 epoch 先保存
immutable checkpoint，再在该 denominator 上做真实 at-most-`B` rollout。训练采用严格的 per-epoch truth
barrier：若当前 epoch 实际查询的 candidate-complete truth 不完整，所有 ranks 保存 resume state 并以
`WAITING_FOR_HELDOUT_TRUTH` 退出；补齐同一 epoch 中 DeepSets 与 Set Transformer 查询子集的并集后，resume
必须先完成 authoritative recovery reduction、更新最佳 checkpoint 与 patience，才允许进入下一个 optimizer
epoch。禁止先训完所有 epoch 再 post-hoc 选模型。primary metric 是 trajectory-equal B1--B4 macro recovery，
Long+ 只作 tie-break；正式 v2 selection contract 在任何训练结果产生前冻结为 primary/Long+
`minimum_delta=0.005`、`patience=3`。未达到 material delta 的 checkpoint 保留最早 epoch，不按最后 epoch
或 train/eval loss 选模型；v2 与 v1 使用完全相同的 100 trajectories / 256 states，只修正选模/早停规则。
[冻结 manifest](../data/manifests/set_utility_train_heldout_v2.json)。

补标输入不再接受任意 `states/` 目录。每次同 epoch 双模型 schedule 必须绑定各自 model family、epoch 与
checkpoint SHA；label runner 完成后先 seal 为 signed manifest/receipt，逐 terminal 校验 `COMPLETED`、
state/trajectory/candidate/coalition identity、runner state hash、source revision 与 scientific/execution config。
Trainer 只通过显式 `--truth-source MANIFEST_CONTENT_SHA256=ROOT` allowlist 加载，并拒绝 stale-only truth。
seal 还显式绑定 merged input content、heldout manifest 与 source manifest file SHA；不同 substrate 即使
state/event ID 恰好相同也不能复用 truth。

## Compute 与 SoT

- Hyper00/Hyper01 每台最多 6 GPUs；当前正式 label rollout 使用每台 6 卡；
- 最终正式 label mapping 使用 Hyper00 6 卡 + Hyper01 6 卡、每卡 2 lanes，共 12 partitions / 24 resumable
  workers；冻结配置为
  [`causalcache_set_utility_direct_on_policy_labels_workers_v4.json`](../code/configs/causalcache_set_utility_direct_on_policy_labels_workers_v4.json)。早期 5+6 卡的 v1 mapping 仅作为下述失败与修复历史保留；
- 第一次 label launcher 把 source revision 误传为 7 位短 SHA，22 lanes 均在 runner input validation 前
  fail-fast，0 terminal / 0 microbatch；正式重启改用完整
  `8acd8d84a8a9e5e837048d9a7f01817ab4ba41e5`；
- retry 前 fresh preflight 显示 Hyper01 GPU 7 已被占用，因此不等待资源：v2 execution 保持相同 source、
  schedule、scientific config 与每卡 2 lanes，只把 256 logical shards 从 11 改分到 10 physical partitions，
  使用 Hyper00/Hyper01 各 5 卡。v1 的零进展失败记录保留，v2 使用全新 output root；
- v2 workers 完成首个 state 的全部 forward 后，在 terminal coverage check 统一失败。根因是 missing-only
  schedule 省略了 runner 默认注入 `D(C)=0` 的 full-history anchor，使 measured set 比 frozen schedule 多一项；
  这是 schedule/runner interface bug，不是 label 数值失败。v3 为每个 state 加一个不触发 forward 的 full anchor，
  359,189 个 missing coalitions 与全部 selection 不变；回归测试直接验证 runner microbatch + anchor 的距离集合
  与 schedule exact equal。v3 使用新 schedule identity、新 source revision 与全新 output root；
- v3 尚未启动时的 fresh preflight 又释放到 Hyper00/Hyper01 各 6 卡；因此最终 v4 execution 直接使用授权上限
  12×H200、每卡 2 lanes（12 partitions / 24 workers）。这只改变 logical-shard execution mapping，不改变
  source、schedule、label definition 或 resume identity；
- v4 运行中确认 12 个静态 partition 存在显著长尾，但 labels、reference repeat 与所有 workers 均保持健康。
  [`v5 conditional tail handoff`](../code/configs/causalcache_set_utility_direct_on_policy_labels_workers_v5_tail_handoff.json)
  只允许在 donor partition 两个 lanes 均完整落盘且 0 skip 后触发：先精确停止同 host 的慢 partition owner，
  再将其 logical shards 按 `count=24` 拆到 donor/recipient 两张卡继续使用同一原子 progress。partition topology
  不进入 state identity；source、schedule、science/execution config、lane hash 与每 host 6-GPU 上限均不变；
- 若第一轮 handoff 后 Hyper01 p10 仍形成长尾，冻结的
  [`v6 handoff`](../code/configs/causalcache_set_utility_direct_on_policy_labels_workers_v6_tail_handoff.json)
  仅在 donor p7 的 397 个 states 全部完成且 p10 尚未自然结束时，将 old p10/count12 精确拆为
  p10+p22/count24；否则 fail closed 或直接放弃 handoff；
- formal trainers 为
  [`train_set_utility_structured_marginal.py`](../code/scripts/train_set_utility_structured_marginal.py) 与
  [`train_set_utility_set_transformer_control.py`](../code/scripts/train_set_utility_set_transformer_control.py)；
  [`materialize_set_utility_heldout_truth_schedules.py`](../code/scripts/materialize_set_utility_heldout_truth_schedules.py)
  只合并同一 epoch 两个模型的实际查询并生成 256 个可恢复 label shards，不参与模型选择；
- DeepSets sampler 直接平衡包含 `|S|=0` 在内的全部 candidate-complete groups；packing 只给 interaction
  复制 weight=0 的 empty anchor，所有真实 sampled groups 在 packing/accumulation/DDP 下保持全局等权；
- 两个 trainer 使用 immutable epoch/rank resume generations。partial next generation 不会发布，重启只从
  上一完整 epoch 重放；119GB contextual cache 每台 host 首次全量 SHA，后续 resume 只验证 signed receipt、
  manifest SHA 与 shard stat，不重复六卡各读一遍缓存；
- merged input 还必须精确绑定 `direct_on_policy_v1` schedule、5,108 states、359,189 新 distance rows、
  10,658 train states 与 post-merge 实际 96,759 complete groups；随后用
  [`materialize_set_utility_formal_training_inventory.py`](../code/scripts/materialize_set_utility_formal_training_inventory.py)
  冻结 optimizer state/group identity、base cardinality、history、STOP 与 joint-stratum census。两份训练 config
  在 merged input content SHA 和 exact inventory 写入前保持 `PENDING_*`，不能执行；
- labels 完成后使用
  [`causalcache_set_utility_direct_on_policy_training_inputs_v1.json`](../code/configs/causalcache_set_utility_direct_on_policy_training_inputs_v1.json)
  将两个 host 的 disjoint terminals 合并到 frozen `3d011990...ad36ce` training input；重复 coalition 仅在
  `1e-6` 内允许，tune bytes 必须逐字节不变，evaluation 不读取；
- labels、schedules、selection payload 与 checkpoints 保存在 `/data02/jaxan` persistent storage；
- reusable data/checkpoints 完成后分别发布到现有 private Hugging Face dataset/model repo；发布前在 README
  记录精确路径并标为 `PENDING_HF_UPLOAD`；
- 每个 material milestone 独立 commit 并 push `main`。
