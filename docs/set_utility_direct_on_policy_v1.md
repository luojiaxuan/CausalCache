# Direct marginal on-policy coverage v1

状态：`SELECTION_AND_SCHEDULE_COMPLETE / LABELS_PENDING`。这是用户在 direct-v3 fixed-tune NO-GO 后显式授权的新假设，
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
3,578/1,499/31；现有/resulting complete groups=62,332/120,172。schedule content SHA256=
`d71c92d0...b521db`，轻量结果见
[`data/results/set_utility_direct_on_policy_v1/`](../data/results/set_utility_direct_on_policy_v1/README.md)。

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

## Compute 与 SoT

- Hyper00/Hyper01 每台最多 6 GPUs；label preflight 的可用量为 5/6，因此使用 5/6；
- 正式 label mapping 使用 Hyper00 5 卡 + Hyper01 6 卡、每卡 2 lanes，共 11 partitions / 22 resumable
  workers；冻结配置为
  [`causalcache_set_utility_direct_on_policy_labels_workers_v1.json`](../code/configs/causalcache_set_utility_direct_on_policy_labels_workers_v1.json)；
- 第一次 label launcher 把 source revision 误传为 7 位短 SHA，22 lanes 均在 runner input validation 前
  fail-fast，0 terminal / 0 microbatch；正式重启改用完整
  `8acd8d84a8a9e5e837048d9a7f01817ab4ba41e5`；
- retry 前 fresh preflight 显示 Hyper01 GPU 7 已被占用，因此不等待资源：v2 execution 保持相同 source、
  schedule、scientific config 与每卡 2 lanes，只把 256 logical shards 从 11 改分到 10 physical partitions，
  使用 Hyper00/Hyper01 各 5 卡。v1 的零进展失败记录保留，v2 使用全新 output root；
- labels、schedules、selection payload 与 checkpoints 保存在 `/data02/jaxan` persistent storage；
- reusable data/checkpoints 完成后分别发布到现有 private Hugging Face dataset/model repo；发布前在 README
  记录精确路径并标为 `PENDING_HF_UPLOAD`；
- 每个 material milestone 独立 commit 并 push `main`。
