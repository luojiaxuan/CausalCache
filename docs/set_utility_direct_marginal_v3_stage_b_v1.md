# Direct conditional-marginal v3 Stage-B v1

状态：`TRAINING_COMPLETED / FIXED_TUNE_PENDING`。

本合同采用
[`Stage-A gate review`](set_utility_direct_marginal_v3_stage_a_gate_review.md) 的任务对齐结论，但不删除或
追认 v1/v2 的 `NO_GO_DIRECT_MARGINAL_V3_STAGE_A`。这是一项明确的、事后提出的 development-gate repair，
不是统计确认，也不能进入论文结果。其唯一作用是允许一次完整 Stage-B 训练，让从未修改的 fixed-tune gate
作最终裁决。

## 1. 为什么允许版本化修订

- 部署只消费 at-most-4 的头部动作与 STOP；全候选 Spearman 主要测量不进入选择的尾部顺序；
- 修订提议前已冻结的 v1 记录为 top-1=`0.9040`、top-4=`0.9873`、B1/oracle=`0.9437`、STOP=`1.0`；
- 唯一 rank repair 把 Spearman `0.238→0.291`，却把 top-1/B1 `0.904/0.473→0.841/0.437`，说明原 proxy
  与部署决策发生冲突；
- 标签重测漂移约 `1e-8`，因此修订理由只能是 task alignment，不能错误声称 noise ceiling。

Stage-A′ 条件固定为 top-1 `>=0.8`、top-4 `>=0.95`、B1/oracle `>=0.9`、STOP `=1.0`。阈值是在看到
v1 后提出，必须永久标记 `post_hoc_development_gate=true`。不允许再次修订。

## 2. 冻结训练输入与目标

- input content SHA256=`3d011990...ad36ce`：5,550 train states、62,332 个 `|S|<=3` complete expansion
  groups；
- contextual cache SHA256=`44405c2c...97604`；
- 初始化使用 Stage-A v1 epoch-30 checkpoint，SHA256=`84cf5140...f5a12`；不用 rank-repair checkpoint；
- 以 frozen salt 按 trajectory 划分 900/100 个 optimization/holdout trajectories，对应
  5,015/535 states、56,235/6,097 groups；tune/evaluation 不参与训练或 checkpoint 选择；
- 每个 group 直接预测 `[STOP,e_1,...,e_n]` conditional marginals，STOP 恒为 0；selected/padded events
  被 action mask 排除；
- primary loss 为 hard conditional listwise + expected decision regret；marginal regression 和 sign
  calibration 为辅助项，不再优化全列表 Spearman。

## 3. 训练与恢复

- Set Transformer `d256/l16/r2/s2`，从 Stage-A v1 初始化；
- Hyper00 8-rank DDP，per-rank batch=1，global batch=8，BF16，最多 12 epochs；
- checkpoint 只按 train-only holdout decision regret 选择，patience=3；
- 每个 rank 每 epoch 原子覆盖 `resume-rank-XX.pt`，中断后以相同 world size 从上一完整 epoch 恢复；
- 完整配置：
  [`causalcache_set_utility_direct_marginal_v3_stage_b_v1.json`](../code/configs/causalcache_set_utility_direct_marginal_v3_stage_b_v1.json)；
- trainer：
  [`train_set_utility_direct_marginal_v3.py`](../code/scripts/train_set_utility_direct_marginal_v3.py)。

## 4. 唯一最终裁决

训练冻结唯一 checkpoint 后，在既有 1,063-state fixed-tune denominator 上做 iterative conditional-marginal
at-most-`B` selection。最终 gate 与 decision-v2 完全相同：

1. B1--B4 每个 point estimate 均严格高于 recent；
2. macro paired bootstrap CI lower 严格大于 0；
3. Long+ point estimate 严格高于 recent。

fixed-tune 任一失败即永久终止 learned general-`B` 路线；不创建 Stage-A″、student v4 或追加 DAgger。
只有 GO 才能访问 untouched evaluation、policy replay、closed-loop 与 matched-NLL。

fixed-tune selector 使用
[`run_set_utility_direct_marginal_tune_selectors.py`](../code/scripts/run_set_utility_direct_marginal_tune_selectors.py)：
每个 state 只编码一次 query/full-history candidate set，随后按冻结 direct marginal 逐步选择，并让显式
`STOP=0` 与剩余 events 同场竞争。`--state-id-file` 支持跨 GPU 的 disjoint resumable shards；
[`merge_set_utility_tune_selection_shards.py`](../code/scripts/merge_set_utility_tune_selection_shards.py)
只接受 checkpoint/config/input/cache identity 完全一致且 state inventory 不重叠的完成分片。

## 5. 正式运行

- source=`main@8ccdb5c`；Hyper00 GPU `0--7`，8-rank DDP；
- container=`sglang-omni-jaxan-07210241`，id=`7866ed74f915`；
- output=`/data02/jaxan/runs/causalcache-direct-marginal-v3-stage-b-v1-8ccdb5c`；
- preflight 时 Hyper00 8/8 cards 空闲；Hyper01 保留 6 张空闲卡用于后续 fixed-tune data parallel；
- 第一次 container 在 import 阶段因缺少 `PYTHONPATH` 立即 exit 1，未创建 output 或消费训练状态；同一
  source/config 只补 runtime `PYTHONPATH` 后重启；正式 container exit 0；
- 12 epochs 用时 785.7s，最佳 epoch=12，holdout decision regret=`0.2581`，checkpoint SHA256=
  `d8abbe8c...1dd23`；train/holdout top-1=`0.6129/0.2982`，STOP accuracy 均近 0；
- 训练完成只授权 unchanged fixed-tune gate，不授权 untouched evaluation 或 policy experiments。结果见
  [`data/results/set_utility_direct_marginal_v3_stage_b_v1/`](../data/results/set_utility_direct_marginal_v3_stage_b_v1/README.md)。
