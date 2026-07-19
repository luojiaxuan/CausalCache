# Variable-history set utility v1

## 正式任务定义

每个 decision state 的候选 universe 是它之前的全部 eligible events：

\[
C_t=\{e_1,\ldots,e_{t-1}\},\qquad n_t=|C_t|=t-1.
\]

`n_t` 是 state 属性，不是配置常数。正式数据不允许 recent-`n` 截断；DeepSets/Set Transformer 使用 padding、
`event_mask`、`subset_mask` 组成 variable-size batch。`B` 与 `n_t` 都不输入 predictor，部署时才执行
`|S|<=B, B in {1,2,3,4}` 的 search。

已有 1,200 trajectories 产生 12,792 states：train/tune/evaluation 为 10,680/1,066/1,046。history bins 为
`5–8`、`9–16`、`17–32`、`33–45`。所有 eligible states 都物化；训练 sampler 再做 trajectory-uniform、
history-bin-balanced sampling，单个 state 每 epoch 最多重复 8 次。

## Full-history reference 的可执行性

完整候选 universe 也必须对应完整 reference，不能一边声称 variable history、一边为 context fit 删除老事件。
GUI-Owl 的 context limit 是 32,768；原 2,560 effective visual tokens/image 无法容纳 45-event reference。因此
v1 固定 512 effective visual tokens/image，并在任何 label forward 前对 12,792 states 做 processor-only full-reference
context census：

\[
\operatorname{tokens}(C_t)+256\le32768.
\]

如果任一 state 不满足，正式 rollout 整体阻塞，必须更换 reference token profile 或 long-context policy；禁止删除
old events。`S=C_t` 是 reference 本身，记录 `D(C_t)=0`，无需额外 coalition forward。

旧 processor tar 的 prefix metadata/OCR 可复用，但它只保存 anchor/terminal image union。新 source 从
`cua-lite/GUIOdyssey@ea08072b` 的 1,200 个 pinned raw rows 重新物化全部 18,792 observations，保证任何历史 event
都能被采样和恢复。

## Broad 40-label sampler

train/tune 每个 state 目标为 40 个 unique subsets；`n_t<=5` 时直接枚举全部 `2^n` subsets。其余 states 先生成：

| Component | Count | Contract |
|---|---:|---|
| anchors | 2 | empty 与 full `C_t` |
| singleton | 8 | 四个连续 age quantile bins 各 2 个；`n_t<=8` 时覆盖全部 |
| conditional chains | 16 | 4 条跨 age seeded permutation，各取 cardinality 1–4 prefixes |
| interaction pairs | 8 | old/recent、同龄、最高 similarity、最低 similarity 跨龄 pairs |
| higher-cardinality | 6 | `|S|=2/3/4` 各 2 个 age-stratified random sets |

sampling similarity 固定为 frozen GUI-Owl mean embedding cosine 与 deterministic OCR-token Jaccard 的等权组合，
只决定采样覆盖，不进入 restoration target。所有 subsets canonicalize 后去重；不足 40 时先按相同 cardinality 与
age pattern 补齐，再从剩余 `|S|<=4` subsets 做 seeded unique backfill。

四条 chain 直接产生 conditional marginal supervision：

\[
U(S\cup\{j\})-U(S).
\]

首轮 broad model 训练后，只在 train split 选择约 20% interaction-dense states，追加 40–80 labels。选择信号可以用
DeepSets/Set Transformer disagreement、uncertainty、redundancy 与 train-only nonadditivity；不得读取 evaluation
结果再追加。

## Evaluation

- exact-oracle track：320 个 trajectory-clustered、history-bin-stratified states，枚举全部 `|S|<=2` 并加入
  full anchor；13 个 evaluation very-long states 全部进入；
- large-history track：720 个 states，只计算冻结方法真正选出的 subsets 和预注册 random controls；
- 同一 checkpoint 评估 `B=1/2/3/4`；
- 学习曲线固定同一 tune set，比较 10%/25%/50%/100% trajectories；
- subset rows 是同一 state 内的相关观测，统计与 bootstrap 仍以 trajectory 为 cluster。

主部署还必须满足 [`set_utility_predictor_v2.md`](set_utility_predictor_v2.md) 的 warm p95 10% latency gate。

## Resumability 与 throughput 边界

正式 runner 以 coalition microbatch 为 atomic resume unit。每批完成后原子更新 state progress；重启时跳过已完成
coalitions，最多重做一个 microbatch。256 个稳定 logical shards 与物理 GPU 解耦，可在 host/GPU 变化后重新分配。
scientific config、reference profile、state/subset identities 必须一致；execution concurrency 可以改变。

throughput v2 必须使用 multi-state in-flight pipeline、CPU image prefetch/cache、尽可能大的 cardinality-compatible
teacher batch，并把 KL 留在 GPU 到 state 完成。目标是提高 wall-clock throughput，不把 GPU utilization 本身当作
科学指标。
