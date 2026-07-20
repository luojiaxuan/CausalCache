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
v1 检查 512 effective visual tokens/image，并在任何 label forward 前对 12,792 states 做 processor-only full-reference
context census：

\[
\operatorname{tokens}(C_t)+256\le32768.
\]

如果任一 state 不满足，正式 rollout 整体阻塞，必须更换 reference token profile 或 long-context policy；禁止删除
old events。实测 512 profile 有 1 个 state 超限 468 tokens，故 v1 BLOCK；v2 将图像预算降为 480 tokens，候选、
state 与 subset sampler 全部不变。v2 tokenizer census 已全部 fit，最大余量 1,004 tokens；visual-token cache
完成后再用真实 `image_grid_thw` 做 exact postflight。`S=C_t` 是 reference 本身，记录 `D(C_t)=0`，无需额外
coalition forward。480-token cache 的实际 `image_grid_thw` postflight 已覆盖全部 12,792 states：最大 prompt
为 30,036 tokens，加 256 action reserve 后为 30,292，全部低于 32,768，正式 labels 已解除 context blocker。

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

## 已实现边界

- `set_utility_variable_history.py` 已实现 full-prefix state 构造、history/age bins、label-blind similarity、40-label
  deterministic sampler 和 evaluation tracks；
- state inventory 固定为 `data/manifests/set_utility_variable_history_v1_states.json`，绑定 config、assignment
  manifest、全部 state identity 的 SHA256；Git 不重复保存 12,792 行 candidate prefixes，运行时按冻结规则重建；
- token predictor 的 batch 维度改为动态 `max(n_t)` 与动态 label 数，分别由 `event_mask`、`label_mask` 排除 padding；
- padded events 不进入 multimodal resampler，避免全空 attention source；
- full source materializer 以 256 个 trajectory Parquet shards 为断点单元，合并 pinned raw images 与已有 terminal
  processor metadata；已有合法 shard/receipt 会在重启时跳过；
- context census 只做 chat-template/tokenizer 计算，不执行 policy forward；它在 12,792 states 全部 fit 前阻塞 labels；
- full visual-token extraction 同样以 logical shard 为原子断点，每个 receipt 绑定完整 Git SHA，并保存逐 trajectory
  实际 visual-token counts；256 shards 完成后执行 exact-grid context postflight；
- full source 已在 Hyper00/Hyper01 完成 1,200 trajectories / 256 shards，共 13.16GB；v2 full visual-token extraction 已在 8xH200 完成 256/256 shards，共约 67GB，失败 shard 为 0；
- label-blind schedule materializer 以 256 logical shards 生成 deterministic 40-label / small-history exact schedules；label runner 分别持久化 reference action、每个 coalition microbatch 与 terminal state，可在中断后跳过已完成 microbatch；
- exact-grid context postflight 已 PASS；256/256 schedule shards 已完成，共 11,746 train/tune states、461,040
  coalition labels，其中 449,294 个需要 policy forward。正式 labels 尚未生成，下一步直接启动 8-GPU runner。
- 首次 16-worker launch 暴露 official-tools encoder 仍硬编码 historical max batch=2；8 个 empty-coalition
  microbatches 后 fail-fast，0 个 terminal states。该 attempt 不续用；修复改为读取 subclass runtime limit，旧 runtime
  的默认上限仍为 2，variable-history profile 才允许 16。

```bash
PYTHONPATH=code python code/scripts/materialize_set_utility_variable_history_source.py \
  --config code/configs/causalcache_set_utility_variable_history_v1.json \
  --assignments data/manifests/set_utility_freeze_b_v2_terminal_index_repair.json \
  --processor-root /data/artifacts/processor-freeze-v2 \
  --source-root /data/source/guiodyssey-full-pool-v1 \
  --output-root /data/artifacts/causalcache-variable-history-source-v1 \
  --workers 32

PYTHONPATH=code python code/scripts/census_set_utility_variable_history_context.py \
  --config code/configs/causalcache_set_utility_variable_history_v2_context_fit.json \
  --source-root /data/artifacts/causalcache-variable-history-source-v1 \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --output-root /data/runs/causalcache-variable-history-context-v1

PYTHONPATH=code python code/scripts/materialize_set_utility_variable_history_schedules.py \
  --config code/configs/causalcache_set_utility_variable_history_v2_context_fit.json \
  --source-root /data/artifacts/causalcache-variable-history-source-v1 \
  --token-root /data/artifacts/causalcache-variable-history-tokens-v2 \
  --output-root /data/artifacts/causalcache-variable-history-schedules-v1 \
  --partition-index 0 --partition-count 8 --workers 4
```
