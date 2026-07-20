# Set Utility held-out selector v1

状态：`FROZEN_BEFORE_EVALUATION_LABEL_ACCESS`。

## 目的

本阶段不再看 train/tune loss，而是在 trajectory-disjoint、variable-`n_t` evaluation 上回答：冻结的 rich-token
DeepSets 与 Set Transformer 是否能把 restoration oracle 蒸馏成优于 recent 与 OCR/RGB 的 at-most-`B`
selector。只有本阶段 GO 才进入 policy replay 和 closed-loop。

## 冻结输入

- evaluation inventory：320-state exact track、720-state large-history track，重叠 235，union 805；
- budgets：`B=1,2,3,4`，模型不接收 `B`；
- DeepSets 100% checkpoint：`47c936f...8648e`；
- Set Transformer 100% checkpoint：`2f710a9...e22b`；
- checkpoints 均来自 train/tune-only learning curve，evaluation 在模型/超参冻结前未加载；
- 候选集合始终是 query 前全部 eligible events，不允许 recent-`n` 截断。

完整机器合同见
[`causalcache_set_utility_heldout_v1.json`](../code/configs/causalcache_set_utility_heldout_v1.json)。

## 执行顺序

1. 只从 source 与 frozen VLM tokens 物化 805 个 evaluation feature rows，不读取 `D(S)`；
2. 每个 checkpoint 对每个 state 只编码一次 source tokens，用 conditional greedy 对 `B=1..4` 选 subset；
3. 同时密封 recent、真实 raw-image/OCR 的 OCR/RGB 与 deterministic random selections；
4. selection artifact 绑定 checkpoint/config/inventory/input hashes 并 push 后，才允许生成 evaluation restoration labels；
5. exact track 生成全部 `|S|<=2`、full anchor 与方法实际选出的 B3/B4 subsets；large track 只生成 empty、full、
   冻结方法 selections 与 random controls；
6. 以 trajectory 为 cluster 聚合真实 utility、normalized recovery、history-bin 指标与 selector latency。

`n=45,B=4` 的全枚举有 164,221 个 subsets/state，不是可部署搜索。本阶段固定 conditional greedy：从空集出发，
每轮批量评分所有单 event extensions；若最佳 extension 的 predicted utility 不严格高于当前 subset，则提前停止。

## GO / NO-GO

候选模型必须同时满足：

- B1--B4 macro normalized recovery 对 recent 和 OCR/RGB 的 trajectory-clustered 95% bootstrap 下界均大于 0；
- exact track 的 B1/B2 oracle regret 均低于两类 heuristic；
- long + very-long states 上相对最佳 heuristic 的点估计为正；
- prediction 与真实 distance 全部 finite。

若两者都 GO，按 primary utility 选择；差距不超过 0.01 时优先 selector p95 更低者，再优先 checkpoint 更小者。
warm selector p95/action-policy p95 `<=10%` 的部署门槛在 policy replay 阶段执行；未通过则不启动 closed-loop。

本阶段不是 ablation。两类模型都运行只是为了从已经冻结的候选中选出唯一 deployment model。

## 实现状态

已实现并通过本地测试：805-state label-blind feature snapshot、raw-image OCR/RGB、query/event source 分离编码、
encoded-state subset scoring、conditional-greedy B1--B4 path、selection merge、exact/sparse evaluation schedule，
以及 held-out reducer。Reducer 从 sealed selections 与 resumable terminal labels 计算真实 utility、trajectory-equal
bootstrap、exact-oracle regret、history-bin 指标、selector latency 和冻结 GO/NO-GO；missing/skipped coverage 不会被
静默过滤。相关 held-out tests 为 14 passed，另有 variable-history/label regression tests 覆盖原 label runner。

Hyper00/Hyper01 已分别完成 333/472 个 label-blind feature states、token cache 与双模型 inference；合并后严格
覆盖 union 805、exact 320、large-history 720、overlap 235，`label_file_read_count=0`。Sealed selections 已写入
[`data/results/set_utility_heldout_v1/`](../data/results/set_utility_heldout_v1/README.md)；evaluation restoration
labels 仍未读取，下一步按该 artifact 物化 exact/sparse truth schedule。

首次 feature snapshot 因原始 full-token root 在 Hyper00/01 各保留 128 个 logical shards 而 fail-fast，未写入
output。learning-curve cache 又只含 train/tune observations，不能替代 evaluation tokens。最终执行按现有数据布局
让两台机器各自物化本机 shard 的 label-blind feature/cache 与双模型 selection partitions，再以 state identity、
track count 和输入 hash 严格合并成 805-state artifact；不重新编码图像，也不放宽任何 label firewall。
