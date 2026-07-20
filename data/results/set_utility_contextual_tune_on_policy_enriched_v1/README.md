# Contextual train on-policy enrichment v1

状态：`COMPLETED / NO_GO_TRAIN_ON_POLICY_ENRICHMENT_V1`。

在固定的 1,063 个 tune states、100 条 trajectory 上，使用 train-only on-policy/conditional-marginal
enrichment 重训 DeepSets 与 Set Transformer。真实 mixed-fidelity GUI-Owl restoration truth 覆盖
11,692 个去重 coalitions，0 skip。

| Method | B1 | B2 | B3 | B4 | B1--B4 macro | Long+ | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSets enriched | 0.2038 | 0.4055 | 0.5364 | 0.6296 | 0.4438 | 0.3298 | 10.31 ms |
| Set Transformer enriched | **0.2405** | 0.4066 | 0.4958 | 0.6227 | 0.4414 | 0.3423 | 19.77 ms |
| Recent | 0.1911 | **0.4361** | **0.5543** | **0.6263** | **0.4519** | **0.4008** | -- |

DeepSets-minus-recent=`-0.00809`，trajectory bootstrap 95% CI=`[-0.02673, 0.00954]`；Set
Transformer-minus-recent=`-0.01051`，95% CI=`[-0.03491, 0.01237]`。两者均未达到冻结 gate，且 long-history
slice 明显低于 recent，因此不启动 untouched evaluation、policy replay、closed-loop 或 matched-NLL。

训练侧新增 37,982 个 targeted distance rows。相比 broad-label full checkpoint，DeepSets macro 从
`0.42470` 提高到 `0.44383`，但 Set Transformer 从 `0.44985` 降到 `0.44141`。这说明 targeted supervision
不是单纯缺失数据量的修复；当前 conditional-marginal objective/representation 仍未把 restoration oracle signal
蒸馏成优于廉价 recent heuristic 的 long-history selector。

执行备注：首次 Set Transformer training backward 遇到 PyTorch cuDNN `mha_graph.execute` runtime error；
committed repair 仅关闭 cuDNN SDP backend，随后正常完成。首次 Hyper01 tune-label launch 因 schedule 未复制而在
model forward 前 fail-fast；保留 failure root，补齐相同 SHA schedule 后只重启 partitions 4--7。Hyper00 不重跑，
最终科学 denominator、schedule 与 source revision 均未变化。

完整 result 当前位于 Hyper00
`/data02/jaxan/runs/causalcache-contextual-tune-evaluation-enriched-v1-aa9a071/result.json`，content SHA256=
`3f73e177...c23817`，file SHA256=`75f75bc2...7753`。完整 result、schedule、labels、selections、training
summary 与失败日志已发布到 private HF dataset revision
[`bbee1ae7`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/bbee1ae7aae2a712aaf2a897a08fa69adcef4ee6/artifacts/set-utility-contextual-tune-enriched-v1-3f73e17)，
tag=`set-utility-contextual-tune-enriched-v1-3f73e17`。为避免逐文件提交 8,007 个小文件，完整 payload 使用
`payload.tar.gz`，SHA256=`bc4d206e314c671040cf1e7fd7c38d4f7b9ca39985042b27eda3949fb4ae3ac6`；README、summary
与最终 `evaluation/result.json` 可直接浏览。两个 checkpoints 已在 private HF model revision
[`1fb466e9`](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/1fb466e93947a13715c46022b6ba5d14f3b5edec/artifacts/set-utility-contextual-enrichment-v1-2711ab55)，
tag=`set-utility-contextual-enrichment-v1-2711ab55`。
