# Full contextual predictor training v4

状态：`COMPLETED_TRAINING / AWAITING_TRUE_SELECTOR_UTILITY`。

使用 10,658 train states、1,063 tune states 与 frozen full contextual hidden cache 训练；未加载 evaluation。
DeepSets 与 Set Transformer 均在 epoch 1 最佳，并在 epoch 6 early-stop：

| Model | Best tune objective | Checkpoint SHA256 | Elapsed |
|---|---:|---|---:|
| DeepSets | 0.3092726 | `bae1ead5...9e32` | 652.7s |
| Set Transformer | 0.3223304 | `aafe3627...c044` | 1037.8s |

Tune loss 不是 deployment winner。两个 checkpoint 的 1,063-state conditional-greedy paths 已冻结；对应
10,368 个去重 coalitions 正在生成真实 restoration truth。只有真实 B1--B4 selector recovery 与 recent 的比较
可以决定下一阶段。

Persistent root：
`/data02/jaxan/runs/causalcache-contextual-training-full-v4-e97f2d4`。

HF 状态：`PENDING_HF_UPLOAD`，intended repo=
`gavinlaw/causalcache-set-utility-predictors-mobile`。
