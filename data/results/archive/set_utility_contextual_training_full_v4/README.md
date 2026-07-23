# Full contextual predictor training v4

状态：`COMPLETED_TRAINING / SET_TRANSFORMER_TRUE_UTILITY_WINNER`。

使用 10,658 train states、1,063 tune states 与 frozen full contextual hidden cache 训练；未加载 evaluation。
DeepSets 与 Set Transformer 均在 epoch 1 最佳，并在 epoch 6 early-stop：

| Model | Best tune objective | Checkpoint SHA256 | Elapsed |
|---|---:|---|---:|
| DeepSets | 0.3092726 | `bae1ead5...9e32` | 652.7s |
| Set Transformer | 0.3223304 | `aafe3627...c044` | 1037.8s |

Tune loss 没有预测 deployment winner：真实 B1--B4 macro recovery 为 DeepSets `0.4247`、Set Transformer
`0.4499`、recent `0.4519`。Set Transformer 是 learned winner，但尚未超过 recent，因此不进入 untouched
evaluation 或 policy replay。真实结果见
[`../set_utility_contextual_tune_on_policy_full_v4/`](../set_utility_contextual_tune_on_policy_full_v4/README.md)。

Persistent root：
`/data02/jaxan/runs/causalcache-contextual-training-full-v4-e97f2d4`。

Immutable model artifact：
[`gavinlaw/causalcache-set-utility-predictors-mobile@729a62fd`](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/729a62fdaed53ccf04e641bea9ecac3ac7da3bcc/artifacts/set-utility-contextual-full-v4-44405c2)，
tag=`set-utility-contextual-full-v4-44405c2`，7 files。
