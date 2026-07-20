# Contextual multi-latent training v3

状态：`COMPLETED_TRAIN_TUNE_ONLY`；untouched evaluation 尚未访问。

相同 25% nested train（2,640 states/250 trajectories）与完整 tune（1,063 states/100 trajectories）、相同
contextual cache、16 latent、hidden 256、2-layer resampler 下：

| Model | Best epoch | Tune objective ↓ | Ranking acc. | Raw MAE | Termination |
|---|---:|---:|---:|---:|---|
| DeepSets | 2 | 0.32128 | 0.69988 | 0.03329 | early stopping |
| Set Transformer | 2 | **0.30910** | 0.68556 | 0.03371 | early stopping |

Set Transformer 相对旧 25% checkpoint 的 `0.32363` 改善 `0.01453`，也低于旧 100% checkpoint 的
`0.31577`。DeepSets 相对旧 25% 的 `0.31731` 没有改善。这个结果说明 contextual multi-token 表示与 set
interaction 的组合值得继续，但标准 tune loss 不能替代 selector truth。

首轮 DeepSets 曾把 16 latent 直接求和，导致 BF16 近似同分；该实现没有被保留为结果。修正为 latent 内
平均、event 间求和后重跑，得到上表结果。Set Transformer 路径不受该修正影响。

Checkpoints：

- DeepSets：SHA256 `773335a296db976b45f67b64fc75308ce1ce3007b50949794cfac33b0b8c0a93`，
  19,217,500 bytes，summary content `30cd6d81...b5ca87`；
- Set Transformer：SHA256 `d52ad1ec7ed131dc4c8402711bc04764bddd1668e40ebff3e89ab0ffe6200b24`，
  25,016,076 bytes，summary content `7552e775...781d78`；
- config SHA256 `2467e8744f89a05bd190703bacbe378541d6948130d0b240164a84bd58a5bc7b`。

Persistent paths：

- DeepSets：Hyper00
  `/data02/jaxan/runs/causalcache-contextual-training-v3-corrected-966b09e/deepsets_contextual_d256_l16_r2_lr3e4`；
- Set Transformer：Hyper00
  `/data02/jaxan/runs/causalcache-contextual-training-v3-911f543/set_transformer_contextual_d256_l16_r2_s2_lr1e4`；
- intended HF model repo：`gavinlaw/causalcache-set-utility-predictors-mobile`，artifact
  `artifacts/set-utility-contextual-training-v3-<result-hash>`；
- upload status：`PENDING_HF_UPLOAD`。

下一步已经启动：在 1,063 个 tune states 上运行两个 frozen conditional-greedy selector，生成它们实际访问
的 at-most-`B` coalitions；随后用真实 policy restoration labels 比较 DeepSets、Set Transformer 与 recent。
