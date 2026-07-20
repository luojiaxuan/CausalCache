# Contextual tune on-policy truth v3

状态：`COMPLETED / REPRESENTATION_ONLY_NO_GO`。

1,063/1,063 tune states 全部完成，无 skip；9,814 个 rows 覆盖两个 frozen conditional-greedy paths、recent
B1--B4 与 empty/full anchors。结果按 100 条 trajectory 等权聚合：

| Method | B1 | B2 | B3 | B4 | B1--B4 macro | Long+ | Total p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSets contextual | 0.1906 | 0.4217 | 0.5440 | **0.6405** | 0.4492 | 0.4000 | 10.96 ms |
| Set Transformer contextual | 0.1905 | 0.4114 | 0.5286 | 0.6155 | 0.4365 | 0.3618 | 21.98 ms |
| Recent | **0.1911** | **0.4361** | **0.5543** | 0.6263 | **0.4519** | **0.4008** | -- |

DeepSets 是两个 contextual models 中的 winner，但仍比 recent 低 0.0027，trajectory bootstrap 95% CI=
`[-0.0160, 0.0100]`。Set Transformer 比 recent 低 0.0154，95% CI=`[-0.0314, -0.0004]`。因此完整
GUI-Owl contextual hidden states 与 multi-latent representation 单独不能闭合 distillation gap；当前结果不授权
untouched evaluation、policy replay 或 closed-loop。

按既定执行要求，下一步仍使用已经物化的 full train+tune contextual inputs 训练两种 family，判断 25%→100%
数据是否改变 selector truth；若 full-data 仍不超过 recent，则优先生成 train-side on-policy coalitions，而不是
继续增加同分布 trajectories。

Source of Truth：

- raw labels：Hyper00
  `/data02/jaxan/runs/causalcache-contextual-tune-on-policy-labels-v3-82a5da1`，23MB，`PENDING_HF_UPLOAD`；
- full result：Hyper00
  `/data02/jaxan/runs/causalcache-contextual-tune-on-policy-evaluation-v3-bd0d376/result.json`；
- result content SHA256：`f34d368fdc5dcf9ccbd6edf6910de0920fd7caa528f5c3b8e11c93201c25e5e1`；
- lightweight summary：[`evaluation-summary.json`](evaluation-summary.json)。
