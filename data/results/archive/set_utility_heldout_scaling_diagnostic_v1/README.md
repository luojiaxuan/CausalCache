# Held-out selector scaling diagnostic v1

状态：`COMPLETED_SET_UTILITY_HELDOUT_SCALING_DIAGNOSTIC_WITH_SKIPS`。

本诊断将 10%/25%/50%/100% train-trajectory checkpoints 运行在同一 trajectory-disjoint held-out exact track，
只消费已有 B1/B2 truth，不新增 labels，也不改变 formal v1 `NO_GO`。

## 结果

| Family | 10% | 25% | 50% | 100% | 10%→100% delta (95% CI) | Held-out best |
|---|---:|---:|---:|---:|---:|---:|
| DeepSets | 0.2490 | 0.2436 | **0.2579** | 0.2405 | -0.0085 `[-0.0322, 0.0166]` | 50% |
| Set Transformer | 0.2392 | 0.2460 | 0.2456 | **0.2629** | +0.0237 `[-0.0050, 0.0529]` | 100% |

指标是 319 个 completed exact-track states、86 条 trajectories 上 B1/B2 trajectory-equal macro normalized
recovery。一个预先存在的 strict-parser skip 原样保留。recent 为 0.2521，OCR/RGB 为 0.2383；100% Set
Transformer 相对 recent 为 +0.0108（95% CI `[-0.0157, 0.0392]`），相对 OCR/RGB 为 +0.0246
（`[-0.0055, 0.0561]`）。

## 结论

- Set Transformer 从 10% 到 100% 有 +0.0237 的方向性改善，100% 同时是 tune 与 held-out 最优；
- 但 scaling curve 不单调，paired bootstrap 仍跨 0，不能说同分布扩数已经稳定解决问题；
- DeepSets 没有数据规模收益，tune 最优的 100% checkpoint 在 held-out 反而不是最优；
- 因此继续增加完全同分布 trajectories 可能帮助 Set Transformer，但优先级低于 representation 与
  deployment-search/on-policy coalition mismatch。

下一步按冻结顺序测小历史 B4 strong/exact oracle，先确认 `B=4` 的真实 policy-recovery 上限，再决定 contextualized
multi-latent v3 的目标差距。

## Artifact

- full result：Hyper00
  `/data02/jaxan/runs/causalcache-set-utility-heldout-scaling-v1-3b0be31/result-d7b9c65.json`；
- content SHA256：`c346bcebd2205cbcca5eeb2ee001eb29c3c2ed41a6ef4e5a75b7d503dfa517c2`；
- file SHA256：`24629e2cfa285579421b3a630e5d23cf6fb4fbeb17d68ffca54181fd1fbc929f`；
- lightweight summary：[`summary.json`](summary.json)；
- 当前状态：`PENDING_HF_UPLOAD`。

12×H200 共运行 16 个 host-partition jobs；两台 multi-GPU containers 均 exit 0。为提高吞吐，部分 GPU 同时运行
两个只读 candidate processes，因此本次记录的 per-model latency 不进入任何部署 claim。
