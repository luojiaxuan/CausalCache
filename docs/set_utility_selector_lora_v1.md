# Selector-side GUI-Owl LoRA v1

## 假设与边界

当前 direct-marginal student 已使用完整 frozen GUI-Owl final hidden states，但扩大 labels、latents 与 set
layers 没有稳定缩小 B2/Long+ gap。v1 检验的新假设是：瓶颈位于 frozen representation，而不只是 set head。

Teacher/action policy `pi_0` 始终保持原始 frozen GUI-Owl。LoRA 只存在于 selector encoder，因此现有
restoration labels `D(S)` 全部有效，不重新生成 teacher labels。

主模型固定：

- GUI-Owl vision tower、merger 与 LM layers 0--31 冻结；
- LM layers 32--35 的 attention `q/k/v/o` 使用 rank-8、alpha-16、zero-init LoRA；
- 保留 direct conditional-marginal Set Transformer head，不回到 scalar `U(S)`；
- 从当前最佳 Set checkpoint warm start；先冻结 head 只训 LoRA，再以更高 head LR 联合训练；
- 每个 epoch 按固定 256-state train-heldout 的真实 B1--B4 restoration recovery 选模，不按 loss 选模。

对照为：相同 checkpoint 的 frozen continuation，以及现有 final hidden 后的 zero-init residual token adapter。
目标是补回 B2 与 Long+，同时保住 B3/B4。

## Boundary cache

现有 119GB cache 只含 final norm 后的 visual positions 与 image 前 64 个 text positions，不能训练顶层
LoRA。新 cache 必须保存 layer 32 输入的完整 sequence，以及 3D M-RoPE position ids、2D attention mask、
input ids 和 output token indices。optimizer + 256-state checkpoint denominator 共约 23,714 个 unique
contexts，预计约 277GiB；旧 restoration labels 不变。

正式 allowlist 已从 9,287 optimizer states 与 256 checkpoint states 物化为 23,714 unique contexts
（query=9,543、event=14,171），content SHA256=`4172d468...6c510`，Hyper00 staging=
`/data02/jaxan/artifacts/causalcache-selector-boundary-allowlist-v1`。它排除其余未被本轮训练或选模消费的
contexts，避免额外生成约 50GiB cache。

当前 selector entity prompt 与实际 action-policy mixed-history prompt 不同，所以二者不能共享 LM prefix
activations。event 表示可在 arrival-time cache；每个 query 的独立 selector forward 必须计入端到端 latency。

## 已通过的实现门

2026-07-21 在 Hyper00 H200 上，用真实 context
`016053e5...dcbcb6df` 验证 top-4 branch replay：

- full sequence shape=`[1,1522,4096]`；
- layer-32 boundary 经原 layers 32--35 + final RMSNorm 重放；
- selected visual/text final tokens bitwise equal；max absolute difference=`0.0`。

该检查证明完整 boundary、M-RoPE 和 causal mask 的重放路径正确；后续仍需在批量 cache 上验证当前 best
head 的 selection parity。

## 2026-07-21 实现与执行状态

- 已实现 full-sequence boundary 的分片提取、原子 chunk、SHA/identity 校验、断点续跑与最终 manifest；
- 已实现 zero-init post-final token adapter 对照；
- 已实现 pruned GUI-Owl top-4 branch、q/k/v/o LoRA、按原始 sequence length 分桶的 entity microbatch，
  以及 current direct-marginal Set head 的严格 warm start；
- 已实现 LoRA-only → joint 两阶段 trainer；每个 epoch 只能由固定 256-state development 的真实
  at-most-`B` restoration recovery 选模，缺 truth 时停在 barrier，不允许 loss 或最后 epoch 代替；
- token-adapter control 已实现，并复用相同 checkpoint-selection contract；
- Hyper00/Hyper01 的 12/12 partitions 已完整提取 23,714/23,714 allowlist contexts，0 failure；Hyper00
  单机 cache 已通过 finalizer：3,065 shards、289,753,201,728 bytes、content SHA256=
  `77dec757c5637d538637463984caa5dff8481e36163589d540bccc5a7c55d16b`；
- executable training config 已单独版本化，未改写 extraction config；token-adapter 与 LoRA-only 首 epoch
  均已完成。token-adapter 的 132-state truth 已 seal 并回填，B1--B4 macro/Long+=
  `0.30893/0.30401`，低于 recent=`0.37330/0.38002`；LoRA-only 的 129-state truth 也已 seal，
  macro/Long+=`0.30542/0.32770`，同样低于 recent。当前不能声称 selector representation 已改善；
  joint adaptation 是剩余的决定性检验。

## 下一步

1. 合并并 seal joint LoRA epoch 1 的跨机 truth；
2. 以 fixed-256 true recovery 判断 joint LoRA 是否改善 B2/Long+；token-adapter control 不再增加 epoch。

Boundary extraction 的冻结参数见
[`causalcache_set_utility_selector_lora_v1.json`](../code/configs/causalcache_set_utility_selector_lora_v1.json)。
Executable training 参数见
[`causalcache_set_utility_selector_lora_training_v1.json`](../code/configs/causalcache_set_utility_selector_lora_training_v1.json)。
