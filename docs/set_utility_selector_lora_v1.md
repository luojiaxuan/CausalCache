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
- Hyper00/Hyper01 的 12/12 partitions 已完整提取 23,714/23,714 allowlist contexts，0 failure。为避免跨机
  训练读取，当前在 Hyper00 重放 Hyper01 半份并形成单机 cache；最终 manifest/content SHA256 尚待
  finalizer 计算；
- LoRA 与 token-adapter 均尚未训练，当前不能声称 selector representation 已改善。

## 下一步

1. 完成 Hyper00 单机 cache，并运行 finalizer 校验 chunk inventory、identity 与 SHA；
2. 用最终 boundary manifest/content SHA256 生成新的 versioned executable training config，不改写已用于
   extraction 的冻结 config；
3. 在 Hyper00 以 6-GPU DDP 运行 LoRA-only phase 1；
4. 每个 epoch 进入固定 256-state development truth barrier，补齐并封存实际访问的 coalitions 后再决定是否
   继续 joint phase。

Boundary extraction 的冻结参数见
[`causalcache_set_utility_selector_lora_v1.json`](../code/configs/causalcache_set_utility_selector_lora_v1.json)。
