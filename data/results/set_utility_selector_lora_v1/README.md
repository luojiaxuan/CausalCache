# Selector-side LoRA v1

状态：`IMPLEMENTATION_TESTS_PASS_BOUNDARY_CACHE_PENDING`。

- teacher/action policy：原始 frozen GUI-Owl，不含 LoRA；
- selector branch：LM top-4，计划 q/k/v/o rank-8 LoRA；
- Hyper00 单 context parity：sequence `[1,1522,4096]`，selected final tokens bitwise equal，max abs
  difference=`0.0`；
- 训练 labels：复用现有 900 trajectories / 9,287 optimizer states / 84,441 candidate-complete groups；
- 新数据需求：约 23,714 contexts 的 layer-32 full-sequence boundary cache，预计约 277GiB；
- boundary allowlist：23,714 contexts（query=9,543、event=14,171），content=
  `4172d46895edc14b507defe9af479cdf072e65fb026b7f2c79cb194de866c510`；
- boundary extractor/finalizer、token-adapter control 与 top-4 LoRA composite 已实现；Hyper00 测试
  `14 passed`；
- allowlist 已同步到 Hyper00/Hyper01，但 23,714-context cache 与 LoRA checkpoint 尚未生成；
- 当前大型 artifact 尚未生成，状态 `PENDING_HF_UPLOAD`。

设计与限制见 [`docs/set_utility_selector_lora_v1.md`](../../../docs/set_utility_selector_lora_v1.md)。
