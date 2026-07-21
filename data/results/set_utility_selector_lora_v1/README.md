# Selector-side LoRA v1

状态：`BOUNDARY_REPLAY_PARITY_PASS_IMPLEMENTATION_IN_PROGRESS`。

- teacher/action policy：原始 frozen GUI-Owl，不含 LoRA；
- selector branch：LM top-4，计划 q/k/v/o rank-8 LoRA；
- Hyper00 单 context parity：sequence `[1,1522,4096]`，selected final tokens bitwise equal，max abs
  difference=`0.0`；
- 训练 labels：复用现有 900 trajectories / 9,287 optimizer states / 84,441 candidate-complete groups；
- 新数据需求：约 23,714 contexts 的 layer-32 full-sequence boundary cache，预计约 277GiB；
- 当前大型 artifact 尚未生成，状态 `PENDING_HF_UPLOAD`。

设计与限制见 [`docs/set_utility_selector_lora_v1.md`](../../../docs/set_utility_selector_lora_v1.md)。
