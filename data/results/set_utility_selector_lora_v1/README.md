# Selector-side LoRA v1

状态：`BOUNDARY_CACHE_COMPLETE_TRAINING_STARTING`。

- teacher/action policy：原始 frozen GUI-Owl，不含 LoRA；
- selector branch：LM top-4，计划 q/k/v/o rank-8 LoRA；
- Hyper00 单 context parity：sequence `[1,1522,4096]`，selected final tokens bitwise equal，max abs
  difference=`0.0`；
- 训练 labels：复用现有 900 trajectories / 9,287 optimizer states / 84,441 candidate-complete groups；
- 新数据需求：约 23,714 contexts 的 layer-32 full-sequence boundary cache，预计约 277GiB；
- boundary allowlist：23,714 contexts（query=9,543、event=14,171），content=
  `4172d46895edc14b507defe9af479cdf072e65fb026b7f2c79cb194de866c510`；
- boundary extractor/finalizer、LoRA trainer、token-adapter control 与 top-4 LoRA composite 已实现；
- Hyper00/Hyper01 两机共 12 个独立可恢复 partitions 已完成 23,714/23,714 contexts，0 failure。执行清单见
  [`boundary-rollout-plan.json`](boundary-rollout-plan.json)；
- Hyper00 单机 cache 已 finalize：3,065 shards、289,753,201,728 bytes、content SHA256=
  `77dec757c5637d538637463984caa5dff8481e36163589d540bccc5a7c55d16b`；本地路径
  `/data02/jaxan/runs/causalcache-selector-boundary-h00-c8cbc63`；
- executable training config 已绑定最终 cache SHA；token-adapter phase 1 已在 Hyper01 4×H200 启动，LoRA
  尚未生成 checkpoint，不能报告方法效果；
- 下一步：`Hyper00 LoRA phase 1 → 256-state truth barrier`；
- boundary cache 完成后需上传 private HF；当前状态 `PENDING_HF_UPLOAD`。

设计与限制见 [`docs/set_utility_selector_lora_v1.md`](../../../docs/set_utility_selector_lora_v1.md)。
