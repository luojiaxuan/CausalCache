# OSWorld AndroidWorld-v3-e1 transfer pilot v1

## 问题

检验在 AndroidWorld 上获得 paired closed-loop `+8.9` percentage points 的 GUI-Owl LoRA
`v3-e1`，是否能零样本迁移到 OSWorld desktop closed-loop，并超过 frozen GUI-Owl。

## 冻结设置

- adapted checkpoint：Hyper01
  `/data02/jaxan/runs/causalcache-margin-sft-v3-eval/lora-epoch1.pt`；
- SHA256：`7122d8978a8c3a86e98a49c83f154e71bbe0b426dadb37e1f77efca88f9c73b8`；
- LoRA：rank 16、alpha 32、LM `q/k/v/o`，不修改视觉塔；
- frozen 与 v3-e1 使用相同 GUI-Owl-1.5-8B base、相同 official no-Google-Drive evenly-spaced
  30 tasks、相同 recent at-most-B4 memory、`max_steps=50`；
- 两臂各 6 environments / 1 policy replica，在 Hyper01 两张 H200 上并发执行；
- task failure 不删除，固定按 score 0 计入 30-task denominator；
- primary：OSWorld mean score、score>0 rate、task-level paired delta 与 bootstrap CI；
- 本实验不含 learned CausalCache selector，不是完整 361-task benchmark。

执行配置：
[`code/configs/causalcache_osworld_v3_e1_transfer_pilot_v1.json`](../code/configs/causalcache_osworld_v3_e1_transfer_pilot_v1.json)。

## 判定

- paired CI 下界大于 0：支持正向 desktop transfer，再考虑扩大 OSWorld；
- CI 跨 0：不声称提升，保留 exploratory 结果；
- CI 上界小于 0：支持负迁移；
- 无论 reward 如何，单独报告 policy HTTP failure、termination、latency 和 Torch allocator peak。

