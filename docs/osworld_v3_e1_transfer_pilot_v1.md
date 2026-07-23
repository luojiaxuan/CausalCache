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

## 结果

| Metric | frozen | v3-e1 |
|---|---:|---:|
| Mean OSWorld score | 0.16667 | 0.13333 |
| Score > 0 | 5/30 | 4/30 |
| Counted HTTP 500 | 13/30 | 9/30 |
| Mean completed steps | 34.47 | 36.07 |
| Policy `done` | 3 | 2 |
| Step-budget exhausted | 14 | 19 |
| Policy steps | 1,034 | 1,082 |
| Torch peak allocated | 19.103 GB | 19.167 GB |

Paired result：

- mean score delta `v3-e1 - frozen = -0.0333333`；
- paired bootstrap 95% CI `[-0.1333333, 0.0666667]`；
- 1 win / 2 losses / 27 ties。

正式判定为：

> `NO_EVIDENCE_OF_POSITIVE_OSWORLD_TRANSFER_V3_E1_V1`

v3-e1 能直接使用 `computer_use` action space，并完成 Chrome、OS 和 VS Code 等 4 个任务，说明
模型没有发生完全的 cross-platform capability collapse。但它只获得 1 个 frozen 未获得的成功，
同时丢失 2 个 frozen 的成功，因此不能支持“AndroidWorld 上的 +8.9pt 能跨平台迁移到 OSWorld”。

## 可执行性与成本

v3-e1 将 task-level policy HTTP 500 从 13 降到 9，但两臂的所有 500 response body 均未由 v1
client 保存，无法继续区分 action parse、coordinate validation 或其他 request-level exception。
可执行失败减少没有转化成 task score 提升。

| Diagnostic | frozen | v3-e1 |
|---|---:|---:|
| Mean policy latency | 11.25 s | 12.90 s |
| p95 policy latency | 13.13 s | 15.23 s |
| Mean generation time | 2.23 s | 2.56 s |
| Mean server queue | 8.76 s | 10.09 s |
| Benchmark wall time | 2,660.7 s | 3,117.6 s |

本轮 GPU0/1 在启动时空闲，运行中未观察到外部共享占用；v3-e1 平均 policy latency 比 frozen
高约 14.7%，Torch allocator peak 只增加约 64 MB。wall time 同时受 v3-e1 更多 completed steps
影响，不能全部解释为 LoRA kernel overhead。

## 执行记录与边界

- 正式 run 使用精确 Git commit `571ea60c536fe46199e69d32ee014b5fecd3eb97`；
- 第一次 policy container launch 因缺少 `PYTHONPATH=code` 在模型加载前退出；
- 第一次 driver launch 因错误使用不含 OSWorld 系统依赖的 PyTorch image，在 0 policy requests 时
  退出；invalid root 随后删除；
- 正式 driver 恢复已验证的 `hongccc/sglang-omni:dev`，两臂从全新空 root 启动；
- task failure 均按 0 分留在固定 30-task denominator；
- 同一 frozen policy 在不同 pilot 的成功数会变化，说明 OSWorld/runtime 存在 cross-run variability；
  本结论只使用本轮同时运行、task identity 完全成对的两臂；
- 该 30-task exploratory pilot CI 较宽，不是完整 361-task benchmark；
- 本实验不含 learned selector，不能据此判断 CausalCache memory selection。

## Artifact

- Git summary：
  [`data/results/osworld_v3_e1_transfer_pilot_v1/summary.json`](../data/results/osworld_v3_e1_transfer_pilot_v1/summary.json)，
  SHA256 `b274b5357d6df59b98bef9f278250a8bff62d72b2cb3454f6d845b7d294d0f2b`；
- raw：Hyper01
  `/data01/jaxan/osworld-runner/v3e1-transfer-pilot-v1/raw`，
  951,036,579 bytes / 2,301 files；
- v3-e1：Hyper00
  `/data02/jaxan/runs/causalcache-margin-sft-v3/lora-epoch1.pt`；Hyper01 verified mirror
  `/data02/jaxan/runs/causalcache-margin-sft-v3-eval/lora-epoch1.pt`；
- checkpoint 仍为 `PENDING_HF_UPLOAD`，不能把 local path 表述成 canonical publication。

## 决策

当前不扩跑完整 OSWorld，也不因为该结果自动追加 α16。α16 只在 α32 出现明显 grammar leakage
时用于减弱适配；本轮 α32 的 HTTP failure 反而少于 frozen，主要缺口是 task reward，不是需要用
半强度修复的硬语法崩溃。若要继续 desktop 方向，应加入 desktop action/task supervision 后另立
新 pilot。
