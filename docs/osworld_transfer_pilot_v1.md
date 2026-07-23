# OSWorld terminal-s60 cross-platform transfer pilot v1

## 研究问题

Android/mobile GUI 轨迹上训练的 terminal-repair LoRA `s60` 已通过单步 desktop grammar leakage
probe。本实验进一步检验：把同一 LoRA 接到 GUI-Owl desktop policy 后，是否能在真实 OSWorld
closed-loop 中超过原始 frozen GUI-Owl。

本实验刻意不接 learned memory selector。两臂都使用相同的 `recent` at-most-B4 memory，隔离
policy LoRA 的 cross-platform transfer。

## 冻结协议

| Field | Value |
|---|---|
| OSWorld | `b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf` |
| Roster | official `test_nogdrive`, evenly-spaced 30/361 tasks |
| Domains | 10 |
| Memory | recent, at-most `B=4` |
| Max steps | 50 |
| Execution | 6 environments/arm, 1 policy replica/arm, concurrent arms |
| Base model | `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc` |
| Adapted model | terminal-s60, rank 16, alpha 32 |
| LoRA SHA256 | `cba455cbfede5e091c06d8714ef000d5582adb46c4a5343fa8c0ab8c4ab47b18` |
| Episode code | `78428f8aed6eff254bfae3ffbae1c6fe3eb52cbd` |
| Reduction code | `27db19a` |
| Host | Hyper01, initially 2 × H200 |

任务级 failure 属于 policy executable behavior 的一部分，不允许从 denominator 删除。只要一个
task 已终止为 result 或 failure，就在固定 30-task denominator 中计入；failure 的 score 固定为 0。

## 结果

| Metric | frozen | terminal-s60 |
|---|---:|---:|
| Mean OSWorld score | 0.13333 | 0.07037 |
| Score > 0 | 4/30 | 3/30 |
| Counted HTTP 500 | 10/30 | 1/30 |
| Mean completed steps | 37.87 | 20.13 |
| Policy `done` | 4 | 20 |
| Policy `fail` | 0 | 1 |
| Step-budget exhausted | 16 | 8 |
| Policy steps | 1,136 | 604 |
| Torch peak allocated | 19.106 GB | 19.182 GB |

成对结果：

- mean score delta `terminal-s60 - frozen = -0.0629633`；
- paired bootstrap 95% CI `[-0.2296300, 0.1037033]`；
- 3 wins / 4 losses / 23 ties。

因此正式判定为：

> `NO_EVIDENCE_OF_POSITIVE_OSWORLD_TRANSFER_TERMINAL_S60_V1`

这不是“s60 一定损害 desktop”的显著性结论；它表示在本次 30-task exploratory pilot 中，正向
迁移主张未成立，且点估计低于 frozen。

## 行为解释

terminal-s60 的 HTTP 500 从 10 降到 1，说明 terminal repair 没有破坏 desktop action grammar，
并可能改善了 closed-loop 输出的可执行稳定性。但 s60 在 20/30 正常结果上以 `policy_done`
终止，平均轨迹长度约为 frozen 的一半。它在 mobile 数据上学到的 terminal 倾向跨平台生效，
却没有稳定学到 desktop task completion 所需的操作策略。

这也修正了先前单步 leakage probe 的解释：30/30 prompt-level parse valid 只能说明输入样本上的
grammar 没有硬渗漏，不能替代长轨迹 closed-loop executable-rate 或 task-success 测试。

## 成本边界

reducer 从失败 checkpoint 一并纳入 policy steps：

| Diagnostic | frozen | terminal-s60 |
|---|---:|---:|
| Mean policy latency | 11.63 s | 13.74 s |
| p95 policy latency | 13.45 s | 16.27 s |
| Mean generation time | 2.27 s | 2.69 s |
| Mean server queue | 9.10 s | 10.79 s |

s60 所在 GPU1 在运行期间被其他 active processes 后续共享，因此 latency/generation 对比存在
外部负载混杂，不能解释为纯 LoRA overhead。Torch allocator 的 model-process peak（19.182 GB）
只比 frozen 高约 76 MB；`nvidia-smi` 曾显示的约 104 GB 总占用主要来自同卡其他进程，未归因给
s60。

## Failure 与可复现边界

- frozen 10 个、s60 1 个 task 收到 policy endpoint HTTP 500；服务均继续存活，不是 server crash
  或 OOM；
- v1 HTTP client 没有把 500 response body写入 failure record，因此本轮不能继续区分 action parse、
  coordinate validation 或其他 request-level exception；
- failure 前的 checkpoint steps 已由 reducer 保留并验证 policy profile；
- 30-task pilot 的 CI 很宽，且不是完整 361-task benchmark；
- reward 结论可用；latency 因 GPU1 后来被共享，只作诊断。

## Artifact

- Git summary：
  [`data/results/osworld_transfer_pilot_v1/summary.json`](../data/results/osworld_transfer_pilot_v1/summary.json)，
  SHA256 `42aff280638a8e70eacc0855837310449a2ecb72cbc5c25bf3e1624173f608d6`；
- raw persistent mirror：Hyper01
  `/data01/jaxan/osworld-runner/transfer-pilot-v1/raw`；
- raw size：715,756,714 bytes / 1,925 files；
- raw 只作为本 pilot 的 local diagnostic evidence 保留，没有伪装成 reusable dataset 或上传到 Git。

## 下一步

这个结果不否定 CausalCache selector，也不应触发完整 OSWorld 361-task 扩跑。若后续仍需要
desktop main result，应先：

1. 保存 HTTP 500 response body，闭合真实 failure taxonomy；
2. 用 desktop action data 做小规模 schema/task adaptation，避免只迁移 mobile terminal bias；
3. 在同一未共享硬件上重跑新的固定 pilot；
4. pilot 显著超过 frozen 后才扩到完整 OSWorld。

