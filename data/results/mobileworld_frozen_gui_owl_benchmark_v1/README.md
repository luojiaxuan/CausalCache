# MobileWorld frozen GUI-Owl benchmark v1

## 结论

本次 benchmark campaign 已结束，但不是完整 117/117 结果：冻结 GUI-Owl 在
`recent-at-most-B4`、每 task 最多 50 步、infra/policy failure 最多重试 2 次的合同下，
得到 **114/117 有结果、27 success**。3 个 task 在三次尝试后仍因 policy action 生成失败而
没有结果：

- `CheckCartPriceTask`
- `MastodonServerInfoReportTask`
- `MattermostTechnicalDebtTriageTask`

observed mean score 为 `27/114 = 0.236842`；若把 3 个 missing 严格计 0，则为
`27/117 = 0.230769`。不得把 observed mean 当成完整 denominator 分数。

从首个 full runner 启动到最后 shard 结束的真实 makespan 为
**22,863.108 秒（6:21:03.108）**。这包含单 GPU 前段、切换和双 GPU 尾段；不是只截取
双卡后的较短时间。

## 并发与时间

单卡 capacity 使用同一组 16 个 GUI-only tasks、每 task 只跑 1 步：

| environments | wall time (s) | tasks/hour | 完整性 |
|---:|---:|---:|---:|
| 1 | 383.715 | 150.111 | 16/16 |
| 2 | 192.133 | 299.793 | 16/16 |
| 4 | 102.320 | 562.940 | 16/16 |
| 8 | 62.540 | 921.010 | 16/16 |

8 environments 已把单 policy 的 warm request throughput 喂满。full campaign 最初因
preflight 只有 Aries GPU 1 空闲，使用 GPU 1 + 8 environments。运行中重新 inspect 后
GPU 0 释放，于 52 个结果落盘时有界停止原 runner，将剩余 65 个 task 以冻结 roster hash
分为 33/32 两个无重叠 shard：

| phase | GPU / env | 结果 | success | wall time |
|---|---|---:|---:|---:|
| single-GPU prefix | GPU 1 / 8 env | 52 | 17 | 13,017.284 s |
| two-GPU tail shard 0 | GPU 1 / 8 env | 32/33 | 7 | 9,689.986 s |
| two-GPU tail shard 1 | GPU 0 / 8 env | 30/32 | 3 | 9,054.263 s |

切换前 52、两个 tail shard 的 32/30 与 3 个 missing 已验证集合互斥，四者并集严格等于
117-task roster。原 runner 收到 SIGINT 后仍在 policy queue 中的 7 个 client 断开，造成
GPU 1 policy counter 在两个 execution records 之间增加 `7 requests / 7 failures`；它们
不属于新 shard。

## 冻结身份与审计边界

- MobileWorld：`8ae506487bf87785292d6cad101c49955d704d39`
- environment image：
  `ghcr.io/tongyi-mai/mobile_world@sha256:b680380eac98a7ad064707f9653772af18554d201a3e6e7cf8f15d58cdc73240`
- task source：上述 Git revision 的 `src/` 只读挂载到 `/app/service/src`
- GUI-Owl：`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`
- runtime：PyTorch `2.11.0+cu130`、Transformers `5.6.0`
- host/GPU：Aries、2 × NVIDIA RTX A6000
- initial execution Git：`83ee957eb461ab49e57508c74ca5ccf56ef2f05a`
- resume shard Git：`e95edd05092cdc0a06a646965345b065eb58ff31`

官方 GHCR image 内旧 `ThanksgivingPrepTask` 缺少 `reset_chrome` import；正式结果只来自
pinned source read-only mount。使用 image 内旧 source 的两次 15/16 capacity 尝试均为
无效 preflight evidence，不进入上表。

## Artifacts

- machine-readable summary：[`summary.json`](summary.json)
- config：
  [`code/configs/causalcache_mobileworld_memory_v1.json`](../../../code/configs/causalcache_mobileworld_memory_v1.json)
- frozen roster：
  [`data/manifests/mobileworld_memory_split_v1.json`](../../manifests/mobileworld_memory_split_v1.json)
- raw execution records、logs、screenshots 和 trajectories：Aries
  `/mnt/data6/jiaxuanluo/runs/mw-memory-v1`，状态 `LOCAL_PRIVATE_RAW_TRACE`

raw traces 不进入 Git。两个 policy 已停止，两组 8+8 emulator exception containers 已按
fleet manifests 精确清理，canonical `sglang-omni-jaxan` 保留。
