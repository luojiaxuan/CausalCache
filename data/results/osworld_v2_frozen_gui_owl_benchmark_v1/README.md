# OSWorld 2.0 frozen GUI-Owl benchmark v1

## 结论

冻结 `mPLUG/GUI-Owl-1.5-8B-Instruct` 在 OSWorld 2.0 full-108 上几乎不可用：
最终只有 63 个 task 进入 evaluator，45 个 task 保留分类失败；严格固定 108 分母的
score 为 **0.006692**，只有 5 个 task 得到正分。

这轮从首个 runner 启动到最后一个 targeted recovery 结束，实测
**2,791.472 秒 = 46:31.472**。不含 recovery 的 12-shard 主 run 为
**2,334.883 秒 = 38:54.883**。Hyper01 的 72-task 主体为
**1,740.337 秒 = 29:00.337**。

## 并发拓扑

| Host | GPU | policy replicas | environments | 主 run tasks | 主 run wall |
|---|---|---:|---:|---:|---:|
| Hyper01 | H200 GPU 1/2 | 4 / GPU，8 total | 2 / replica，16 total | 72 | 29:00.337 |
| Aries | A6000 GPU 0/1 | 2 / GPU，4 total | 2 / replica，8 total | 36 | 38:54.883 |

主 run 共 12 个 policy replicas、24 个并发 Docker VMs、4 张物理 GPU。Hyper01
GPU 0 在正式重启窗口被其他 workload 占用，因此未触碰；Taurus/Hyper00 复查时也没有
低于 1 GiB 的空卡。H200 每卡 4 个模型副本把原先单 replica inference lock 的低利用率
提升到实测最高约 99%。

`max_steps=500` 的首轮吞吐诊断会把重复点击拖成数小时，因此正式结果使用本项目既有
OSWorld 口径 `max_steps=50`。模型 revision、prompt、parser、recent-B4 memory、
task denominator 与 evaluator 均未改变。这个 46:31.472 结果不能与官方 Claude
`max_steps=500` wall time 直接比较。

## 分数

| Split | denominator | evaluator result | classified failure | positive | score sum | strict mean |
|---|---:|---:|---:|---:|---:|---:|
| full | 108 | 63 | 45 | 5 | 0.7227 | **0.006692** |
| memory core | 43 | 24 | 19 | 1 | 0.1500 | 0.003488 |
| memory stress union | 65 | 34 | 31 | 2 | 0.6752 | 0.010388 |
| non-memory control | 43 | 29 | 14 | 3 | 0.0475 | 0.001105 |

正分 task 只有：`057=0.5252`、`070=0.15`、`083=0.02`、`087=0.01`、
`090=0.0175`。由于总体 floor 极低且不同 split 的 failure coverage 不同，本轮不能据此
声称 memory split 带来收益或退化。

## Failure closure

| 最终分类 | 数量 | 处理 |
|---|---:|---|
| policy endpoint HTTP 500 | 39 | policy server 全程存活；按冻结 policy failure 计 0 |
| GitLab 未配置 | 2 | task `026/041` 需要 self-hosted GitLab URL/token；严格分母计 0 |
| environment upload HTTP 500 | 2 | task `061/064` targeted retry 后仍失败；计 0 |
| evaluator runtime error | 1 | task `035` whole-table evaluator 失败；计 0 |
| environment worker exhausted | 1 | task `082` 两台 host 均无法稳定拉起本地服务；计 0 |

Aries 对官方 hosted mocked websites 返回 503；这些 task 全部按 task ID 转移到 Hyper01，
不把 503 当作模型失败。补跑中 task `057` 得到 0.5252，证明 failover 是必要的。
运行时还修复了 gated task lazy import 和 percent-encoded local asset path；所有对应
task 都重新进入 setup/policy，未把已修复 infra failure 留在最终分类中。

完整机器可读结果见 [`summary.json`](summary.json)。

## Source of Truth

- Git branch：`luojiaxuan/mobileworld-memory-osworld2`；
- 主 run launch revision：`981eaabf49f3fe86361c059e811c9994c09a10fb`；
- OSWorld 2.0：
  `v2026.06.24@2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6`；
- frozen GUI-Owl：
  `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；
- Hyper01 raw run：
  `/data01/jaxan/runs/osworld-v2-memory-benchmark-v1`；
- Aries raw run：
  `/mnt/data6/jiaxuanluo/runs/osworld-v2-memory-benchmark-v1`
  （container 内 `/data/runs/osworld-v2-memory-benchmark-v1`）；
- raw screenshots/trajectories：`LOCAL_PRIVATE_RAW_TRACE`，不进入 Git；
- config SHA256：
  `b445a2496725e0cf9b685d63ca6cb9fb71f9b906f8ebd79878bedc05c2f5f943`；
- memory plan SHA256：
  `d4a9889d0f068d4c09ece5302ca2febe6a94d387b3a5a2dfc85cf06fca93f7e8`。

主 summary 原实现是在结束时读取 worktree revision；部分较晚结束的 shard 因 recovery
期间推进 checkout 而记录了较新的 commit，虽然其进程实际从 `981eaab` 启动。最终 runner
已改为启动前冻结 Git revision 与 config/manifest hash，后续 run 不再有这一 provenance
歧义。
