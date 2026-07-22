# OSWorld + frozen GUI-Owl capacity v1

## 目的

本轮只调查单台 H100 server 上真实 policy inference 与 KVM environment 的并发上限，不把短程容量 run
解释成 OSWorld task success。Policy 固定为 `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5fae`，权重 BF16、
`requires_grad=False`、greedy decoding；memory selector 固定为 `recent`、at-most-`B=4`。

## 系统拓扑

每张 H100 加载一个独立 GUI-Owl replica。OSWorld worker 复用一个 KVM VM，通过动态 task queue 领取任务，
并以 `worker_id mod replica_count` 路由到 policy。每个请求包含全部历史的结构化摘要、recent-4 的恢复截图和
当前截图。GPU replica 内串行生成，不声称 continuous batching；环境并发用于覆盖 VM reset、action settle、
evaluator 与 policy 等待时间。

## 容量协议

- H100 授权上限：6 GPUs；候选 GPU 由 launch 前 fleet preflight 决定；
- GPU 点：`1, 2, 4, 6` replicas；
- environment 点：逐级增加到吞吐不再提高、排队延迟明显恶化，或到 30 VMs（4 vCPU/VM，接近本机
  128 logical CPUs 的边界）；
- 每点使用 pinned official 361-task no-GDrive roster 中完整的 46-task Chrome domain；这给出单机
  browser-workload capacity，不冒充跨域 full-roster wall time；
- capacity 主 sweep 使用 `max_steps=1`、`pause_seconds=0.1`，避免把 agent action quality 和 history 长度混进
  并发上限，并显式跳过 task evaluator；另做 recent-B4 多步功能 smoke。正式 benchmark 始终开启 evaluator；
- 同时记录 wall time、tasks/hour、policy requests/s、client p50/p95、server queue p50/p95、generation
  p50/p95、failure、CPU/KVM 与 GPU utilization；
- 最佳并发点必须满足 0 runner failure，且相对更低并发有实质吞吐收益。若吞吐进入平台或 p95 queue
  急剧上升，较高并发不作为默认值。

实际矩阵避免全笛卡尔积：先在 `6 replicas / 46 Chrome tasks` 下测 `6/12/18/24/30 envs`，找最后一个
零失败 environment 点；再固定保守的 `12 env / 46 tasks` 测 `1/2/4/6 replicas`（6-GPU/12-env 点复用环境轴
结果）。每点由独立 output root 保存 episode completion、`nvidia-smi`、`vmstat` 与 runner log。高并发点
invalid 会记录后继续，不用一个不可靠点阻止较低并发和 GPU scaling。

容量配置在 `code/configs/causalcache_osworld_capacity_h100_v1.json`。它与正式 benchmark config 分离，
任何 `max_steps=1` 或 `pause_seconds=0.1` 数字都不得进入 paper success table。

## 正式结果

8/8 个点全部完成 46/46 episodes、0 runner failure。6 replicas 的 environment scaling 为：

| Envs | Tasks/hour | Client p95 | Queue p95 | CPU idle | Mean GPU util/GPU |
|---:|---:|---:|---:|---:|---:|
| 6 | 863.79 | 1.96s | 0.00s | 84.27% | 1.73% |
| 12 | 1359.04 | 3.06s | 1.33s | 78.88% | 2.65% |
| 18 | 1565.58 | 3.24s | 1.36s | 73.80% | 3.10% |
| **24** | **1835.24** | 4.27s | 2.30s | 68.73% | 3.60% |
| 30 | 1803.63 | 3.86s | 1.60s | 68.30% | 3.75% |

因此 `30 envs` 是本轮零失败测试上界，`24 envs` 是吞吐 knee；后者不是 hard reliability cap。

固定 12 envs 的 GPU scaling 为：

| Replicas | Tasks/hour | Client p95 | Queue p95 | Generation p95 |
|---:|---:|---:|---:|---:|
| 1 | 1395.08 | 12.18s | 10.31s | 1.83s |
| 2 | 1360.42 | 6.94s | 5.09s | 2.02s |
| 4 | 1398.74 | 3.39s | 1.52s | 2.02s |
| 6 | 1359.04 | 3.06s | 1.33s | 1.98s |

GPU 数量没有改变 one-step task throughput，却把 policy p95 大幅压低。原因是 Chrome task reset/setup 占据
wall time，而每个 task 只有一次 policy request。该 workload 下单 GPU nonzero utilization fraction 已达
69.17%；六个 replica 分摊相同请求后，每卡平均利用率只有 2.65%。这是 duty-cycle 证据，不是 inference
没有使用 GPU。

recent at-most-`B=4` 另做真实多图请求：4 张恢复历史截图 + 当前截图成功生成 action，`image_count=5`、
`prompt_tokens=2866`、generation=`2.012s`、peak allocated HBM=`17.22 GiB`。因此容量主 sweep 虽然只执行
一步，mixed-fidelity B4 输入路径本身已经通过。

## 推荐 topology

- 正式多步 closed-loop 默认从 `6 GPUs + 12 envs` 开始：它保留最低 p95 与更充足的 steady-state policy
  余量；
- 只追求 VM reset/setup throughput 时使用 `6 GPUs + 24 envs`；
- 不默认使用 30 envs，因为吞吐相对 24 envs 已下降 1.72%；
- full closed-loop 不能直接沿用 `1835 tasks/hour`。真实 episode 有多步、action settle、task evaluator 和跨域
  setup，必须在正式 roster 上重新测 steady-state queue 与 wall time。

## 实现与 Source of Truth

- `code/causalcache/osworld_gui_owl.py`：mixed-fidelity prompt、normalized desktop action parser、冻结
  Transformers runtime；
- `code/scripts/serve_osworld_gui_owl_policy.py`：每张可见 GPU 一个 HTTP replica，返回生成与排队时间；
- `code/scripts/run_osworld_multienv.py`：支持 runtime endpoint count override、evenly-spaced roster 和容量
  aggregate；
- focused unit tests 已覆盖 recent 恢复截图绑定、坐标转换、键盘/滚动/终止 action；
- H100 raw root：`/data/jaxan/osworld-capacity/sweep-780ef47/`；B4 smoke：
  `/data/jaxan/osworld-capacity/recent-b4-smoke-780ef47.json`；
- 轻量 reduction 与完整 provenance：
  [`data/results/osworld_gui_owl_capacity_v1/`](../data/results/osworld_gui_owl_capacity_v1/)；
- source revision=`780ef4729c79c92102d16ab06ef11332a4af5680`，OSWorld=
  `b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`，model revision=
  `06d5faecff74840bab2be2425e9c42667a5d04fc`；
- raw traces 标记为 `LOCAL_INFRASTRUCTURE_ARTIFACT`，不上传 Hugging Face。Git 保存 reducer、轻量结果和
  provenance。

## Invalid attempt

`sweep-38c97c6` 的首个 `1 GPU / 24 env / 48 tasks` point 在 28/48 episodes 后停止，分类为
`INVALID_CAPACITY_EVALUATOR_CONFOUND`。原因是短程 `max_steps=1` 后仍执行完整文件型 evaluator；未完成任务本来
不存在目标文件，20 个 workers 花时间等待 file fetch/metric timeout。该点没有 runner/model crash，但其 wall
time 混入了与 policy/KVM 并发无关的错误态 evaluator I/O，不进入容量曲线。修订后只在 capacity config 关闭
evaluator，正式 success config 和既有结果均不追溯修改。

`sweep-bfab165` 在关闭 evaluator 后，首点 33/48 很快完成，但其余 evenly-spaced 跨域 tasks 长时间停在
reset/setup；此时 policy request 不再增长、GPU idle、host CPU 约 96% idle。该点分类为
`INVALID_CAPACITY_TASK_SETUP_CONFOUND`。容量 workload 因而固定为官方 no-GDrive roster 的全部 46 个 Chrome
tasks；跨域下载/setup 长尾属于 full benchmark wall-time 问题，不能用于判定 GPU/KVM capacity knee。

`sweep-e90fdb5` 又揭示两个确定性 infrastructure bug：24-env recreation 时，官方 provider 的全局安全端口
分配 lock 仍保留，但 10 秒硬编码 timeout 导致 3 个 worker 失败；另一个 state 的模型输出合法
`computer_use(action=click)`，而 adapter 只接受 `left_click`。前者改为 config-bound 180 秒（不移除 lock），
后者按 OSWorld 官方 GUI-Owl parser 的既有做法兼容 `click/drag` aliases。该 attempt 不进入曲线。

`sweep-1596be2` 验证 lock timeout 与 parser alias 修复均生效，但 `24 env` 在 41/46 后出现一次 VM container
reset `Docker NotFound`。这被视为“24 env 可能超过可靠性 knee”的正当容量观测，而不是继续修到该点强行
通过；最终矩阵因此按 6→30 env 递增，并把 GPU scaling 固定在 12 env。

`sweep-28b2817` 从 6 env 复现相同 `Docker NotFound`，且三个 failure 都在同一 worker 的多次 reset，否定了
“只是 24 env 超过硬件 knee”。根因是 official `_get_used_ports()` 先 list containers、再逐个 inspect；其他
worker 同时删除旧 VM 时产生 list→inspect race。adapter 改为 Docker sparse `/containers/json` snapshot，直接
从响应提取 public ports，不逐容器 inspect。安全的全局 port-allocation lock 仍保留。

`sweep-7e2277b` 的首个 6-env point 已完成 46/46、0 failure，却在汇总前不退出。原因是 runner 先 join
workers、后读取 result queue；46 条结果填满 multiprocessing pipe 后，workers 阻塞在 `put()`。runner 改为
运行中持续 drain，收到每个 task terminal message 后再 join；episode bytes 保留，但该 attempt 无合法 wall
aggregate，不进入容量曲线。
