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

实际矩阵避免全笛卡尔积：GPU 轴固定 `24 env / 46 Chrome tasks` 测 `1/2/4/6 replicas`；environment 轴固定
`6 replicas / 46 Chrome tasks` 测 `6/12/18/24/30 envs`。每点由独立 output root 保存 episode completion、
`nvidia-smi`、`vmstat` 与 runner log；任一点失败即停止，不用后续点掩盖。

容量配置在 `code/configs/causalcache_osworld_capacity_h100_v1.json`。它与正式 benchmark config 分离，
任何 `max_steps=1` 或 `pause_seconds=0.1` 数字都不得进入 paper success table。

## 实现状态

- `code/causalcache/osworld_gui_owl.py`：mixed-fidelity prompt、normalized desktop action parser、冻结
  Transformers runtime；
- `code/scripts/serve_osworld_gui_owl_policy.py`：每张可见 GPU 一个 HTTP replica，返回生成与排队时间；
- `code/scripts/run_osworld_multienv.py`：支持 runtime endpoint count override、evenly-spaced roster 和容量
  aggregate；
- focused unit tests 已覆盖 recent 恢复截图绑定、坐标转换、键盘/滚动/终止 action；
- H100 raw 根目录预定为 `/data/jaxan/osworld-capacity/`，当前为
  `LOCAL_INFRASTRUCTURE_ARTIFACT`，不上传 Hugging Face。

实测完成后，本文件追加数字表、瓶颈归因和推荐 topology。

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
