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
- 每点使用 pinned official 361-task no-GDrive roster 的 24 个 evenly-spaced tasks；
- capacity 主 sweep 使用 `max_steps=1`、`pause_seconds=0.1`，避免把 agent action quality 和 history 长度混进
  并发上限；另做 recent-B4 多步功能 smoke；
- 同时记录 wall time、tasks/hour、policy requests/s、client p50/p95、server queue p50/p95、generation
  p50/p95、failure、CPU/KVM 与 GPU utilization；
- 最佳并发点必须满足 0 runner failure，且相对更低并发有实质吞吐收益。若吞吐进入平台或 p95 queue
  急剧上升，较高并发不作为默认值。

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
