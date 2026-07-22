# OSWorld benchmark acceleration v1

## 结论

OSWorld 官方所说的“一小时内”主要来自 AWS Host–Client environment parallelization，而不是单纯增加 GPU。
OSWorld-Verified 报告称 AWS 可扩到约 50 个并发 environment；单台 Docker server 的官方经验是同时运行 8 或
16 个 environment。CausalCache 当前 H100 profile 采用：

```text
official 361-task no-GDrive roster
              ↓ dynamic task queue
      12 persistent KVM workers
        ↙                    ↘
H100 policy replica 0   H100 policy replica 1
              ↓
task-level result.json / resume
```

两张 GPU 只运行两个常驻 policy replicas，12 个 VM 共享它们。继续增加 GPU 只有在 policy inference 已成为
瓶颈时才有效；VM reset、应用初始化、action settle 和 evaluator 主要消耗 CPU/KVM、I/O 与 wall time。

官方 source of truth：

- [OSWorld README / AWS 与 multi-env](https://github.com/xlang-ai/OSWorld)；
- [OSWorld-Verified infrastructure report](https://xlang.ai/blog/osworld-verified)；
- [官方 setup guideline](https://github.com/xlang-ai/OSWorld/blob/main/SETUP_GUIDELINE.md)；
- [官方 no-GDrive roster](https://github.com/xlang-ai/OSWorld/blob/main/evaluation_examples/test_nogdrive.json)；
- [官方 multi-environment runner](https://github.com/xlang-ai/OSWorld/blob/main/scripts/python/run_multienv.py)。

本项目固定 OSWorld revision `b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`。

## 哪些 Google Drive tasks 可以忽略

官方不是允许忽略所有 Google 页面任务，而是精确允许从 369 tasks 中排除下面 8 个 `multi_apps` tasks，得到
361-task `test_nogdrive.json` denominator：

| Task ID | 任务摘要 |
| --- | --- |
| `46407397-a7d5-4c6b-92c6-dbe038b1457b` | 从 Thunderbird 邮件附件提取图片并上传 Drive `figures/` |
| `4e9f0faf-2ecc-4ae8-a804-28c9a75d1ddc` | 从 Drive 下载 invoice，抽表格到本地 XLSX |
| `78aed49a-a710-4321-a793-b611a7c5b56b` | 保存 Thunderbird 邮件附件到 Drive 并移动邮件 |
| `897e3b53-5d4d-444b-85cb-2cdc8a97d903` | 转换 DOCX 为 PDF 并上传 Drive `forms/` |
| `a0b9dc9c-fc07-4a88-8c5d-5e3ecad91bcb` | 备份 Thunderbird 邮件为 EML 到 Drive |
| `b52b40a5-ad70-4c53-b5b0-5650a8387052` | 合并邮件中的 PDF 并上传 Drive |
| `0c825995-5b70-4526-b663-113f4c999dd2` | 从 PDF 摘取内容并创建 Drive Google Doc |
| `22a4636f-8179-4357-8e87-d1743ece1f81` | 转换 DOCX 为 PDF 并上传 Drive `meetings/` |

这 8 个任务需要 Google account、OAuth/Drive API、远端文件清理或 evaluator 访问。没有正确凭据时应该用官方
361-task roster并明确报告 denominator；不能运行 369 后把这 8 个失败记成模型失败，也不能再自行删除其他
Google Search、Chrome 或网络任务。

## 已实现的加速

1. `run_osworld_multienv.py` 使用动态任务队列；快 worker 会继续领取 task，避免固定 shard 的长尾。
2. 每个 worker 只启动一次 VM并跨 tasks 复用；task 内仍调用 clean `reset(task_config)`。
3. 两个 policy replicas 常驻 GPU，多个 environments 共享，避免每 task 重载模型。
4. 官方 qcow2、Hugging Face file cache、Python environment 和 Docker images 均在 H100 persistent `/data`。
5. 每个 task 的 `result.json` 是完成标记；中断后只重跑未完成 tasks。
6. runner 只保存必要 screenshots/checkpoints，不默认录制 MP4。
7. H100 固定 `CPU_MODEL=qemu64`，避免默认 host model 的 300 秒 early-boot timeout。
8. roster revision、counts、Google Drive差集、policy replica count 和 worker routing 全部 fail closed。

## H100 实测

Git `d2e0841`，两个 H100 replicas、pinned OSWorld 与相同 VM image：

| Smoke | Envs | Tasks | Policy requests | Runner failures | Episode makespan | Sum task time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| routing | 2 | 4 | 8 | 0 | 46.870s | 78.758s |
| default topology | 12 | 12 | 24 | 0 | 28.509s | 75.164s |

12-env run 中两个 replicas 各处理 12 requests，12 个 workers 全部 exit 0；重复运行直接跳过 12/12 tasks。
该测试只用 `WAIT -> DONE`、2 steps，验证的是环境并发、GPU routing、evaluator、清理和 resume，不是 agent
performance 或完整 benchmark throughput。一个 task 在初始状态下 evaluator score 非零，也不得作为模型结果。

## 正式估时与扩容规则

真实总时长近似为：

\[
T_{wall}\approx
\frac{\sum_i T_i(\text{reset, policy, action settle, evaluator})}
{N_{effective\ environments}}
+T_{startup/long-tail}.
\]

下一步接入 GUI-Owl desktop policy 后，先在 trajectory-disjoint 的 24--36 个 no-GDrive tasks 上保持正式
`max_steps=50`、`pause_seconds=2.0`，测量：

- task wall time 与 step count 分布；
- policy p50/p95 latency、batch size、GPU utilization/queue wait；
- VM reset、action settle、evaluator latency；
- 4/8/12 env 的吞吐与失败率。

如果 8→12 env 仍近线性且 GPU queue 不增长，保留 12 或测 16；如果 policy p95/queue wait 明显上升，先做
continuous batching或增加 replica，再增加 VM。AWS 的一小时说法对应最高约 50 env，不应在只有 12 env、尚未
接真实 policy 前承诺同样时长。

## 运行入口

```bash
PYTHONPATH=code python3 -m scripts.run_osworld_multienv \
  --repository-root . \
  --osworld-root /data/jaxan/osworld-runner/OSWorld \
  --config code/configs/causalcache_osworld_benchmark_h100_v1.json \
  --preflight
```

正式运行由 config 提供 12 env、2 endpoints、361-task roster、50 steps 和 2 秒 action pause。任何 smoke override
不得复制到 paper result。

## Source of Truth

| 内容 | 位置 |
| --- | --- |
| fail-closed roster / endpoint helpers | `code/causalcache/osworld_benchmark.py` |
| concurrent runner | `code/scripts/run_osworld_multienv.py` |
| H100 config | `code/configs/causalcache_osworld_benchmark_h100_v1.json` |
| focused tests | `code/tests/test_osworld_benchmark.py` |
| lightweight result | `data/results/osworld_benchmark_h100_smoke_v1/` |

raw screenshots/results 保留在 H100 persistent storage，只是 infrastructure smoke，状态为
`LOCAL_INFRASTRUCTURE_SMOKE_NOT_A_PAPER_ARTIFACT`，无需上传 Hugging Face。
