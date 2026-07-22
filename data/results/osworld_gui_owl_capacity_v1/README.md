# OSWorld frozen GUI-Owl capacity v1

## 结论

- H100 host 上限测试到 `30` 个并发 KVM environments，8 个正式点均为 `46/46` episodes、0 runner
  failure；这不是 hard maximum，只是本轮测试上界。
- `6` 个 frozen GUI-Owl replicas 下，environment throughput 在 `24 envs` 达到峰值
  `1835.24 fresh tasks/hour`；`30 envs` 降至 `1803.63`，因此推荐的 environment throughput knee 是
  `24`。
- 固定 `12 envs` 时，`1/2/4/6` replicas 的 throughput 都在 `1359--1399 tasks/hour`，说明本轮
  one-step workload 主要受 VM reset/setup 限制，而不是 GPU compute 限制。
- GPU scaling 仍显著降低 policy latency：`1/2/4/6` replicas 的 client p95 分别为
  `12.18/6.94/3.39/3.06s`。正式多步 closed-loop 建议从 `6 GPUs + 12 envs` 启动；若目标是最大化
  task setup throughput，可使用 `6 GPUs + 24 envs`，但必须重新测多步 steady-state queue。
- recent at-most-`B=4` 的独立多图 smoke 真实输入 `4` 张恢复历史截图和 `1` 张当前截图，生成成功：
  `image_count=5`、`prompt_tokens=2866`、`generation=2.012s`、peak allocated HBM=`17.22 GiB`。

## Environment scaling（6 GPU replicas）

| Envs | Tasks/h | Client p95 (s) | Queue p95 (s) | Generation p95 (s) | CPU idle | Mean GPU util/GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 863.79 | 1.96 | 0.00 | 1.81 | 84.27% | 1.73% |
| 12 | 1359.04 | 3.06 | 1.33 | 1.98 | 78.88% | 2.65% |
| 18 | 1565.58 | 3.24 | 1.36 | 2.18 | 73.80% | 3.10% |
| **24** | **1835.24** | 4.27 | 2.30 | 2.51 | 68.73% | 3.60% |
| 30 | 1803.63 | 3.86 | 1.60 | 2.63 | 68.30% | 3.75% |

## GPU scaling（12 environments）

| Replicas | Tasks/h | Client p50/p95 (s) | Queue p50/p95 (s) | Generation p50/p95 (s) | Mean util/GPU | GPU nonzero fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1395.08 | 2.98 / 12.18 | 1.16 / 10.31 | 1.69 / 1.83 | 17.02% | 69.17% |
| 2 | 1360.42 | 3.09 / 6.94 | 1.27 / 5.09 | 1.72 / 2.02 | 8.04% | 32.93% |
| 4 | 1398.74 | 1.96 / 3.39 | 0.00 / 1.52 | 1.76 / 2.02 | 3.86% | 16.60% |
| 6 | 1359.04 | 1.91 / 3.06 | 0.00 / 1.33 | 1.73 / 1.98 | 2.65% | 11.79% |

## 口径与边界

- Workload 是 official no-GDrive roster 的完整 46-task Chrome domain；每个 episode 只执行一步，
  `pause_seconds=0.1`，并关闭 task evaluator。它只测 KVM reset、截图、一次真实 frozen GUI-Owl inference、
  action execution 和 cleanup 的容量，不是 task success，也不能外推为完整 361-task benchmark 的准确用时。
- 每张 GPU 是一个独立 BF16 GUI-Owl replica，replica 内串行生成，没有 continuous batching。权重
  `requires_grad=False`，selector 是 recent at-most-`B=4`。
- `successful_tasks=0` 是容量协议主动不执行 evaluator 的结果，不是 46 个任务全部失败。
- 低 GPU average utilization 是 workload duty cycle 的实测结果，不是模型没有运行。单 GPU 时 nonzero
  fraction 已达 `69.17%`；增加 replicas 后，同样 46 次请求被摊薄，但排队尾延迟显著下降。
- 30 envs 仍然零失败，因此 `24` 是 throughput knee，不是可靠性 hard cap。要声称 full closed-loop 的最优
  并发，还需用真实多步 episode 重新测 steady-state request rate。

## Source of Truth

- Git source revision：`780ef4729c79c92102d16ab06ef11332a4af5680`；
- OSWorld revision：`b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`；
- model：`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；
- host/GPU：`host-85-234-79-62`，`6 x NVIDIA H100 80GB HBM3`；
- Docker：`hongccc/sglang-omni@sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- runtime：PyTorch `2.11.0+cu130`、Transformers `5.6.0`、BF16、480 effective visual tokens/image；
- raw sweep：H100 persistent path `/data/jaxan/osworld-capacity/sweep-780ef47/`；
- recent-B4 raw smoke：`/data/jaxan/osworld-capacity/recent-b4-smoke-780ef47.json`；
- machine-readable reduction：[`summary.json`](summary.json)；
- raw status：`LOCAL_INFRASTRUCTURE_ARTIFACT`。这是 host-specific benchmark trace，不上传 Hugging Face；
  Git 保存 reducer、轻量结果和完整 provenance。

复现 reducer：

```bash
python3 code/scripts/summarize_osworld_capacity_sweep.py \
  --raw-root /data/jaxan/osworld-capacity/sweep-780ef47 \
  --source-raw-root /data/jaxan/osworld-capacity/sweep-780ef47 \
  --output data/results/osworld_gui_owl_capacity_v1/summary.json
```
