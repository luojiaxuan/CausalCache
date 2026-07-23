# Set capacity + DeepSets prefetch v1

状态：`L64_S4_TRAINING_RUNNING / EPOCH1_TRUTH_NO_CAPACITY_GAIN / PREFETCH_BENCHMARK_COMPLETE`。本实验不访问
untouched evaluation。

## Set Transformer capacity run

- config：[`causalcache_set_utility_decision_distillation_v2_long_oracle_set_capacity_v1.json`](../../../../code/configs/causalcache_set_utility_decision_distillation_v2_long_oracle_set_capacity_v1.json)；
- source：`main@8a6b8a0`；model=`d256 / latent_count=64 / resampler_layers=2 / set_layers=4`；
- execution：Hyper00 GPU 4/5，2-rank DDP，per-device batch=1，gradient accumulation=4，global batch=8，
  evaluation batch=1；原 d256/l16/s2 run 已在 GPU 0--3 完成；
- persistent root：
  `/data02/jaxan/runs/causalcache-set-transformer-decision-v2-long-oracle-l64-s4-ddp2-v1-8a6b8a0`；
- 启动后已进入训练 loop；观察到 GPU utilization `82%/95%`，随后较长 state 峰值显存约
  `142,719/125,575 MiB`。checkpoint 完成后状态为 `PENDING_HF_UPLOAD`。

该 run 只改变信息瓶颈和 set interaction depth，不改 hidden size、训练 labels、loss、LR、seed、最大 epochs、
early stopping 或 global batch。若 Long+ 仍不达标，再决定是否比较 latent 32/64，而不是先扩大 hidden size。

2026-07-21 已对 epoch-1 immutable snapshot 运行完整 fixed-tune selector truth。L64/S4 的 B1--B4=
`0.24547/0.39986/0.54143/0.63626`，macro=`0.45575`、Long+=`0.35577`、total p95=`114.19ms`；基础
Set Transformer 为 `0.26293/0.42930/0.56152/0.64999`，macro=`0.47593`、Long+=`0.38853`、p95=
`71.39ms`。因此 epoch 1 没有形成 utility--latency capacity gain。

L64/S4 tune total=`2.77187` 不能与 base=`4.02887` 直接比较：两者 `evaluation_batch_size` 分别为 1 和 8，
当前 conditional-listwise complete-group 计数依赖 eval batch。该指标只用于同一 run 内 early stopping；跨模型
选择以真实 selector truth 为准。容量 run 继续自然训练到 early stop，但不因该较低 loss 预判最终 winner。
真实结果见
[`set_utility_decision_distillation_v2_epoch1_tune`](../set_utility_decision_distillation_v2_epoch1_tune/README.md)。

## DeepSets infra diagnosis

正式 4-card DeepSets 的首两个 epochs 分别约 `8.85/8.39` 分钟；短时 GPU utilization 多在 `10%--39%`。
同时整机约 `96%` CPU idle、块设备 I/O 接近 0，说明不是 CPU 总量或磁盘带宽耗尽，而是同步 Python
padding/mmap batch preparation、H2D copy 与小模型 kernel/collective 之间存在 bubbles。

`main@8a6b8a0` 增加单-batch CPU worker prefetch、pinned memory、non-blocking H2D；gradient accumulation
仅在 optimizer step 做 DDP gradient synchronization。单元测试为 20 passed + 6 subtests。

在 Hyper01 GPU 6/7 上，用同一 committed config、同一 256-state overfit slice、同一 2-rank geometry 做
一次非科学 throughput A/B：

| Infra | Source | Elapsed | Persistent root |
|---|---|---:|---|
| synchronous lazy cache | `1aafe4c` | 30.373s | `/data02/jaxan/runs/causalcache-deepsets-prefetch-benchmark-sync-v1-1aafe4c` |
| async prefetch + pinned H2D | `7e6d365` | 26.965s | `/data02/jaxan/runs/causalcache-deepsets-prefetch-benchmark-async-v1-7e6d365` |

elapsed 降低 `11.22%`，等价吞吐提高 `12.64%`。该 A/B 只证明 infra 改善，不比较模型质量。正式 DeepSets
随后自然 early-stop，并未为 infra 优化重启；prefetch 作为后续训练默认执行路径。
