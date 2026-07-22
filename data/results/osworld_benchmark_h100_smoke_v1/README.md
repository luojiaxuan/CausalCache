# OSWorld benchmark H100 smoke v1

状态：`VALID_OSWORLD_12_ENV_2_H100_INFRASTRUCTURE_SMOKE`。

在 H100 host 上使用两个独立 GPU-bound HTTP policy replicas 与 12 个并发 KVM environments，完成 pinned
OSWorld official no-GDrive roster 的前 12 个 tasks：12/12 completed、0 runner failure、24 policy requests；
replica 0/1 各处理 12 requests，12 workers 均 exit 0。重复运行返回 12/12 resumed skips。

episode makespan=`28.509s`，逐 task time sum=`75.164s`。这是 `WAIT -> DONE`、2-step infrastructure smoke；
不能外推 361-task 正式评测时间或解释 score。raw evidence 位于 H100
`/data/jaxan/osworld-benchmark/smoke12-d2e0841`，不是 reusable dataset/paper artifact，因此不上传 HF。

完整轻量 provenance 见 [`summary.json`](summary.json)，设计与 no-GDrive 排除清单见
[`../../../docs/osworld_benchmark_acceleration_v1.md`](../../../docs/osworld_benchmark_acceleration_v1.md)。
