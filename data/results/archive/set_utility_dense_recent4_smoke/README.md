# Dense recent-4 smoke rollout

状态：`DEPRECATED_SMOKE_ONLY_RECENT4`。

该 rollout 在每个 state 只暴露最近四个 non-current events，因此改变了完整历史 memory selection 的任务定义。
它不能用于正式 predictor 训练、held-out evaluation、oracle gap、matched-NLL、closed-loop 或方法 GO/NO-GO。

2026-07-19T23:28:18Z 主动停止时：

| Host | Persistent output | Completed | Skipped | Processed |
|---|---|---:|---:|---:|
| Hyper00 | `/data02/jaxan/runs/causalcache-set-utility-dense-v1-0d32187` | 2,047 | 41 | 2,088 |
| Hyper01 | `/data02/jaxan/runs/causalcache-set-utility-dense-v1-9671e45-partition-01` | 1,917 | 28 | 1,945 |
| Total | — | 3,964 | 69 | 4,033 |

两个 Docker container `sglang-omni-jaxan-07191359` 均已停止。state 文件采用 temporary file、`fsync`、atomic
replace，保留为 generator 与 throughput smoke evidence。本地目录不是正式数据 source of truth，也不计划上传
Hugging Face。
