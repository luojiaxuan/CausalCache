# Set Utility held-out v1：sealed selections

状态：`FORMAL_NO_GO`（selections 仍保持 `SEALED_BEFORE_EVALUATION_LABEL_ACCESS`）。

本目录冻结了 805 个 trajectory-disjoint、variable-`n_t` evaluation states 上的 DeepSets、Set Transformer、
recent、raw OCR/RGB 与 deterministic random 的 at-most-`B` selections。生成 selections 时未读取任何
evaluation restoration distance；两个 feature partitions 的 `label_file_read_count` 都为 0。

## Artifact

- Git artifact：[`selections.json`](selections.json)；
- record count：805；exact track 320、large-history track 720、overlap 235；
- content SHA256：`98573107b9309f4c3c49dd7e91405f8bfdcdbbccce56a47bc77678cc412e658f`；
- file SHA256：`0c2223dc6617d06f9c56701f237e128f1e3cb63d564a14d50ce795cce82be60d`；
- held-out config SHA256：`8374083cb38825f852fdbf50e862c50732579ade4f3d650269e6997cb56de308`；
- state inventory SHA256：`7ec5bfcf6d9ba347c793264e7a836e4a29e01b3e4d75e3c0e71dc376cb49c68a`。

Persistent working artifacts 位于 Hyper00：

- full label-blind input：`/data02/jaxan/artifacts/causalcache-set-utility-heldout-features-full-480e26d`；
- selections：`/data02/jaxan/runs/causalcache-set-utility-heldout-selections-480e26d`。
- exact/sparse schedules：`/data02/jaxan/runs/causalcache-set-utility-heldout-eval-schedules-1b5896d`。

它们当前状态为 `PENDING_HF_UPLOAD`；可复用 checkpoint 已在 README 的 immutable HF model revision 中记录。

## Truth 与正式结果

- canonical truth：Hyper00 `/data02/jaxan/runs/causalcache-set-utility-heldout-labels-v1`；
- terminal coverage：805/805，801 completed + 4 `GUIOwlV21GenerationParseError` skipped，missing 0；
- state-id SHA256：`11f2987886bcdd1ea13240558a87523e843959574264c49de14bfae2e7069ca0`；
- full result：Hyper00 `/data02/jaxan/runs/causalcache-set-utility-heldout-evaluation-v1-7113e09/result.json`；
- result content SHA256：`d89263ce6e76b127a6741e3d2e9b005c7c813e9a49b75163c056ba799158ef49`；
- result file SHA256：`4fb4a55f0280b96ceaed711b675102639e542aa507b93f592289262116294e01`；
- lightweight summary：[`evaluation-summary.json`](evaluation-summary.json)。

Truth 与 full result 当前均为 `PENDING_HF_UPLOAD`。冻结 reducer 返回
`INCOMPLETE_SET_UTILITY_HELDOUT_EVALUATION / NO_GO`，没有 deployment winner。

801 个 completed states 的诊断如下；它们不能越过正式 coverage gate：

| Method | Primary recovery | Exact regret B1 / B2 | Long+very-long | Cached-source p95 |
|---|---:|---:|---:|---:|
| DeepSets | 0.3997 | 0.0378 / 0.0320 | 0.2983 | 44.10 ms |
| Set Transformer | **0.4093** | **0.0343** / 0.0281 | **0.3912** | 210.94 ms |
| recent | 0.4040 | 0.0389 / 0.0264 | 0.3693 | -- |
| OCR/RGB | 0.3826 | 0.0388 / **0.0230** | 0.3147 | -- |

Set Transformer 相对 OCR/RGB 为 +0.0267，trajectory-clustered 95% CI `[0.0034, 0.0525]`；相对 recent
仅 +0.0054，CI `[-0.0248, 0.0318]`。因此它有弱正信号，但即使忽略 parser skips，仍因 recent comparison
与 B2 regret 失败而不能进入 policy replay。

## Label-blind latency

这里的 warm latency 不包含 event arrival 时可缓存的 event-source encoding：

| Model | selector p50 | selector p95 | search p95 |
|---|---:|---:|---:|
| DeepSets | 7.08 ms | 44.10 ms | 7.15 ms |
| Set Transformer | 15.33 ms | 210.94 ms | 209.21 ms |

该 latency 是密封 selections 时的 label-blind measurement。正式 truth 已表明没有 model winner；action-policy
forward p95 尚未测量，因此这里也不作相对 10% deployment gate 结论。

## Truth schedule

[`schedule-summary.json`](schedule-summary.json) 已冻结 256/256 shards、805 states、55,175 个 unique coalitions；
其中 full anchor 不需要额外 forward，需执行的 coalitions 为 54,370。Schedule 的 method inventory 固定为
DeepSets、Set Transformer、recent、OCR/RGB 与 random，不能在读取 truth 后追加 selector。Label runtime 见
[`causalcache_set_utility_heldout_labels_execution_v1.json`](../../../code/configs/causalcache_set_utility_heldout_labels_execution_v1.json)。
最终 preflight 释放了两台 host 各 6 张 H200；正式 rollout 使用 12×H200、每卡 2 个 processes。初始 24 个
静态 partitions 在 276 个 completed states 处因负载不均改为 48 个 deterministic state lanes；所有 containers
最终 exit 0，48/48 lane receipts 完成。映射见
[`causalcache_set_utility_heldout_labels_workers_24_v2.json`](../../../code/configs/causalcache_set_utility_heldout_labels_workers_24_v2.json)。
先前 22-worker mapping 未启动，仅作为调度记录保留。不使用每卡 3 processes，因为既有 long-history run 已证明
该配置会在约 140GB 峰值 OOM。
