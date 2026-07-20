# Set Utility held-out v1：sealed selections

状态：`SEALED_BEFORE_EVALUATION_LABEL_ACCESS`。

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

## Label-blind latency

这里的 warm latency 不包含 event arrival 时可缓存的 event-source encoding：

| Model | selector p50 | selector p95 | search p95 |
|---|---:|---:|---:|
| DeepSets | 7.08 ms | 44.10 ms | 7.15 ms |
| Set Transformer | 15.33 ms | 210.94 ms | 209.21 ms |

这不是 model winner 或 GO 结论。下一步必须先从 sealed artifact 生成 sparse/exact restoration truth，按冻结合同比较
真实 utility、exact-oracle regret 和 long-history recovery；只有通过后才执行 downstream closed-loop。

## Truth schedule

[`schedule-summary.json`](schedule-summary.json) 已冻结 256/256 shards、805 states、55,175 个 unique coalitions；
其中 full anchor 不需要额外 forward，需执行的 coalitions 为 54,370。Schedule 的 method inventory 固定为
DeepSets、Set Transformer、recent、OCR/RGB 与 random，不能在读取 truth 后追加 selector。Label runtime 见
[`causalcache_set_utility_heldout_labels_execution_v1.json`](../../../code/configs/causalcache_set_utility_heldout_labels_execution_v1.json)。
最终 preflight 释放了两台 host 各 6 张 H200；正式 rollout 使用 12×H200、每卡 2 个 processes、24 个
disjoint partitions，映射见
[`causalcache_set_utility_heldout_labels_workers_24_v2.json`](../../../code/configs/causalcache_set_utility_heldout_labels_workers_24_v2.json)。
先前 22-worker mapping 未启动，仅作为调度记录保留。不使用每卡 3 processes，因为既有 long-history run 已证明
该配置会在约 140GB 峰值 OOM。
