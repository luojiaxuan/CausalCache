# Small-history exact B4 diagnostic v2

状态：`COMPLETED_SET_UTILITY_B4_ORACLE_EVALUATION`。

在 exact track 全部 `5≤n_t≤8` states 上，同一 reference session 完整生成 `|S|≤4` restoration truth：

- 103/103 completed states、73 trajectories、0 skip；
- 8,525 policy forwards、8,628 distance rows；
- Hyper00/H200 `0--5` 与 Hyper01/H200 `2--7`，24 workers，两个 supervisors exit 0；
- trajectory-equal normalized recovery 如下。

| 方法 | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| Exact subset oracle | 0.5304 | 0.6999 | 0.7823 | 0.8248 |
| True conditional greedy | 0.5304 | 0.6686 | 0.7329 | 0.7677 |
| Set Transformer v1 | 0.1997 | 0.4386 | 0.5947 | 0.7128 |
| DeepSets v1 | 0.1427 | 0.4485 | 0.5854 | 0.7064 |
| OCR/RGB | 0.1499 | 0.4340 | 0.5635 | 0.6942 |
| Recent | 0.1486 | 0.4237 | 0.5823 | 0.6718 |

结论：B4 exact ceiling 达到 0.8248，B2→B4 gain 为 0.1249；true greedy 的 B4 search gap 为 0.0571，
而 Set Transformer 的 B4 distillation gap 为 0.1121。主瓶颈是 representation/distillation，不是 oracle
signal，也不主要是 greedy search。Set Transformer 在 B4 点估计领先，但本 post-hoc 小历史诊断不改变
formal v1 `NO_GO`。

完整 artifact：Hyper00
`/data02/jaxan/runs/causalcache-set-utility-b4-oracle-labels-v2-aggregate-59e6a73`；raw roots 在 Hyper00/01
同名 `...b4-oracle-labels-v2-59e6a73`。result content SHA256=`195bcbb...0473b`，file
SHA256=`b14896f...e1b6`。完整 artifact 已发布到 private HF dataset revision
[`9b53ec82...c599c0`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/9b53ec82c12fefaba571233e5e0d78d0f6c599c0/artifacts/set-utility-b4-oracle-v2-195bcbb)，
tag=`set-utility-b4-oracle-v2-195bcbb`；remote readback SHA 与本地一致。
