# Set Utility Freeze-B v1

本目录记录 Freeze-B/A 的 policy-blind 完成态。它从已提交的 P0 census 中固定 1,200 条新 trajectory：
`train/tune/evaluation=1000/100/100`，三个 candidate-capacity stratum 各 400 条；每条 trajectory 固定
一个 stratum anchor 和一个 terminal query，共 2,400 states。

Canonical manifest 是
[`../../manifests/set_utility_freeze_b_v1.json`](../../manifests/set_utility_freeze_b_v1.json)，SHA256=
`144b0de1e66eff1624f6bd10fa6dbebd3d215c6d3d9965d9e9cd3dae295373e5`。assignment inventory SHA256=
`d9d58f40b2b40f537eb719b3071369f5cffd1b3a181fe645ffdb5f6260f2a660`，query-plan inventory SHA256=
`9e96681dc3ce5befc838545dbae249d219fbd8e27343e622072d08b655f5fa52`。

本阶段只读取 P0 manifest，没有读取 raw shard、截图或 instruction 原文，也没有加载 processor/model、生成
restoration label、训练 predictor 或打开 one-shot evaluation。初始 `n<=16` 的最坏上限是 328,800 raw
label rows / 338,400 total model operations；它不是最终精确预算。下一步必须在独立 execution contract 下读取
冻结 row，做 processor-only candidate freeze，才能产生最终 candidate ids 与 exact operation budget。
