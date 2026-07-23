# Restoration-v2 exact selection 与 exposure

正式 CPU materialization 的 canonical verdict 是
`PASSED_PREOUTPUT_SELECTION_VALIDATION`。Hyper00 从 pushed
`main@30879c09e896a61929c66935f98c21e6c3fc7ff5` 重验 16 个 pinned Parquet、重建 212 rows /
111 eligible trajectories，并复现 eligible-pool SHA256
`84d6855b20227d7f884d4ce26cfae7188c1075bb7751cab78a3cf325eef192cb`。

固定选择得到：

- label-train 30 states、development 15 states、confirm 20 states；
- confirm 是排除 exact 8+15 后，`decision_count>=5` structural order 的首 20 条；
- 首 20 条覆盖 29 个 normalized app labels，未 top-up；
- reference/oracle/confirm union 为 43 条且两两不交；
- 每个 screening/confirm state 都冻结 current image、candidate post-state、current-equivalence 与 action
  SHA256 witness。

Canonical files：

- [`../../manifests/restoration_v2_selection.json`](../../../manifests/restoration_v2_selection.json)，SHA256
  `292c7e52f76d158863b0ee76b15e76e8531f9d7291b3fd68b0d8c3fe4f05ca7b`；
- [`../../manifests/restoration_v2_exposure.json`](../../../manifests/restoration_v2_exposure.json)，SHA256
  `bc1224826e0642c2374f6cfbebd676389d8c17ce6bf8a789f08903740cf97a95`；
- [`summary.json`](summary.json)：host/runtime、完整 argv/时间与 input hashes、exact IDs、deterministic
  rebuild 和失败/superseded attempt 索引；SHA256
  `b3c5c580273d7187439ed8cc28c1382cdc006cd12fd99c5971186249629e74f1`。

两个全量构建在不同空 output directory 下 byte-identical，并各自通过独立 validator。原
`ed0706e` passing record 因未保存完整 argv/起止时间/dtype 被透明 supersede，exact IDs 不变。attempt 1 暴露并
修复了 dict-order serialization bug，保留为失败记录且未覆盖；它不是 scientific failure。所有步骤均未
加载 policy、未使用 GPU、未产生 GUI-Owl v2 或 restoration output。

验证：

```bash
make validate-restoration-v2-selection
```
