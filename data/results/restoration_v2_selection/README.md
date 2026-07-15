# Restoration-v2 exact selection 与 exposure

正式 CPU materialization 的 canonical verdict 是
`PASSED_PREOUTPUT_SELECTION_VALIDATION`。Hyper00 从 pushed
`main@ed0706ecb72d9a828f452f7308859f95f9554c55` 重验 16 个 pinned Parquet、重建 212 rows /
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

- [`../../manifests/restoration_v2_selection.json`](../../manifests/restoration_v2_selection.json)，SHA256
  `13197eedc413717a3f190aa53453c6f34b1db82c57f0b47945d552f38bec74f3`；
- [`../../manifests/restoration_v2_exposure.json`](../../manifests/restoration_v2_exposure.json)，SHA256
  `0b1a4dfdb9f23be1a7456f78801f26e0b0800b7f32f011af913257b6919535ae`；
- [`summary.json`](summary.json)：host/runtime、exact IDs、deterministic rebuild 与失败 attempt 索引。

两个全量构建在不同空 output directory 下 byte-identical，并各自通过独立 validator。attempt 1 暴露并
修复了 dict-order serialization bug，保留为失败记录且未覆盖；它不是 scientific failure。所有步骤均未
加载 policy、未使用 GPU、未产生 GUI-Owl v2 或 restoration output。

验证：

```bash
make validate-restoration-v2-selection
```
