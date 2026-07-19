---
pretty_name: CausalCache Selected-Image Format Census v2 Column-Projection Repair
---

# CausalCache Selected-Image Format Census v2 Column-Projection Repair

这是 CausalCache set-utility 路线的 private selected-image format census artifact。它只包含 Freeze-B v2
选中 1,200 trajectories 的 18,792 个 pseudonymous image-format records、四个 worker receipts、run identity、
manifest 与 Git-safe result metadata；不包含 raw instruction/action/outcome、OCR、model output、restoration
labels、predictor checkpoint、matched-NLL 或 closed-loop traces。

## 身份与来源

- Git repository：`https://github.com/luojiaxuan/CausalCache`；
- formal producer commit：`1a03b7eb8ea2c41c8fed5213fed1bcc9d73f846b`；
- Git result commit：`873e28cd83a849ce8ca578e1540810d9c8b882e6`；
- frozen config SHA256：`eb823bd794c555265107e5edd4ff9b0be60b9c907b476a313ce5e156779a549f`；
- HF repository：`gavinlaw/causalcache-set-utility-new-development-mobile`；
- payload commit：`08e1b721760eb92ce8db9cb4d83ad4e01e957249`；
- tagged immutable revision：`c1d19eb96d7fa7926f1eb9db3328dbff4e88eae0`；
- frozen tag：`phase1-b2-image-format-census-v2-column-projection-repair`。

## Layout 与校验

artifact 位于：

```text
artifacts/selected-image-format-census-v2-column-projection-repair/
```

其中原始 formal root 恰有 10 files / 6,943,195 bytes：`manifest.json`、`run-identity.json`、四份
`records/worker-*.jsonl` 与四份 `receipts/worker-*.json`。同目录额外提供本说明与 `git-summary.json`，不计入
formal 10-file inventory。

- canonical 10-file inventory SHA256：`6c687bec18fd9bcb8f06dfd31c9a1685a6de976eed67b610de18fd3d7933bd4f`；
- manifest SHA256：`83fc82ab32e7c5fde09ce79483d0da76d896830a249157883b05881bbbdfd7c6`；
- record inventory SHA256：`49c0960fa3eda487d5363160526f0c7c99fb1ba425d422b5b0496ed00b4131fd`；
- selector inventory SHA256：`83bddc74f0ef81506334d742013eb38dfbab44982688ace8b359058701919553`；
- unique encoded-image SHA256 count：18,523。

## 正式结果

- status：`VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR`；
- denominator：1,200 trajectories / 527 selected shards / 18,792 observations；
- format：`PNG=18,792`；
- mode：`RGBA=18,768`、`RGB=24`；
- alpha：opaque=`18,768`、null=`24`；
- EXIF present：`false=18,792`；
- worker line counts：`4700/4700/4693/4699`。

正式 runner 使用 exact `ParquetFile.iter_batches(batch_size=8, columns=["images"])`，并在
`to_pylist()` 前验证 exact batch schema、逐 row 验证 exact keys。committed completed-root postflight 独立重建
namespace、inventory、selector order/uniqueness、records/receipts、histograms、manifest/run identity 与完整
projection/source/predecessor hash chain。

## 生成与验证

完整 formal argv、runtime versions、container identity、negative-operation counts 和 postflight argv 见
`git-summary.json`。consumer 应从 frozen tag 或明确 commit 下载，并用 Git producer revision 中的：

```bash
PYTHONPATH=code python3 code/scripts/validate_set_utility_selected_image_census_v2_output.py \
  --repository-root <CLEAN_PRODUCER_CHECKOUT> \
  --execution-config <CLEAN_PRODUCER_CHECKOUT>/code/configs/causalcache_set_utility_selected_image_format_census_v2_column_projection_repair.json \
  --output-root <DOWNLOADED_10_FILE_ROOT> \
  --expected-git-revision 1a03b7eb8ea2c41c8fed5213fed1bcc9d73f846b
```

publication 后已从 frozen tag fresh-download 到另一条本地路径，并重算 10-file canonical inventory；结果为
`6c687bec18fd9bcb8f06dfd31c9a1685a6de976eed67b610de18fd3d7933bd4f`，与原件逐 byte 相同。
