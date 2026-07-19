# Selected-Image Format Census v2 Column-Projection Repair

## 正式结论

本次 replacement 从 clean pushed `main@1a03b7eb8ea2c41c8fed5213fed1bcc9d73f846b` 完整运行，并通过同一
commit 中的独立只读 completed-root postflight。正式状态为：

```text
VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR
```

v1 的 `INVALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_COLUMN_PROJECTION_CONTRACT_DRIFT` 不被追认或覆盖；本结果使用
独立 protocol/config/output/tag，重新扫描完整 1,200 trajectories / 18,792 observations / 527 shards。

## 核心验证

- PyArrow exact call：`iter_batches(batch_size=8, columns=["images"])`；
- 每个 batch 在 `to_pylist()` 前验证 schema names 恰为 `['images']`；
- 每个 materialized row 的 keys 恰为 `{'images'}`；
- 4-worker line counts=`4700/4700/4693/4699`，总数 18,792；
- final root 恰为 10 个 regular files / 6,943,195 bytes，无 symlink，sibling staging absent；
- committed postflight 重建 selector 顺序与全局唯一性、records/receipts、histograms、manifest、run identity 和
  source/predecessor hash chain，并验证前后 tree snapshot 不变；
- OCR、AutoProcessor、model/policy、GPU、labels、training、matched-NLL、closed-loop、sealed test 与 HF mutation
  全部为 0。

完整 machine-readable provenance、argv、runtime、digests 与 negative operations 见
[`summary.json`](summary.json)。

## 有效 histogram

18,792 个 observation records 的正式分布为：

- format：`PNG=18,792`；
- mode：`RGBA=18,768`，`RGB=24`；
- alpha extrema：`[255,255]=18,768`，`null=24`；
- EXIF：`false=18,792`；
- dimensions：9 种，完整 counts 见 summary。

因此后续 processor repair 必须同时接受原始 `PNG/RGBA` opaque 与 `PNG/RGB`，不能跳过 24 个 RGB records、
补 alpha 或重编码。v2 与 v1 forensic root 的 18,792 个 non-selector records 逐项一致；该比较仅作诊断，正式
有效性来自 v2 contract 与 postflight。

## Artifact 状态与下一步

当前 10-file root 位于 Hyper00：

```text
/data/artifacts/causalcache-selected-image-format-census-v2-column-projection-repair-1a03b7e
```

Git result milestone push 后状态为 `PENDING_HF_UPLOAD`，intended private destination 是：

```text
gavinlaw/causalcache-set-utility-new-development-mobile
phase1-b2-image-format-census-v2-column-projection-repair
```

上传并回写 immutable HF revision 前，外置 root 仍是唯一 artifact copy，不能删除。有效 census 只授权下一步
另立 versioned processor image-contract repair；final candidates、restoration labels、predictor training、
matched-NLL 和 closed-loop 仍未解锁。
