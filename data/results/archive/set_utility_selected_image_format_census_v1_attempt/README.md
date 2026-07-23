# Selected-Image Format Census v1 Attempt

## 正式终态

本次 Hyper00 CPU run 的进程与 10-file output 均完整结束，但 execution contract 在独立审计中被证实违反，
正式状态为：

```text
INVALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_COLUMN_PROJECTION_CONTRACT_DRIFT
```

不能把它写成成功 census，也不能用其 histogram 冻结 downstream processor repair。

## 失败原因

冻结 config 要求：

```text
images_column_only_no_message_metadata_action_or_outcome_decode
```

worker 的 Python 分支确实只显式读取 `row.get("images")`，也没有调用 `inspect_candidate`、
`load_selected_rows_once` 或 `build_selected_pilot`。但底层 `ParquetRowFactory` 使用：

```python
parquet.iter_batches(batch_size=8)
batch.to_pylist()
```

它没有传 `columns=["images"]`，所以 PyArrow 在 Python 分支之前已经读取并物化 selected shards 的完整 rows，
包括 `messages` 与 `metadata`。因此 `access_outcome_or_utility_allowed=false` 与 image-column-only 执行声明没有
得到满足。现有 AST regression 只禁止显式 semantic field access，没有验证真正传给 PyArrow 的 projection，
未能在 source freeze 阶段发现这个问题。

该错误在正式进程结束后的 `2026-07-19T03:36:38Z` 被独立结果审计发现。它不是 histogram 数值不一致，
而是数据访问授权与实际 I/O 不一致，所以即使 runner exit code 为 0，也必须 fail closed。

## 保留的 forensic evidence

唯一 attempt 使用 clean pushed
`main@e636df1fa3890c0f889c99db227d6bb959d23383`，从 `2026-07-19T03:24:38Z` 到
`03:31:16Z`，四 worker line counts=`4700/4700/4693/4699`。runner 原子发布的外置 root 为：

```text
/data/artifacts/causalcache-selected-image-format-census-v1-e636df1
```

该 root 的 10 files / 6,940,661 bytes、manifest SHA256
`282b86c62e3856f35da901c925f23d0baa54970211260e97f44a4aa6ee2b7180` 与 tree SHA256
`973b0f1b29059fde2f7e1001559b10a38629192e8e3db6f5774d41764ff627e3` 原样保留，仅作 INVALID forensic
evidence。完整 argv、worker hashes、observed-but-formal-ineligible histograms 和 failure classification 见
[`summary.json`](summary.json)。

这些 bytes 不得：

- 上传预定 canonical HF tag；
- 被追认为 `PENDING_HF_UPLOAD` 或 reusable artifact；
- 用来声称 18,792-image census 已闭合；
- 用来直接冻结 processor input repair；
- 删除、覆盖或在原 namespace 续跑。

本次 HF mutation、OCR、AutoProcessor、model/policy forward、restoration label、training、matched-NLL、
closed-loop 与 sealed test operation 仍为 0。注意这不等于 outcome-column access 为 0：完整 Parquet rows 已被
PyArrow 物化，这是本 attempt invalid 的根因。

## 唯一允许的后继

另立 versioned column-projection repair，至少必须：

1. 使用 `parquet.iter_batches(batch_size=8, columns=["images"])`；
2. 在运行时断言每个 batch schema 恰好只有 `images`；
3. 测试真实捕获并断言传给 PyArrow 的 projection，而不是只做 AST 字符串检查；
4. 使用新的 config/protocol/output/tag identity；
5. 从 clean pushed commit 重新跑完整 1,200 trajectories / 18,792 observations；
6. 在声明 VALID 前运行 committed completed-root postflight。

在 versioned repair 完成前，final candidates、restoration labels、predictor training、matched-NLL 与
closed-loop 继续 locked。
