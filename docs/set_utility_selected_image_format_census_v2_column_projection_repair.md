# Set Utility Selected-Image Format Census v2 Column-Projection Repair

## 当前状态

当前状态为 `VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR`。formal run 从 clean
pushed `main@1a03b7eb8ea2c41c8fed5213fed1bcc9d73f846b` 启动，并通过同一 commit 中的 independent read-only
completed-root postflight。有效 output 与 histogram 已产生，并已发布到 immutable HF revision
`c1d19eb96d7fa7926f1eb9db3328dbff4e88eae0`；fresh re-download 的 formal 10-file inventory 与原件逐 byte
相同。当前只授权另立 versioned processor image-contract repair，不授权 labels/training/closed-loop。

v1 的唯一 formal attempt 保持
`INVALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_COLUMN_PROJECTION_CONTRACT_DRIFT`。其 10-file root 只能作为 forensic
evidence：不得上传 Hugging Face，不得从中冻结 processor repair，也不得通过修改旧 config、旧 runner 或旧
output namespace 追认。失败证据与完整原因见
[`../data/results/set_utility_selected_image_format_census_v1_attempt/`](../data/results/set_utility_selected_image_format_census_v1_attempt/)
和 [`set_utility_selected_image_format_census_v1.md`](set_utility_selected_image_format_census_v1.md)。

## 新版本身份

- protocol id：`causalcache_set_utility_selected_image_format_census_v2_column_projection_repair`；
- schema version：`2.0.0`；
- config：`code/configs/causalcache_set_utility_selected_image_format_census_v2_column_projection_repair.json`；
- core：`code/causalcache/set_utility_selected_image_census_v2.py`；
- contract：`code/causalcache/set_utility_selected_image_census_contract_v2.py`；
- runner：`code/scripts/run_set_utility_selected_image_census_v2.py`；
- source validator：`code/scripts/validate_set_utility_selected_image_census_v2_contract.py`；
- completed-root postflight：`code/scripts/validate_set_utility_selected_image_census_v2_output.py`；
- formal output：
  `/data/artifacts/causalcache-selected-image-format-census-v2-column-projection-repair-<COMMIT>`；
- intended private dataset：`gavinlaw/causalcache-set-utility-new-development-mobile`；
- intended tag：`phase1-b2-image-format-census-v2-column-projection-repair`。

以上身份与 v1 全部隔离。冻结 config 为 10,292 bytes，SHA256=
`eb823bd794c555265107e5edd4ff9b0be60b9c907b476a313ce5e156779a549f`；它逐 byte 绑定 frozen inputs、v1 invalid
predecessor、v1 config/core/contract/runner 与 24-file runtime import closure。formal producer commit 由正式
run 的 `--git-revision` 记录，不在运行前猜测。

## 固定分母与授权边界

replacement 不改变科学分母：继续使用 Freeze-B v2 的 1,200 trajectories、18,792 observations、527 selected
transport shards 和 4-worker whole-shard deterministic LPT schedule；expected worker observation load 仍为
`4700/4700/4693/4699`。允许读取的唯一 Parquet column 是 `images`。

执行期间以下操作全部保持 0：

- `messages`、`metadata`、instruction、action、terminal outcome 或 utility 的 materialization/parse/access；
- OCR、`AutoProcessor`、`AutoModel`、policy load/forward/generate 和 PyTorch/GPU compute；
- restoration label、feature cache、predictor training、matched-NLL、closed-loop 和 sealed test access；
- Hugging Face mutation。

输出 record schema、pseudonymous selector identity、atomic staging、paired resume、no-clobber write 与 no-replace
publish 语义沿用 v1；v2 只修复被审计确认的 Parquet projection contract，不借机改变 roster、image preparation、
histogram 字段或 denominator。

## 必须满足的 column-projection contract

每个 Parquet batch 必须通过 exact call 读取：

```python
parquet.iter_batches(batch_size=8, columns=["images"])
```

在任何 `to_pylist()`、row selection 或 image decode 前，runtime 必须 fail closed 地断言：

```python
batch.schema.names == ["images"]
```

投影后的每个 Python row 还必须满足 `set(row) == {"images"}`；缺列、多列、重复/漂移 schema 或任何不带
`columns=["images"]` 的 fallback 都立即使 attempt INVALID。只在这些断言通过后，runner 才能读取
`row["images"]` 并进入既有 `prepare_image_bytes` 路径。

测试不能再只扫描 AST 中是否出现 `messages`/`metadata`。projection regression 必须实际捕获
`ParquetFile.iter_batches` 的调用参数，并断言 `batch_size=8`、`columns=["images"]`，同时覆盖错误 schema、
额外 row key 和省略 `columns` 的 fail-closed case。source validator 还应 byte-bind v1 failure summary 与新的
runtime import closure，防止绕过 replacement 身份回退到旧 runner。

## Committed completed-root postflight

正式结果只有在新的 committed postflight 从 final root 独立重算并返回
`VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR` 后才可称为 VALID。postflight 至少复核：

- exact 10-file regular-file inventory、无 symlink、staging absent 和 output namespace；
- 四个 worker 的 record/receipt pairing、line counts 与 18,792 global denominator；
- selector uniqueness、record/selector inventories、histograms、manifest/run-identity 与 tree hash chain；
- protocol/config/source/runtime identity，以及 frozen `images` projection witness；
- forbidden-operation counts 与 HF mutation count 全部为 0。

进程 exit code `0`、runner 内部自检通过或与 v1 forensic histogram 相同，都不能替代该 postflight。任一检查
失败时，v2 output 必须在其新 namespace 原样保留并标为 INVALID，不得上传或用于 downstream contract。

## Publication 与下游解锁顺序

正式运行期间以及 completed-root postflight 前禁止任何 HF upload。只有 VALID postflight 与 Git-safe result
summary 已 commit/push 后，才可在独立 SoT milestone 上传 intended private dataset/tag，并回写 immutable HF
revision；在此之前状态不是 `PENDING_HF_UPLOAD`，而是尚未产生可发布 artifact。

上述 publication gate 已完成：private dataset tag
`phase1-b2-image-format-census-v2-column-projection-repair` 已冻结到 revision
`c1d19eb96d7fa7926f1eb9db3328dbff4e88eae0`，并通过独立 fresh-download inventory 验证。该完成态只解锁
processor image-contract repair，不改变后续 labels/training/closed-loop 的锁定状态。

即使 v2 最终 VALID，也只能授权下一步另立 versioned processor image-contract repair。final candidate universe、
exact operation budget、restoration labels、utility predictor training、matched-NLL、closed-loop 和 sealed test 仍需
各自后续 contract，不能由 census 直接解锁。processor repair 只能依据 VALID v2 histogram，不得引用 v1
formal-ineligible histogram 作为设计依据。

## Source-freeze 验证结果

source freeze 已同时完成：

1. 新 protocol/config/runner/validator/postflight identity 全部落在 Git；
2. config 逐 byte 绑定 frozen inputs、v1 failure predecessor 与完整 runtime source closure；
3. projection capture、runtime schema/row-key、contract 与 completed-root postflight tests 通过；
4. source validator 明确报告 CPU-only、image-column-only，并且不授权 formal result claim；
5. source validator 返回 `VALID_CPU_ONLY_SELECTED_IMAGE_CENSUS_V2_COLUMN_PROJECTION_REPAIR_SOURCE`，contract
   validator 返回 `VALID_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR_CONTRACT`；
6. v2-specific tests=`23 passed`，v1+v2 focused regression=`37 passed`，全部 set-utility regression=
   `299 passed, 15 skipped, 24 subtests passed`；
7. README、`code/README.md`、`data/README.md` 与 `docs/progress.md` 已回写真实 SHA、验证结果和下一步。

本 material milestone commit/push canonical `main` 后，正式运行只从该 clean immutable commit 启动。runner
完成时 artifact 状态必须是 `AWAITING_COMMITTED_POSTFLIGHT`；不得在 postflight 前声称 VALID 或
`PENDING_HF_UPLOAD`。

## Formal run 与 completed-root postflight

正式进程在 Hyper00 no-GPU container 中从 `2026-07-19T04:02:14Z` 运行到 `04:08:52Z`，耗时 398 秒，
exit code 0。producer、runtime 与 artifact identity 为：

- Git revision：`1a03b7eb8ea2c41c8fed5213fed1bcc9d73f846b`；
- config SHA256：`eb823bd794c555265107e5edd4ff9b0be60b9c907b476a313ce5e156779a549f`；
- Python/PyArrow/Pillow：`3.12.3 / 24.0.0 / 12.2.0`；
- device/GPU：`cpu / 0`；
- worker line counts：`4700/4700/4693/4699`；
- final root：10 files / 6,943,195 bytes；无 symlink，sibling `.incomplete` absent；
- manifest SHA256：`83fc82ab32e7c5fde09ce79483d0da76d896830a249157883b05881bbbdfd7c6`；
- canonical file inventory SHA256：`6c687bec18fd9bcb8f06dfd31c9a1685a6de976eed67b610de18fd3d7933bd4f`；
- read-only tree snapshot SHA256：`bd5f28c19b1303cae22437ec8d6659144c7301f20dee9a3be7401adc95be5cda`。

committed postflight 返回：

```text
VALID_COMPLETED_SELECTED_IMAGE_FORMAT_CENSUS_V2_COLUMN_PROJECTION_REPAIR
```

它独立重建 exact namespace/inventory、selector order/uniqueness、records/receipts、histograms、manifest、run
identity、projection/source/predecessor hash chains，并验证前后 tree snapshot 不变。完整 argv、worker hashes、
negative-operation counts 与 Git-safe provenance 见
[`../data/results/set_utility_selected_image_format_census_v2_column_projection_repair/`](../data/results/set_utility_selected_image_format_census_v2_column_projection_repair/)。

## 正式 histogram 与 processor repair 输入

18,792 个 observation records 的有效分布为：

- format：`PNG=18,792`；
- mode：`RGBA=18,768`，`RGB=24`；
- alpha extrema：`[255,255]=18,768`，`null=24`；
- EXIF：`false=18,792`；
- dimensions：9 种，完整 counts 见 Git summary。

因此 processor image-contract repair 必须保留 original encoded bytes，并同时接受 exact
`PNG/RGBA + opaque alpha + no EXIF` 与 `PNG/RGB + null alpha + no EXIF`。不得将 24 个 RGB records 跳过、
转码成 RGBA、补 alpha 或更改既有 OCR/AutoProcessor semantics。v2 与 v1 forensic root 的 non-selector records
逐项一致仅作诊断；v2 的 formal eligibility 只来自新 projection contract 与 postflight。

## 当前 artifact publication 状态

外置 root 当前保留于 Hyper00：

```text
/data/artifacts/causalcache-selected-image-format-census-v2-column-projection-repair-1a03b7e
```

artifact 已发布到 private dataset/tag/revision：

```text
gavinlaw/causalcache-set-utility-new-development-mobile
phase1-b2-image-format-census-v2-column-projection-repair
c1d19eb96d7fa7926f1eb9db3328dbff4e88eae0
```

从该 frozen revision fresh-download 的 formal 10 files 共 6,943,195 bytes，canonical inventory SHA256=
`6c687bec18fd9bcb8f06dfd31c9a1685a6de976eed67b610de18fd3d7933bd4f`，与 Hyper00 原件逐 byte 相同。外置
root 继续保留，但已不是唯一 artifact copy。final candidates、exact operation budget、restoration labels、
predictor training、matched-NLL 与 closed-loop 继续 locked。
