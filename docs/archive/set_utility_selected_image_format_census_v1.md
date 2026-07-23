# Set Utility Selected-Image Format Census v1

## 当前状态

`processor-freeze Execution-CF v1` 已因第 228 个 observation 是合法 `PNG/RGB`、而旧 contract
只接受 opaque `PNG/RGBA` 而 fail closed。这个失败只否定旧图像输入假设，不允许据单一样本直接放宽
contract。v1 的 output root 仍不存在，`.incomplete`、日志和 failure hashes 保持原样。

本协议的唯一 formal attempt 已永久判为
`INVALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_COLUMN_PROJECTION_CONTRACT_DRIFT`。它虽然完成 1,200 trajectories /
18,792 observation records 和 atomic output，但 PyArrow 未执行冻结要求的 image-only column projection；
observed histogram 只能作 forensic evidence，不能供 downstream processor repair。该 attempt 没有运行 OCR、
`AutoProcessor`、policy/model forward、restoration labels、training、matched-NLL 或 closed-loop，也没有接触
sealed AndroidWorld test。

冻结 config 位于
`code/configs/causalcache_set_utility_selected_image_format_census_v1.json`，SHA256 为
`c0ecbf59dc6bd77881503e92b0e3eb8c011c2768d0721fa5a74c82c8fe173d10`。它逐 byte 绑定 Freeze-B v2、P0
census、610-shard inventory、base config，以及 runner 的完整 repository import closure。任何 bound input
或 runtime source 变化都会使 validator fail closed。

## 固定分母与执行边界

- roster：Freeze-B v2 的 1,200 trajectories，不能增删或替换；
- observation denominator：每条 trajectory 的全部 `decision_count + 1` observations，共 18,792；
- source schedule：527 个 selected transport shards，以 whole-shard deterministic LPT 分给 4 个 CPU worker；
  worker observation load 固定为 `4700/4700/4693/4699`；
- source read：仅从 pinned local source root 读取，逐 shard 校验 size、SHA256、row count，再按冻结 row index
  只读取 selected row 的 `images` column；不解析 `messages`、`metadata`、instruction、action 或 terminal outcome；
- image path：复用 `prepare_image_bytes`，记录原始 format/mode/alpha/EXIF/dimensions 和 source/RGB hashes；
- 禁止接口：OCR、`AutoProcessor`、`AutoModel`、policy forward/generate、PyTorch、GPU 与 HF mutation；
- 未选 source row 只用于维持顺序 row scan 和 selected-row identity verification，不允许进入语义输出。

正式 execution 的授权计数固定为：model/policy load `0`、OCR `0`、restoration label `0`、training `0`、
HF mutation `0`。本 census 不是 utility/result experiment，也不产生任何 paper method delta。

## 无 raw identity 的 pseudonymous record schema

每个 selected observation 恰好写一行 canonical JSONL，只允许以下九个字段：

```text
selector_identity_sha256
image_sha256
decoded_rgb_sha256
format
mode
alpha_extrema
exif_present
width
height
```

其中 selector identity 由 frozen `p0_selection_sha256`、observation ordinal 与 protocol id 哈希得到，因此
它与拥有 frozen source manifest 的审计者仍可关联，不声称不可逆匿名。记录不含
trajectory/source id、transport path/row、instruction、action、OCR text、state id、raw image bytes 或 outcome。
最终 Git-safe manifest 只保留 counts、histograms、digests、runtime 与 provenance；四份 pseudonymous JSONL/receipt 属于
reusable data artifact，不进入 Git。

## 原子发布与恢复

正式 output root 必须是 repository/source 之外的绝对 persistent path，并且启动时不存在。runner 使用 sibling
`.incomplete` staging：

1. 每个 worker 以 hard-link no-clobber 方式写 canonical JSONL，再写绑定它的 receipt；
2. 已存在且逐 byte 一致的 record/receipt pair 可以 resume；只存在一半或内容漂移立即失败；
3. orchestrator 聚合四个 worker，检查 18,792 denominator、全局 selector uniqueness、histograms 与 digests；
4. 写入 `run-identity.json` 与 `manifest.json`、删除成功日志、验证 exact final inventory；
5. 仅在全部检查通过后，以 no-replace directory rename 发布 final root。

任何成功 artifact 在执行期间都不上传 Hugging Face。完成后先登记为 `PENDING_HF_UPLOAD`，预定 private dataset
为 `gavinlaw/causalcache-set-utility-new-development-mobile`，tag 为
`phase1-b2-image-format-census-v1`；上传和 immutable revision 必须作为后续独立 SoT milestone 记录。

## 正式运行接口

正式运行必须来自包含本 source freeze 的 clean pushed commit，并显式传入完整身份参数：

```bash
env -u PYTHONPATH /usr/bin/python3 \
  <CHECKOUT>/code/scripts/run_set_utility_selected_image_census.py \
  --repository-root <CHECKOUT> \
  --execution-config <CHECKOUT>/code/configs/causalcache_set_utility_selected_image_format_census_v1.json \
  --source-root /data/source/guiodyssey-full-pool-v1 \
  --output-root /data/artifacts/causalcache-selected-image-format-census-v1-<COMMIT> \
  --python-executable /usr/bin/python3 \
  --git-revision <FULL_SOURCE_COMMIT> \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id <FULL_CONTAINER_ID> \
  --container-image-digest <SHA256_IMAGE_DIGEST> \
  --worker-count 4 \
  --python-version 3.12.3 \
  --pyarrow-version 24.0.0 \
  --pillow-version 12.2.0
```

正式结果必须另行记录 full argv、source/result Git commit、UTC start/end、host/container/image、package
versions、CPU device、`dtype/seed=not_applicable`、output/staging/log paths、file inventory/tree hash、manifest
hash、HF 状态和全部 forbidden-operation counts。

## v1 正式 attempt 与失败

唯一 formal attempt 使用 clean pushed
`main@e636df1fa3890c0f889c99db227d6bb959d23383`，在 Hyper00 CPU-only container 从
`2026-07-19T03:24:38Z` 运行到 `03:31:16Z`，exit code `0`。四个 worker 的 line counts 为
`4700/4700/4693/4699`，final root 以 no-replace rename 原子发布：10 个 regular files、6,940,661 bytes、
无 symlink，staging absent。

但 `ParquetRowFactory` 实际调用为：

```python
parquet.iter_batches(batch_size=8)
```

没有传 `columns=["images"]`。所以 `batch.to_pylist()` 在 runner 只显式取 `row.get("images")` 之前已经
物化完整 rows，包括 `messages` 与 `metadata`。这违反 config 中冻结的
`images_column_only_no_message_metadata_action_or_outcome_decode`，也使
`access_outcome_or_utility_allowed=false` 无法成立。原 AST test 只拒绝显式 semantic field access，没有验证
PyArrow projection，因此 source validation 未捕获该缺口。

observed-but-formal-ineligible 18,792-record histogram 为：

- format：`PNG=18,792`；
- mode：`RGBA=18,768`，`RGB=24`；
- alpha extrema：`[255,255]=18,768`，`null=24`；
- EXIF：`false=18,792`；
- dimensions：9 种，完整 counts 见 Git summary。

records/receipts 自身的 canonical bytes、全局 selector uniqueness、histograms 与 manifest/run-identity hash chain
可以重算一致，但 column-projection contract 未满足，所以整体 postflight 必须为 INVALID。manifest SHA256=
`282b86c62e3856f35da901c925f23d0baa54970211260e97f44a4aa6ee2b7180`，10-file tree SHA256=
`973b0f1b29059fde2f7e1001559b10a38629192e8e3db6f5774d41764ff627e3`。完整 argv、runtime、worker digests 与
negative operation counts 见
[`../../data/results/archive/set_utility_selected_image_format_census_v1_attempt/`](../../data/results/archive/set_utility_selected_image_format_census_v1_attempt/)。

该 histogram 不得用于冻结 processor repair，exact 10-file root 也禁止 HF publication。唯一允许的后继是新的
column-projection census version：`columns=["images"]`、runtime batch-schema assertion、真实 projection regression、
new protocol/config/output/tag identity，并重新跑完整 denominator。replacement v2 已从 clean `main@1a03b7e`
完成全量 run 与 committed postflight，正式 VALID；其独立身份、histogram 与结果证据见
[`set_utility_selected_image_format_census_v2_column_projection_repair.md`](set_utility_selected_image_format_census_v2_column_projection_repair.md)。

## 验证

source freeze 的 focused validation 为：

```bash
PYTHONPATH=code .venv/bin/pytest -q \
  code/tests/test_set_utility_selected_image_census.py \
  code/tests/test_set_utility_selected_image_census_contract.py \
  code/tests/test_set_utility_processor_freeze.py \
  code/tests/test_set_utility_processor_substrate.py

PYTHONPATH=code .venv/bin/python \
  code/scripts/validate_set_utility_selected_image_census_contract.py \
  --repository-root . \
  --execution-config code/configs/causalcache_set_utility_selected_image_format_census_v1.json
```

source-freeze 当时得到 `30 passed`，validator 返回 `VALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_CONTRACT` 和
`VALID_CPU_ONLY_SELECTED_IMAGE_CENSUS_SOURCE`；但这些检查遗漏 PyArrow column projection。正式进程随后虽
atomic publish，post-run audit 已将该 attempt 永久封为 INVALID。replacement 必须加入 projection-aware tests
与 committed completed-root postflight，不能修改原 v1 config/source/result bytes 后续跑。
