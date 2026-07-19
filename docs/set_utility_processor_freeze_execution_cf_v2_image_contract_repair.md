# Set Utility Processor Freeze Execution-CF v2 Image-Contract Repair

## 当前状态

当前状态为 `SOURCE_FROZEN_FORMAL_RUN_PENDING`。canonical Execution-CF v2 已冻结，config 为
`code/configs/causalcache_set_utility_processor_freeze_execution_cf_v2_image_contract_repair.json`，8,014 bytes，
SHA256=`82107b02b0e25fb23e6582af0fe6bd4c3cc3d04c300fb2495d0febc8498506dc`。本 milestone 只授权从
clean pushed producer commit 启动 processor-only formal run；尚未产生 completed root、final candidates、exact
operation budget、restoration labels、predictor checkpoint、matched-NLL 或 closed-loop result。

v1 `INVALID_PROCESSOR_FREEZE_EXECUTION_CF_V1_IMAGE_CONTRACT_DRIFT` 与其外置 `.incomplete`、logs、hashes
保持原样。v2 使用全新 source/config/output/staging identity，不修改、追认或 resume v1 bytes。

## 修复依据与唯一变化

修复依据是已通过 committed postflight、immutable HF publication 与 fresh re-download 验证的 selected-image
census v2：

- Git summary：
  `data/results/set_utility_selected_image_format_census_v2_column_projection_repair/summary.json`；
- HF repo：`gavinlaw/causalcache-set-utility-new-development-mobile`；
- tag：`phase1-b2-image-format-census-v2-column-projection-repair`；
- immutable revision：`c1d19eb96d7fa7926f1eb9db3328dbff4e88eae0`；
- formal inventory SHA256：
  `6c687bec18fd9bcb8f06dfd31c9a1685a6de976eed67b610de18fd3d7933bd4f`；
- denominator：18,792 observations，其中 18,768 个 opaque `PNG/RGBA`、24 个 `PNG/RGB`，全部
  `EXIF=false`。

v2 只将 selected-image eligibility 从单一 opaque RGBA 改为以下 exact union：

```text
PNG / RGBA / alpha=(255,255) / EXIF=false
PNG / RGB  / alpha=null      / EXIF=false
```

JPEG、P/LA、半透明 RGBA、带 EXIF、未知 format/mode 继续 fail closed。source encoded bytes 原样传给
RapidOCR、写入需要的 artifact image members 并以原 SHA256 绑定；禁止 source re-encode、补 alpha、
`ImageOps.exif_transpose` 或跳过 24 个 RGB observations。AutoProcessor replay 仍保持 v1 行为：验证原 bytes 后，
只在内存中执行唯一一次 `source.convert("RGB")`，不把转换结果写回 artifact。

除此之外以下内容逐值等于 v1：

- Freeze-B v2 的 1,200 trajectories / 2,400 query states；
- `1000/100/100` train/tune/evaluation role partition；
- 527 selected shards / 18,792 observations / 4-worker whole-shard schedule；
- OCR model、runtime、参数与 canonical record schema；
- prompt、newest-16 initial candidates、drop-oldest rule、context limit 与 reserved action tokens；
- exact subset cardinalities `|S|=0,1,2` 与 operation-budget materialization；
- policy/model forward、labels、training、matched-NLL、closed-loop、sealed test 与 HF mutation 全部禁止。

## 版本化接口

- image contract：`code/causalcache/set_utility_processor_image_contract_v2.py`；
- Execution-CF contract：`code/causalcache/set_utility_processor_freeze_contract_v2.py`；
- runner：`code/scripts/run_set_utility_processor_freeze_v2.py`；
- source/config validator：`code/scripts/validate_set_utility_processor_freeze_v2_execution.py`；
- completed-root postflight：`code/causalcache/set_utility_processor_postflight_v2.py` 与
  `code/scripts/validate_set_utility_processor_freeze_v2_output.py`；
- canonical output basename：
  `causalcache-set-utility-processor-freeze-v2-image-contract-repair-<GIT7>`；
- intended private HF tag：`phase1-b2-processor-freeze-v2-image-contract-repair`，只有 VALID postflight 与
  Git-safe result commit/push 后才可创建。

Execution-CF v2 真实调用并 byte-bind canonical v1 config SHA256=
`66fd93c64669be734f82830af7d623e9cb97263f90613a5809eb665078b2fba7`，再额外绑定 v1 failure summary、
有效 census summary/card 与 6 个 v2 runtime sources。v1 contract/config/runner/postflight、冻结 OCR config、
restoration backend、failure result 共 12 个关键文件的 Git blob hashes 已在 source freeze 前后核对一致。

## 三处一致的 v2 image contract

1. pre-OCR：逐 observation decode 并验证 exact signature，同时累计 worker format tally；
2. generation/replay：RapidOCR record runner 与 OCR record validator 均使用 v2 union；
3. processor replay：stored source bytes 先通过 v2 validation，再按 v1 exact transient RGB semantics 进入
   `AutoProcessor.apply_chat_template`。

只改 precheck 不足以形成有效修复；generation 或 replay 任一处回退到旧 RGBA-only validator 都会使 attempt
INVALID。RGBA fixture 的 v2 OCR record 必须与 v1 canonical bytes 相同；RGB fixture 则必须保留 source SHA 与
encoded bytes。

## Source audit 与 completed-root postflight

source audit 返回
`VALID_PROCESSOR_ONLY_SOURCE_V2_IMAGE_CONTRACT_REPAIR`。它递归绑定 v1/v2 runner，要求恰好一个
`AutoProcessor.from_pretrained`，禁止 model `forward/generate`、v1 image execution helper、image mutation、
import alias 与 `getattr` 绕过；唯一例外是 processor replay 内 literal `source.convert("RGB")`。

completed-root postflight 先执行既有 whole-root structural reconstruction，再执行 v2 overlay：

- exact output basename 必须绑定 producer `git_revision[:7]`；
- receipt exact keys、backend SHA、完整 runtime identity、image-contract SHA、worker/shard/tally 必须一致；
- terminal OCR records 全量验证 exact schema、backend、dimensions、format/alpha/EXIF、canonical nodes、full
  tokens 与 hashes；
- artifact 中有 source bytes 的 records 额外按原 bytes 完整重建 v2 canonical OCR record；
- receipt 与 terminal records 分别重建的 global tally 都必须精确等于
  `PNG:RGBA=18,768`、`PNG:RGB=24`、total=`18,792`；
- 缺失一个 RGB、重复 observation、额外 format、tampered record/receipt/source 或旧 namespace 均 fail closed。

root `manifest.json` 保持 v1 structural schema，便于复用已经验证的 whole-root parser；v2 identity 由 execution
config SHA、run identity、receipt identity 与 v2 postflight status 联合绑定。consumer 不能用旧 RGBA-only OCR
validator 读取该 artifact，必须显式使用本 v2 contract/postflight。

## Source-freeze 验证

- canonical config validator：
  `VALID_SET_UTILITY_PROCESSOR_FREEZE_EXECUTION_CF_V2_IMAGE_CONTRACT_REPAIR`；
- config bytes 与 live skeleton byte-for-byte 相同；
- source audit：0 forbidden image mutation、0 forbidden v1 execution helper、1 exact transient RGB convert；
- v2 focused tests：`42 passed`；
- processor v1/v2 focused + regression：`91 passed`；
- all set-utility regression：`341 passed, 15 skipped, 24 subtests passed`；15 个 skip 是本机没有 PyTorch 的
  integration paths，目标 processor runtime 在 formal preflight 中显式绑定 PyTorch/Transformers versions；
- `py_compile` 与 `git diff --check` 通过。

## 正式执行模板

source-freeze commit/push 后，必须从对应 clean detached checkout 启动。`<COMMIT>`、`<GIT7>` 与 container
identity 在运行时替换为真实值；不得从 dirty worktree 启动：

```bash
env -u PYTHONPATH /usr/bin/python3 \
  /data/worktrees/causalcache-processor-v2-<GIT7>/code/scripts/run_set_utility_processor_freeze_v2.py \
  --repository-root /data/worktrees/causalcache-processor-v2-<GIT7> \
  --execution-config /data/worktrees/causalcache-processor-v2-<GIT7>/code/configs/causalcache_set_utility_processor_freeze_execution_cf_v2_image_contract_repair.json \
  --source-root /data/source/guiodyssey-full-pool-v1 \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --snapshot-manifest /data/worktrees/causalcache-processor-v2-<GIT7>/code/configs/gui_owl_1_5_8b_snapshot.json \
  --ocr-model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 \
  --ocr-wheel-dir /data/tmp/causalcache-ocr-v2-wheels \
  --output-root /data/artifacts/causalcache-set-utility-processor-freeze-v2-image-contract-repair-<GIT7> \
  --ocr-python-executable /data/.venv/causalcache-ocr-v2/bin/python \
  --processor-python-executable /usr/bin/python3 \
  --git-revision <COMMIT> \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id <CONTAINER_ID> \
  --container-image-digest <IMAGE_DIGEST> \
  --worker-count 4 \
  --ocr-python-version 3.12.3 \
  --ocr-runtime-version 3.8.4 \
  --pyarrow-version 24.0.0 \
  --processor-python-version 3.12.3 \
  --transformers-version 5.6.0 \
  --torch-version 2.11.0+cu130 \
  --pillow-version 12.2.0
```

当前 Hyper00 preflight 显示 `/data02` 仍约有 1.6 TB 可用，既有 CPU-only container
`sglang-omni-jaxan-07181624` 与 `/data/CausalCache` checkout 可用；正式启动前仍需重新记录 host、disk、
container identity、clean checkout、完整 argv 与 start time。formal root 通过独立 postflight 前不得称为 VALID，
不得生成 labels 或上传 HF。
