# Set Utility Processor Freeze Execution-CF v2 Image-Contract Repair

## 当前状态

source freeze 已完成；唯一 formal run 当前状态为 `RUNNING_INCOMPLETE_NOT_UPLOADABLE`。canonical Execution-CF v2
已冻结，config 为
`code/configs/causalcache_set_utility_processor_freeze_execution_cf_v2_image_contract_repair.json`，8,014 bytes，
SHA256=`82107b02b0e25fb23e6582af0fe6bd4c3cc3d04c300fb2495d0febc8498506dc`。run 于
`2026-07-19T04:58:04Z` 从 clean detached
`main@1c84dfe37f86bdc255a00184521170eeaa3b0663` 在 Hyper00 CPU-only container 启动；尚未产生 completed root、
final candidates、exact
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
- Git-safe result recorder：`code/causalcache/set_utility_processor_result_v2.py` 与
  `code/scripts/record_set_utility_processor_freeze_v2_result.py`；它要求 fresh committed postflight、exact 23-file
  tree、staging absent、runner argv/start/end/exit/log evidence 全部闭合，且只写 `README.md` 与 `summary.json`；
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

## Formal v2 运行状态

- 初始 clean producer：`1c84dfe37f86bdc255a00184521170eeaa3b0663`；config SHA256 为
  `82107b02b0e25fb23e6582af0fe6bd4c3cc3d04c300fb2495d0febc8498506dc`；
- host/container：Hyper00 `node-radixark-16-0000`，container ID=
  `a954543dd85b38a761a06a16384347169706e26e1bc34ffc8f8d279064cca50a`，Docker
  `DeviceRequests=null`，4 个 OCR CPU workers，GPU count=0；
- output namespace：
  `/data/artifacts/causalcache-set-utility-processor-freeze-v2-image-contract-repair-1c84dfe`；完成前只允许 sibling
  `.incomplete` 存在，禁止上传；
- `2026-07-19T05:12:26Z` 的 prefix-safe tar-header snapshot：worker completed records=
  `30/34/24/34`，映射到 `1,011/18,792` observations（5.38%）；四 worker 均持续约 100% CPU，零 traceback，
  仅一次合法 empty-detection warning；
- worker 0 已完整完成 v1 violation 所在 trajectory，因此 v2 明确越过第 228 个 observation 的旧 RGB failure
  boundary；这只关闭已知 blocker，不等于 formal VALID；
- 运行 ETA 粗估为 OCR 4--5 小时、AutoProcessor 1.5--3 小时、总计 6--8 小时。正式终态只看 atomic root、
  runner exit 与 committed v2 postflight，不看 ETA；
- 外置 `argv.txt`、`start.json`、`outer.log`、`exit_code` 与 one-second end watcher 均在 `/data/logs`；result
  recorder 将验证其 exact bytes/hash、start/end/elapsed、run identity、postflight 与 formal tree，成功时仍只标记
  `PENDING_HF_UPLOAD`；recorder 明确允许 producer worktree 与 recorder worktree 不同，但 runner path 只能取自
  formal `run-identity.runtime_cli.repository_root`；两个 checkout 都必须 exact clean Git HEAD，执行中的 recorder、
  contract 与 postflight module origins 也必须实际来自 recorder checkout，不能用 clean checkout 替 dirty code 背书；
- recorder focused suite=`59 passed`；当前 all set-utility regression=
  `356 passed, 15 skipped, 24 subtests passed`，`py_compile`、CLI help、source validator 与 `git diff --check` 通过。

### Execution-only 32-slot replacement

初始运行暴露了一个纯执行问题：四个 logical artifact workers 各只有一个 RapidOCR engine，因此在 224 logical
CPU 的 Hyper00 上总共只使用约 4 cores。`main@dc90664bc4c16f5f74463e10072f271843d0b316` 不改变四份 logical
tar、query/candidate order、23-file output、operation budget 或 postflight，而在每份 logical shard 内冻结 8 路
bounded ordered execution；每个 slot 拥有独立 engine，in-flight window 固定为 16 trajectories，yield 顺序仍与
source schedule 一致。修订后 config SHA256=
`c5c85f99c2f457fcff6d6ae0b096005481947220a6af17335c5f6736fc289142`。

真实截图 smoke 对 32 张 source PNG 同时运行串行和 8 路并发，canonical records 完全相等；engine init=1.69s、
串行=107.66s、并发=17.05s，speedup=6.32×。accelerated formal output 为：

```text
/data/artifacts/causalcache-set-utility-processor-freeze-v2-image-contract-repair-dc90664
```

它于 `2026-07-19T06:10:19Z` 启动，warmup 后四个 logical workers 各约 715--726% CPU；当前仍只有 sibling
`.incomplete`，没有 exit/end/final。初始 run 保留作备份，不删除也不追认 partial bytes。accelerated run 只有通过
同一 exact 23-file committed postflight 后才可成为 canonical；若两者都完成，先执行 semantic equivalence audit，
只允许一个 immutable HF publication。并发输出一旦与串行语义漂移即 fail closed。

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
