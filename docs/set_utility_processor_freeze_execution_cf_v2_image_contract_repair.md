# Set Utility Processor Freeze Execution-CF v2 Image-Contract Repair

## 当前状态

replacement source freeze 已完成；clean formal run 已于 `2026-07-19T07:10:28Z` 从 pushed
`main@e976b990ddb089cdea3b04ea15e5c911d5670d40` 在 Hyper00 启动，当前严格为
`RUNNING_INCOMPLETE_NOT_UPLOADABLE`。canonical Execution-CF v2 config 为
`code/configs/causalcache_set_utility_processor_freeze_execution_cf_v2_image_contract_repair.json`，9,290 bytes，
SHA256=`e2c271e00749ca7643899c86fd216a635d007337630ba9a1a19b2314ff4afb70`。此前 `1c84dfe` 与
`dc90664` 两次 attempt 均已主动终止且 exit=`143`：真实 processor smoke 证明旧 runtime guard 会把
Transformers 5.6 必需的通用 `transformers.models.auto.modeling_auto` registry 误判为 architecture model。
两次均停在 0 receipt、0 candidate-part、0 policy output，只有不可续写的外置 partial staging；replacement run
使用全新 namespace，当前仍未产生 completed root、final candidates、exact operation budget、restoration labels、
predictor checkpoint、matched-NLL 或 closed-loop result。

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

除 versioned image eligibility 与后文明确冻结的 CPU execution scheduling 外，以下科学/输出内容逐值等于 v1：

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
- immutable HF publication manager：`code/causalcache/set_utility_processor_publication_v2.py` 与
  `code/scripts/manage_set_utility_processor_freeze_v2_publication.py`；它只接受 committed
  `PENDING_HF_UPLOAD` summary 与 exact 23-file root，以单次 25-operation commit 上传 formal tree、summary 和
  card，创建无覆盖 annotated tag，并分别从 immutable commit 与 tag fresh-download 全部 25 files 逐 byte 验证；
  commit/tag/download/receipt 中断后只允许 exact-state reconcile，未知或 drifted prefix/tag 拒绝续跑；fresh
  replay 路径在远端 mutation 前验证，receipt 通过 `0600` temp + file/parent `fsync` + no-overwrite hard-link
  原子发布；
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
`VALID_PROCESSOR_ONLY_SOURCE_V2_IMAGE_CONTRACT_REPAIR`。它递归绑定 v1/v2 runner，要求 processor runtime 只从
`AutoProcessor.from_pretrained` 构造，禁止 model `forward/generate`、v1 image execution helper、image mutation、
import alias 与 `getattr` 绕过；唯一图像例外是 processor replay 内 literal `source.convert("RGB")`。首个
AutoProcessor 构造前 modeling module 数必须为 0；构造后 exact allowlist 只有
`transformers.models.auto.modeling_auto`，任何 architecture `modeling_*` 仍 fail closed。AST audit 同时固定
thread setter/getter、调用顺序、bounded ordered map、runtime pool 和 child environment 清理。completed-root
postflight 还要求四个 processor worker log 的第一行分别是唯一 canonical runtime getter evidence；任何缺失、
重复、顺序错误、线程值漂移、非法 UTF-8 或控制字符都会拒绝 root。

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
- committed execution-focused suite：`79 passed`；加入 publication recovery tests 后 combined focused suite：
  `102 passed`；
- config validator 会重建 live skeleton，并要求与 canonical config byte-for-byte 相同；
- `py_compile` 与 `git diff --check` 通过。

## 已终止 attempts 与 replacement execution contract

- `1c84dfe37f86bdc255a00184521170eeaa3b0663` 于 `2026-07-19T04:58:04Z` 启动、
  `2026-07-19T06:27:56Z` 主动终止，exit=`143`；staging 为
  `/data/artifacts/.causalcache-set-utility-processor-freeze-v2-image-contract-repair-1c84dfe.incomplete`，
  6,110,870,181 bytes；
- `dc90664bc4c16f5f74463e10072f271843d0b316` 于 `2026-07-19T06:10:19Z` 启动、
  `2026-07-19T06:46:35Z` 主动终止，exit=`143`；staging 为
  `/data/artifacts/.causalcache-set-utility-processor-freeze-v2-image-contract-repair-dc90664.incomplete`，
  14,201,155,380 bytes / 9 files；
- 两次均为 0 receipt、0 completed shard、0 candidate-part、0 policy output、0 HF mutation。现有 resume 只复用
  completed tar+receipt pair，会删除 `.tar.partial` 后重算；因此两份 staging 仅作
  `LOCAL_FORENSIC_NOT_UPLOADABLE`，不得复制到新 namespace 或追认为可续写 substrate；
- primary blocker 不是 image contract，而是 processor-only import guard 的 false positive。旧 guard 要求 load
  后仍有 0 个 `modeling_*` module；真实 Transformers 5.6 smoke 证明只出现通用 auto registry。replacement
  contract 采用上述 exact allowlist，不允许 architecture model module。

Hyper00 有 112 个 physical / 224 个 logical CPU 和约 2 TiB RAM。最终执行参数不是任意调大 worker，而是由真实
输入 smoke 收敛并冻结：

- OCR：128 张真实图像下，并发 `8/16/32` 分别为 `75.52/49.74/33.93s`；32 路相对 8 路为 `2.23x`，
  canonical records 全等。因此 4 logical workers 各 32 路，总 128 OCR slots；
- processor：8 条真实 query 下，单路/4路/8路为 `85.66/28.11/26.93s`；显式 PyTorch intra-op=`28`、
  inter-op=`1` 后 8 路为 `25.71s`。每 process 的 28-thread pool 与 4 logical processes 对齐 112 physical
  cores；每 process 8 个独立 AutoProcessor runtimes 共享该 process-level thread pool，总 32 processor slots；
- 所有并发 smoke 与串行输出 canonical-equal；bounded map 只改变 execution scheduling，不改变 logical shard、
  trajectory/query/candidate order、23-file tree 或 operation budget；processor child 会移除 ambient
  `OMP/MKL/OpenBLAS/NUMEXPR/Torch/vecLib` thread variables，再通过 PyTorch runtime API 设置并读取校验。

新 formal run 必须使用全新 output basename，从 replacement commit 的 clean detached checkout 全量重跑；不能
复用上述 partial。只有 atomic root、exit=0、committed postflight 和 Git-safe result commit/push 完成后，状态才可
进入 `PENDING_HF_UPLOAD`。

## 当前 formal run

- producer：clean detached `e976b990ddb089cdea3b04ea15e5c911d5670d40`；config SHA256=
  `e2c271e00749ca7643899c86fd216a635d007337630ba9a1a19b2314ff4afb70`；
- host/container：Hyper00 `node-radixark-16-0000`，CPU-only container
  `a954543dd85b38a761a06a16384347169706e26e1bc34ffc8f8d279064cca50a`；
- output：`/data/artifacts/causalcache-set-utility-processor-freeze-v2-image-contract-repair-e976b99`；当前只有
  sibling `.incomplete`，没有 final root 或 exit evidence；
- durable evidence：`/data/logs/causalcache-processor-v2-e976b99`；one-shot supervisor SHA256=
  `95f066f1e0df17dd0efb8e3d113cb5620306382be1c63cdb64c3c890b7ab7d03`，已预写 exact argv/start/postflight argv，
  run 结束后原子写 exit/end 并在 exit=0 时自动运行 committed postflight；
- 四进程联合 processor smoke：32 query / 32 runtimes，joint wall=`41.26s`、effective CPU=`30.79 cores`、
  aggregate peak RSS=`50,295,324 KiB`，四进程 getter 均为 `28/1`；
- OCR 稳定启动 snapshot：四个 logical worker lifetime CPU 分别为 `2740/2692/2837/2744%`，合计约
  `110.13` physical-core equivalents；aggregate RSS=`38,560,624 KiB`，error/traceback match=0。该 snapshot 只
  证明 execution health，不替代 final root/postflight；
- Git-safe 记录：`data/results/set_utility_processor_execution_scaling_v2/`。GPU、model/policy forward、label、
  training、matched-NLL、closed-loop 与 HF mutation 均为 0。

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
