# Set Utility Processor Freeze Execution-CF

## 当前结论

Freeze-B v2 已修复 terminal `decision_step_id` 的 one-based offset。下一步不是直接生成
restoration labels，而是先在完全不加载 policy model 的条件下，冻结每个 query 的最终候选集合、
processor token length、精确 subset schedule 和 model-operation budget。

本 Execution-CF 配置位于
`code/configs/causalcache_set_utility_processor_freeze_execution_cf.json`，SHA256 为
`66fd93c64669be734f82830af7d623e9cb97263f90613a5809eb665078b2fba7`。它绑定：

- Freeze-B v2 manifest：`915892ef2e0f1495da4b9e409b3e7a112dc86cda0b06384e8a1b7f8053581d30`；
- P0 census、610-shard inventory、independent base parser；
- GUI-Owl snapshot、OCR backend config 与 OCR completion manifest；
- runner 及其 26 个 runtime source files 的 path、byte count 和 SHA256。

任何被绑定输入或源码变化后，必须新建 versioned contract，旧配置会 fail closed。

## 固定分母与调度

- 1,200 trajectories；train/tune/evaluation=`1000/100/100`；
- 每条 trajectory 恰好一个 stratum anchor 和一个 terminal query，共 2,400 states；
- anchor decision 按 stratum 固定为 `6/10/18`，terminal 固定为
  `decision_count + 1`；
- 527 个 selected transport shards，18,792 observations；
- processor shard schedule 以完整 transport shard 为不可拆单元做 deterministic LPT，四个 worker
  的 observation load 为 `4700/4700/4693/4699`；
- final label schedule 在 processor freeze 后按真实 `n` 重新用 model operations 做四 worker LPT，
  与 raw-shard schedule 是两个正交对象。

## 两段隔离执行

### 1. Raw decode + OCR

使用显式 `--ocr-python-executable`。每个 selected shard 只归一个 worker；runner 逐 shard 完成：

1. size/SHA256 校验与顺序 Parquet decode；
2. selected raw-row/P0 identity 复核；
3. PNG/RGBA/opaque/no-EXIF 输入检查；
4. frozen RapidOCR CPU inference；
5. 只写 query-prefix substrate、validated OCR records 与必要 candidate/current images；
6. 释放该 shard 的 raw rows 后再处理下一个 shard。

不会把 raw row、`SelectedPilot.manifest`、当前 target action、terminal outcome、policy output、KL 或
utility 写入 artifact。anchor record 只保留自己的 causal prefix；同一 trajectory 的 terminal future
history 不会进入 anchor record。

### 2. AutoProcessor-only candidate freeze

使用显式 `--processor-python-executable`，唯一 pretrained loader 为：

```text
AutoProcessor.from_pretrained(
  local_files_only=True,
  trust_remote_code=False,
  min_pixels=2621440,
  max_pixels=2621440
)
```

每个 state 从 newest-16 non-current events 开始，用完整 summary prefix、全部 candidate 高保真图像和
current observation 运行 official-tools `apply_chat_template`。若 `input_tokens + 256 > 32768`，每次只
删除一个最老 candidate 并重跑；`n=4` 仍不 fit 时立即 fail closed，不继续试 `n<4`。

静态 AST 与运行时 `sys.modules` 双重拒绝 AutoModel、architecture modeling module、`forward` 和
`generate`。本 contract 也不授权 policy throughput pilot；后者必须等本步 commit 后另立 train-only
contract。

## 正式运行状态

唯一正式 run 已从 clean detached checkout
`b3472bfbd0a26541a8c6f2e207b88741e256f59a` 于 `2026-07-19 02:03:17 UTC` 在 Hyper00 启动。
运行身份如下：

- host alias/hostname：`hyper00` / `node-radixark-16-0000`；
- CPU-only container：`sglang-omni-jaxan-07181624`，完整 ID
  `a954543dd85b38a761a06a16384347169706e26e1bc34ffc8f8d279064cca50a`；
- image digest：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- output root：`/data/artifacts/causalcache-set-utility-processor-freeze-v2-b3472bf`；
- resumable staging：
  `/data/artifacts/.causalcache-set-utility-processor-freeze-v2-b3472bf.incomplete`；
- outer log/exit code：
  `/data/logs/causalcache-set-utility-processor-freeze-v2-b3472bf.{outer.log,exit_code}`；
- OCR runtime：Python `3.12.3`、RapidOCR `3.8.4`、PyArrow `24.0.0`；
- processor runtime：Python `3.12.3`、Transformers `5.6.0`、PyTorch `2.11.0+cu130`、
  Pillow `12.2.0`；
- Docker `DeviceRequests=null`，本步 GPU allocation 和 policy/model forward 均为 0。

启动命令的科学与身份参数全部显式传入：

```bash
env -u PYTHONPATH /usr/bin/python3 \
  /data/worktrees/causalcache-processor-freeze-b3472bf/code/scripts/run_set_utility_processor_freeze.py \
  --repository-root /data/worktrees/causalcache-processor-freeze-b3472bf \
  --execution-config /data/worktrees/causalcache-processor-freeze-b3472bf/code/configs/causalcache_set_utility_processor_freeze_execution_cf.json \
  --source-root /data/source/guiodyssey-full-pool-v1 \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --snapshot-manifest /data/worktrees/causalcache-processor-freeze-b3472bf/code/configs/gui_owl_1_5_8b_snapshot.json \
  --ocr-model-dir /data/artifacts/causalcache-ocr-ppocrv5-mobile-v1 \
  --ocr-wheel-dir /data/tmp/causalcache-ocr-v2-wheels \
  --output-root /data/artifacts/causalcache-set-utility-processor-freeze-v2-b3472bf \
  --ocr-python-executable /data/.venv/causalcache-ocr-v2/bin/python \
  --processor-python-executable /usr/bin/python3 \
  --git-revision b3472bfbd0a26541a8c6f2e207b88741e256f59a \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id a954543dd85b38a761a06a16384347169706e26e1bc34ffc8f8d279064cca50a \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --worker-count 4 \
  --ocr-python-version 3.12.3 \
  --ocr-runtime-version 3.8.4 \
  --pyarrow-version 24.0.0 \
  --processor-python-version 3.12.3 \
  --transformers-version 5.6.0 \
  --torch-version 2.11.0+cu130 \
  --pillow-version 12.2.0
```

启动窗口审计时四个 OCR worker 全部存活且无 error/traceback；`2026-07-19 02:08 UTC` 可完整读取
`30/2400` 个 partial query records，约 `244 MB`。这只是运行活性证据，不是结果 denominator。

## v1 正式终态

worker 0 于 `2026-07-19 02:14:20 UTC` 在第 228 个 selected observation 的 pre-OCR image validation
fail closed。该 source image 是合法 `PNG/RGB`、`2208x1840`、无 EXIF；v1 contract 只允许
`PNG/RGBA` 且 alpha extrema=`[255,255]`。首个 violation 的 selector identity SHA256 为
`fb6fd35ccfc826257b7fc60c348ac23a804dbffeaef9ee748d4851f3375ff2bc`，source image SHA256 为
`c850a794771657930b5694195afc2503c9339d47ece94e09d9350829c590b9c3`。

primary failure 已使 atomic publication 必然不可能；为避免另外三个 worker 继续数小时无效 CPU 工作，
它们随后收到 SIGTERM。outer process 于 `02:16:22 UTC` 以 code `1` 退出。正式 output root 未创建，
0 个 receipt、0 个 completed shard；8 个 staging files 共 `666,629,393` bytes 原样保留于 `.incomplete`，
tree SHA256=`9190b1b140de9b507b4b396694443a5e1293848ae9b02b1309ef81a53bc57e84`。

终态为 `INVALID_PROCESSOR_FREEZE_EXECUTION_CF_V1_IMAGE_CONTRACT_DRIFT`。AutoProcessor、policy/vision
forward、final candidates、restoration labels、training、matched-NLL、closed-loop 和 HF mutation 全为 0。
结构化 failure evidence 见
[`data/results/archive/set_utility_processor_freeze_execution_cf_v1_attempt/`](../../data/results/archive/set_utility_processor_freeze_execution_cf_v1_attempt/)。
下一步必须先对全部 18,792 selected images 做 read-only format census，再冻结 versioned repair；不得
修改原 v1 bytes、跳过 observation 或把 partial tar 追认为完成结果。
该独立 census 的 source/config 与执行边界见
[`set_utility_selected_image_format_census_v1.md`](set_utility_selected_image_format_census_v1.md)。它只读取
selected row image 的 Python field，不会把 processor v1 `.incomplete` 追认为有效 artifact，也不解锁
labels/training。其唯一 census v1 attempt 后续因 PyArrow 未传 `columns=["images"]`、实际物化完整 rows 而永久
`INVALID_SELECTED_IMAGE_FORMAT_CENSUS_V1_COLUMN_PROJECTION_CONTRACT_DRIFT`；observed histogram 不得用于
processor repair。v2 replacement 已从 clean `main@1a03b7e` 重跑完整 18,792 denominator，并通过 committed
postflight；正式分布为 18,768 opaque `PNG/RGBA` + 24 `PNG/RGB`。HF publication 已完成，revision=
`c1d19eb96d7fa7926f1eb9db3328dbff4e88eae0`。processor image-contract v2 已完成独立 source freeze；不得直接
修改本 v1 Execution-CF、重编码、补 alpha 或跳过 RGB observations。当前下一步是从 clean pushed v2
producer 执行 formal processor-only run。v2 合同见
[`set_utility_processor_freeze_execution_cf_v2_image_contract_repair.md`](set_utility_processor_freeze_execution_cf_v2_image_contract_repair.md)，census 证据见
[`set_utility_selected_image_format_census_v2_column_projection_repair.md`](set_utility_selected_image_format_census_v2_column_projection_repair.md)。

## 完成态只读 postflight

已新增 `code/scripts/validate_set_utility_processor_freeze_output.py` 与
`code/causalcache/set_utility_processor_postflight.py`。它只接受 atomic-published completed root，并从
frozen roster 重建四个 shard、candidate JSONL/final schedule、exact operation budget、run identity、OCR
format tally 与 Git-safe manifest；同时二次流式检查 tar 内每个 JSON member 的 canonical compact bytes、
成员顺序、trailing member、symlink/partial/extra file。

postflight + processor artifacts/freeze/contract focused suite 为 `40 passed`。该 validator 当前明确绑定
v1 `load_execution_contract` 和 canonical config，不声称自动支持尚未冻结的 v2 contract。v1 attempt
没有 completed output root，因此不能也没有被它追认为成功；未来 v2 repair 必须显式适配并重新测试。

## 外置 artifact 与恢复

正式 output 必须位于 repository、source、model 和 OCR inputs 之外的绝对 persistent path，且目标目录
不得预先存在。runner 使用 sibling `.incomplete` staging：

- 每个完成的 OCR worker 写一个 deterministic tar 与 receipt；
- 已完成且逐 byte 一致的 worker 可复用；
- final candidate JSON、exact subsets、worker schedule 与 operation budget 全部完成后，才 atomic rename
  为正式 output root；
- `manifest.json` 和未来 Git summary 不包含 raw instruction、OCR text、image 或 state ID。

本步骤禁止 Hugging Face mutation。v1 未产生正式 artifact，原计划 private dataset tag
`gavinlaw/causalcache-set-utility-new-development-mobile:phase1-b2-processor-freeze-v1` 没有 canonical
revision，终态为 `INVALID_FAILED_PRESERVED_NO_HF_PUBLICATION`。只有未来 versioned repair 完成并通过
独立只读 postflight 后，新的 output 才能进入 `PENDING_HF_UPLOAD`。

## 失败分类

- source/config/code hash drift；
- Python executable、package version、Git HEAD、container identity 或 snapshot drift；
- selected shard size/SHA/row count/raw identity drift；
- OCR input-format 或 OCR replay contract drift；
- anchor future-event leakage、query topology 或 current-equivalence drift；
- AutoModel/modeling/forward/generate 出现；
- `n=4` 仍超过 context；
- worker artifact/receipt 不完整或现有 bytes 不一致；
- denominator、role counts、candidate inventory 或 exact schedule drift。

任一失败都不得临时修改 threshold、候选规则、OCR normalization 或 state denominator 后续跑；需要保留
失败 evidence，并单独建立 versioned repair。
