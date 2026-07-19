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

## 外置 artifact 与恢复

正式 output 必须位于 repository、source、model 和 OCR inputs 之外的绝对 persistent path，且目标目录
不得预先存在。runner 使用 sibling `.incomplete` staging：

- 每个完成的 OCR worker 写一个 deterministic tar 与 receipt；
- 已完成且逐 byte 一致的 worker 可复用；
- final candidate JSON、exact subsets、worker schedule 与 operation budget 全部完成后，才 atomic rename
  为正式 output root；
- `manifest.json` 和未来 Git summary 不包含 raw instruction、OCR text、image 或 state ID。

本步骤禁止 Hugging Face mutation。正式 artifact 产生后先持久化于 Hyper00 `/data`，状态记为
`PENDING_HF_UPLOAD`，计划上传 private dataset
`gavinlaw/causalcache-set-utility-new-development-mobile` 的 versioned processor-freeze tag；上传成功前，
本地路径只是 staging，不是 canonical publication。

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
