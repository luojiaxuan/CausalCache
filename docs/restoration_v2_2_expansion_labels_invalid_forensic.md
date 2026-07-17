# Restoration v2.2 expansion exact-label v1 invalid forensic archive

> 当前状态：**source-only P0**。本协议只定义对原 v1 `INVALID` attempt 的只读、不可覆盖 transport
> archive；没有读取真实 Hyper output、没有生成正式 archive、没有访问网络或 Hugging Face，也没有授权
> gate training、matched-NLL 或 closed-loop。

## 目的与 claim 边界

原 expansion exact-label v1 的两个 worker 已完成 192/192 states，但 frozen monitor cadence gate 失败，最终
external global ledger 因此永久为 `INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`。root 内仍保留 validator 前写入的
`COMPLETED_EXPANSION_EXACT_LABEL_ATTEMPT` snapshot。这两个 ledger 都是历史事实，不能相互覆盖，也不能把
root 内部 snapshot 当成 terminal authority。

本协议只做三件事：

1. 原样保存 producer root 的全部 regular-file bytes；
2. 原样保存 root 外的 terminal global/even/odd high-water ledgers；
3. 用 deterministic USTAR 和自描述 manifest 建立可重复验证的 transport envelope。

它不把原 v1 追认为 PASS，不独立声称 scientific payload 已可供 formal gate 使用，也不替代后续 CPU-only
validation repair。

## 冻结 source contract

- config：`code/configs/causalcache_restoration_v2_2_expansion_labels_invalid_forensic_v1.json`；
- config SHA256：`5290a51a250e31be4fcf0a970c77ef31c08c92c5892edd19a12ecb14d9d5a6a2`；
- protocol：`causalcache_restoration_v2_2_expansion_exact_labels_invalid_forensic_v1`；
- artifact class：`invalid_attempt_forensics`；
- source status：`source_only_frozen_before_any_invalid_forensic_packaging_or_upload`；
- archive status：`ARCHIVED_INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`；
- formal label loader eligible：`false`；
- HF publish authorized：`false`。

真实 source binding 固定为 402 root files / 3,579,535 bytes，root sorted inventory canonical-JSON SHA256 为
`655c5d946b6a2fac448c2b144631f101fdf03d196945b8bea1f0047dd5916aba`。三份 external ledger 的 exact
path、size 与 SHA256，以及 root 内关键文件 hash，都写在 frozen config 中。

source-only contract 可离线验证：

```bash
cd /path/to/CausalCache/code

python3 -m scripts.manage_restoration_v2_2_expansion_labels_invalid_forensic \
  validate-contract \
  --repository-root ..
```

## 双 ledger namespace

archive member 只允许位于下面两个 payload namespace，manifest 单独位于 archive 根：

```text
restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-v1/
  forensic_manifest.json
  attempt_root/
    global_attempt_ledger.json
    worker_sibling_ledgers/even.json
    worker_sibling_ledgers/odd.json
    workers/even/worker_attempt_ledger.json
    workers/odd/worker_attempt_ledger.json
    ...其余 producer root regular files...
  terminal_external_ledgers/
    global_attempt_ledger.json
    workers/even.json
    workers/odd.json
```

validator 强制：

- `attempt_root/global_attempt_ledger.json` status 为 `COMPLETED...`；
- `terminal_external_ledgers/global_attempt_ledger.json` status 为 `INVALID...`；
- external INVALID ledger 的 `claimed_ledger_sha256` 精确等于 embedded completed ledger bytes 的 SHA256；
- 每个 worker 的 root ledger、root sibling copy 与 external high-water copy 三者逐 byte 相同；
- attempted/completed state inventories 分别精确为 even/odd 96-state shard，retry/top-up 为 0。

402 个 root files 加 3 个 external ledgers构成 405 个 payload members，再加 manifest 后固定为 406 个
regular-file members。任一 namespace collision、缺失、额外 member、symlink 或 non-regular entry都会失败。

## Append-only execution log chain

producer 在创建 `execution_evidence.json` 后才遇到 cadence failure，随后向 `logs/execution.log` 追加唯一
invalidation line。因此 final log 合法形式固定为：

```text
pre_failure_prefix || one_terminal_invalidation_line
```

冻结 binding 为：

- final log：292 bytes，SHA256
  `7c0e6a369652a188fda78d9bf5d3ade3bf15749150136bc0caa922e92d59bfed`；
- pre-failure prefix：162 bytes，SHA256
  `c765cd854bf5b76fd58a8b0065321998aa4f610d50da8a0cebe926a3a301a44a`；
- terminal suffix：130 bytes，SHA256
  `91d9a755bf78da36c0600f19d7284b4fe851a66eb154dc3cd9b901dcbf3ef1aa`。

validator 还要求 `execution_evidence.execution_log_sha256` 绑定 prefix，而不是错误地绑定 append 后的 final
log；suffix 必须是单行 UTF-8、LF 结尾，timestamp 不早于 external INVALID ledger terminal，failure message
必须精确等于 frozen cadence failure。

## Deterministic USTAR 与 no-overwrite

所有 member 按完整 path 排序，只允许 regular file，并固定：

```text
mode=0644, uid=0, gid=0, uname="", gname="", mtime=0, pax_headers={}
```

reader 会读取全部 members、验证顺序/metadata/path/duplicate，然后从解出的 bytes 重建整个 USTAR；重建 bytes
必须与输入 archive 完全一致。publisher 先把完整 archive 写入同目录临时文件并 `fsync`，重新读取全部 source
证明未变后，以 hard-link no-replace 原子发布并 `fsync` parent directory。目标已存在时永久拒绝覆盖。

正式 package 命令形状如下；当前 source-only P0 **没有执行该命令**：

```bash
cd /path/to/CausalCache/code

python3 -m scripts.manage_restoration_v2_2_expansion_labels_invalid_forensic \
  package \
  --repository-root .. \
  --raw-output-dir /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1 \
  --external-global-ledger /data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1.attempt.json \
  --external-even-ledger /data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1.even.attempt.json \
  --external-odd-ledger /data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1.odd.attempt.json \
  --output /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-v1.tar
```

## 尚未授权的下一步

P0 不包含 HF repo 创建、upload、tag 或 fresh download。后续若要发布 forensic archive，必须先另行冻结
crash-recoverable remote publication contract；不能复用原计划 PASS repo/tag/path，也不能因为 remote path 或 tag
已存在就静默覆盖。

forensic archive immutable fresh replay 闭合后，才可启动独立 CPU-only validation-repair child。formal loader 必须
先检查 manifest protocol/status/class 与 `formal_label_loader_eligible`；本 forensic protocol 在读取任何 label rows
之前就必须被拒绝。
