# Restoration v2.2 expansion-label scientific repair core

> 状态：CPU-only validation core 已实现并通过 synthetic/test-only regression；formal runner、durable
> claim/completion、repaired-label artifact 与 immutable HF fresh replay 尚未冻结或执行。本 core 不能单独解锁 gate。

## 目标与边界

原 expansion exact-label v1 已完成 192/192 scientific payload，但 source-locked 3 秒 monitor cadence gate 在
3,849 个 intervals 中有 5 个超限，因此 producer attempt 永久为
`INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`。repair 不重跑 GPU、不改阈值、不删改 ledger，也不把 v1 追认为 PASS。

本 core 只回答一个 versioned child 问题：在 exact P0 forensic bytes 上，能否保持原失败为 negative control，
改用不依赖 scheduler jitter 的 structural monitor lifecycle rule，并重新验证 raw labels、reduction、外部输入与
provenance。正式状态为：

```text
PASS_LEDGER_NEUTRAL_EXPANSION_LABEL_SCIENTIFIC_REPAIR_V1
```

只有 exact P0 source、production external replay 和 frozen provenance 全部通过时才可能得到该状态；synthetic
fixture 只能得到：

```text
TEST_ONLY_LEDGER_NEUTRAL_EXPANSION_LABEL_SCIENTIFIC_REPAIR_V1
```

## 输入契约

公开入口为
`validate_ledger_neutral_scientific_payload(evidence, forensic_contract, external_input_replay_callback, external_replay_attestation)`。

1. `evidence` 必须是 P0 strict reader 返回的 `InvalidForensicEvidence`，不是任意 path/dict；
2. `forensic_contract` 必须重新构建为同一 exact contract；formal tier 额外要求 config SHA256
   `5290a51a250e31be4fcf0a970c77ef31c08c92c5892edd19a12ecb14d9d5a6a2` 与 forensic tree SHA256
   `bc481dd77e26764dad3458e91a3e0247ade6049af7adb1744672aa6f2fb437d1`；
3. formal external replay 固定 parent substrate archive、derived expansion artifact、OCR backend config 与 production
   replay implementation；test-only tier 只能用显式 fixture attestation；
4. core 输入全程在内存中只读，进入与退出 inventory 必须相同。

formal external provenance 独立冻结为：

- parent substrate：private HF revision `25ac19cf6ef98adc243d421cd0039ac104ddb539`，archive SHA256
  `4e77a38be34cb2f3c084a13abd47c0530e6729ff6cca72793977c62fa78ff47d`，tree
  `56f291053121ccf813698a48beb6269b1fa1d096b7e974f6eb9424f55bc45543`；
- derived artifact：private HF revision `630363a6adb692d72774f16dd0653a50216313ff`，tree
  `9394b369e2b741e6aacf9ece4fc5dae3e6337b25a7e65402307e4e7862b94abc`；
- OCR config：6,922 bytes，SHA256
  `51e08a9565f804ecb1ee7f12887cc12742b84c04996fffbbde7c0f304e4bd036`；
- production replay module：119,937 bytes，SHA256
  `a1ba86e672a51dcde807fdb2d844874696ba0e9a35839f6e631390a7ef0b275d`；callable source SHA256
  `94887261ce11326e6aaf81a28e791cbee4e6200c6eeb339e4f655f277b891f1c`。

## 验证顺序

1. 用 supplied P0 contract 重新运行完整 `validate_invalid_forensic_files`；
2. exact compare rebuilt dataclass、manifest、inventory 与 tree，确认 formal-ineligible/producer-INVALID declarations；
3. 分离 `attempt_root/**` 与 `terminal_external_ledgers/**`，验证 completed-to-invalid global ledger chain 和三份
   worker ledger byte equality；
4. 只在内存中剥离 execution log 最后一条 invalidation suffix，验证 suffix 时间不早于 terminal invalid ledger，
   并确认 execution evidence 绑定 pre-invalidation prefix；
5. 原冻结 v1 validator 必须以 exact cadence error 再次失败；若意外 PASS 或以别的原因失败，repair fail closed；
6. 对 producer view 重验 run contract、state schedule、GPU binding metadata、192 raw states、1,792 distances、stored
   aggregate 与 independent stdlib math audit；
7. formal tier 在任何 external replay 之前检查 exact P0 config/tree、production callable identity/source/code 与独立
   parent/derived/OCR provenance；然后直接调用 import-time captured production replay，而不动态 dispatch 可被
   monkeypatch 的 executor `__call__`；
8. 对 192 state projections / 1,792 coalition witnesses 做 byte-identical external replay，formal callback 前后再次
   检查 provenance globals；
9. 退出前重新比较 forensic inventory，确认 bytes modified=false、ledger mutation count=0。

## Monitor repair rule

旧 cadence 公式仍被完整重算并报告，不隐藏负结果：ready→first sample、全部相邻 samples、last sample→summary、
first claim、last worker terminal、stop request 与 terminal tail 全部进入 diagnostics，3 秒 legacy gate 仍为 false。

新 acceptance 只要求：

- ready、JSONL、stop 与 summary identity/hash 一致；
- sample indices/timestamps 连续且严格递增；
- monitor 在 first state claim 前已经 ready/sample；
- last sample、stop 与 summary 都覆盖 last worker terminal；
- runtime UUID 与 physical GPU inventory 与原 run contract 一致。

新 rule 明确记录 `acceptance_uses_numeric_gap_threshold=false`。它修复的是 validation semantics，不是降低、修改或
重新解释原 3 秒阈值。

## 仍由 formal runner 负责

core 不声明全局 GPU/model operation count 为 0。下一层 runner 必须另外证明：

- new no-GPU container、Docker `DeviceRequests=[]`、`NVIDIA_VISIBLE_DEVICES=void`、无 `/dev/nvidia*`；
- `torch`、`transformers`、`accelerate`、`triton`、`bitsandbytes`、`xformers` 等 model/runtime import 被阻断；
- invalid forensic 输入来自 immutable HF commit 的新空目录 fresh download，并绑定已完成的 read-only
  tag-resolution child；
- claim/completion 使用新 versioned paths 和 `O_EXCL`/no-replace 状态机，原 P1/producer state 不变；
- repaired labels 使用 shard-oriented artifact，重新 fresh replay 后才可解锁 gate training。

## 当前验证

- core SHA256 / size：`6b1c022758844284a0c8364a33aa12457db1fa08dba9e5567c8fe3a48cfa744c` /
  47,044 bytes；
- focused scientific-repair tests：19/19；
- expansion-label wildcard tests：144/144；
- `py_compile` 与 `git diff --check`：通过；
- formal external replay 的路径探索 smoke 在禁用 GPU、阻断 model-framework imports 条件下得到 192 states / 1,792
  coalition witnesses，wall time 约 276 秒；该 smoke 不是 formal repair result。
