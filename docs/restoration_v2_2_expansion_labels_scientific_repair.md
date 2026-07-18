# Restoration v2.2 expansion-label scientific repair

> 状态：CPU-only validation core 与 no-GPU formal runner 已从 clean pushed source 完成正式 run 和完整只读
> revalidation；local repaired payload 为 `VALID + REVALIDATED`。独立 private-HF publication、幂等 immutable
> replay 与只读 postflight 已闭合，formal-58 label-data prerequisite 已满足；下游 train-only gate fit 已另行
> 闭合，见 [`../data/results/gate_v1_formal58_train_v1/`](../data/results/gate_v1_formal58_train_v1/)。原 producer
> attempt 永久 `INVALID`。

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

## no-GPU formal runner

core 不声明全局 GPU/model operation count 为 0。正式 runner 已把下列条件冻结为执行契约：

- new no-GPU container、Docker `DeviceRequests=[]`、`NVIDIA_VISIBLE_DEVICES=void`、无 `/dev/nvidia*`；
- `torch`、`transformers`、`accelerate`、`triton`、`bitsandbytes`、`xformers` 等 model/runtime import 被阻断；
- invalid forensic 输入来自 immutable HF commit 的新空目录 fresh download，并绑定已完成的 read-only
  tag-resolution child；
- claim/completion 使用新 versioned paths 和 `O_EXCL`/no-replace 状态机，原 P1/producer state 不变；
- repaired labels 使用 shard-oriented artifact，重新 fresh replay 后才可解锁 gate training。

machine-readable contract 为
`code/configs/causalcache_restoration_v2_2_expansion_labels_scientific_repair_runner_v1.json`，SHA256 为
`4f4202944674192090cf3df19b864c96e7082628ab25ca8287affa32944e64c2`。它固定：

- clean `main` checkout 必须满足 local HEAD = `origin/main` = 实时 `git ls-remote origin refs/heads/main`，且
  parent core commit `c537a1f1c12b9afe1b324cb481a9af7e596e9c5f` 是祖先；所有运行时加载的
  `causalcache.*` / `scripts.*` 模块必须是 Git tracked、非 symlink、且 bytes 等于该 HEAD；
- formal container 只能在 Hyper00 上以 unprivileged `runc`、空 Docker `DeviceRequests`、无 explicit device、
  `NVIDIA_VISIBLE_DEVICES=void`、空 `CUDA_VISIBLE_DEVICES` 和无 `/dev/nvidia*` 运行；bootstrap 在任何项目
  package import 前阻断 13 个 model-framework roots，但允许 PIL 做输入重放；
- invalid forensic 输入只从 private HF immutable commit
  `5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416` force-download 到新的空目录；remote mutation count 固定为
  0，并在 durable claim 前完成 private/identity/P0 strict readback；
- 原 producer root、P1 claim、缺失的 P1 completion、tag-resolution claim/completion 和本地 forensic archive
  在 formal run 前后都必须逐 byte 不变；
- 状态顺序固定为 `fresh immutable read -> claim -> full core recompute -> strict output readback -> completion`。
  completion 是唯一终态 mutation；`completion-without-output` 永久 invalid，`output-without-completion` 只能在
  同一 claim 下完整重算并逐 byte exact match 后恢复；
- deterministic USTAR 只有 `audit.json`、`derived_labels.jsonl`、`manifest.json`、`raw_states.jsonl` 四个成员。
  formal reader 从 raw rows 重新执行 projection validation、reducer 与独立 math audit，而不是相信 stored summary；
- 本 runner 只生成本地 scientifically valid child。它不授权 HF publication；gate、matched-NLL 与 closed-loop
  只有在新的 private HF repo 完成 archive+sidecar same-commit publication 和 immutable fresh replay 后才解锁。

formal output 与状态文件固定为：

```text
/data/artifacts/causalcache/restoration-v2-2-expansion-exact-labels-scientific-repair-v1.tar
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-scientific-repair-v1.claim.json
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-scientific-repair-v1.completion.json
```

独立 publication contract 已冻结，见
[`restoration_v2_2_expansion_labels_scientific_repair_publication.md`](restoration_v2_2_expansion_labels_scientific_repair_publication.md)，
config SHA256 为 `547b291022e2430fe6af5158d9350132f405c4b6019198488386dd7fb50be0aa`。目标 artifact
identity 是 private dataset
`gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile`，tag
`v2.2-expansion-exact-labels-scientific-repair-v1`。publication source freeze 不表示该 repo、tag 或 repaired
artifact 已存在。

formal CLI 必须直接执行 bootstrap script，不能用 `python -m`：

```bash
PYTHONPATH=code python3 \
  code/scripts/run_restoration_v2_2_expansion_labels_scientific_repair.py run \
  --repository-root /data/<clean-checkout> \
  --source-git-commit <clean-pushed-main-commit> \
  --launch-receipt \
    /data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-scientific-repair-v1.launch.json \
  --hf-token-file /data/.secrets/hf_key.txt \
  --fresh-download-parent /data/tmp/<new-empty-parent>
```

`validate` 使用同一显式参数并重新读取已完成的 claim/output/completion；它仍执行 immutable remote read，不能
被当作纯离线 shortcut。

## 当前验证

- core SHA256 / size：`6b1c022758844284a0c8364a33aa12457db1fa08dba9e5567c8fe3a48cfa744c` /
  47,044 bytes；
- focused scientific-repair tests：19/19；
- expansion-label wildcard tests：144/144；
- `py_compile` 与 `git diff --check`：通过；
- formal external replay 的路径探索 smoke 在禁用 GPU、阻断 model-framework imports 条件下得到 192 states / 1,792
  coalition witnesses，wall time 约 276 秒；该 smoke 不是 formal repair result。
- runner/config/CLI/test SHA256 分别为
  `f3e86d90488e5e86f6a04cf0d8122e353a9891693e7c68e7400339a791fe2205` /
  `4f4202944674192090cf3df19b864c96e7082628ab25ca8287affa32944e64c2` /
  `8d0f6434120c2969172841ef403c7e9d6f30add1f409fa3be68ae114b53f97b6` /
  `d05c6fbe4f6417834031a2bc9a3cb37f627d85a49786552b1a2b690b82277786`；
- runner focused tests 29/29、全部 expansion wildcard 173/173、direct-script source-only validation、
  `py_compile` 与 `git diff --check` 均通过；source-only status 为
  `VALID_SOURCE_ONLY_NO_GPU_SCIENTIFIC_REPAIR_RUNNER_V1`，其中 formal run、network、GPU/model operation 均为 0。

## 正式执行结果

formal run 从 clean pushed `main@b14f489fe55b51a83917b57e6a54fb73d268342a` 在 Hyper00
`node-radixark-16-0000` 执行。container `sglang-omni-jaxan-07172101` / ID
`e67bd5ef26b23014a530ade43dde239b366bd2bcc2f2c84a383ead2cf0f93ce9` 使用 image digest
`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`、unprivileged
`runc`、`DeviceRequests=null`、无 explicit devices、`NVIDIA_VISIBLE_DEVICES=void`、空
`CUDA_VISIBLE_DEVICES` 且无 NVIDIA device node。Python 为 3.12.3；forbidden framework imports 为空。

`run` 返回 `VALID_REPAIRED_EXPANSION_EXACT_LABEL_SCIENTIFIC_PAYLOAD_V1`，随后 `validate` 再次从
immutable HF fresh-download、完整重跑 core/reducer/external replay/math audit，并返回
`REVALIDATED_REPAIRED_EXPANSION_EXACT_LABEL_SCIENTIFIC_PAYLOAD_V1`。后者的 claim/output/completion
`created=false`，三者 bytes 未改变。

- archive：3,020,800 bytes，SHA256
  `1a9fdcc08aeb83f88bcd50957c3d890e3b103f2063a2aecbe08610d450950e01`，tree
  `09bb681b2715997df326ab9648d8501955ec6ba127631a97141484631a98f44b`，exact 4 members；
- claim：mode 0600，7,014 bytes，SHA256
  `f2a19d6f2b444eb000027abc8b2358e409be22713870bb7e84e62a82998d70f5`；
- completion：mode 0600，2,524 bytes，SHA256
  `6a4775357de392aef5d9ce7d9999db6cfd3a27f8948fd532e9341c7acacb4905`；
- source inventory 为 11 paths / `3ac6fc21...f4c3`；loaded project closure 为 43 modules /
  `6b5e5bd9...6c06`；local HEAD、origin/main 与实时 remote main 全部为同一 source commit；
- 192 states、1,792 raw distances、1,856 deployment edges、3,072 full edges、1,984 interactions、576
  attributions、192 exact oracles 全部 exact；独立 math projection SHA256 为 `b16d7d26...c99d9`；
- 原 3 秒 cadence negative control 精确失败：5/3,849 intervals 超限、max 3.883721 秒；新 structural lifecycle
  validation PASS 且 `acceptance_uses_numeric_gap_threshold=false`；
- repair runtime 的 GPU/model/policy/teacher/KL/gate/matched-NLL/closed-loop/confirm/test/remote-mutation counts
  全部为 0。原 P1 completion 仍缺失，producer 没有被重分类。

轻量结果见
[`data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_v1/`](../data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_v1/)。
local archive 只是 staging copy；canonical reusable artifact 已发布到 private HF repo
`gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile`，tag
`v2.2-expansion-exact-labels-scientific-repair-v1` 的 resolved immutable commit 为
`7a6c254b8cec0dd3d8111dfc9c080de357e5cef3`。exact-pair upload、annotated tag、幂等 replay 与独立只读
postflight 已闭合，结果见
[`data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/`](../data/results/restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/)。
publication 只解除 formal-58 label-data prerequisite；不能修改已被 config 绑定的本 scientific summary，也不
代表 gate 已训练。
