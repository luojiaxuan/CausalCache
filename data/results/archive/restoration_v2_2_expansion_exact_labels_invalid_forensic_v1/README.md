# Expansion exact-label v1 invalid forensic archive

本目录记录原 v1 `INVALID_EXPANSION_EXACT_LABEL_ATTEMPT` 的只读 transport archive。它只保全历史 bytes，
不会把原 attempt 追认为 PASS，也永远不能直接作为 formal gate input。

## 当前状态

clean pushed `main@bb73bfee899f436ad292afb5d2d603d706001a5f` 在 Hyper00 对冻结 source 做了三次一致性
读取（collect、pre-publish、post-publish），随后生成并 readback 验证 deterministic USTAR：

- 402 个 `attempt_root/**` files；
- 3 个 `terminal_external_ledgers/**` files；
- 1 个 `forensic_manifest.json`；
- 共 406 regular-file members；
- archive SHA256：`8e205d73196604c0d8d342f9415755b090b96e83555ca55c386b12b8427ca489`；
- archive size：3,880,960 bytes；
- tree inventory SHA256：`bc481dd77e26764dad3458e91a3e0247ade6049af7adb1744672aa6f2fb437d1`。

archive 的原始 staging copy 仍保留在 Hyper00：
`/data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-v1.tar`。
相同 bytes 已固定到 private Hugging Face dataset 的 immutable revision
`5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416`，并由 read-only child 完成 fresh replay；HF 是该 reusable
invalid-evidence artifact 的 canonical source of truth。它仍永久 formal-ineligible。

## 后续链路

独立的 crash-recoverable private-HF publication contract 已冻结，详见
[`docs/archive/restoration_v2_2_expansion_labels_invalid_forensic_publication.md`](../../../../docs/archive/restoration_v2_2_expansion_labels_invalid_forensic_publication.md)。
唯一 P1 invocation 已把相同 archive SHA 与 sidecar manifest 作为一个 commit 上传，并创建 no-overwrite tag；
但 annotated-tag object SHA 与 tag-resolved commit SHA 的接口差异使 v1 parser 在 completion 前 fail closed，见
[`publication v1 attempt`](../restoration_v2_2_expansion_exact_labels_invalid_forensic_publication_v1_attempt/README.md)。
独立 read-only tag-resolution child 已完成 immutable fresh replay、P0 strict readback 与幂等复验，见
[`result`](../restoration_v2_2_expansion_exact_labels_invalid_forensic_tag_resolution_v1/README.md)。随后独立 CPU
scientific validation repair 与新的 repaired-label private-HF publication 均已闭合，见
[`publication result`](../restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/README.md)。该链路只
解除 formal-58 label-data prerequisite；本 invalid-forensic artifact 仍永久 formal-ineligible，gate 尚未训练。

机器可读记录见 [`artifact.json`](artifact.json)。
