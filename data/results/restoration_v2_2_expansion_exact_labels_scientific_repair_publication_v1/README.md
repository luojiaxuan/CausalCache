# Repaired expansion exact-label publication v1

## 结论

scientific-repair archive 与 sidecar 已由 clean pushed
`main@e47c50665c24fed5d9886665233962f85c1210e0` 发布到新的 private Hugging Face dataset，并通过同一
completion 的幂等 replay 和独立只读 postflight。正式状态为
`COMPLETED_REPAIRED_EXPANSION_LABELS_PUBLICATION_V1`。

这次里程碑只解除 formal-58 的 **label-data prerequisite**：
`formal_label_loader_eligible=true`、`gate_training_unlocked=true`，但 gate 仍未训练；matched-NLL、
closed-loop 与 confirm/test 仍未解锁。原 GPU producer attempt 继续永久为
`INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`，invalid-forensic archive 也继续 formal-ineligible。

## Canonical Hugging Face identity

- private dataset：
  <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile>；
- tag：`v2.2-expansion-exact-labels-scientific-repair-v1`；
- annotated-tag object：`6a907ba2a3dce07c9f0810a84a8755c389da4a57`；
- tag-resolved / immutable pair commit：`7a6c254b8cec0dd3d8111dfc9c080de357e5cef3`；
- archive path：
  `repaired/v2.2-expansion-exact-labels-scientific-repair-v1/repaired-labels-v1.tar`；
- archive：3,020,800 bytes，SHA256 / LFS oid
  `1a9fdcc08aeb83f88bcd50957c3d890e3b103f2063a2aecbe08610d450950e01`，exact 4 members；
- sidecar path：
  `repaired/v2.2-expansion-exact-labels-scientific-repair-v1/artifact-manifest-v1.json`；
- sidecar SHA256：`9d86c75597a2a65eae952dd311373a7adbb0536d56d88e7fdb00d674f8dc4ee5`。

remote base `1a070e603639ace6ab713eaba039a8d751310c25` 到 pair commit 只新增上述两个目标文件；base 的完整递归
blob inventory 与 pair 的 non-target blob inventory SHA256 均为
`ef4c5eb5f271fd7b091958e9ea938179febd1ae883e813743742cbaeafd22f95`。pair commit title 精确为
`Publish repaired expansion exact-label scientific payload`。

## 执行与 replay

- host/container：Hyper00 `node-radixark-16-0000` / `sglang-omni-jaxan-07172101`；
- runtime：Python 3.12.3、`huggingface_hub==1.16.1`、CPU-only、GPU/model operation count 0；
- source config SHA256：`547b291022e2430fe6af5158d9350132f405c4b6019198488386dd7fb50be0aa`；
- 首次 publication bracket：`2026-07-17T14:21:08.324233422Z`--
  `2026-07-17T14:21:14.296303376Z`；
- completion：mode 0600、7,948 bytes、SHA256
  `995b3ee25b643550de3df680b52382ef8b6b1373a3aad4c00bdc9caa34cfa2ff`；retained staging 与 final
  completion 为同一 inode；
- 第二次 invocation 返回相同 completion，completion bytes 和所有本地 state identities 未变化，且未调用
  remote mutation；
- 独立 postflight 从新空目录 force-download，重新验证 private visibility、tag/main/revision、exact pair、两个
  remote blobs/LFS identity 与 strict 4-member USTAR，返回
  `PASS_INDEPENDENT_READ_ONLY_REPAIRED_PUBLICATION_POSTFLIGHT_V1`，且未执行 remote mutation。

本地 `/data` 文件只保留为 staging/audit evidence，不是 reusable source of truth。机器可读证据见
[`summary.json`](summary.json)。下一步只能按已冻结的 gate v1 contract 构建 formal-58 cache、训练和评估 gate，
不能把本次 publication 写成 learned-gate 效果。
