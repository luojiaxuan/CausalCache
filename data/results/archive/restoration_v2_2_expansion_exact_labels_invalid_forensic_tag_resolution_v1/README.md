# Invalid-forensic read-only tag-resolution v1

正式 read-only child 已完成，并在同一 clean source 上做了第二次幂等 replay。它闭合的是永久
`INVALID_EXPANSION_EXACT_LABEL_ATTEMPT` 证据的 private-HF transport，不是 expansion label 的科学有效性。
原 P1 completion 在两次 replay 前后均不存在，remote mutation call count 始终为 0。

## 执行与 source binding

- status：`COMPLETED_READ_ONLY_INVALID_FORENSIC_TAG_RESOLUTION_V1`；
- Git source：`main@d61dff5857bee7815da0c40709d0e7032fb9dc86`；
- config SHA256：`f257dcdebb218533a080739e8a5067cc44fe35fd533ad57808db5ce789cfd956`；
- host/container：Hyper00 `node-radixark-16-0000` / `sglang-omni-jaxan-07171437`；
- container image ID：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- Python / `huggingface_hub`：`3.12.3` / `1.16.1`；
- argv：`PYTHONPATH=. python3 -m scripts.manage_restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution validate-remote --repository-root /data/CausalCache-invalid-forensic-v1 --fresh-download-parent /data/tmp --hf-token-file /data/.secrets/hf_key.txt`；
- claim/completion UTC bracket：`2026-07-17T09:37:01.465982928Z`--`2026-07-17T09:37:02.891999696Z`；
- 本 child 没有 GPU/model operation；运行容器虽然可见 GPU 2/3，但 validator 只执行 Git、HF read 与 stdlib
  archive validation。

## Immutable HF 证据

- private dataset：
  <https://huggingface.co/datasets/gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-invalid-attempts-mobile>；
- tag：`v2.2-expansion-exact-labels-v1-attempt-1-forensic-v1`；
- annotated-tag object：`ca652858c3d59eab066eb7b31399690a65ea5f44`；
- tag-resolved / immutable pair commit：`5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416`；
- direct predecessor：`1e1bb828d196779e4fc855890922c4865fcd6458`；
- archive SHA256 / size：`8e205d73196604c0d8d342f9415755b090b96e83555ca55c386b12b8427ca489` /
  3,880,960 bytes；
- sidecar SHA256：`c67914d05ce5e93518c710cd99132548f4bf5b51250e506c996a9450a369b561`；
- 406 members，tree inventory SHA256
  `bc481dd77e26764dad3458e91a3e0247ade6049af7adb1744672aa6f2fb437d1`；
- identities、pair provenance、private visibility 与原 P1 completion absence 在 fresh immutable download 前后
  完全相同；P0 strict archive readback 通过。

## Local state 与结论边界

- child claim：mode `0600`、3,205 bytes、SHA256
  `3dd268f7cac895609a08b1a5c532a0992099ecdc63e409eb6bc365fa39bf3a15`；
- child completion：mode `0600`、3,971 bytes、SHA256
  `c0f70c53da30517c6625e90931f128a39b235a4613439ddb0520d01d48612195`；
- 第二次 read-only replay 返回同一 completion，两个 state file 的 bytes 与 SHA 均未变化；
- `formal_label_loader_eligible=false`、`gate_training_unlocked=false`、
  `producer_attempt_reclassified=false`。

因此 invalid evidence 的 reusable transport 已闭合；下一步是独立 CPU-only scientific repair，以新 artifact
identity 重算标签并发布 immutable fresh replay。机器可读记录见 [`summary.json`](summary.json)。
