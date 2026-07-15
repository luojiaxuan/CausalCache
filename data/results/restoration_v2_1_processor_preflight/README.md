# Restoration v2.1 processor preflight

本目录保存 v2.1 fixed-15 interface pilot 的 processor-only 前置证据。raw prompt/input IDs 不进入 Git；Git
只保存 private Hugging Face immutable artifact 的轻量 binding 与 compact reduction。

## 正式结果

- verdict：`PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT`；
- source Git commit：`d0205afd789cda602fbf8964505d9de9a1b1fe53`；
- host/container：Hyper00 `node-radixark-16-0000`，
  `69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df`；
- container image digest：
  `sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- UTC bracket：`2026-07-15T21:36:49.347422Z` -- `2026-07-15T21:37:48.936167Z`，耗时
  `59.588757` 秒；
- prompts：45 states × 2 fidelity = 90；official tools 注入 90/90；
- image counts：1-image 45，3-image 15，4-image 15，5-image 15；
- input token length：最小 3,759，最大 15,013；context overflow 0；最大值加 256 generation budget 后
  为 15,269，小于 32,768；
- execution boundary：CPU-only；policy model 未实例化、weights 未 materialize 为 tensors，GPU operation、
  forward、generate、policy/restoration output 均为 false；完整 artifact（含 confirm bytes）仅被 loader
  validation/hash，进入 decoder/processor 的 confirm state/prompt/image 为 0。

关键 reduction hashes：

- processor classes：`d968f9dc8ad020b04852d35912016f90266c0c642c919295c5db59e96b4f341a`；
- prompt records：`12ee2eebcf7532833f0f589a468654e7f899d21be3ef6825cebb62a8c501502e`；
- shape records：`c8c645965676c054aab35d8deee5d44e30316f48d3f8e386e448cd67d154abf8`；
- teacher golden records：`91807002b322a720683638fdea55bd4cea9624b67495ede4837d7c9352b164c9`。

## Source of Truth

- private HF dataset：`gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile`；
- tag：`v2.1-processor-preflight-v1`；
- immutable revision：`85576161b7cb8bbae14e46a482c42b5be5bf1d7e`；
- path：`processor-preflight-v1/formal-result.json`；
- raw SHA256：`5349ffc6b91bf93ed26d25104fe6c907f31ccc497007a5c0ae6ed5ddf5c84991`；
- raw size：7,609,803 bytes；
- Git binding：`artifact.json`，SHA256
  `2689fe984215f308e5f5230de9656cfe0439bfd2df5e5f061ebac1a1cbc8299b`。

上传后已按 40-hex immutable revision fresh download；下载件的 byte SHA256/size 与正式 raw 完全一致，HF API
也确认 tag 指向上述 revision。`artifact.json` 只含 hashes、revision、protocol/source identity 与 compact
reduction，不含 raw prompts、input IDs、screenshots、model weights 或 policy output。

## Claim 边界

该 PASS 仅证明冻结的 official-tool prompts 能被 pinned `AutoProcessor` 在 context 预算内一致构造，并授权在
manifest commit/push 与 reuse validation 后启动唯一 fixed-15 pilot。它不证明模型会输出可解析 tool call，
不证明 repeat stability、finite KL、memory sensitivity、restoration oracle 或论文主张，也不授权 confirm、
teacher、KL 或 restoration。
