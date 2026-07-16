# Spatial reference audit v1

本目录保存 spatial reference audit v1 的轻量正式结果索引；完整 profile、terminal、ledger 与运行日志不进入
Git，而由下述 private Hugging Face dataset artifact 保存。

## 正式结论

结论为 `EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`：

- `bf16_auto` 的 exact generated-token / canonical-action stability 均为 7/13；运行
  138.249687 秒，peak allocated GPU memory 为 22,299,479,040 bytes；
- `bf16_eager_control` 的两项 stability 均为 13/13；运行 461.792909 秒，peak allocated GPU memory 为
  91,697,764,864 bytes；
- `fp32_eager_control` 为 4/4；运行 306.077164 秒，peak allocated GPU memory 为
  99,273,054,208 bytes。该 profile **只作 descriptive probe，不参与 decision**。

全次审计共执行 60 次 generation、120 次 teacher forward；confirm access、restoration coalition construction
和 gate training 均为 0。validation repair 逐项验证 60 个 generation shape node 与 120 个 teacher shape
node，实际每图 effective visual token 取值为 2,516、2,560、2,584。

raw audit source commit 为 `c093bd8f92ab97427acb427bd2d66fb6b20b556a`；只读、零 forward 的
validation repair commit 为 `a2528d7e95e73c25639568650f63abce58e4e491`。raw `summary.json` SHA256 为
`1d6dc90c602a3cb97ff58da8fafe03eefe31b19d1673084a132cc328c9229bf4`，其 scientific payload SHA256
为 `2d73e395d5ee91349adf904b344b7ae25569e79ce8bfa35be9df7902ab8dc79e`。

## Canonical artifact

- private HF dataset：
  <https://huggingface.co/datasets/gavinlaw/causalcache-spatial-reference-audit-mobile>；
- immutable revision：`d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`；
- tag：`spatial-reference-audit-v1`；
- path：`raw/spatial-reference-audit-v1.tar`；
- deterministic USTAR：1,873,920 bytes、72 members，SHA256
  `d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc`；
- tree inventory SHA256：`6423c13f4b7067f5a2139442bab0be11a1ef5da9f880f51a1131768afa963b82`；
- `manifest.json` SHA256：`37f89b40e5a75b6c3073d32b2d3b4aeff90c07b3ada20075a21e428d99cbcf1a`；
- HF `README.md` SHA256：`c8d86701f6eb28c43e181b386a77fee7cfe69a0070228adac255be82690a4c92`。

已从该 immutable revision fresh download，并验证 private visibility、仅含 `README.md`、`manifest.json`、
`raw/spatial-reference-audit-v1.tar` 的 exact allowlist、逐文件 hash、archive inventory/tree，以及 canonical
USTAR rebuild 一致性。

## Claim 与下一步边界

该结果只说明本次冻结 GUI-Owl reference workload 的 exact stability 在 BF16 eager control 下恢复；不声称
strict CUDA determinism，也不把 FP32 probe 作为因果证据。父实验的
`NO_GO_V2_1_FULL_45_SUBSTRATE` **保持不变**，本次 13-state audit 不能改写它。

下一步从新的 clean、pushed commit 冻结 `v2.2-eager` source，再运行新的 45-state attempt；不得把 v2.1 或
本审计 raw state 当作 v2.2 state evidence 复用。
