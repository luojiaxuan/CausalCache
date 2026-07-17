# Expansion exact-label scientific repair v1

## 结论

Hyper00 上的正式 no-GPU child 从 clean pushed `main@b14f489fe55b51a83917b57e6a54fb73d268342a`
完成，`run` 返回 `VALID_REPAIRED_EXPANSION_EXACT_LABEL_SCIENTIFIC_PAYLOAD_V1`，随后完整只读重算返回
`REVALIDATED_REPAIRED_EXPANSION_EXACT_LABEL_SCIENTIFIC_PAYLOAD_V1`。

这证明原 GPU attempt 中已归档的 192-state scientific payload 能在不加载模型、不使用 GPU、不修改原 ledger 的
条件下，通过 structural monitor lifecycle rule、production external-input replay、原 reducer 和独立 stdlib math
audit。它不追认原 attempt：producer 仍永久为 `INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`，旧 3 秒 cadence
negative control 仍精确复现 5/3,849 个超限 interval，max 3.883721 秒。

## 本地 artifact

- deterministic USTAR：3,020,800 bytes，SHA256
  `1a9fdcc08aeb83f88bcd50957c3d890e3b103f2063a2aecbe08610d450950e01`；
- exact 4 members，tree inventory SHA256
  `09bb681b2715997df326ab9648d8501955ec6ba127631a97141484631a98f44b`；
- claim：mode 0600，7,014 bytes，SHA256
  `f2a19d6f2b444eb000027abc8b2358e409be22713870bb7e84e62a82998d70f5`；
- completion：mode 0600，2,524 bytes，SHA256
  `6a4775357de392aef5d9ce7d9999db6cfd3a27f8948fd532e9341c7acacb4905`；
- `validate` 完整重算后 claim/output/completion 均未创建或改写，所有 SHA、size 与 tree 保持相同。

scientific denominator 为 64 trajectories、192 states、1,792 raw distances、1,856 deployment edges、3,072
full edges、1,984 interactions、576 exact-permutation attributions 和 192 exact-subset oracles。原 producer 的
1,984 teacher forwards / 1,792 KL measurements 只作为 archived provenance；本 repair 的 GPU、model、policy、
teacher、KL、gate、matched-NLL、closed-loop、confirm/test 和 remote-mutation operation count 全部为 0。

## 边界与下一步

Hyper00 路径
`/data/artifacts/causalcache/restoration-v2-2-expansion-exact-labels-scientific-repair-v1.tar` 只保留为 staging。
archive 与 sidecar 后续已同 commit 发布到 private HF dataset
`gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile`；tag
`v2.2-expansion-exact-labels-scientific-repair-v1` 的 resolved immutable commit 为
`7a6c254b8cec0dd3d8111dfc9c080de357e5cef3`。幂等 replay 与独立只读 postflight 已闭合，正式 publication 记录见
[`../restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/`](../restoration_v2_2_expansion_exact_labels_scientific_repair_publication_v1/)。
这只令 `gate_training_unlocked=true`；gate 尚未训练，matched-NLL、closed-loop 与 confirm 仍 locked。本阶段的
完整轻量证据见 [`summary.json`](summary.json)。
