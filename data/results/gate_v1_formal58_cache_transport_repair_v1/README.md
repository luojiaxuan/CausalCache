# Gate v1 formal-58 transport-repair cache：COMPLETED + REVALIDATED

## 结论

formal-58 train-only cache 的 v1 invocation 保持永久 `INVALID`；它的旧 claim 没有被删除、覆盖或续跑。
独立 transport-repair 从 Source-A=`4f8c01b026167d6e9429716a082f3abd7c0c1bc9` 机械生成唯一
Execution-B=`f96c197c0fd31bfd299b5ab6e9e1416f6183bc6d`，只修复
`expansion_feature_trajectories` 的一个 SHA256 transcription leaf：

```text
wrong:     00fe93e9deeeb9a3c018227fb781b29f6efefbb728db987584b650d9df353a6d
corrected: fe93e9deeeb9a3c018227fb781b29f6efefefbb728db987584b650d9df353a6d
size:      1,245,673 bytes (unchanged)
```

2026-07-17 的 Hyper00 no-GPU formal `run` 返回
`VALID_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`，随后相同 B 的只读 `validate` 返回
`REVALIDATED_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`。revalidation 未新建 claim/cache/state，remote mutation
count 为 `0`。两个 run 都在任何新 claim 前复核了 retained v1 claim、六个旧 successor 的缺失、两份旧 cache 的
缺失及 Git-pinned producer 三重 transport witness。

这闭合的只是 formal-58 的 train-only cache prerequisite：`formal58_training_input_eligible=true`。它**不等于**
gate 已训练，也没有运行 OOF、checkpoint、fresh-16、旧 dev-5、matched-NLL、closed-loop 或 confirm。

## Canonical artifacts

| Artifact | Canonical location | Immutable identity |
| --- | --- | --- |
| feature/label cache + bundle manifest | private HF dataset [`gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile) | tag `gate-v1-formal58-cache-transport-repair-v1` → commit `a61b31bf2e69be00f94469f4a2f2d6b336fcc386`; annotated-tag object `c603397b9b1b1c3ad472f49125f827b8a096997d` |
| compact completion record | Git: [`summary.json`](summary.json) | source B `f96c197…bc6d`; config SHA256 `aaf82fd5…123a` |

HF immutable exact-three targets are:

```text
formal58-transport-repair/v1/feature-cache-v1.tar
formal58-transport-repair/v1/label-cache-v1.tar
formal58-transport-repair/v1/cache-bundle-manifest-v1.json
```

Their SHA256 values, local state hashes, runtime, argv, operation counts and log provenance are in
[`summary.json`](summary.json). Local `/data/artifacts/...` copies and `/data/logs/...` logs are staging/audit copies,
not canonical reusable artifacts.

## Reproduction boundary

The source and execution protocol is documented in
[`docs/gate_v1_formal_cache_transport_repair.md`](../../../docs/gate_v1_formal_cache_transport_repair.md). A completed
state may only be replayed through its committed `validate` command; it must not be regenerated from a newer source
commit. The next scientific step is the already-preregistered train-only OOF/model-selection workflow, with a separate
source/execution record and without opening fresh-16 before its allowed phase.
