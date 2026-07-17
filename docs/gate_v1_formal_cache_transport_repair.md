# Formal-58 cache transport-identity repair

> 当前状态：transport-repair 已完成并经只读 replay 再验证。Source-A=
> `4f8c01b026167d6e9429716a082f3abd7c0c1bc9` 与其唯一 direct-child Execution-B=
> `f96c197c0fd31bfd299b5ab6e9e1416f6183bc6d` 均已 push。Hyper00 no-GPU `run` 返回
> `VALID_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`；相同 B 的只读 `validate` 返回
> `REVALIDATED_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`，没有创建 claim/cache/state，remote mutation count 为
> 0。repair cache 已成为 formal-58 的 train-only input prerequisite，但没有 gate training、OOF、checkpoint、
> matched-NLL、closed-loop 或 confirm execution。

## 为什么必须新开 repair namespace

旧 formal-58 cache v1 的 Source-A=`990f015b02e0dc89dedaab3853fab9f07fb0884d` 与唯一 Execution-B=
`079c0952017a9e2936f5741bbe15345255b0e481` 已经完成 commit/push。Hyper00 的唯一 CPU-only invocation 在
`_verify_downloads()` 创建 global claim 后、任何 semantic decode 之前 fail closed。失败不是数据 bytes 改变，
也不是 model 或 gate 的负结果；它是原 config 中一个合法 64-hex digest 的 transcription error。

因此 repair 不得：

- 覆写、删除或续跑旧 v1 global claim；
- 修改 parent config、原 Execution-B、原失败 summary 或 producer artifact；
- 将旧 namespace 的缺失 successor state 当作可重新创建的状态；
- 把 repair 当作 v1 的 retry，或把 source-only validation 写成执行成功。

repair 的唯一目标是以独立的 local/HF namespace，重新验证同一冻结输入并物化相同 schema 的 train-only cache。
它不改变 formal roster、selective semantic allowlist、cache schema、teacher labels、gate objective 或下游 evaluation
contract。

## 唯一允许改变的 input leaf

repair config 是 parent `causalcache_gate_v1_formal_cache_v1` 的显式 overlay。除下表这一项外，所有
`input_artifacts` 记录会被归一化回 parent 后逐字节比较；`data_bytes_changed=false` 与
`semantic_contract_changed=false` 都是硬约束。

| 字段 | 冻结值 |
| --- | --- |
| input key | `expansion_feature_trajectories` |
| artifact | `expansion_derived_features` |
| path | `derived/restoration-v2-label-expansion-v1/trajectories-00000-of-00001.jsonl` |
| size | `1,245,673` bytes（不变） |
| v1 错误 SHA256 | `00fe93e9deeeb9a3c018227fb781b29f6efefbb728db987584b650d9df353a6d` |
| repair 正确 SHA256 | `fe93e9deeeb9a3c018227fb781b29f6efefefbb728db987584b650d9df353a6d` |

两者均为长度正确的 64-hex 字符串；错误值是在开头误插 `00`，并漏掉真实 digest 中相邻重复的第二个 `fe`。
corrected value 必须同时匹配 Git 固定 producer artifact
`data/results/restoration_v2_2_label_expansion_derived/artifact.json` 的以下三处同一 file record：

- `artifact.fresh_download.files`；
- `artifact.preupload.files`；
- `payload_manifest_snapshot.payload_files`。

这条三重 witness 与 Hyper00 的旧 attempt failure summary 一起构成 repair 的证据，而非一项事后接受规则。

## 旧失败边界必须在新 claim 前验证

repair `run`/`validate` 在读取 token、构造 Hub client、下载输入、创建 repair global claim 前，必须先验证：

| 旧 v1 evidence | 要求 |
| --- | --- |
| old global claim | `/data/experiments/causalcache/.gate-v1-formal58-cache-v1.claim.json` 仍为 mode `0600`、7,062 bytes、SHA256 `a9372c7a4b606323b60e27b8984c5c6d456d62ad16c297fb4737c3d4abd54b1b` |
| old execution provenance | B=`079c095…0e481` 是 A=`990f015…0884d` 的 direct child，旧 runner freeze 的 bytes/SHA 仍匹配 |
| failure evidence | Git-tracked summary 仍为 `INVALID_PRE_SEMANTIC_GATE_V1_FORMAL58_CACHE_ATTEMPT_V1`，并绑定上述错误/正确 digest、size 与 pre-semantic boundary |
| old successor states | feature completion、label-access claim、label completion、remote-base receipt、staged/final completion 均不存在 |
| old cache artifacts | `/data/artifacts/causalcache/gate-v1-formal58-feature-cache-v1.tar` 与 `...label-cache-v1.tar` 均不存在 |

任一旧 evidence 漂移、旧 successor 被创建或旧 artifact 出现，repair 都必须 fail closed。这个 precondition 防止
repair 无意中把 v1 失败路径恢复为可写路径。

## 独立 namespace 与不变量

repair 使用新的 local state、artifact path、private dataset repo、tag 和 exact-three remote targets：

| 层级 | repair identity |
| --- | --- |
| local claim 前缀 | `.gate-v1-formal58-cache-transport-repair-v1.*` under `/data/experiments/causalcache/` |
| local feature cache | `/data/artifacts/causalcache/gate-v1-formal58-transport-repair-feature-cache-v1.tar` |
| local label cache | `/data/artifacts/causalcache/gate-v1-formal58-transport-repair-label-cache-v1.tar` |
| private HF repo | `gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile` |
| tag | `gate-v1-formal58-cache-transport-repair-v1` |
| exact-three targets | `formal58-transport-repair/v1/{feature-cache-v1.tar,label-cache-v1.tar,cache-bundle-manifest-v1.json}` |

repair 保留 v1 的 deterministic USTAR/readback、feature/label 物理分离、train-only semantic firewall、HF
exact-three atomic commit、annotated tag 与 fresh immutable replay。新 namespace 不是放宽 validator 的理由：
partial/conflicting remote state、split commit/tag、non-identical existing local state 或非空 fresh replay directory
都必须 fail closed。

## 已完成的 execution record

repair source 与执行闭包如下；B 相对 A 的唯一 source-tree diff 是机械生成的 runner freeze：

| 记录 | Immutable identity |
| --- | --- |
| Source-A | `4f8c01b026167d6e9429716a082f3abd7c0c1bc9` |
| repair config | `code/configs/causalcache_gate_v1_formal_cache_transport_repair_v1.json`，SHA256 `aaf82fd5e994588bc22f0f745139e349219863eee5fa8cb4cc93c4a9e48e123a` |
| Execution-B | `f96c197c0fd31bfd299b5ab6e9e1416f6183bc6d` |
| repair runner freeze | `code/configs/causalcache_gate_v1_formal_cache_transport_repair_runner_v1.json`，SHA256 `14141d0d790c4cf4d6c12b3bc336e1e6d2ebd26fde5f4be40b58627bd86a2b34` |
| canonical HF dataset | [`gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-formal58-cache-transport-repair-mobile)，tag `gate-v1-formal58-cache-transport-repair-v1` → immutable commit `a61b31bf2e69be00f94469f4a2f2d6b336fcc386` |
| annotated tag object | `c603397b9b1b1c3ad472f49125f827b8a096997d` |

immutable exact-three remote targets 与 strict local readback 的 SHA256 为：

| Target | SHA256 | Size |
| --- | --- | ---: |
| `formal58-transport-repair/v1/feature-cache-v1.tar` | `81fded50c4450700220742d3e0be9a5585d1bc51086150515b463bbdf4b4df8e` | 993,280 bytes |
| `formal58-transport-repair/v1/label-cache-v1.tar` | `4f9ef172aaa94c3ea8ce53aa43336c9fcb7800c24e181e1462d8239e31053cee` | 163,840 bytes |
| `formal58-transport-repair/v1/cache-bundle-manifest-v1.json` | `15c8bf56ddad4f6f278599c32aaadcd813e0e016db0d523b4892eb47f8d9d144` | manifest |

run 先完成 retained v1 failure boundary 的 pre-claim validation，然后物化 58 trajectories、174 feature/label
states、522 candidate features、1,624 raw distance values、1,682 conditional targets、522 independent targets 与
174-state join audit。semantic allowlist 只解码 58 trajectories、290 OCR records、174 label states；development/
confirm semantic decode、model load/forward、optimizer step、training example、OOF metric、checkpoint、matched-NLL
与 closed-loop operation count 均为 0。fresh immutable replay 后最终 completion 与 retained staging 为 same inode。

紧随其后的 `validate` 是只读 replay：它返回
`REVALIDATED_GATE_V1_FORMAL58_CACHE_PUBLICATION_V1`，没有创建新的 global claim、feature/label cache 或 local state，
且 remote mutation count 为 0。完整 compact evidence 位于
[`../data/results/gate_v1_formal58_cache_transport_repair_v1/`](../data/results/gate_v1_formal58_cache_transport_repair_v1/)。

## Source-A → Execution-B 的已完成 provenance

Source-A 的 canonical config 是
`code/configs/causalcache_gate_v1_formal_cache_transport_repair_v1.json`。source-only validator 只验证 source
inventory、parent/failure/producer binding 与 repair overlay；它不读 token、不访问网络或 formal artifact、不写 state，
且不授权 GPU、model、policy、semantic decode 或 training。

在 clean pushed repair Source-A 上，唯一允许的 source-tree 变更曾是机械物化：

```text
code/configs/causalcache_gate_v1_formal_cache_transport_repair_runner_v1.json
```

该文件是 repair Execution-B 的唯一 source-tree diff。formal execution 已证明：B 是 A 的 direct child、
`HEAD == origin/main == live origin/main == B`、runner source inventory 与 A 的 Git blobs 一致，且所有 loaded
`causalcache.*`/`scripts.*` module 都来自已绑定 source bytes。已完成的 `run` 按以下顺序前进：

```text
validate retained old-v1 failure boundary
  -> create repair global claim
  -> feature cache/readback/completion
  -> label-access claim
  -> label cache/readback/completion
  -> join audit
  -> exact-three HF publish + annotated tag
  -> fresh immutable replay
  -> retained staged completion -> same-inode final completion
```

上面的流程现已发生；其 compact completion record、cache hashes、runtime、argv、operation counts 与 read-only
revalidation status 由结果目录中的 `summary.json` 固定。已完成 state 只能通过 committed `validate` replay，不能从
较新的 source commit 重新生成。

## 已解锁范围与仍然锁定的工作

final completion 令 `formal58_cache_loader_eligible=true` 与 `formal58_training_input_eligible=true`：它只使这个
immutable cache bundle 可作为已 preregistered 的 train-only gate workflow 的输入。它不表示 gate 已训练，也没有
产生 OOF/model-selection result、checkpoint、development metric、fresh-16、旧 dev-5、matched-NLL、closed-loop 或
confirm/test output。后者仍需各自的 source/execution contract；transport repair 不会自动授权或重算任何下游结果。
