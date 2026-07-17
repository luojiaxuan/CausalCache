# Gate v1 formal-58 train-only cache

> 当前状态：Source-A=`990f015` 与 Execution-B=`079c095` 已冻结并 push。首次 Hyper00 CPU-only run 创建 global
> claim 后，在 feature transport byte check 发现 expansion trajectories 的合法 64-hex SHA transcription mismatch
> 并 fail-closed；尚未执行任何 semantic decode，也没有 feature/label cache、HF repo/tag/completion、gate training、
> OOF、checkpoint、development metric、matched-NLL、closed-loop 或 confirm access。旧 claim 永久保留；只有新的
> versioned transport-repair Execution-B 完成 immutable HF fresh replay 与 final completion 后，formal-58 才能作为
> training input。失败证据见 `data/results/gate_v1_formal58_cache_v1_attempt/`。

## 目标与边界

本步骤把已经闭合的四类 immutable input 投影为两个物理分离的 train-only cache：

1. feature cache：只含 join identity、`q64`、candidate `h64/g8`；
2. label cache：只含 join identity、candidate geometry 与完整 raw $D(S)$。

它不训练 gate，不计算 OOF/development metric，不构造 checkpoint，也不访问 fresh-16、legacy dev-5、
confirm-20、matched-NLL 或 closed-loop。marginal、independent projection、ranking pair 与 330/200 维 model input
仍由已冻结 gate code 在训练阶段从两个 cache join 后派生，不写入 cache。

## 冻结输入

| 逻辑输入 | Immutable identity | Formal train slice |
| --- | --- | ---: |
| legacy features | `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@89f136abaff797e14fe758a198996e51032a10a6`，prefix `derived/restoration-v2-v1` | 前 10 trajectories |
| expansion features | 同一 repo `@630363a6adb692d72774f16dd0653a50216313ff`，prefix `derived/restoration-v2-label-expansion-v1` | 前 48 trajectories |
| legacy exact labels | `gavinlaw/causalcache-restoration-labels-mobile@8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`，`raw/v2.2-eager-train-dev-exact-v2.tar` | states `0..29` |
| repaired expansion labels | `gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile@7a6c254b8cec0dd3d8111dfc9c080de357e5cef3` | `raw_states.jsonl` 前 144 行 |

repaired input 还必须联合验证本地 publication completion SHA256
`995b3ee25b643550de3df680b52382ef8b6b1373a3aad4c00bdc9caa34cfa2ff`、Git publication result 与 remote
sidecar SHA256 `9d86c75597a2a65eae952dd311373a7adbb0536d56d88e7fdb00d674f8dc4ee5`。archive 或 sidecar
单独存在都不能解锁 formal cache。

gate preregistration SHA256 固定为
`37be1ff7bf52fd425be85a6407100a47ec6edd724b4c1e93ddcf1b6c93e3ab1b`。formal roster 只能是 legacy
train-10 后接 expansion train-48；source order、五折 assignment 与全部 digest 必须机械重放。

## Train-only semantic firewall

四个 source artifact 都把 train 与 development 共置在同一 transport 中，因此必须区分：

- `transport_byte_possession`：为了 immutable SHA/size/USTAR 验证，可以持有或流式 hash 整个文件；
- `semantic_record_decode`：只允许对 frozen formal-train allowlist 调用 JSON decoder。

禁止调用会完整解码输入的 generic readers。专用 selective reader 固定：

- legacy trajectory JSONL 只 decode 前 10 行，expansion trajectory JSONL 只 decode 前 48 行；
- OCR 先做 byte-level path routing，只 decode 58 条 selected trajectories 实际引用的 290 个 records；
- legacy label USTAR 只 extract/decode state indices `0..29`；
- repaired `raw_states.jsonl` 只 decode 前 144 行；`derived_labels.jsonl` 不作为训练 teacher；
- development、confirm 与 test semantic decode/access count 全部为 0。

`low_fidelity_v2` 经 canonical `sort_keys` JSON roundtrip 后字段顺序会变成字母序，因此 reader 只校验 exact
key set，并始终按 `LOW_FIDELITY_V2_KEYS` 的冻结顺序 hash。label adapter 也必须显式选择 schema：legacy 使用
`trajectory_id / coalition_event_step_ids / distance_kl`，expansion 使用
`source_id / coalition / distance`；禁止 alias fallback 或 mixed schema。

## 固定分母与 cache schema

| 项目 | legacy train-10 | expansion train-48 | formal-58 |
| --- | ---: | ---: | ---: |
| trajectories | 10 | 48 | 58 |
| states | 30 | 144 | 174 |
| candidate occurrences | 90 | 432 | 522 |
| raw $D(S)$ values | 280 | 1,344 | 1,624 |
| conditional targets | 290 | 1,392 | 1,682 |
| independent targets | 90 | 432 | 522 |

每个 cache 是独立 deterministic USTAR，sorted exact members、mode 0644、uid/gid 0、mtime 0、空
uname/gname、无 pax/symlink/hardlink：

```text
feature cache: manifest.json + feature_states.jsonl
label cache:   manifest.json + label_states.jsonl
```

向量与 distance 不直接依赖 JSON decimal float：先编码为 big-endian IEEE754 binary64 的 16 位 lowercase hex；
readback 拒绝 NaN、Inf、negative zero、非 canonical hex 或 roundtrip drift。feature payload 禁止 `D(S)`、
marginal、oracle、role、split、raw instruction/OCR/low-fidelity text；label payload 禁止 `q64/h64/g8`、semantic
text、role 与 split。两个 archive 的 path、inode、bytes 与 SHA 必须不同，174 个 join keys、state geometry 与
formal source-major step-4/5/6 order必须完全一致。

## Source-A / Execution-B

Source-A 包含 scientific config、contract validator、selective core、runner、CLI、tests 与本文档，但不包含 generated
runner freeze。source-only validator 不接受 token/data/output 参数，不访问网络或 formal artifact，不写 claim/cache，
并返回 `execution_authorized=false`。

Source-A commit/push 后，从 clean pushed A 机械生成唯一 runner-freeze 文件：

```text
code/configs/causalcache_gate_v1_formal_cache_runner_v1.json
```

该文件单独作为 Execution-B commit/push。正式 `run` 只允许在 clean
`HEAD == origin/main == git ls-remote origin/main == B` 上执行，并证明 A 是 B 的直接科学 ancestor、A 到 B 只新增
runner freeze、所有 loaded modules 与 A 的 Git blobs 完全一致。

Source-A push 后的机械入口为：

```bash
cd code
PYTHONPATH=. python3 -m scripts.validate_gate_v1_formal_cache_contract \
  --repository-root .. \
  --contract code/configs/causalcache_gate_v1_formal_cache_v1.json
PYTHONPATH=. python3 -m scripts.manage_gate_v1_formal_cache \
  materialize-runner-freeze \
  --repository-root .. \
  --contract code/configs/causalcache_gate_v1_formal_cache_v1.json \
  --source-a-git-commit <SOURCE_A_COMMIT>
```

第二条命令只允许生成 runner-freeze 这一个未跟踪文件；提交并 push B 后才能调用 `run`。

## Local-first / HF-second 状态机

Execution-B 的状态顺序固定为：

```text
global claim
  -> feature cache strict readback + feature completion
  -> durable label-access claim
  -> label cache strict readback + label completion
  -> 174-state join-only audit
  -> HF exact-three atomic commit + annotated tag
  -> fresh immutable two-cache replay
  -> retained final-completion staging
  -> same-inode final hard-link
```

feature completion 通过后才能写 label-access claim；该 claim durable 落盘后才允许下载并验证 label transport，
随后才可 semantic-decode formal-train allowlist。label-access claim 落盘前，label transport download 与 semantic
decode count 都必须为 0。所有 local state 都是 mode 0600 no-replace file；existing
byte-identical state 可恢复，non-identical、缺 predecessor、partial output 或 split remote commit 均 fail closed。
final completion 先保留 staging，再用 no-replace hard-link 创建；stage/final 必须 same inode，final link 与目录
fsync 是首次成功执行的最后一次 namespace mutation。

planned canonical destination：

```text
private dataset: gavinlaw/causalcache-gate-v1-formal58-cache-mobile
tag: gate-v1-formal58-cache-v1
feature: formal58/v1/feature-cache-v1.tar
label: formal58/v1/label-cache-v1.tar
sidecar: formal58/v1/cache-bundle-manifest-v1.json
```

三个 target 必须由一个 frozen-title commit 原子新增；base history/tree 与 pair 后的 non-target blob inventory 必须
完全相同。annotated-tag object 与 tag-resolved commit 分开记录。只有 fresh immutable replay 严格重建两个 cache
和 joint manifest 后，final completion 才可声明 `formal58_training_input_eligible=true`。该声明仍不授权 fresh-16、
legacy dev-5、confirm、matched-NLL 或 closed-loop。

## Runtime 与 operation boundary

正式 materialization 固定在 Hyper00 no-GPU、unprivileged `runc` container：`DeviceRequests=[]`、无 NVIDIA device
node、`NVIDIA_VISIBLE_DEVICES=void`，不 import `torch`、`transformers` 或 policy/model runtime。cache stage 的
training example、optimizer step、model/policy/teacher forward、oracle/development metric、checkpoint、matched-NLL、
closed-loop 与 confirm/test operation count 全部为 0。

该 CPU/network job 不申请 GPU；执行前仍检查 host、disk、container 与 source checkout。正式命令必须以 committed
CLI 和 runner freeze 为准，不能用临时 Python 直接调用 gate core。
