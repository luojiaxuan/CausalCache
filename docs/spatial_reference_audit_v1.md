# Spatial reference audit v1

## 当前结论

`spatial_reference_audit_v1` 是一个冻结在任何新 audit policy forward 之前的受限数值审计。它只重用
v2.1 已经暴露的 13 个 coordinate mismatch states，不向 processor/decoder/model/teacher 暴露 confirm，不构造
restoration coalition，不训练
gate，也不允许据此调 coordinate radius。父结论
`NO_GO_V2_1_FULL_45_SUBSTRATE` 永久保留；本审计不是对 v2.1 的事后重算。

唯一 Hyper01 attempt 已完成全部三个 profile，但原 independent validator 在 summary 写入前 fail closed：它把
processor target `2560` 误当成固定 realized grid token 数，并错误要求 teacher aligned inputs 包含主
`input_ids`。已有 profile/state evidence 未删除、未重跑；纯离线 repair 的证据与边界见
[`spatial_reference_audit_v1_validation_repair.md`](spatial_reference_audit_v1_validation_repair.md)。repair、artifact
packaging 和 immutable HF verification 现已闭合，正式 decision 为
`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；它不改写 v2.1 NO-GO，也不授权 restoration/confirm。

源合约位于 [`code/configs/spatial_reference_audit_v1.json`](../code/configs/spatial_reference_audit_v1.json)，
父 mismatch 投影位于
[`data/fixtures/spatial_reference_audit_v1_parent_mismatches.json`](../data/fixtures/spatial_reference_audit_v1_parent_mismatches.json)，
append-only exposure child ledger 位于
[`data/manifests/spatial_reference_audit_v1_exposure.json`](../data/manifests/spatial_reference_audit_v1_exposure.json)。旧
[`restoration_v2_exposure.json`](../data/manifests/restoration_v2_exposure.json) 保持 byte-immutable。

## 只回答一个问题

v2.1 的 13 个 failure 全部是同 action type 下的 coordinate jitter（12 个 click、1 个 swipe）。本审计只回答：

> 在不改变 prompt、processor、action grammar、decoding 和样本分母的前提下，显式 eager numerical control
> 能否让这 13 个 state 的两次 generation 恢复 exact token stability？

它不回答两个坐标是否指向同一个 UI element，也不把小距离解释为 semantic equivalence。

## 冻结输入与父证据限制

父 raw archive 固定为 private HF dataset
`gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile@814506ef1450838d4bc6ed3d89fe53e0773d92fb`
中的 `raw/restoration-v2-1-full-45-substrate-v1.tar`，archive SHA256 为
`8cd53d6e56d5ad509da2af91d73d9e83b4db989ffc26aa18e1bca84e4c4f4fa4`。runner 会重验完整 archive、
45-state 父结果、13-state failure set、各 state member SHA、两条 action/text/token hash 和固定 canonical
projection SHA。

独立 validator 不把三份 profile terminal 中复制的父字段当作证据：它会自行重新打开固定 raw archive，重算
45-state failure set、13 个 member SHA、两条 raw output/action/token hash 与 canonical projection，再把这些值逐字段
绑定回每条新 record。父 archive 整体 SHA 正确但 profile 内父字段被替换，同样必须失败。

旧 raw evidence 只保存 output text、canonical action 和 generated-token SHA256，没有保存 token IDs、generation
scores 或 full logits。因此：

- first-divergent generated token 只能用 pinned tokenizer 重编码，并先逐条通过旧 token SHA；
- top-1/top-2 margin 不能声称是旧 generation-time margin；
- 不允许把 `output_scores=True` 加进 primary generation，因为这会改变被审计的执行路径；
- margin 只在两条父 action 的共同 teacher-forced prefix 上 post-hoc 测量，不进入 pass/fail。

## 固定 profile 与调用预算

三个 profile 必须在同一 H200、同一 device、同一 container、同一 clean pushed source commit 上分别启动，且不允许
静默 fallback：

| Profile | Dtype | Attention | Eager numerical controls | States | Generation | Teacher forward |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `bf16_auto` | BF16 | checkpoint/default | off | 13 | 26 | 52 |
| `bf16_eager_control` | BF16 | eager | on | 13 | 26 | 52 |
| `fp32_eager_control` | FP32 | eager | on | 4 sentinels | 8 | 16 |
| 总计 |  |  |  |  | 60 | 120 |

FP32 sentinels 固定为 indices `10,17,26,39`，覆盖最大 click delta、最小非零 click delta、最大 prompt/visual
load 与唯一 swipe。FP32 只作描述性 probe，不控制结论。

严格 `torch.use_deterministic_algorithms(True)` 在 CUDA GEMM 上需要 `CUBLAS_WORKSPACE_CONFIG`；项目规则禁止用
environment variable 传 scientific 参数，因此本协议明确不启用也不声称 strict CUDA determinism。eager control
只固定 seed、cuDNN deterministic on/benchmark off、TF32 off、float32 matmul precision `highest` 和 eager
attention，并记录全部实际 flags。它回答的是受控 eager backend 的 repeat stability，不是数学确定性证明。

正式 attempt 还逐字段冻结并在 durable claim 前核对：container image digest
`sha256:6a8f60af...d349acfa`、Python `3.12.3`、PyTorch `2.11.0+cu130`、PyTorch CUDA
`13.0`、cuDNN `91900`、Transformers `5.6.0` 与 NVIDIA driver `570.172.08`。`bf16_auto` 必须实际观察到
至少一个 non-null 且全部非 eager 的 attention implementation；两个 eager profile 的所有 non-null 观察值必须
都是 `eager`，否则在任何 state forward 前失败。
正式设备名称在 runner 与 validator 两侧都严格固定为 `NVIDIA H200`；包含该子串但名称不同的 SKU 不会先运行再
在聚合阶段失败。

cache 与 device routing environment variables 不在 scientific audit 范围；可能改变 CUDA 数值、attention backend
或执行调度的固定名单必须全部 absent。名单为 `CUBLAS_WORKSPACE_CONFIG`、`NVIDIA_TF32_OVERRIDE`、
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`、`PYTORCH_CUDA_ALLOC_CONF`、`PYTORCH_ALLOC_CONF`、
`CUDA_LAUNCH_BLOCKING`、`CUDA_DEVICE_MAX_CONNECTIONS`、`FLASH_ATTENTION_DETERMINISTIC`、
`TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT`、`TORCH_CUDNN_V8_API_DISABLED` 与
`TORCH_CUDNN_V8_API_ENABLED`。runner 和 runtime 都记录 exact audited names、empty present names 与
`all_absent=true`；不记录 environment value。

## 每个 state 保存什么

- 两条父 action 的 normalized coordinate delta、AndroidWorld bridge pixel delta 与 L2；
- pinned tokenizer 重编码后的父 token IDs、SHA 验证和 first-divergent token；
- 每个 profile 的两次原生 greedy generation、完整新 token IDs 与 exact token/action stability；
- 在两条父 canonical action 的共同 prefix 上做两次相同 shape 的单次 forward，并从同一 logit vector 报告两个
  竞争 token 的 raw / generation-aligned logit、log-prob、rank、top-1/top-2 和 pairwise margin；
- 每条父 action 另做一次 full-branch teacher forward，只作 shape-specific branch diagnostic；
- latency、peak GPU memory、host/container/runtime identity 和精确 operation counts。

full-vocabulary logits 不允许传到 host；只传 target/top-2 和最终 diagnostic scalars。
独立 validator 还要求两次 shared-prefix forward 的 image count、`image_grid_thw`、effective visual tokens、
policy-visible text tokens、prompt tokens 与 aligned-input inventory exact 相同，并验证视觉 token accounting。
两条 full branch 分别绑定其完整 action-token 长度，但 base image/prompt shape 必须与 generation/shared-prefix
完全相同。

## 唯一 decision rule

判定只读取两个 BF16 profile 的 exact generated-token repeat stability：

- BF16 eager 非 13/13：`PERSISTENT_INSTABILITY_REQUIRES_SEMANTIC_REFERENCE`；
- BF16 eager 为 13/13、BF16 auto 非 13/13：`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；
- 两者均为 13/13：`BOTH_BF16_PROFILES_STABLE_THIS_ATTEMPT_NUMERICAL_AUDIT_INCONCLUSIVE`。

最后一种情况只说明本次两个 backend 都稳定，不能把稳定性归因给 eager，也不能声称 recovery。FP32 probe、margin
大小和 action-level stability 永不控制判定。

## Artifact 与 source of truth

三份 raw profile JSON 与 repaired aggregate 的 canonical 位置是 private HF dataset
[`gavinlaw/causalcache-spatial-reference-audit-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-spatial-reference-audit-mobile)：

- immutable revision：`d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`；
- tag：`spatial-reference-audit-v1`；
- path：`raw/spatial-reference-audit-v1.tar`；
- deterministic USTAR：1,873,920 bytes、72 members、SHA256
  `d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc`；
- tree inventory SHA256：`6423c13f4b7067f5a2139442bab0be11a1ef5da9f880f51a1131768afa963b82`。

tag 已通过 API exact 解析到上述 40-hex revision；随后从全新 cache 强制下载，验证 private visibility、
README/manifest/raw exact allowlist、archive bytes、canonical USTAR rebuild、member tree、summary 与 scientific
payload hash。Git 只保存 source/config/tests、exposure ledger、compact summary/artifact manifest 和进展结论。

本地 archive 路径固定为 `/data/experiments/causalcache/spatial-reference-audit-v1.tar`。冻结 packager 只接受
canonical audit root 与 root 外 sibling ledger；二者会放在同一 `spatial-reference-audit-v1/` USTAR prefix 下，
ledger 的 member name 固定为 `global_attempt_ledger.json`。目录树中的 symlink 与任意 non-regular file 都会
fail closed。成员排序、mode `0644`、uid/gid/mtime `0` 固定，archive 必须在内存构造后和 exclusive-create 写盘后
各做一次 member 读取与 canonical-byte rebuild；已有 output 不覆盖、不删除、不重试。

聚合前，validator 必须逐个读取 `start.json`、三个隐藏的 profile claim、三个 profile start 和每个 state start，
检查 exact key schema、固定顺序、完整 argv、canonical path、预期 operation counts，并与 sibling ledger、state/profile
terminal 逐字段闭合。只保留 terminal、删除或改写任一 pre-forward durable record 的目录不能成为有效 artifact。

## 后续 semantic reference 边界

若数值审计未恢复 exact stability，v2.2 需要先冻结三个接口：

```text
action_to_semantic_key(image, action)
semantic_key_to_canonical_action(image, key)
actions_equivalent(image, action_1, action_2)
```

它必须使用确定性 target key 和唯一 canonical representative，而不是 pairwise radius。OCR 只允许 unique polygon
membership；多重命中、坏 polygon 或无法保持 swipe motion bin 时 fail closed。无 OCR 命中时使用 GUI-Owl 的 merged
vision cell，不使用 raw patch grid。该协议必须先通过 synthetic 邻接控件 negative fixture，再在独立
AndroidWorld UI-node calibration artifact 上报告 false-positive；后者属于 reusable data artifact，必须进 HF。

v2.2 的 fixed 45 只作 protocol development/screening，不能冒充 independent validation。未来 gate 应报告固定
45 分母、parse coverage、semantic stability、trajectory coverage、canonical finite logits 与 memory sensitivity；
confirm 20 仍是唯一 untouched policy-output evidence，全部架构和阈值冻结前不得打开。

## Exact restoration 计数更正

只有 v2.2 substrate 通过后才允许 exact restoration。当前 states 的 candidate event 数随 decision step 变化：

| Decision step | Candidate events | `B=2` 可行 subset 数 |
| ---: | ---: | ---: |
| 4 | 2 | 4 |
| 5 | 3 | 7 |
| 6 | 4 | 11 |

每条 trajectory 各含 step-4/5/6，因此 30 个 label-train states（10 trajectories）的 exact table 是
`10 × (4+7+11) = 220` rows；15 个 development states（5 trajectories）是 `5 × (4+7+11) = 110`
rows；合计 330，而不是把所有 45 states 都按 11 个 subset 计算得到的 495。任何 exact restoration 仍须等
semantic/deterministic substrate gate 通过后另行冻结 source、预算与 go/no-go。

## 执行顺序

1. source-only 合约、父证据、唯一 no-retry attempt ledger 和 exposure child ledger commit/push；
2. Hyper01 单张 H200 preflight、10 秒 idle cleanup、同 container 三 profile 顺序运行；
3. 独立 validator 聚合，deterministic USTAR packager 封装 root + sibling ledger，再上传 private HF 并 fresh
   immutable download 复核；
4. Git 回写 compact result、artifact revision、失败模式和下一步，commit/push；
5. 按唯一 decision rule 进入 deterministic runtime 或另一个 source-only semantic protocol。

loader 仍会读取并 hash-validate 包含 confirm bytes 的完整 derived artifact；这里的零 confirm 声明特指
confirm state/prompt/image 对 processor/decoder/model/teacher 的可见计数为 0。任何一步都不生成 confirm policy
output，不运行 restoration 或 gate training。
