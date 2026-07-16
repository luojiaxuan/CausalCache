# Spatial reference audit v1：离线验证修复

## 状态

`spatial_reference_audit_v1` 的唯一 Hyper01 policy attempt 已从 clean pushed
`main@c093bd8f92ab97427acb427bd2d66fb6b20b556a` 完成全部冻结 forward：

- `bf16_auto`：7/13 exact generated-token stable；
- `bf16_eager_control`：13/13 exact generated-token stable；
- `fp32_eager_control`：4/4 exact generated-token stable，仅作描述性 probe；
- 总计 60 次 generation、120 次 teacher forward；
- confirm、restoration coalition、gate training 均为 0。

三个 profile terminal 与 sibling attempt ledger 都已 durable terminal，ledger 状态为
`COMPLETED_SPATIAL_REFERENCE_AUDIT_ATTEMPT`。原 formal wrapper 随后在独立 validator 阶段以 exit code 1
fail closed。随后从 clean pushed `main@a2528d7e95e73c25639568650f63abce58e4e491` 执行唯一一次纯离线
repair，exit 0；72-member USTAR 与 private HF immutable fresh-download 也已闭合。正式 decision 是
`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`，但原 v2.1 NO-GO 不变。

## 失败原因

旧 validator 把 processor 配置中的每图 target `2560` effective visual tokens 错当成每张实际图都必须
恰好产生 `2560` tokens。真实 processor 会根据 resize 与 grid rounding 产生多个合法 realized grid：

| `image_grid_thw` | merge 后每图 tokens |
| --- | ---: |
| `[1, 148, 68]` | 2516 |
| `[1, 80, 128]` | 2560 |
| `[1, 136, 76]` | 2584 |
| `[1, 152, 68]` | 2584 |

这是 post-hoc validator contract bug，不是 policy raw shape 漂移。只读诊断确认：

- raw 中 180 组 shape metadata 的
  `effective_visual_tokens = sum(t × h × w / merge_size²)` 全部成立；
- `prompt_input_tokens = effective_visual_tokens + policy_visible_text_tokens` 全部成立；
- 60/60 audit generation 的 grid、visual/text/prompt token metadata 与已验证的 v2.1 parent raw 中同一 state、
  同一 prompt witness 逐项 exact；
- 原 validator 在任何 summary 或 archive 写入前失败。

离线继续执行修正后的第一项检查还暴露了第二个同类错误：旧 validator 要求 teacher forward 的
`extended_prompt_aligned_inputs` 为 `attention_mask + input_ids`，但 frozen runtime 的
`prompt_aligned_input_keys()` 按定义排除主 `input_ids`，实际 120/120 teacher forwards 都记录
`attention_mask + mm_token_type_ids`。这与先前 processor audit 的 tensor inventory 一致，也必须作为窄的
validator repair 明确冻结。

不能把常数从 `2560` 直接改成 `2584`：这样仍会错误拒绝合法的 2516/2560 grid。修复必须从实际 grid
重算，并由固定 parent raw 的同 prompt witness 独立锚定。

## 修复边界

修复是一个新的 source-only offline validation child，不是第二次 policy attempt：

- 不修改 `code/configs/spatial_reference_audit_v1.json`、旧 validator、runner、runtime、fixture、exposure
  ledger 或其 33-file frozen source inventory；
- 不删除、不改写、不重跑任何 profile/state terminal、start/claim 或 sibling ledger；
- 不实例化 processor/model，不做 GPU、generation、teacher forward、restoration、gate 或 confirm access；
- 把旧 config 与其中 33-file source inventory 逐 Git blob 绑定回 raw commit `c093bd8f...`，并核对实际
  import 的旧 validator 路径；
- 先绑定 pre-repair raw tree、三个 terminal、ledger、formal log/exit 与 parent archive 的 exact hash；
- 使用 dynamic grid accounting，并要求每个 audit shape witness 的 grid/effective tokens 与固定 parent generation
  witness 一致；generation 还要求 text/prompt tokens exact；teacher aligned-input inventory 固定为
  `attention_mask + mm_token_type_ids`；
- 继续复用旧 validator 对 token IDs、parent evidence、runtime、durable claim、operation count、profile order、
  logits 与 decision rule 的其余检查；
- 所有读验证通过后才 exclusive-create canonical `summary.json`；repair provenance、原 failure message 与
  log/exit bindings 直接嵌入 summary，不另写 manifest，也不复制或改写原 log/exit；已有 output 时拒绝覆盖。
- 写后重新读取 canonical bytes 与 scientific hash，要求旧 70 个 root files byte-identical、只新增 summary
  后为 71 个，连同 sibling ledger 的 packaging inventory 为 72 个，且 archive 仍不存在。

repair summary 必须同时记录 policy source commit `c093bd8f...` 和新的 clean pushed validation-repair commit，
并把 repair config/source SHA、原失败类型、parent-shape comparison counts、raw pre-repair tree hash 与零新增
policy-operation boundary 纳入 scientific payload hash。

## Artifact

repair 后仍使用 policy attempt 前已冻结的 deterministic USTAR packager；它打包 canonical audit root（包括
repair provenance）与 root 外 sibling ledger。packager 的 `source_git_commit` 保持产生 raw 的 `c093bd8f...`，
没有改写成 repair commit。canonical private HF artifact 为：

- repo：`gavinlaw/causalcache-spatial-reference-audit-mobile`；
- immutable revision：`d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`；
- tag / path：`spatial-reference-audit-v1` / `raw/spatial-reference-audit-v1.tar`；
- archive：1,873,920 bytes、72 members、SHA256
  `d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc`；
- fresh-download canonical rebuild 与逐 member tree hash：通过。

原 v2.1 结论 `NO_GO_V2_1_FULL_45_SUBSTRATE` 永久不变。repaired independent validation 与 artifact 已闭合，
依据冻结三分支 rule 将 eager 13/13、auto 7/13 归约为
`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`。下一步必须单独冻结 `v2.2-eager` source；不能复用本次 raw。
