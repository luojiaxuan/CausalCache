# Set-utility train-only throughput pilot v1

## 当前结论

唯一 v1 formal attempt 已在 Hyper00 四张 H200 上并发消费，但在任何 policy runtime/model load 或 native call
之前 fail closed。四个 worker 均写入 no-retry attempt，随后在 semantic input load 阶段因 candidate schedule
exact bytes 不满足 consumer 额外要求的 canonical-pretty JSON 而返回 `ValueError`；正式计数为 `0/12` pairs、
`0/84` native calls。aggregate CLI 因四个 failed terminals 返回 `ValueError`，没有产生 `aggregate.json`，因此
没有 mb1/mb2 选择，也不能写成 throughput `NO_GO`。

正式状态为
`INVALID_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_V1_CANDIDATE_SCHEDULE_CANONICALIZATION_CONTRACT_DRIFT`，
轻量证据见
[`../../data/results/archive/set_utility_train_only_throughput_pilot_v1/`](../../data/results/archive/set_utility_train_only_throughput_pilot_v1/)。
source A=`5158f2a`，execution-envelope B=`c7d5b31`，envelope SHA256=`adf50363...228e`。同一 v1 identity
不得重跑；下一步只能另立 versioned consumer repair，同时保持 schedule bytes/hash、roster、阈值与 84-call
预算不变。restoration labels、predictor training、matched-NLL 与 closed-loop 仍 locked。

后续 versioned repair 已冻结为
[`set_utility_train_only_throughput_pilot_candidate_schedule_key_repair_v2.md`](set_utility_train_only_throughput_pilot_candidate_schedule_key_repair_v2.md)；
它保留本页 v1 evidence，只修复 producer typed integer-key reconstruction，不追溯重跑 v1。

## Immutable prerequisite

pilot 唯一允许使用的 processor prerequisite 是已经完成双 fresh replay 的 private Hugging Face immutable
artifact：

| 字段 | 冻结值 |
| --- | --- |
| dataset repo | `gavinlaw/causalcache-set-utility-new-development-mobile` |
| immutable revision | `c20bab8df424dc9e45ece1084f3d1dc035dd1ed8` |
| annotated tag | `phase1-b2-processor-freeze-v2-image-contract-repair` |
| prefix | `artifacts/processor-freeze-v2-image-contract-repair` |
| formal inventory | 23 files / 18,730,620,511 bytes |
| formal inventory SHA256 | `7c2a971658ad9a6bbc639446e9326e9dd4718b8d2d77186f52a17f10f3b32fc1` |
| candidate schedule | `processor-candidate-freeze-schedule.json` |
| candidate schedule SHA256 | `186f2952108273672c6cdbf963094754298d23693fd6268a72b6223e99c2299d` |

Git-side prerequisite 是
[`data/results/archive/set_utility_processor_freeze_v2_image_contract_repair_publication_v1/summary.json`](../../data/results/archive/set_utility_processor_freeze_v2_image_contract_repair_publication_v1/summary.json)。正式执行只能读取与上述 revision 和
inventory 逐 byte 一致的本地副本；不得把 tag 的可变解析结果、未验证 staging bytes 或另一 processor run 混入
pilot。source-only validation 只读取已经 Git-bound 的 metadata，不访问 Hub。

policy snapshot 同样固定为 `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`。

## Freeze-B 预注册 roster

roster 直接来自 Freeze-B v2 的 train-only `stratum_anchor`，不是观察 policy output 后抽样。三个 candidate-capacity
strata 各取 4 个 state：

| stratum | decision step | 初始 candidates | 冻结 state IDs |
| --- | ---: | ---: | --- |
| `decisions_6_9` | 6 | 4 | `0296753837938323:decision:006`<br>`0310939638496410:decision:006`<br>`0336706763935531:decision:006`<br>`0363106249041737:decision:006` |
| `decisions_10_17` | 10 | 8 | `0214209291313446:decision:010`<br>`0268406573756492:decision:010`<br>`0271654003819383:decision:010`<br>`0279447750102246:decision:010` |
| `decisions_18_plus` | 18 | 16 | `0018651612081817:decision:018`<br>`0047881550315557:decision:018`<br>`0053721915679562:decision:018`<br>`0117550419342475:decision:018` |

把上表按 stratum 顺序展平为 `state_ids` 后，固定四个 worker：

```text
worker[index] = state_ids[index::4], index in {0,1,2,3}
```

因此每个 worker 恰好处理三个 state，并且每个 stratum 恰好一个。正式执行不得重排、替换失败 state、增加
warmup state 或从同一 stratum 补样；aggregate 顺序固定为 worker `0,1,2,3`。

## Paired mb1 -> mb2 contract

每个 state 在同一 worker、同一 policy instance、同一 GPU 上严格按 `microbatch 1 -> microbatch 2` 执行。每个
variant 都先对同一个 frozen reference input 独立 generation 两次；两次得到的 action 只作为进程内 opaque
handle，由 pinned runtime 的 `reference_actions_equal()` 判等。禁止把 action text、tokens 或 native output
序列化后再比较。

顺序和调用预算为：

1. mb1：2 次 reference generation，确认 variant 内 action equality，再对 2 个 logical teacher examples 分两次
   teacher forward；共 4 个 native calls；
2. mb2：2 次 reference generation，确认 variant 内 equality，并在进入 mb2 teacher 前确认 mb1 与 mb2 的
   opaque action equality；随后把同样 2 个 logical examples 合并为 1 次 teacher forward；共 3 个 native calls；
3. 每 state 共 7 calls；12 states 的 success path 必须精确等于 **84 native calls**，即 48 generation calls、
   36 teacher calls 和 48 teacher examples。

不允许改变 decoding、prompt、processor input 或 logical teacher examples 来帮助 mb2。跨 variant action 不等时
fail closed；此时 latency/memory/counts 可以保留为 metric-only failure evidence，但不得继续写 utility 或 labels。

source 接口固定为：

- `run_train_only_set_utility_throughput_pilot_pair_v1()`：单 state 的 mb1 -> mb2 pair；
- `aggregate_train_only_set_utility_throughput_pilot_pairs_v1()`：精确 12-state aggregate 与 84-call 检查；
- `GUIOwlV21SetUtilityThroughputAdapter`：拥有 encode/H2D/preparation/forward/decode-or-logit-disposal 的完整计时和
  CUDA peak 边界。

## Metric-only 输出边界

每个 variant 仅允许四个顶层字段：

```text
counts
failure_class
latency_seconds
peak_memory_bytes
```

允许写出的信息只包括：完整 adapter call 的 end-to-end wall time、full-call CUDA allocated/reserved peak、精确
调用计数、cross-variant equality 的布尔聚合，以及不含 message 的安全 error-class identifier。teacher logits 必须
在 adapter 内丢弃后才返回；opaque action handle 只在同一进程内存活，永不序列化。

以下字段或其同义 payload 一律禁止进入 stdout、stderr、receipt、result、exception message 或 Git/HF artifact：

```text
action, action_text, decoded_output, native_output,
messages, tokens, logits, KL, utility
```

measurement boundary 必须覆盖 encode、CPU-to-GPU/H2D、输入准备、native forward，以及 generation 的
decode/parse 或 teacher logits disposal；不得只报告 kernel time 来替代该指标。

## 执行授权与 no-retry 状态机

当前 canonical source 路径为：

- [`code/configs/causalcache_set_utility_train_only_throughput_pilot_v1.json`](../../code/configs/causalcache_set_utility_train_only_throughput_pilot_v1.json)；
- [`code/causalcache/set_utility_throughput_pilot_contract_v1.py`](../../code/causalcache/set_utility_throughput_pilot_contract_v1.py)；
- [`code/scripts/validate_set_utility_throughput_pilot_contract_v1.py`](../../code/scripts/validate_set_utility_throughput_pilot_contract_v1.py)；
- [`code/causalcache/set_utility_throughput_pilot_pair_v1.py`](../../code/causalcache/set_utility_throughput_pilot_pair_v1.py)；
- [`code/causalcache/set_utility_gui_owl_v2_1_throughput_adapter.py`](../../code/causalcache/set_utility_gui_owl_v2_1_throughput_adapter.py)。

source commit/push 后还必须物化一个 exact execution envelope。Git canonical path 为
`code/configs/causalcache_set_utility_train_only_throughput_pilot_v1_execution.json`，同一份 bytes 以
`O_CREAT|O_EXCL` 写入本次 `/data` run root 的 `execution-envelope.json`。envelope commit B 必须是
source commit A 的 direct child，A→B 只允许增加该 Git execution JSON，且 B 必须 clean push 到
canonical `main`；运行时会逐 byte 检查 Git blob、working-tree file 与 `/data` envelope 三者一致。
正式轻量结果预定写入 `data/results/archive/set_utility_train_only_throughput_pilot_v1/`；在 envelope B
验证并 push 前，不构成任何 GPU/policy 执行授权。

exact envelope 至少要绑定 fresh preflight 证据、clean Git revision、container/image digest、CUDA/driver/library
versions、四个确切 GPU IDs、model local snapshot identity、processor local inventory、四 worker mapping、canonical
attempt/result/ledger paths、argv、起止时间和 failure taxonomy。它必须在任何 model load 或 policy call 前，以
`O_CREAT|O_EXCL`（并拒绝 symlink/no-clobber）持久化唯一 attempt claim：

materializer 还会把原始 10 秒 cleanup stdout、四张卡的 UUID/host index/name/total memory、sampled
max utilization=`0`、post-cleanup memory `<=500 MiB` 和 compute-app count=`0` 写入 strict canonical
preflight evidence，并现场核对 container、Python、PyTorch、Transformers、CUDA 与 GPU identity。model +
processor 只在 materialization 时完整 SHA256 一次；随后每个 worker 只复验同一份 per-file stat
identity receipt，并在 model access 前重新检查 preflight freshness，避免四个 worker 重复读取约 55 GB。

- claim 已存在，无论前次成功、失败或中断，都禁止 retry/resume；
- 不得删除 claim 后重跑，不得换 output path 规避 no-retry；
- 不得替换失败 state，不得增加 top-up state，不得超过 84-call ceiling；
- failure 只按预注册规则解释，不以额外试跑修改 threshold 或 denominator。

## Host 与并行边界

正式 pilot 固定在 **同一台 Hyper00** 上使用 4 张经 fresh GPU preflight 选出的 H200；四个 worker 各独占一张
确切 GPU，并执行上面的 `state_ids[index::4]` shard。不得把 Hyper01、H100、Taurus/Aries 或本机得到的 latency/
memory 填入同一 aggregate，也不得把不同 GPU SKU、driver、container digest 或 runtime revision 的指标合并。

如果 fresh preflight 无法在 Hyper00 同时提供四张合格卡，本次 formal pilot 不降级为跨机拼接；等待资源或另立
versioned contract。host-readiness snapshot 不是 reservation，不能代替 fresh preflight。

## 预注册选择规则

1. memory metric 固定为每个完整 native call 的 `full_call_cuda_reserved_max`。可部署 variant 在每张设备上的
   observed peak 必须不超过该设备总显存的 **80%**，即至少保留 20% headroom；mb1 任一 failure 或 memory
   violation 直接 `NO_GO`。
2. 若 12 个 pair 的 mb1、mb2 全部成功，且 12 次跨 variant opaque action equality 全部成立，只比较精确 12
   states 的 teacher-forward wall-time 总和：

   ```text
   teacher_wall_ratio = mb2_teacher_wall_total / mb1_teacher_wall_total
   ```

   只有 `teacher_wall_ratio <= 0.95`（mb2 至少快 5%）且 mb2 满足 memory bound 时才选择 mb2；否则选择满足
   bound 的 mb1。
3. mb2 failure 仅在**所有** failure 都发生于 mb2 teacher stage、每个 state 的跨 variant action equality 都已完成
   且 mb1 完整通过时，才允许 fallback 到 mb1。generation/input/equality failure、action mismatch、未完成的
   comparison 或 memory violation 都不冒充 teacher-stage-only fallback，均 `NO_GO`。
4. 不做 retry、top-up、跨 host normalization 或异构 metric merge；不得事后调整 80%/5% threshold。

该规则只选择后续 formal label producer 的 microbatch，并不评价 restoration 方法效果。

## Definition of Done

### Source-freeze milestone

- exact immutable processor publication、model revision、Freeze-B v2 roster 和三 strata 通过 fail-closed validator；
- 12 state IDs、`state_ids[index::4]`、paired mb1 -> mb2、84-call ceiling 和选择阈值均进入 canonical config；
- runtime transitive source inventory 被逐文件 SHA256 绑定；
- source validation 的 GPU/model/policy/processor/label/training/HF mutation/result-write counts 全为 0；
- focused tests、source validator、Git diff 检查通过，并 commit/push canonical `main`。

### Formal-pilot milestone

- source-freeze commit 之后完成 fresh Hyper00 GPU preflight，并冻结、验证、commit/push 独立 exact execution
  envelope；
- O_EXCL attempt claim 在首次 model/policy access 前成功落盘，四张 H200 与四 worker mapping exact；
- 12 个预注册 state 各执行一次，未 retry、resume、替换或 top-up；
- success path 的 aggregate 精确闭合为 84 calls，或唯一失败按预注册 metric-only taxonomy fail closed；
- 跨 variant equality、latency、reserved/allocated peak、计数和 selection outcome 通过独立 validator；
- 轻量 result/receipt 更新 Git docs，commit 并 push；任何可复用大 artifact 若产生则按 immutable revision 发布到
  Hugging Face。

即使 formal-pilot milestone 完成，**restoration labels、predictor/gate training、matched-NLL 和 closed-loop success
仍保持锁定**。它只能解锁下一份 formal label-generation contract；不得把 throughput pilot 写成 scientific GO、
restoration signal 或 task-success 证据。
