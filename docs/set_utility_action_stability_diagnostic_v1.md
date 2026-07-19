# Set-utility action-stability diagnostic D1

## 当前状态

D1 的 source A2=`121c8621bae4f56a57060ea2fb402cfa3d8c7146` 与 direct-child execution B=
`5fa1fdd45dba1fa95c7a69ee84d6ddb5b3f5166e` 均已冻结并 push。canonical config SHA256=
`e72b0ba226048b6a188ec5ee35e9a0d5bffa04fcc8ee48f1af2b409151fad2c6`，51-file source inventory
SHA256=`3313f90e43b3f3dec44e1fba8502d796847f5bfba55bc8c05308e19f222cded0`，execution envelope
SHA256=`093eadeea6c05e1266c681aeea31fa48a86b6dff74d83404e115ce63c3b0a6ea`。唯一 formal run 已完成 exact
aggregate，aggregate SHA256=`2debde6de3f552e9551d0ee37d25b82fa2ca85dfb42746d389dde39eff577ba2`，
formal verdict=`INVALID_RUNTIME_FAILURE`。

初始 source `e3bb7bc` 的真实 pre-policy smoke 在 model/GPU/preflight 前 fail closed：旧 parent envelope 的
worker argv 绑定旧 absolute checkout，而初版 consumer 错误地用 D1 checkout 重建 parent argv。A2 从 parent
envelope 的 4 个 worker argv 与 aggregate argv 共同推导并验证唯一 parent checkout，再只把验证后的 artifact
projection rebind 到 D1 source。该历史 repair 已由 B 和唯一 formal run 消费，不得再重跑或 top-up D1。D1 不修改
parent v2 结果，也不解锁 restoration labels、predictor training、matched-NLL 或 closed-loop。

parent v2 的有效事实保持不变：12 个 pair 中 8 个 completed；3 个
`MICROBATCH_1__REFERENCE_ACTION_MISMATCH`，1 个 `CROSS_VARIANT_REFERENCE_ACTION_MISMATCH`；formal
selection=`NO_GO / MICROBATCH_1_FAILURE`。D1 绑定 parent aggregate SHA256=
`35e0c250232efbd2b9bdccfb1b04a7c372fbed892635342cce4f9f81097ff33f`。

## Formal result

8/8 profile-worker attempts 与 8/8 terminals 完成，auto→eager 文件时间 barrier 通过，retry=`0`。实际调用为
33/36 generation 与 24/24 encode；teacher/KL/restoration/label/training/HF mutation 全为 `0`。三个
`decision:010` state 的 `eager_frozen_encoded_control` 均在第一次 generation 形成 class-only
`OutOfMemoryError`，所以 aggregate 按预注册优先级返回 `INVALID_RUNTIME_FAILURE`，不能把这些 state 归类为
action instability。

| parent role | state_id | formal diagnosis |
| --- | --- | --- |
| mismatch | `0296753837938323:decision:006` | `FRESH_VS_FROZEN_PATH_ASSOCIATION` |
| mismatch | `0310939638496410:decision:006` | `PARENT_MISMATCH_NOT_REPRODUCED` |
| control | `0336706763935531:decision:006` | `PARENT_MISMATCH_NOT_REPRODUCED` |
| mismatch | `0271654003819383:decision:010` | `INVALID_CONDITION_EXECUTION_FAILURE` |
| mismatch | `0279447750102246:decision:010` | `INVALID_CONDITION_EXECUTION_FAILURE` |
| control | `0268406573756492:decision:010` | `INVALID_CONDITION_EXECUTION_FAILURE` |

`029675...:decision:006` 的 fresh encode 两次输出不一致，而 frozen encoded 与 eager frozen 都稳定；这只支持
path association，不证明 processor/re-encode、attention backend 或 hidden state 的单一因果。三个 decision-10
OOM 分布在三张独立 H200，其中 worker 3 没有前序 state，因此不能简单解释为同一 worker 的跨 state 累积；但
formal artifact 没有 allocation trace，也不能把机制进一步声称为已证明的 quadratic-attention OOM。

Git-safe aggregate、provenance 和解释位于
[`../data/results/set_utility_action_stability_diagnostic_v1/`](../data/results/set_utility_action_stability_diagnostic_v1/)。
raw attempts、terminals 与 logs 仅保留在 Hyper00
`/data/runs/causalcache-action-stability-d1-121c862`，状态为 `LOCAL_FORENSIC_NOT_UPLOADABLE`。

## Frozen diagnostic shape

D1 使用 4 个 parent mismatch states 和 2 个同 candidate-capacity stratum 的 stable controls：

| role | state_id | parent observation |
| --- | --- | --- |
| mismatch | `0296753837938323:decision:006` | cross-variant mismatch |
| mismatch | `0310939638496410:decision:006` | mb1 repeat mismatch |
| control | `0336706763935531:decision:006` | v2 pair completed |
| mismatch | `0271654003819383:decision:010` | mb1 repeat mismatch |
| mismatch | `0279447750102246:decision:010` | mb1 repeat mismatch |
| control | `0268406573756492:decision:010` | v2 pair completed |

每个 state 按以下顺序运行三个 condition，每个 condition 恰好两次 greedy generation：

1. `auto_fresh_encode`：保留 parent v2 的 automatic attention runtime，每次 generation 重新 processor encode；
2. `auto_frozen_encoded`：同一 automatic attention runtime，只 encode 一次并复用 exact GPU tensor mapping；
3. `eager_frozen_encoded_control`：显式 eager attention 的既有 numerical-control runtime，只 encode 一次并复用
   exact GPU tensor mapping。

总 ceiling 为 36 generation calls 与 24 processor encode calls。teacher forward、KL、utility、restoration label、
training 和 HF mutation 全部为 0。D1 latency 不与 v2 throughput latency 比较，因为 frozen-input condition 改变了
encode/H2D 边界。

第三个 condition 的正式名称故意是 `eager_frozen_encoded_control`。现有 eager runtime 固定 seed、关闭 TF32、
使用 eager attention 并启用部分 numerical controls，但明确没有启用 strict CUDA deterministic algorithms；D1
不得把它写成 `eager_deterministic`。如果 eager control 仍不稳定，才另立 strict-determinism D2，且必须在新进程
首次 CUDA 之前绑定 `CUBLAS_WORKSPACE_CONFIG`。

auto runtime 在模型加载后必须 fail closed 验证 top/text/vision 三层实际 attention 均解析为 `sdpa`，并将
observed mapping 纳入 runtime metadata；因此 silent fallback 不能被误写成 auto-vs-eager 结果。

## Metric firewall

原始 action、coordinate、text、token ids、decoded output、logits 和它们的 digest 只允许存在于进程内 opaque
observation，不能序列化。每个 condition 只落盘：

- exact generated-sequence、decoded-output、canonical-action 三层 equality boolean；
- prepared input 在 before/between/after 的 exact tensor equality boolean；
- encode/generation attempted/completed counts；
- class-only failure；
- `metric_safe=true`。

prepared input snapshot 同时检查 keys、shape、dtype、device 和 `torch.equal`，并拒绝跨 runtime handle。旧
`gui_owl_v2_1_throughput_runtime.py`、adapter 与 pilot core 不修改，避免破坏 historical source inventory。

## Preregistered interpretation

- fresh unstable、auto frozen stable、eager stable：`FRESH_VS_FROZEN_PATH_ASSOCIATION`；
- auto frozen unstable、eager stable：`AUTO_VS_EAGER_PROFILE_ASSOCIATION`；
- eager 仍不稳定：`PERSISTENT_GENERATION_INSTABILITY`；
- parent mismatch states 全部稳定：`PARENT_MISMATCH_NOT_REPRODUCED`；
- 任一 stable control 未能在全部三个 condition 保持稳定：`INVALID_STABLE_CONTROL_INSTABILITY`。

前两项故意只称 `ASSOCIATION`：同一 auto runtime 固定先跑 fresh、后跑 frozen，因此仍可能混入 warmup、
allocator/autotune 或模型持久状态。default 与 eager 必须在独立进程、分 stage 执行，避免 eager numerical controls
污染后续 auto condition。任一 OOM、parse error、unsupported op 或 input-mutation failure 都记为
`INVALID_CONDITION_EXECUTION_FAILURE`，不能被归因为 action instability。若 condition 已形成完整 metric-safe
partial，aggregate 允许实际 calls 低于 ceiling 并返回 `INVALID_RUNTIME_FAILURE`；若 runtime 在形成该 state
partial 前退出，则 worker terminal fail closed，不能 top-up 或重试。

D1 只做 failure localization。无论得到哪一类结果，正式 labels 都不能直接启动；必须另立新的 12-state
throughput identity，回到完整 parent roster 验证后才能决定 label Execution-B。

## Formal execution boundary

- B 必须是 source A 的唯一 direct child，且该 commit 只能新增
  `code/configs/causalcache_set_utility_action_stability_diagnostic_v1_execution.json`；Git 与 `/data` envelope
  bytes 必须一致；
- fresh preflight 必须连续覆盖至少 10 秒，并选择同一 Hyper host、同一 container 内 4 张 H200；本实验不把
  H100/A6000 结果合入同一 verdict；
- parent artifact binding 必须读取旧 run 的 canonical
  `/data/runs/causalcache-throughput-pilot-v2-key-repair-d5e0cca/execution-envelope.json`，不能传 Git 中的
  byte-identical copy；
- 每个 worker 是独立 OS process，且 `PYTHONPATH` 必须精确等于 execution worktree 的 `code/`，避免旧 venv
  editable install 把 import 路由回 parent worktree；
- 四个 auto worker 并发运行；任一 eager attempt 创建前，runner 会重新验证全部 4 个 auto terminal、attempt
  hash、source/envelope/parent identity、roster 与实际 calls。随后四个 eager worker 并发运行；
- aggregate 不执行 GPU work，严格合并 8 个 profile-worker terminals 与 6 个 states，并同时复核文件时间 barrier。

## Implementation and validation

- runtime：`code/causalcache/policy/gui_owl_v2_1_action_stability_runtime_v1.py`；
- adapter：`code/causalcache/set_utility_gui_owl_v2_1_action_stability_adapter_v1.py`；
- diagnostic core：`code/causalcache/set_utility_action_stability_diagnostic_v1.py`；
- source/envelope/runner/aggregate：`code/causalcache/set_utility_action_stability_{contract,envelope,execution}_v1.py`
  与对应 `code/scripts/` entrypoints；
- focused tests 按 core、source contract、runner、envelope/aggregate 四个 CPU process 并行执行，共
  `41 passed`；source validator、`py_compile` 与 `git diff --check` 通过。

本 D1 已封存。下一步另立 D1b source：保留 exact roster/input，但把第三个 condition 改为 memory-efficient
SDPA numerical-control profile，在新进程首次 CUDA 前冻结其控制参数；每个 state 使用 fresh OS process，按
4+2 两波调度，避免跨 state 残留。D1b 是新 identity，不与 D1 拼接，也不能把 Hyper01/H100/A6000 的异构结果
并入 D1 verdict。D1b 闭合前，12-state throughput、labels、training、matched-NLL 与 closed-loop 继续 locked。
