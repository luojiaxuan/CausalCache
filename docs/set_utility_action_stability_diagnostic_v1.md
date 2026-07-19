# Set-utility action-stability diagnostic D1

## 当前状态

D1 的 formal source contract 已冻结，canonical config SHA256=
`e72b0ba226048b6a188ec5ee35e9a0d5bffa04fcc8ee48f1af2b409151fad2c6`，51-file source inventory
SHA256=`3313f90e43b3f3dec44e1fba8502d796847f5bfba55bc8c05308e19f222cded0`。初始 source
`e3bb7bc` 的真实 pre-policy smoke 在 model/GPU/preflight 前 fail closed：旧 parent envelope 的 worker argv
绑定旧 absolute checkout，而初版 consumer 错误地用 D1 checkout 重建 parent argv。当前 repair 从 parent
envelope 的 4 个 worker argv 与 aggregate argv 共同推导并验证唯一 parent checkout，再只把验证后的 artifact
projection rebind 到 D1 source。本状态仍是 source A：
独立 execution envelope B 尚未物化，因此当前 source commit 本身不授权 GPU run。D1 只定位 throughput pilot
v2 的 reference-action instability，不修改 v2 结果，也不直接解锁 restoration labels、predictor training、
matched-NLL 或 closed-loop。

parent v2 的有效事实保持不变：12 个 pair 中 8 个 completed；3 个
`MICROBATCH_1__REFERENCE_ACTION_MISMATCH`，1 个 `CROSS_VARIANT_REFERENCE_ACTION_MISMATCH`；formal
selection=`NO_GO / MICROBATCH_1_FAILURE`。D1 绑定 parent aggregate SHA256=
`35e0c250232efbd2b9bdccfb1b04a7c372fbed892635342cce4f9f81097ff33f`。

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

下一步 commit/push 本 source A；随后 fresh 10 秒 preflight，机械生成并单独 push execution envelope B，在
Hyper00 四张同构 H200 上运行 auto stage 与 eager stage。不能把 Hyper01/H100/A6000 的异构数值结果合并进同一
D1 verdict。
