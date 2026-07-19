# Set-utility action-stability diagnostic D1

## 当前状态

D1 的 versioned source core 已实现，formal source/execution envelope 尚未冻结，因此当前不授权 GPU run。它只
定位 throughput pilot v2 的 reference-action instability，不修改 v2 结果，也不直接解锁 restoration labels、
predictor training、matched-NLL 或 closed-loop。

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

- fresh unstable、auto frozen stable、eager stable：`ENCODING_OR_PREPARATION_PATH_IMPLICATED`；
- auto frozen unstable、eager stable：`AUTO_ATTENTION_OR_NUMERICAL_CONTROL_IMPLICATED`；
- eager 仍不稳定：`PERSISTENT_GENERATION_INSTABILITY`；
- parent mismatch states 全部稳定：`PARENT_MISMATCH_NOT_REPRODUCED`；
- 任一 stable control 未能在全部三个 condition 保持稳定：`INVALID_STABLE_CONTROL_INSTABILITY`。

D1 只做 failure localization。无论得到哪一类结果，正式 labels 都不能直接启动；必须另立新的 12-state
throughput identity，回到完整 parent roster 验证后才能决定 label Execution-B。

## Implementation and validation

- runtime：`code/causalcache/policy/gui_owl_v2_1_action_stability_runtime_v1.py`；
- adapter：`code/causalcache/set_utility_gui_owl_v2_1_action_stability_adapter_v1.py`；
- diagnostic core：`code/causalcache/set_utility_action_stability_diagnostic_v1.py`；
- focused tests：3 个独立 test files，按 3 个 CPU worker 并行运行，共 `13 passed`；
- `py_compile` 通过。

下一步先完成 formal source contract、runner 与 source inventory，再 commit/push source A；随后机械生成独立
execution envelope B，fresh GPU preflight 后在 Hyper00 四张同构 H200 上运行 default stage 与 eager stage。不能
把 Hyper01/H100/A6000 的异构数值结果合并进同一 D1 verdict。
