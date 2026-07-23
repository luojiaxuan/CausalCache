# Set-utility action-stability diagnostic D1b

## 目的

D1 的全-eager frozen control 在三个 `decision:010` state 上 OOM，因此 D1 的正式结论是
`INVALID_RUNTIME_FAILURE`，不能判断这些 state 的 action repeat stability。D1b 只回答一个更窄的问题：在保持
parent GUI-Owl automatic loader、实际 top/text/vision 全为 SDPA 的前提下，若在任何 CUDA 初始化前应用已经恢复的
numerical controls，并让每个 state 在独立 OS process 中运行，memory-safe frozen-encoded generation 是否重复稳定。

D1b 是 train-only failure-localization diagnostic，不生成 restoration label，不训练 predictor，也不运行
matched-NLL、closed-loop 或 sealed test。

## 冻结输入与信息边界

- 唯一历史输入是 Git-frozen D1 aggregate 与 summary；aggregate SHA256 为
  `2debde6de3f552e9551d0ee37d25b82fa2ca85dfb42746d389dde39eff577ba2`；
- 只投影 `auto_fresh_encode` 与 `auto_frozen_encoded` 两个完整 metric-safe condition；
- 不消费第三个 eager condition 的语义，不把三个 eager OOM backfill 为 SDPA observation，也不把历史调用计入 D1b
  ceiling；
- 原始 action、coordinate、text、token ids、decoded output、logits 及其 digest 不落盘；只允许 exact equality
  boolean、prepared-input equality、class-only failure、调用计数和 runtime/process identity。

## 唯一新 condition

`sdpa_numerical_control_frozen_encoded` 对每个 state：

1. 在 fresh process 中、CUDA 初始化前固定 seed，关闭 TF32，并应用既有 non-strict numerical controls；
2. 使用 parent automatic loader，模型加载后 fail closed 验证 top/text/vision 均实际解析为 `sdpa`；
3. processor encode 一次，复用同一个 exact GPU tensor mapping 连续 generation 两次；
4. 明确不声称 strict CUDA determinism。

六个 state 的总 ceiling 是 6 次 encode、12 次 generation；无 retry、无 top-up。

## 进程与 GPU 调度

每个 state 必须使用不同 fresh OS process。正式调度固定为同一 Hyper00 container 内四张 H200 的非连续 `4+2`
两波：

- wave 0：state index `0,3,4,5`，四进程并发；
- wave 1：state index `1,2`，仅在 wave 0 四个 schema-valid terminal 都存在后启动，两进程并发。

Execution-B 前置 evidence 必须包含至少连续 10 秒的 0% GPU utilization sample、all-container cleanup 记录、四张
H200 的 host index/visible index/UUID 和容器身份。materialization 与 state launch 都要求 evidence 不超过 900 秒；
aggregate 为 CPU-only readback，不要求 freshness，也不复用当前 GPU 环境作为科学输入。

这里不把 Hyper01、H100、Taurus/Aries 的结果混入同一个 formal verdict。原因不是算力不足，而是本诊断需要固定
host/container/runtime/GPU class；异构机器可承担独立工程 smoke，但不能拼接为六状态 aggregate。

## 解释与 gate

state diagnosis 区分：execution failure、stable-control instability、D1 auto-frozen instability 未被 controlled SDPA
修复、fresh-path 与 SDPA 双重 instability、SDPA-only instability、auto-vs-controlled-SDPA profile association、
fresh-vs-frozen association，以及 parent mismatch 是否在新 profile 下复现。

aggregate verdict 按以下优先级唯一决定：

1. `INVALID_RUNTIME_FAILURE`；
2. `INVALID_STABLE_CONTROL_INSTABILITY`；
3. `NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY`；
4. `PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY`。

只有最后一个 PASS 才授权另立全新 12-state throughput v3 Source-A。即使 PASS，也不直接解锁 restoration labels、
utility predictor training、matched-NLL 或 closed-loop。

## Source-A → Execution-B 生命周期

- Source-A config 为
  `code/configs/causalcache_set_utility_action_stability_diagnostic_v2.json`，只允许 CPU/source validation，禁止 GPU
  execution；
- Source-A 必须在 clean、已 push 的 canonical `main` 上冻结；
- Execution-B 必须是 Source-A 的 direct child，且唯一 changed path 是
  `code/configs/causalcache_set_utility_action_stability_diagnostic_v2_execution.json`；
- Git 中的 Execution-B 与 `/data` canonical envelope 必须逐 byte 相同；
- aggregate 重新校验 source/envelope/D1/artifact/attempt/terminal/process/wave identity，不能由日志或失败进程补写结果。

Source-A canonical config 已冻结：SHA256=
`07ec807effd9a1655b0572e49a835e61aec5192503d16bd5c2ee6a3d7a18c77f`，54-file transitive source inventory
SHA256=`a58cbda4db59bf6f951783c94f606e9e00bb310191e371de099256c4286c55b0`。runtime/core、contract、
envelope/execution、aggregate/CLI 与相关 v1 回归分四个 CPU process 并发运行，共 `75 passed`；source validator、
`py_compile` 与 diff check 通过。

## Formal result

Source-A=`fe6f6b69adf2acd5457026835529f2ca3e682dd2`，direct-child Execution-B=
`db2c848af310d7559247895f418f69255465eb73`。execution envelope 为 19,349 bytes / SHA256=
`7adb460bbe65d08cc7fe2bea992cf77f544d8be18f5c43f28c84d4ee4e9171f8`。唯一 formal run 完成 6/6
processes、12/12 generation、6/6 encode、retry=`0`；exact aggregate 为 17,011 bytes / SHA256=
`b419f9640cb46276b7a52e292d6feabd81311f66efb22563642ce80c4561919c`，从全部 attempts/terminals 只读重建后
逐 byte 相同。

三个 decision-10 state 全部在 controlled SDPA 下完成，说明 D1 eager OOM blocker 已消失。五个 state
repeat-stable；`0296753837938323:decision:006` 的 frozen input 三个检查点均 unchanged，但 exact generated
sequence、decoded output 与 canonical action 都 repeat-unequal。因此正式 verdict 为
`NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY`，不是 runtime-invalid，也不是 PASS。

startup utilization sampler 因 operator shell quoting 错误没有形成有效记录，本 run 不作 throughput 或 GPU
utilization claim；不因此重跑。根据冻结 gate，本结果不授权 12-state throughput v3、labels、training、matched-NLL
或 closed-loop。Git-safe aggregate、summary 与解释位于
[`../../data/results/archive/set_utility_action_stability_diagnostic_v2/`](../../data/results/archive/set_utility_action_stability_diagnostic_v2/)。
