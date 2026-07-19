# Strict-determinism action-stability diagnostic D2

## 目的

D1b 在 memory-efficient SDPA 与 non-strict numerical controls 下完成全部 6 个 state，但
`0296753837938323:decision:006` 在 frozen GPU input 三个检查点均不变时，两个 generation 的 exact sequence、
decoded output 和 canonical action 仍全部不相等。D2 只判断这一个不稳定 state 在 PyTorch strict deterministic
profile 下是否稳定，并用两个 D1b stable control 检查新 profile 本身是否产生回归。

D2 是 train-only failure-localization diagnostic。它不重跑或 top-up D1b，不恢复 12-state throughput，不生成
restoration labels，不训练 predictor，也不运行 matched-NLL、closed-loop 或 sealed test。

## 冻结 roster

| state | role | 选择理由 |
| --- | --- | --- |
| `0296753837938323:decision:006` | unstable target | D1b 唯一 controlled-SDPA repeat-unstable state |
| `0336706763935531:decision:006` | stable control | 与 target 相同 `decision:006` stratum |
| `0268406573756492:decision:010` | stable control | 更长 context/candidate-capacity stratum |

三者必须在同一 Hyper H200 host、同一 container/image/runtime stack 上，各自使用 fresh OS process 与独立 GPU，
可以三进程并发。总 ceiling 是 3 次 encode、6 次 generation；无 retry、无 top-up、无替换 state。

## 唯一新 condition

`strict_determinism_sdpa_frozen_encoded` 保持 D1b 的 parent automatic loader 与实际 top/text/vision 全 SDPA，且仍然
只 encode 一次、复用同一个 exact GPU tensor mapping 连续 generation 两次。新增 controls 必须在 CUDA 初始化前
精确成立：

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- `torch.use_deterministic_algorithms(True, warn_only=False)`；
- fixed seed `0`；
- cuDNN deterministic on、benchmark off；
- CUDA matmul 与 cuDNN TF32 off；
- float32 matmul precision=`highest`。

不允许因 runtime error 改为 eager、关闭 strict mode、开启 warn-only、换 seed 或换 state。如果 SDPA 路径包含
PyTorch 不支持的 deterministic operation，只记录异常 class，D2 verdict 为 `INVALID_RUNTIME_FAILURE`。

## 信息边界与 verdict

Git-frozen parent 是 D1b exact aggregate，SHA256=
`b419f9640cb46276b7a52e292d6feabd81311f66efb22563642ce80c4561919c`。D2 只序列化 equality boolean、
input-unchanged boolean、调用计数、class-only failure 与 source/runtime/process identity；action、decoded text、
token ids、coordinates、logits、KL 和 utility 都禁止落盘。

verdict 优先级固定为：

1. 任一 runtime/process failure：`INVALID_RUNTIME_FAILURE`；
2. 任一 stable control 在新 condition 下不稳定：`INVALID_STABLE_CONTROL_INSTABILITY`；
3. controls 稳定但 target 仍不稳定：`NO_GO_STRICT_DETERMINISM_REPEAT_INSTABILITY`；
4. 三者全部稳定：`PASS_STRICT_DETERMINISM_REPEAT_STABILITY_DIAGNOSTIC`。

即使得到 PASS，也只允许另立一个新的 strict-profile full-roster substrate/throughput Source-A，并由该新版本自行
重跑完整 roster。D2 PASS 不追认 D1b 为 PASS，也不直接解锁 labels、training、matched-NLL 或 closed-loop。

## Source-A → Execution-B

- source config：`code/configs/causalcache_set_utility_action_stability_diagnostic_v3.json`；
- source-only config 禁止 CUDA/model load/result write；
- canonical config SHA256=`4531b1c7e8c067b28c31662ee04b210dd2009031a9f92b4689128f701074b18d`；
- 53-file transitive source inventory SHA256=
  `86dbd5a6ed340dd46ce93709e6a7b59c07d200a04652723407e28683b917066b`；
- Source-A 是包含本协议、实现、测试与 canonical config 的 clean pushed `main` commit；
- Execution-B 必须是 Source-A 的 direct child，且唯一 changed path 是
  `code/configs/causalcache_set_utility_action_stability_diagnostic_v3_execution.json`；
- GPU 启动前运行项目统一 5 秒 fleet preflight；envelope 精确绑定 3 个 GPU UUID、host、container、image、runtime、
  parent artifact 与 run root；
- raw attempts/terminals/logs 留在 `/data`，状态 `LOCAL_FORENSIC_NOT_UPLOADABLE`；Git 只回写 metric-safe aggregate、
  summary 与进展记录。

Source focused verification 为 `40 passed`（runtime 7、diagnostic 22、contract 11；均含相关 v2 回归），另有
execution/aggregate/CLI v2 回归 `9 passed`；source validator 与 `py_compile` 通过。

## Formal result

Source-A=`60fd971b1614afba2601c838a34f7952d2334eb0`，direct-child Execution-B=
`38e7f53b83c3b943e6c5a60c15c01ac382e00576`。execution envelope 为 9,563 bytes / SHA256=
`1f194b90b2bdd780478ce112d25407ca78d7a77a749713bc048aed8ffa692f53`。唯一 formal run 完成 3/3 fresh
processes、6/6 generation、3/3 encode、retry=0；target 与两个 controls 的三层 action equality 和 frozen-input
checks 全部为 true，无 runtime failure。

exact aggregate 为 6,643 bytes / SHA256=
`cc8ae9e18d72002d642c9111fa7bcb78574d5f3dabc68a54d67586e62c3078e5`，独立只读 rebuild 逐 byte 相同；正式
verdict=`PASS_STRICT_DETERMINISM_REPEAT_STABILITY_DIAGNOSTIC`。这只授权另立新的 full-roster strict-profile
Source-A；D1b 仍保持 NO-GO，labels、training、matched-NLL 与 closed-loop 继续 locked。Git-safe 结果见
[`../data/results/set_utility_action_stability_diagnostic_v3/`](../data/results/set_utility_action_stability_diagnostic_v3/)。
