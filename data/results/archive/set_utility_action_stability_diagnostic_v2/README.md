# D1b memory-safe SDPA action-stability result

## 正式结论

D1b 的唯一 formal run 有效完成，正式 verdict 为：

```text
NO_GO_SDPA_CONTROL_REPEAT_INSTABILITY
```

这不是 runtime-invalid。六个 fresh state processes 全部完成，实际执行 `12/12` generation、`6/6` encode、
retry=`0`；三个此前在 D1 eager control 中 OOM 的 `decision:010` state 均在 memory-safe SDPA profile 下完成。
但 `0296753837938323:decision:006` 的两次 generation 仍在 exact sequence、decoded output 和 canonical action 三层
全部不一致，因此预注册 PASS gate 未通过。

## State-level result

| state | role | controlled-SDPA repeat | diagnosis |
| --- | --- | --- | --- |
| `0296753837938323:decision:006` | mismatch | unstable | `MIXED_D1_FRESH_PATH_AND_SDPA_CONTROL_INSTABILITY` |
| `0310939638496410:decision:006` | mismatch | stable | `PARENT_MISMATCH_NOT_REPRODUCED_UNDER_SDPA_CONTROL` |
| `0336706763935531:decision:006` | stable control | stable | `STABLE_CONTROL_REPRODUCED` |
| `0271654003819383:decision:010` | mismatch | stable | `AUTO_VS_CONTROLLED_SDPA_PROFILE_ASSOCIATION` |
| `0279447750102246:decision:010` | mismatch | stable | `PARENT_MISMATCH_NOT_REPRODUCED_UNDER_SDPA_CONTROL` |
| `0268406573756492:decision:010` | stable control | stable | `STABLE_CONTROL_REPRODUCED` |

六个 condition 的 frozen encoded input 在 before/between/after 三个检查点都保持 exact unchanged。唯一不稳定 state
没有 execution failure，完成了两次 generation；因此其 NO-GO 不能解释为 OOM、input mutation、parse failure 或缺失
调用。它仍只是 invocation-level association，不识别具体 kernel、tie-breaking 或 CUDA nondeterminism 机制。

## Provenance

- Source-A：`fe6f6b69adf2acd5457026835529f2ca3e682dd2`；
- direct-child Execution-B：`db2c848af310d7559247895f418f69255465eb73`；
- source config SHA256：`07ec807effd9a1655b0572e49a835e61aec5192503d16bd5c2ee6a3d7a18c77f`；
- 54-file source inventory SHA256：`a58cbda4db59bf6f951783c94f606e9e00bb310191e371de099256c4286c55b0`；
- execution envelope：19,349 bytes，SHA256=
  `7adb460bbe65d08cc7fe2bea992cf77f544d8be18f5c43f28c84d4ee4e9171f8`；
- exact aggregate：17,011 bytes，SHA256=
  `b419f9640cb46276b7a52e292d6feabd81311f66efb22563642ce80c4561919c`；
- historical D1 aggregate SHA256：
  `2debde6de3f552e9551d0ee37d25b82fa2ca85dfb42746d389dde39eff577ba2`；
- runtime：Hyper00 / 4×H200 / container `d57521b...e657` / driver `570.172.08` /
  PyTorch `2.11.0+cu130` / Transformers `5.6.0`；
- raw run：`/data/runs/causalcache-action-stability-d1b-fe6f6b6`，状态
  `LOCAL_FORENSIC_NOT_UPLOADABLE`。

fresh preflight 覆盖 `2026-07-19T12:53:48Z` 至 `12:54:13Z`，绑定四张 H200；wave 0 四进程并发，wave 1
两进程并发，且第一条 wave-1 attempt 晚于最后一条 wave-0 terminal。aggregate 已从 6 attempts + 6 terminals
独立只读重建，17,011 bytes 与 SHA256 均逐 byte 相同。

启动利用率 sampler 因 operator shell quoting 错误没有形成可用证据，因此本 run 不提供 GPU utilization 或
throughput claim；该监控失败没有修改 state process、terminal、aggregate 或预注册科学 verdict，也不允许据此重跑。

## 权限结果

`PASS_MEMORY_SAFE_SDPA_NUMERICAL_CONTROL_REPEAT_STABILITY` 是冻结新 12-state throughput v3 Source-A 的唯一
许可。本次为 NO-GO，因此：

- 不启动或冻结 throughput v3；
- 不生成 restoration labels；
- 不训练 utility predictor；
- 不运行 matched-NLL 或 closed-loop；
- 不 retry、top-up 或替换失败 state。

本目录的 `aggregate.json` 是 exact metric-safe formal aggregate；`summary.json` 是 Git-safe provenance 与权限
摘要。raw attempts、terminals 和 logs 不进入 Git/HF，也不是 reusable dataset/model artifact。
