# D2 strict-determinism action-stability result

## 正式结论

唯一 3-state D2 attempt 有效完成，正式 verdict 为：

```text
PASS_STRICT_DETERMINISM_REPEAT_STABILITY_DIAGNOSTIC
```

不稳定 target `0296753837938323:decision:006` 与两个 frozen stable controls 均完成 1 次 encode、2 次 generation；
三者的 exact generated sequence、decoded output、canonical action 与 frozen-input before/between/after checks 全部
repeat-equal。总调用为 3/3 encode、6/6 generation、retry=0，无 runtime failure。

该 PASS 只说明：在本次冻结的三个 state 上，D1b 中唯一 controlled-SDPA repeat instability 与 strict deterministic
profile 存在可重复的 profile association，同时两个 controls 没有回归。它不识别具体 kernel 因果，不证明全部
12-state substrate 稳定，不追认 D1b 为 PASS。

## Provenance

- Source-A：`60fd971b1614afba2601c838a34f7952d2334eb0`；
- direct-child Execution-B：`38e7f53b83c3b943e6c5a60c15c01ac382e00576`；
- source config SHA256：`4531b1c7e8c067b28c31662ee04b210dd2009031a9f92b4689128f701074b18d`；
- 53-file source inventory SHA256：
  `86dbd5a6ed340dd46ce93709e6a7b59c07d200a04652723407e28683b917066b`；
- execution envelope：9,563 bytes，SHA256=
  `1f194b90b2bdd780478ce112d25407ca78d7a77a749713bc048aed8ffa692f53`；
- exact aggregate：6,643 bytes，SHA256=
  `cc8ae9e18d72002d642c9111fa7bcb78574d5f3dabc68a54d67586e62c3078e5`；
- parent D1b aggregate SHA256：
  `b419f9640cb46276b7a52e292d6feabd81311f66efb22563642ce80c4561919c`；
- runtime：Hyper00 / 3×H200 / container `42b7ee6...a992` / driver `570.172.08` /
  PyTorch `2.11.0+cu130` / Transformers `5.6.0` / CUDA `13.0`；
- raw run：`/data/runs/causalcache-action-stability-d2-60fd971`，状态
  `LOCAL_FORENSIC_NOT_UPLOADABLE`。

fleet preflight 在 `2026-07-19T17:20:55.294797Z` 完成 5 秒 sample，Hyper00 当时 8/8 H200 空闲，D2 绑定 host
GPU 0/1/2。标准 runner 首次在 Docker CLI GPU argument parsing 处失败，发生在 container/model/process 创建前；
随后使用相同 H200 profile 和显式 quoted device list 创建冻结容器，再 materialize Execution-B。该基础设施修复没有
更改 GPU allocation、image、mount、runtime 或科学参数。

aggregate 从三个 attempts 与三个 terminals 独立只读 rebuild，bytes 和 SHA256 逐项相同。三个 state stderr 各
687 bytes，仅保留于 raw run；aggregate stderr 为 0。Git/HF mutation、labels、training、matched-NLL、closed-loop、
sealed test 均为 0。

## 权限结果

D2 PASS 只授权未来另立新的 strict-profile full-roster substrate/throughput Source-A，并由新版本重跑其完整 roster。
当前仍然：

- 不生成 restoration labels；
- 不训练 utility predictor；
- 不运行 matched-NLL 或 closed-loop；
- 不把 D1b 的正式 NO-GO 改写为 PASS；
- 不 retry/top-up D2。

`aggregate.json` 是 exact metric-safe aggregate，`summary.json` 是 Git-safe provenance 与权限摘要。raw actions、
decoded outputs、tokens、attempts、terminals 和 logs 不进入 Git/HF。
