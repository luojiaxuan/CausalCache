# Set-utility action-stability diagnostic D1：formal result

## 结论

唯一 formal run 已完成 exact 6-state aggregate，但科学 verdict 为
`INVALID_RUNTIME_FAILURE`。三个 `decision:010` state 的
`eager_frozen_encoded_control` 均在第一次 generation 以 `OutOfMemoryError` 结束；因此 D1 不能对这三个
state 给出 auto-vs-eager 诊断，也不能据此解锁 12-state throughput、restoration labels、predictor training、
matched-NLL 或 closed-loop。

该失败不是空跑：两个 `decision:006` mismatch states 与一个 stable control 完整跑完三个 condition。其中
`0296753837938323:decision:006` 的 `auto_fresh_encode` 两次输出不一致，而同进程复用 frozen encoded input 与
独立 eager process 都稳定，预注册 diagnosis 为 `FRESH_VS_FROZEN_PATH_ASSOCIATION`；另一个 mismatch state 和
stable control 均为 `PARENT_MISMATCH_NOT_REPRODUCED`。这是 association 证据，不是对 processor、kernel 或
hidden state 的单一因果归因。

## 正式计数

- source A2：`121c8621bae4f56a57060ea2fb402cfa3d8c7146`；execution B：
  `5fa1fdd45dba1fa95c7a69ee84d6ddb5b3f5166e`；
- aggregate：14,672 bytes，SHA256=
  `2debde6de3f552e9551d0ee37d25b82fa2ca85dfb42746d389dde39eff577ba2`；
- 8/8 profile-worker attempts 与 8/8 terminals 完成，stage barrier 通过，retry=`0`；
- 6 states，33/36 generation calls，24/24 encode calls；teacher/KL/restoration/label/training/HF mutation 全为
  `0`；
- diagnosis histogram：`FRESH_VS_FROZEN_PATH_ASSOCIATION=1`、
  `PARENT_MISMATCH_NOT_REPRODUCED=2`、`INVALID_CONDITION_EXECUTION_FAILURE=3`。

完整 metric-safe equality matrix 位于 [`aggregate.json`](aggregate.json)，执行与 forensic hashes 位于
[`summary.json`](summary.json)。二者均不含 action、coordinate、text、token ids、decoded output、logits 或其
digest。

## 执行与证据边界

formal run 在 Hyper00 的同一既有 container 内使用 host GPU `0,1,3,4` 四张 H200。auto 与 eager 是独立
process/stage；所有 auto terminals 完成后 190.222 秒才出现首个 eager attempt。10 秒 preflight 于
`2026-07-19T12:01:24Z--12:01:38Z` 完成，未发现 active compute process；执行从
`2026-07-19T12:04:47.338779Z` 到 aggregate 落盘的 `2026-07-19T12:11:08.507276Z`。

这是仅 6 个 state 且 4-way 不等长分片的短诊断。auto startup monitor 捕获到的 subset window 平均 67%、峰值
100%，eager 在下一次采样前已结束，因此没有形成满足“每张卡代表性窗口至少 80%”的 throughput evidence。这个
事实不改变 metric-safe D1 failure aggregate，但禁止拿本 run 声称 GPU throughput；下一版会重新平衡分片或减少
GPU 数，并把 memory-safe control 与吞吐证据分开。

raw attempts、terminals、stdout/stderr 与 preflight raw log 只保存在 Hyper00：

```text
/data/runs/causalcache-action-stability-d1-121c862
```

状态为 `LOCAL_FORENSIC_NOT_UPLOADABLE`。它们不是 reusable dataset，不上传 Hugging Face；Git 中的
metric-safe aggregate/summary 是本次轻量结果的 source of truth。

## 下一步

不得重跑或 top-up D1。下一步先冻结新的 versioned、memory-safe control diagnostic，隔离“全 eager attention
在较长 decision-10 input 上 OOM”与真实 generation instability；在该控制闭合前，不进入 12-state throughput、
label generation 或 utility predictor training。
