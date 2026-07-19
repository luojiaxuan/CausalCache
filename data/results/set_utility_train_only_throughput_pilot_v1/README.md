# Set-utility train-only throughput pilot v1：formal failure

## 结论

唯一 v1 formal attempt 已消费，但在任何 GUI-Owl native call 之前 fail closed。四个 worker 均完成
no-retry attempt 写入，随后在 semantic input load 阶段返回 `ValueError`：immutable processor candidate
schedule 的 exact bytes 不是 v1 consumer 额外要求的 canonical-pretty JSON。

正式状态为
`INVALID_SET_UTILITY_TRAIN_ONLY_THROUGHPUT_PILOT_V1_CANDIDATE_SCHEDULE_CANONICALIZATION_CONTRACT_DRIFT`。
这不是 microbatch `NO_GO`：实际 native calls 为 `0/84`，没有进入 model runtime factory，没有产生
`aggregate.json`，也没有选择 mb1 或 mb2。

## Evidence boundary

- source A：`5158f2ad04e456fd088765973da6a99064b4efcf`；
- execution-envelope B：`c7d5b31f8235124699ecbee5beb5240c72eb8977`；
- envelope SHA256：`adf50363d9b3623b84b9229d709f796d7b1971faf7e4056c468cfe3b55ce228e`；
- 四个 worker：`4/4` claimed、`4/4` terminal、`0/12` pairs、`0/84` native calls、`0` retries；
- post-attempt 只读 failure decomposition 精确复现：`candidate schedule is not canonical pretty JSON`；
- raw metric-only terminal、attempt 和日志保留在 Hyper00
  `/data/runs/causalcache-throughput-pilot-v1-5158f2a`，状态为 local forensic，不上传 HF；
- Git-safe hashes 与完整边界见 [`summary.json`](summary.json)。

v1 identity 不得重跑、删除 claim 或覆盖 terminal。下一步只能保留 v1 全部 bytes，另立 versioned consumer
canonicalization repair；不得改变 candidate schedule、12-state roster、mb1/mb2、84-call 预算、80% memory
阈值或 5% latency 判据。
