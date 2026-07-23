# Gate v1 formal-58 cache v1 failed attempt

该 attempt 在 Hyper00 CPU-only runtime 从 clean pushed Execution-B
`079c0952017a9e2936f5741bbe15345255b0e481` 启动。runner 创建 mode-0600 global claim 后，下载并逐 byte
校验 feature transports；`expansion_feature_trajectories` 的实际 immutable SHA256 为
`fe93e9deeeb9a3c018227fb781b29f6efefefbb728db987584b650d9df353a6d`，而 source config 错误冻结为
`00fe93e9deeeb9a3c018227fb781b29f6efefbb728db987584b650d9df353a6d`。两者都是 64 位合法 SHA，错误值在开头
误插 `00`，并漏掉真实 digest 中连续出现的第二个 `fe` byte。

失败发生在 `_verify_downloads()`，早于 trajectory/OCR JSON semantic decode。因此 formal label semantic decode、
development/confirm access、cache write、gate training、HF mutation、matched-NLL 与 closed-loop 均为 0。HF destination
在失败后的 read-only 查询中仍不存在。

旧 claim 永久保留，不删除、不覆盖，也不从原 namespace 续跑。后续只能通过独立 transport-repair source/runner：

- secure-read 并绑定旧 claim 的 mode、size、SHA 和旧 B commit；
- 证明旧 feature completion 及其后所有 state/cache 仍不存在；
- 从 Git-pinned upstream completion 交叉验证 9 个 transport bindings；
- 只修复该一个 SHA leaf，并使用新的 local/HF namespace；
- 重新执行 Source-A → machine-generated Execution-B。

机器可读记录见 [`summary.json`](summary.json)。共享机上的 traceback 与空 stdout 保留在
`/data/logs/causalcache/gate-v1-formal58-cache-v1/`，不是 reusable canonical artifact。
