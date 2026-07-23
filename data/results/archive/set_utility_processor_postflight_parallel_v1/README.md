# Set-utility processor parallel postflight v1

状态：`VALID_COMPLETED_SET_UTILITY_PROCESSOR_FREEZE_V2_PARALLEL_POSTFLIGHT_V1`。

该结果只验证 versioned 4-worker completed-root postflight 的语义等价与吞吐，不替代 `e976b99` formal 的
canonical committed postflight。validator 来自 clean detached `de4c0348d47d2bc19b462cb49a6178d670118c7c`，
historical context 来自 clean producer `e976b990ddb089cdea3b04ea15e5c911d5670d40`。

4 个 semantic workers 分别读取 4 个 frozen tar，主线程按 `[0,1,2,3]` 聚合。运行耗时
`908.558s`，canonical reference 为 `2549.669s`，实测加速 `2.8063x`。stdout 5,490 bytes，SHA256=
`90185d5271bd084e73cb07ea361d03a58ef4eba40ba5ff151f8878a283d5f512`；stderr 为空。

与 canonical 相比，四个 shard、candidate artifact、operation budget、image/runtime identity、format tally、
stored/metadata-only OCR validation counts 与 structural status 逐字段相同。并行运行开始后 formal root 中
mtime 更新文件数为 0，producer/validator checkout 结束后均 clean。完整轻量记录见
[`summary.json`](summary.json)；raw evidence 保留在 Hyper00
`/data/logs/causalcache-parallel-postflight-de4c034`。

本结果没有生成 policy output、restoration label、checkpoint、matched-NLL、closed-loop result 或 HF mutation。
