# Independent confirm-20 restoration continuation v1

本目录保存 restoration-guided independent 主线一次性 confirm-20 的轻量完成记录。正式结论是
`NO_GO_INDEPENDENT_CONFIRM`：20/20 reference 与 memory-sensitive checks 通过，但 independent / exact raw
utility ratio 为 `0.821298 < 0.85`，且 independent 相对 recent、OCR/RGB、policy-vision 的 mean raw delta
均为负。因此 paired closed-loop、matched-NLL 与 sealed AndroidWorld test 没有获得执行授权。

完整 20-state report 与 raw records 不复制进 Git，canonical artifact 位于 private Hugging Face dataset
[`gavinlaw/causalcache-independent-confirm20-mobile@a0b408e`](https://huggingface.co/datasets/gavinlaw/causalcache-independent-confirm20-mobile/tree/a0b408e58d629299be334a74ecbd0ec2fa2ed1fc/independent-confirm20/v1/report)，
tag 为 `independent-confirm20-v1`。`completion.json` 是 formal output 的逐字节副本；`summary.json` 保存关键
metric、gate、runtime、operation count、Git/HF identity 与 postflight。

formal runner 在 completion 持久化、report publication、tag 与 immutable replay 全部成功后，最后打印
`MappingProxyType` completion 时触发 stdout JSON serialization `TypeError`。该错误发生在科学终态之后；HF
postflight 与 durable completion 均通过。仓库只修复 CLI transport 并增加回归测试，没有重跑 confirm，也没有
改变任何科学输入、阈值或结果。

验证结果：CLI regression file `5 passed`；post-B continuation focused suite `78 passed, 2 deselected`，两项
deselect 都是历史 Source-A 测试在 Execution-B 已合法存在后仍要求 B absent。全仓 canonical run 为
`1474 passed, 8 skipped, 37 failed, 620 subtests passed`；其中 36 项是同类历史 Source-A lifecycle tests，
1 项是 full-suite import-order 下的 CPU-only boundary test，后者在隔离进程中通过。没有本次改动引入的新失败。
