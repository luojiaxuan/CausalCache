# Gate v1 fresh-16 evaluation v1 首次执行

本目录记录 2026-07-18 在 Hyper00 上执行的 v1 formal attempt。该 attempt 在任何 fresh trajectory、OCR、
image 或 label semantic decode 之前 fail closed，不能产生 GO/NO-GO 结论，也不能续跑。

- Source-A：`97694eff052ecbdc5f12f58b6f9ee10f4dd616ab`；
- Execution-B：`a8bb27ccf9b8820c1d8487f63883f813e6845645`；
- Source-A config SHA256：`c98647aecf6b07e0ccccf1601b5e21289b595b7a4a45cc7ab9a542b1bd7dff2e`；
- Hyper00 container：`sglang-omni-jaxan-07181130`，unprivileged `runc`，精确 2 张 H200；
- formal start：`2026-07-18T03:40:39Z`；exit code `1`；
- failure：derived-input immutable repo inventory validation 在下载/解码前拒绝执行。Source-A 只把本阶段消费的
  4 个 label-expansion 文件传给 exact-tree validator，但 revision
  `630363a6adb692d72774f16dd0653a50216313ff` 还合法包含 9 个历史 artifact path；
- old state namespace 只保留 `runtime_receipts` 与 `global_claim`，不删除、不覆盖、不续跑；
- planned HF destination `gavinlaw/causalcache-gate-v1-fresh16-evaluation-mobile` 仍不存在，remote mutation 为
  `0`；
- fresh semantic decode、label access、policy worker、GPU forward、model checkpoint load、primary report、
  matched-NLL、closed-loop、confirm 与旧 dev-5 operation 都为 `0`。

完整轻量字段见 [`summary.json`](summary.json)。原始 mode-0600 log、Docker receipt 与 ordered-state receipts
保留在 Hyper00 `/data02/jaxan` 下；它们不是成功 artifact，也不上传到 planned result repo。

下一步必须另立 inventory-repair Source-A/Execution-B：精确绑定该 immutable revision 的完整 15-path tree，
先验证旧 failure receipt/旧 successor absence，再使用全新的 local state/artifact/runtime-receipt namespace。
在修复 B 之前仍禁止 fresh semantic access。
