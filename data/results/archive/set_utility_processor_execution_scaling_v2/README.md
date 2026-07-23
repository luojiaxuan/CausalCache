# Processor v2 execution scaling 与 formal launch

本目录只保存 Git-safe execution summary，不复制 forensic tar、raw image、OCR、candidate、worker log 或模型。

四进程联合 smoke 从旧 `dc90664` partial tar 的四个安全前缀各只读加载 8 条真实 query；旧 bytes 没有被修改、
复制或追认为可恢复 artifact。四个 process 同时构造每进程 8 个独立 AutoProcessor runtime，总计 32 slots；
joint wall 为 41.26 秒，四个 runtime getter 均确认 PyTorch intra-op=28 / inter-op=1，aggregate peak RSS 约
48 GiB，没有 OOM 或 import-guard drift。该 smoke 只验证执行并行度，不是 restoration、label 或 selector 结果。

正式 run 随后从 clean detached `e976b990ddb089cdea3b04ea15e5c911d5670d40` 启动。source/config/snapshot
validator 与 Hyper00 committed focused suite 均通过；启动稳定窗口的四个 OCR process 合计约 110.13 CPU cores，
表明 4 logical shards × 每 shard 32 OCR slots 已基本吃满 112 个 physical cores。当前只有 `.incomplete` staging，
状态严格为 `RUNNING_INCOMPLETE_NOT_UPLOADABLE`；只有 atomic root、exit=0、committed postflight 和 Git-safe result
commit/push 后才能进入 `PENDING_HF_UPLOAD`。

完整 machine-readable 数值见 [`summary.json`](summary.json)。
