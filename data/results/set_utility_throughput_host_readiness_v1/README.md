# Set-utility throughput pilot host readiness v1

这是 `2026-07-19T09:12Z` 的只读资源快照，不是 GPU preflight、资源预留或正式 pilot result。检查过程中没有
启动 job、cleanup、停止容器或写远端。

当前首选是 Hyper00：7 张 H200 满足“无 compute app 且至少 99% 显存空闲”，pinned GUI-Owl model 与
18,730,620,511-byte processor root 已在本地盘，额外数据搬运为 0。GPU 2 被现有容器占用，不在候选中。

Hyper01 的 model/image 与 Hyper00 相同，但 8 张 H200 当前均有 compute/memory 占用且缺 processor root；
H100 有 3 张空闲 80GB 卡和充足磁盘，但缺 repo/model/processor，并使用不同 local image identity，需要搬运约
36.28GB 并单独做 runtime/behavioral binding。Aries 当前无空闲 A6000，repo dirty 且磁盘紧张。

因此 12-state throughput pilot 应优先在 Hyper00 的新 clean exact checkout 上单机完成，不混合跨 host
throughput 指标。正式启动前仍必须 fresh GPU preflight；若届时资源变化，再按同一检查重选。完整数值见
[`summary.json`](summary.json)。跨机时等待 processor immutable HF publication 后下载并逐 byte 验证，不把
临时本地副本当 source of truth。
