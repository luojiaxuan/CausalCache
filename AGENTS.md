# CausalCache collaboration rules

## 每一步的完成条件

每个 material step 必须同时完成：实现或实验、验证、更新顶层 README/相关 docs/轻量 result summary、
commit、push canonical `main`。未 push 的本地 commit 不是 source of truth。

## 目录职责

- `README.md`：总索引、当前结论、下一步和团队交接入口；
- `paper/`：AAAI LaTeX package；
- `code/`：package、CLI、tests、configs、requirements；
- `data/`：只放小 fixture、轻量 summary/CSV/manifest；
- `docs/`：实验契约、执行逻辑、进展、失败与决策。

Git 保存代码、config、论文、文档与轻量 metadata。Reusable datasets/traces 放 Hugging Face dataset
repo；model weights/checkpoints/adapters 放 Hugging Face model repo。README/docs 必须记录 repo、tag 或
immutable revision、schema/provenance 与生成命令。

## 实验与计算

- 科学参数必须由 committed config 或显式 CLI 参数传入，不用环境变量覆盖；Docker cache 路由是
  基础设施例外；
- CPU-only 工作默认使用输入分片、可用物理核心和内存/I/O 能有效支撑的最大并行度；先用短程 scaling
  measurement 找到吞吐拐点，再冻结 worker 数。不得仅为日志简单或实现方便把可分片工作串行化；只有
  已 committed 的 historical/formal contract 才保持原调度，并以新的 versioned path 引入并行修复；
- 通用 GPU 资源发现、5 秒 idle cleanup、跨机 shard 调度和单机启动遵循全局 GPU skills。正常路径直接
  使用完整选定 allocation，不要求预先单卡/双卡 smoke、广泛 host/disk/container sweep 或 startup
  utilization gate；失败、无进展、OOM 或明显过慢时再按需诊断；
- 本项目经用户持续授权，Hyper00 与 Hyper01 各自最多可使用 8 张 GPU；可并行的 rollout、label、training
  和 evaluation 应按实际吞吐尽量使用可用配额，不把该上限误解为必须占满。其他 hosts 仍遵循全局上限；
- 已 committed 的 frozen formal execution contract 不因通用策略更新而被追溯修改；
- Hyper01 默认承担 H200 policy/offline 工作；当前 AndroidWorld closed-loop MVP 留在已验证的 Aries
  stack，迁移条件见 `docs/execution.md`；
- 不同芯片不要求 bitwise logits 一致；正式结果分别报告 latency、显存和 runtime metadata。新 host、新
  image 或已经出现兼容性问题时再按需执行 behavioral smoke。

## 安全与审计

- `~/hf_key.txt`、token、`.secrets/`、raw traces、checkpoints 不得进入 Git；
- 不打开 sealed AndroidWorld test split，不为结果事后修改 prompt、equivalence、threshold 或 task
  denominator；
- 每个正式结果记录 Git/HF revision、完整 argv、config hash、host/GPU、container digest、library
  versions、dtype、seed、起止时间和 failure classification；
- 详细执行与回写流程以 `docs/execution.md` 为准。
