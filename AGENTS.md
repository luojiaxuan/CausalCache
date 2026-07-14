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
- GPU job 前执行 host/GPU/disk/container preflight、10 秒 idle cleanup，默认最多 2 张 GPU；
- 长任务启动 utilization monitor，低于 90% 立即检查；
- Hyper01 默认承担 H200 policy/offline 工作；当前 AndroidWorld closed-loop MVP 留在已验证的 Aries
  stack，迁移条件见 `docs/execution.md`；
- 不同芯片不要求 bitwise logits 一致，但必须通过固定 behavioral smoke，并分别报告 latency、显存和
  runtime metadata。

## 安全与审计

- `~/hf_key.txt`、token、`.secrets/`、raw traces、checkpoints 不得进入 Git；
- 不打开 sealed AndroidWorld test split，不为结果事后修改 prompt、equivalence、threshold 或 task
  denominator；
- 每个正式结果记录 Git/HF revision、完整 argv、config hash、host/GPU、container digest、library
  versions、dtype、seed、起止时间和 failure classification；
- 详细执行与回写流程以 `docs/execution.md` 为准。
