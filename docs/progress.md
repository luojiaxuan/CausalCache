# 项目进展

## 当前目标

在 AAAI-27 截止前完成一个最小但可证伪的 CausalCache 实验链路：validated offline attribution、multi-budget gate、AndroidWorld closed-loop frontier 与 matched-NLL mechanism test。

## 已完成里程碑

### 2026-07-14：论文骨架

- 使用 AAAI-27 官方 author kit 建立 anonymous submission LaTeX；
- 编译并逐页检查 3 页论文骨架；
- 明确 Shapley-style attribution 不是 estimator novelty；
- 将 claim 收窄为 decision-time behavioral restoration，并由 closed-loop 验证长期意义；
- 引入 budget-conditioned value、validated teacher、policy-visible context budget 和可为空的选择。

### 2026-07-14：实验契约 v0.1

- 冻结 archive / low-fidelity index / policy-visible context 三层接口；
- 冻结五字段低保真事件 schema 与 executable action canonicalization；
- 冻结 reference validation、near-budget coalition、稳定性指标和 matched-NLL 协议；
- 增加机器可读配置、fixture、验证 CLI 与单元测试。

### 2026-07-14：Synthetic estimator validation

- 实现 shared antithetic permutation 与 variable-cost maximal near-budget coalition；
- 实现 exact permutation enumeration、standard error、positive-value knapsack、Spearman 与 top-budget Jaccard；
- 在 8-event 非线性 synthetic frozen behavior 上验证两个负 restoration-gain event 不会被强制选择；
- 5 seeds 下，$K=4$ 的 mean top-budget Jaccard 为 0.90，$K\ge 8$ 为 1.00；mean standard error 从 2.99（$K=4$）下降到 1.19（$K=32$）；
- exact marginal-score selection 的实际 utility 仅为 global subset optimum 的 85.9%，确认 interaction 会破坏 attribution 的可加性，后续真实实验必须报告 reconstruction error 并保留 budget-aware set loss。

以上结果只验证实现与指标链路，不构成论文效果证据。

### 2026-07-14：GUIOdyssey deterministic pilot artifact

- 从 `cua-lite/GUIOdyssey` 固定 revision 的单个 Parquet shard 提取 upstream GUIOdyssey 成功轨迹 `0054832199799795`；
- 实现 tap、long press、type、swipe、system button 的 executable canonicalization，以及 action 与 terminal signal 分离；
- 生成 10 screenshots、9 events、9 decisions 的 deterministic tar shard，连续两次构建 SHA256 一致；
- 明确拒绝失败轨迹，且 policy-visible manifest 不包含 expert inline reasoning；
- 上传到私有 Hugging Face dataset `gavinlaw/causalcache-guiodyssey-pilot-mobile` revision `6c840b1be9d96d23425c51ba7c02b35063cfa731`；
- 该 artifact 只验证真实数据接口，不构成 restoration 或任务成功率证据。

### 2026-07-14：Qwen3-VL real-policy forward smoke

- 固定 `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` 的 14 个加载文件，并校验四个 weight shard 的 LFS SHA256；
- 实现统一的 summary-only、mixed-fidelity、full-history prompt contract 与 JSON executable action parser；
- 在 GUIOdyssey trajectory `0054832199799795` 的 decision step 4 上，三种 fidelity 输入都生成与 recorded action 完全匹配的 `type_text("Cryptocurrency Market")`；
- 输入长度分别为 501、865、1,593 tokens，单张 A6000 peak allocated memory 为 17.71、17.85、18.10 GB；
- 结果见 `results/qwen_policy_smoke/`。这只验证 policy forward 链路，不能说明 restoration 有收益。

### 2026-07-14：Qwen3-VL full-history coverage（negative result）

- 在同一成功 trajectory 的全部 9 个 decisions 上运行 full-history deterministic generation；
- 9 个输出中 6 个满足 executable JSON schema，只有 decision step 4 的 `type_text` 与 recorded action 匹配；
- tap 为 0/7、swipe 为 0/1、type_text 为 1/1，总 coverage 为 1/9（11.1%）；
- 3 个长历史状态输出 non-executable canonical target，另外 5 个视觉 action 的 coordinate bin 不匹配；
- 按预注册 validation contract，拒绝 Qwen3-VL-8B-Instruct 作为主 frozen teacher，不放宽标准；
- 结果见 `results/qwen_policy_coverage/`。这不是对 CausalCache 机制的 falsification，而是 backbone selection 的负结果。

## 当前 artifact 状态

- Git 代码、配置、论文与轻量测试 fixture：本仓库 `main`；
- GUIOdyssey pilot：私有 Hugging Face dataset `gavinlaw/causalcache-guiodyssey-pilot-mobile@6c840b1be9d96d23425c51ba7c02b35063cfa731`；
- Rejected policy candidate：上游 Hugging Face model `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；共享机器副本只是可重建 cache；
- 完整 attribution dataset：尚未生成，pilot 扩展后仍需单独登记 revision；
- gate checkpoint：尚未生成，目标 Hugging Face model repo 待 owner 确认；
- 当前没有仅存在共享机器或本地磁盘上的正式实验 artifact；Taurus 目录只作为 HF artifact 的 staging/cache。

## 未决策项

- transfer backbone；
- canonical action path 上 teacher-forced component distance 的具体 token boundary；
- pilot 应扩展到多少 app、trajectory 和 horizon 才足以进入 attribution 主表。

这些项目必须经过可获得 logits、许可证、磁盘和算力检查后再冻结，不能为了填配置而猜测。

## 下一步

筛选具有公开权重、可访问 logits、GUI action grounding 能力和许可清晰的 GUI-tuned policy；先用同一 9-decision contract 做 coverage gate，只有覆盖率足够的 backbone 才进入 teacher-forced distance 实现。
