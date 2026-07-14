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

## 当前 artifact 状态

- Git 代码、配置、论文与轻量测试 fixture：本仓库 `main`；
- 可复用 attribution 数据集：尚未生成，目标 Hugging Face dataset repo 待 owner 确认；
- gate checkpoint：尚未生成，目标 Hugging Face model repo 待 owner 确认；
- 当前没有仅存在共享机器或本地磁盘上的正式实验 artifact。

## 未决策项

- 主 frozen policy backbone 与 transfer backbone；
- 离线真实轨迹首选 GUIOdyssey 还是更容易复现实验 logits 的等价数据；
- 目标 Hugging Face owner 与最终 dataset/model repo ID。

这些项目必须经过可获得 logits、许可证、磁盘和算力检查后再冻结，不能为了填配置而猜测。

## 下一步

实现 budget-conditioned restoration estimator，并在完全可控的 synthetic frozen policy 上验证：精确值恢复、负 gain 不强制选择、$K$ 与 ranking stability、top-budget overlap 和 oracle utility。
