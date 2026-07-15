# Ablation 索引

本目录记录尚未进入 frozen scientific contract 的方法比较、诊断设计与执行前边界。文档是 Git 中的
source of truth，但不等于实现完成、实验授权或论文证据。

| Ablation | 状态 | 回答的问题 |
| --- | --- | --- |
| [Interaction-aware memory gate](interaction_aware_gate.md) | source-only proposal | coalition-conditioned teacher 是否需要 set-conditioned student，以及 interaction 强弱如何影响 oracle gap |
| [Subset search](subset_search.md) | source/config/runner frozen；formal CPU result pending | exact、true conditional greedy、bounded exchange 与 beam 的 search regret/query cost，以及它与 averaged-score projection gap 的区别 |

任何 ablation 在运行真实 policy、读取 untouched confirm output 或修改论文主方法前，都必须先进入独立的
versioned contract。当前 v2.1 的 `NO_GO_V2_1_FULL_45_SUBSTRATE` 保持不变。
