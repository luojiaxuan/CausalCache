# Synthetic Phase-0 Attribution Validation

> 这只是 estimator implementation validation，不是论文效果证据，也不能替代真实 GUI policy 实验。

## 设置

- Visual-token budget: `512`
- Exact attribution selection: `[0, 2, 3]`
- Global subset optimum: `[0, 1]`
- Negative-gain events: `[5, 6]`
- Exact-score selection / subset-optimum utility: `0.859`
- Exact-selection reconstruction error: `15.319`

## 采样稳定性（5 seeds）

| K | Spearman mean +/- std | Jaccard mean +/- std | Mean SE | Utility / exact-score | Utility / subset optimum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.981 +/- 0.031 | 0.900 +/- 0.224 | 2.991 | 0.977 | 0.839 |
| 8 | 0.995 +/- 0.011 | 1.000 +/- 0.000 | 2.196 | 1.000 | 0.859 |
| 16 | 0.990 +/- 0.013 | 1.000 +/- 0.000 | 1.656 | 1.000 | 0.859 |
| 32 | 0.986 +/- 0.013 | 1.000 +/- 0.000 | 1.186 | 1.000 | 0.859 |

## 判读边界

该实验检查采样器、预算约束、负 gain 与选择器的接口是否闭合。它不检查真实视觉证据、模型 logits、teacher coverage、closed-loop success 或 matched-NLL 假设。

该可控例子中 exact marginal-score selection 只达到 global subset optimum 的一部分，说明 event interaction 会破坏可加性；真实实验必须报告 reconstruction error，并保留 budget-aware set loss。
