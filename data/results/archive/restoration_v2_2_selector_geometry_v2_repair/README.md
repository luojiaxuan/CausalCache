# Restoration v2.2 selector geometry v2 repair

## 结论边界

这是 v1 table-only selector geometry 的 reporting-only versioned repair。它没有改变 selector algorithm、
utility/recovery、bootstrap 或 method-shaping 值，也没有新增 policy forward、KL、gate training、matched-NLL、
closed-loop、confirm/test access。v1 目录未被覆盖。

## Primary `n=4,B=2`

| Selector | Train recovery | Development recovery | Overall recovery |
| --- | ---: | ---: | ---: |
| Exact subset | 0.877376 | 0.925342 | 0.893364 |
| True conditional greedy | 0.877376 | 0.925342 | 0.893364 |
| Budget-conditioned independent | 0.835977 | 0.766303 | 0.812753 |
| Full-path Shapley independent | 0.815367 | 0.884128 | 0.838287 |
| Recent | 0.554136 | 0.019666 | 0.375979 |
| Random exact expectation | 0.430032 | 0.139471 | 0.333178 |

## 修复内容

- interaction report materialize 完整 `role × n × B × strength × negative-marginal` joint Cartesian：
  `144` cells，其中 nonempty `83`、empty
  `61`；每个 cell 均含 10 个 selectors，空 cell 使用统一 null schema；
- analytic exact-cardinality random 显式报告 `k=B`、expected exact-match 与 expected Jaccard。primary overall
  cardinality/match/Jaccard 分别为 `2.000000`、
  `0.155556`、`0.379630`；
- train-only tertile 在每个 `n` 内独立计算，并原样应用到 matching-`n` development states；独立 validator
  从 state records 重算 cutpoint、assignment、random expectation 与完整 cell coverage。

## Artifact identity

- source commit：`9a4eca5a53c2a9a3340c6274b9fa5ff9012a5a64`；
- v2 repair contract SHA256：`2d312f54559f67aafe7efec2656d23171000e8b0f41d2d3c923f6a3c8b43be4c`；
- parent v1 contract SHA256：`8022dcdec272916b7975d696a3ce6b54022c7414cd348a715c55b0d3d694dad5`；
- immutable label revision：`8f6baae5c0b23b08915fa1b0fb848dd519b4c8db`；
- complete scientific payload SHA256：`cc505443a7efdc68c8eeca754f24c9143cabf72a090f024fde05011e783cbb21`；
- state-budget rows：`180`，SHA256
  `b3f67714bb5667ceda945a3cb953b8987108aef607d5819a617272d48450cb03`。
