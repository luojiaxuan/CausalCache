# Set utility decision distillation v2

## 目的

最后一次检验 variable at-most-`B` 主线。v1 已证明 restoration oracle 存在，但 scalar subset regression、
top-3 enrichment 与单一 nested greedy prefix 没有在固定 tune truth 上超过 Recent。v2 不增加同分布 trajectory，
只修正 teacher--student decision contract。

## 冻结方法

- utility predictor 仍学习 `U(q,C,S)`，不输入固定 `B`；同一 checkpoint 服务 `B=1,2,3,4`；
- deployment search 改为 width-4 beam。beam 允许保留 non-positive prefixes，各预算从已访问的所有
  `|S|<=B` coalitions 独立选择，因此结果不要求形成 nested prefix；
- train-only DAgger 固定只做一轮：从 v1 checkpoints 的 beam frontier 取 base coalitions，并对每个 base
  标注所有 one-event expansions，而不是只标 predicted top-3；
- loss 新增带 STOP action 的 conditional listwise distillation 与 differentiable expected regret；STOP 的真实
  marginal 固定为 0，只在完整 expansion group 上计算；
- DeepSets 与 Set Transformer 共享 contextual GUI-Owl hidden cache、labels、loss、seed 与 search，只替换
  set aggregator。

## 数据与防火墙

- parent contextual input content SHA256=`2711ab55...09d407f`；
- contextual hidden cache content SHA256=`44405c2c...597604`；
- targeted train fraction=`10%`，每 trajectory 最多 3 states，long/very-long 权重合计 70%；
- evaluation 不加载，tune labels 不进入 optimizer；
- 不根据本轮 fixed-tune 结果追加第二轮 DAgger、改 loss 权重或改 beam width。

## GO/NO-GO

在固定 1,063-state tune truth 上，以真实 GUI-Owl restoration utility 而非 student 自评分判定。候选模型必须同时：

1. B1、B2、B3、B4 recovery 点估计全部严格高于 Recent；
2. B1--B4 macro learned-minus-recent trajectory bootstrap 95% CI lower bound 严格大于 0；
3. long+very-long macro 点估计严格高于 Recent。

任一失败即 `NO_GO_DECISION_DISTILLATION_V2`，不访问 untouched evaluation，不启动 policy replay、closed-loop
或 matched-NLL。通过只授权一次 untouched selector evaluation；后续仍需单独 gate。
