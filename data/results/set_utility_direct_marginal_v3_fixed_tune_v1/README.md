# Direct conditional-marginal v3 fixed-tune v1

状态：`COMPLETED / NO_GO_DECISION_DISTILLATION_V2 / LEARNED_GENERAL_B_ROUTE_STOPPED`。

Stage-B 唯一 checkpoint 在冻结的 1,063-state / 100-trajectory fixed-tune denominator 上完成真实
restoration truth，0 skip。最终 gate 沿用 decision-v2 的原定义，没有修改预算、阈值、bootstrap 或
Long+ slice。

| Method | B1 | B2 | B3 | B4 | Macro | Long+ |
|---|---:|---:|---:|---:|---:|---:|
| Direct marginal v3 | 0.21827 | 0.38641 | 0.54221 | 0.63645 | 0.44583 | 0.39073 |
| Scalar v2 | **0.26293** | 0.42930 | **0.56152** | **0.64999** | **0.47593** | 0.38853 |
| Recent | 0.19105 | **0.43606** | 0.55426 | 0.62631 | 0.45192 | **0.40081** |

Direct v3 相对 recent 的 macro delta=`-0.00609`，trajectory bootstrap 95% CI=
`[-0.03012,+0.01642]`。它只在 B1/B4 胜 recent，B2/B3、macro CI 与 Long+ 均失败；并且 macro 明显低于
旧 scalar-v2。1,063/1,063 states 的 B1--B4 全部选满，显式 STOP 从未触发，与 Stage-B train-holdout
STOP accuracy≈0 的风险一致。

因此按预承诺停止 learned general-`B` 路线：不创建 v4，不再次修订 gate，不访问 untouched evaluation，
也不启动 policy replay、closed-loop 或 matched-NLL。Restoration teacher 的 Long+ headroom 结论不变；
被否定的是当前 learned general-`B` distillation route。

## 执行与 Source of Truth

- selector：Hyper00 8 + Hyper01 6 张 H200，14 个 disjoint shards，约 95 秒完成；selection content=
  `d9875adb...ffa52e5`，checkpoint=`d8abbe8c...1dd23`；
- truth：14×H200、每卡 2 条 atomic-resume lanes；动态 rebalancing 只移动未完成 lanes，不改变 state
  identity 或科学配置；1,063/1,063 completed、0 skip，wall interval 约 25 分钟；
- schedule：12,763 unique coalitions，content=`8d579c9f...cf9775`；
- full result：Hyper00
  `/data02/jaxan/runs/causalcache-direct-marginal-v3-fixed-tune-b0f84e4/result.json`，content/file SHA256=
  `794fb905...10e76` / `679c3b44...94762`；
- labels：Hyper00 merged root
  `/data02/jaxan/runs/causalcache-direct-marginal-v3-fixed-tune-labels-b0f84e4`；Hyper01 保留 partitions 8--13；
- full result、selection、schedule 与 labels 已发布到 private
  [HF dataset@76615721](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/766157217d99dc8c10d82349d9909ba30ceaa8e9/artifacts/set-utility-direct-marginal-v3-fixed-tune-794fb90)，
  tag=`set-utility-direct-marginal-v3-fixed-tune-794fb90`；远端 3 files 回读完成，`payload.tar.gz` SHA256=
  `e9b080ec...4b4827`；
- Git-safe summary：[`evaluation-summary.json`](evaluation-summary.json)；合同：
  [`docs/set_utility_direct_marginal_v3_stage_b_v1.md`](../../../docs/set_utility_direct_marginal_v3_stage_b_v1.md)。
