# Long-oracle in-sample 蒸馏诊断 v1(base Set epoch-1)

状态:`COMPLETED / IN_SAMPLE_TRAIN_DIAGNOSTIC_ONLY_NOT_EVALUATION`。

**用途边界:本诊断在 250 个 long-oracle train states 上进行,这些 state 的 25,032 条标签已参与
v2 训练——数字是 in-sample 的、必然偏乐观,只用于机理定位,不得进入任何 gate、评测表或 paper 主结果。**

对 NO-GO 那版 base Set Transformer epoch-1 checkpoint(SHA `77080c58...`),用带
`--state-id-file`/`--conditional-candidates-per-step` 的 selector CLI 在 Hyper01 GPU 2 重放这 250 个
train states(beam-4,持久化完整 singleton 预测排名),与 immutable long-oracle 真值逐 state 配对。
250/250 join 成功、0 skip。

## 核心发现:模型在自己训过的 state 上,singleton 边际排序仍接近噪声

| 指标 | long (220) | very_long (30) | 随机基线 |
|---|---:|---:|---|
| 预测-真值 singleton Spearman(逐 state 平均) | **0.116** | **0.047** | 0 |
| 模型 top-1 命中真实最优事件 | 9.7% | 5.2% | ~4%/~2.6% |
| 真实最优事件进入模型 top-4 | 31.7% | 18.8% | ~17%/~10.5% |
| 真实最优在 recent-4 之外时被模型 top-4 召回 | 23.2% | 15.4% | -- |
| 模型 top-1 落在 recent-4 内的比例 | 46.6% | 30.2% | ~17%/~10.5% |
| B1 recovery:learned / oracle / recent | 0.178 / 0.534 / 0.087 | 0.093 / 0.456 / -0.839 | -- |

- B2--B4 的 learned 真值覆盖率只有 28.4%/14.4%/5.6%(beam 选择大多不在已标注集合内),且覆盖样本
  有偏,数字不可用;B1 与 singleton 指标为全覆盖、可信。
- 模型 B1 仍明显强于 recent(in-sample),方向与 tune truth 一致;但只兑现 oracle headroom 的约
  1/4--1/3,very_long 上接近随机。

## 参数化病理补充验证(2026-07-21)

对同一 checkpoint 的 250 个 in-sample 重放再验证 scalar-U(S) 参数化的行为:预测 utility 严格随基数
上升 **250/250**,at-most-B 全部退化为 exact-B(每预算选满)**250/250**;而 wave-1/2 真值显示
**56.6% 的第二事件加入降低真实 utility**,20/250 个 state 连最优第二加入都有害(真最优停在 B=1)。
模型的预测边际恒正,结构上无法学会 STOP 与避开有害加入;head 输入含 `log1p(|S|)` 基数特征、回归
目标平均趋势又随 |S| 上升,"more is better" 是最易学解。该证据支持 v3 转向 direct conditional
marginal + STOP 参数化(保留 set-aware encoder)。

## 机理判定

三个候选病因中,本诊断**排除了"标签覆盖不足是主要矛盾"**:这些 state 的 candidate-complete
singleton 组(listwise 权重 2.0 的直接监督对象)就在训练集里,模型仍未拟合其排序。剩余解释:

1. **欠训练**:early stop 在 epoch 1,decision-supervised 部分实际只过了一遍;混合 tune-total 的
   early-stopping 可能被 regression 项主导,decision 目标远未收敛;
2. **表示瓶颈**:16 latents/event 压缩掉判别细节(但 L64 interim 未见改善,尚不能归因于此);
3. **listwise 梯度结构**:大组(n=17--45)+ temperature 0.25 下 teacher 分布长尾近均匀,信号集中在
   极少数事件,KL 梯度稀释。

## 对下一轮的直接建议

把本诊断固化为**零标签成本的训练侧拟合探针**(重放约 2 分钟/checkpoint):在花任何 GPU 做
tune-truth rollout 之前,先要求候选 checkpoint 在这 250 个 in-sample states 上达到如 singleton
Spearman > 0.5、top-4 recall > 60% 这类拟合门槛——连 in-sample 都拟合不了的 checkpoint,不可能在
tune Long+ 上翻盘。优先试:decision 目标单独的 early-stopping 指标、更多 epoch/组过采样、listwise
temperature/归一化修订;然后才轮到加标签或加容量。

## 复现

- selector 重放:`run_set_utility_tune_selectors.py --role train --state-id-file ... --conditional-candidates-per-step 64`
  (Hyper01 `/data02/jaxan/runs/causalcache-long-oracle-insample-diag-v1-b98e0ec/`,selections content
  SHA `1b7e8d3d...`);
- reducer:[`evaluate_set_utility_long_oracle_insample.py`](../../../code/scripts/evaluate_set_utility_long_oracle_insample.py);
- 真值:immutable tag `set-utility-long-oracle-v1-179b0d8`;
- 本目录 [`summary.json`](summary.json) content SHA256=`042a37fb...`(per-state 明细在 Hyper01 raw
  selections 与本地重放中可再生,未入库)。
