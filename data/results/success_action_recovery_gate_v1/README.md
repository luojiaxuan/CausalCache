# U_act oracle 门禁 v1:冻结 policy 对历史图内容完全不敏感

155 条成功轨迹 × 3,221 个重渲染样本(correct/b0/shuffled/irrelevant),冻结 GUI-Owl teacher-forced
打分成功动作的平均 token 对数概率,pair-group 内做差(bootstrap 2000 次):

| 对照 | n | mean | 95% CI | frac>0 |
|---|---:|---:|---:|---:|
| correct − b0 | 698 | **−0.0380** | [−0.0429, −0.0334] | 0.27 |
| correct − shuffled | 590 | −0.0003 | [−0.0031, +0.0025] | 0.59 |
| correct − irrelevant | 667 | −0.0002 | [−0.0038, +0.0033] | 0.55 |

两条结论:

1. **加入正确历史图使成功动作概率显著下降**(−0.038 nats/token):图像对冻结 policy 是纯干扰,
   在 logprob 级复现了闭环的格式税/成功率不增益;
2. **内容盲**:正确图 vs 打乱 vs 无关图,成功动作概率差为 0(CI 半宽 ±0.003)——冻结 policy 完全
   不读历史图内容。任何冻结-policy selector(recency/相似度/learned)在此死透:内容不被读取,选什么
   都一样。这是"policy 必须先学会用历史"主线的直接动机证据,也与上限实验 STORY_DEAD、selector LoRA
   B2/Long+ 无改善互相印证。

预注册后果:纯 CE 的 SFT 历史梯度信号为零(按门禁刹车);margin 对照训练是唯一在设计上仍可行的
路线(显式制造 content 敏感性),是否投入待用户决定。

- 打分数据:hyper01 `/data02/jaxan/artifacts/sft/causalcache-success-sft-v1/scores/`(3,221 行,
  `PENDING_HF_UPLOAD`);数据集 manifest 同目录。
