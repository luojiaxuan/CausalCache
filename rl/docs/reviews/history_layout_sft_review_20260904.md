# 外审:历史布局配对 SFT(insight 论文的核心实验)(2026-09-04 12:40 PT)

模型 GPT-5.6 Pro,思考 2m26s(首次尝试检索 21 站后"已停止思考"无输出,重发并要求不检索)。

## 1. 审前判断(先写后审)

- 主张:agent 能否利用**非连续**历史截图不是模型固有属性,而是训练时的历史布局造出来的;因此在冻结
  agent 上评测记忆检索,量到的是训练留下的**地板**而非方法的天花板。
- 核心实验:配对 SFT。15,473 样本、442 条成功轨迹,状态与目标动作完全相同、图像张数相同,**只改是哪两张**
  历史帧:RECENCY(最近两帧)vs RANDOM(随机两帧非连续),可选 ORACLE 臂;89% 的样本实际不同。
  未见模板上测 oracle−recency 帧敏感度、上下文不敏感步占比、闭环成功率。
- 待审:三大攻击点 / 配对设计还留下什么混杂 / 赢家诅咒下换什么指标 / SFT 是否正确工具。

## 2. 外审原文(英文,逐字)

The paired SFT experiment is worth doing. The headline is too strong. It can establish that history-sampling
training changes useful history dependence—not that training "creates" the ability, or that frozen performance
is a guaranteed floor.

### (a) Three strongest reviewer attacks

Your evidence does not identify the origin of the ability. Your frozen models already exploit oracle pairs:
without traces, oracle beats recency by 30.0 percentage points for UI-Venus and 17.7 for GUI-Owl.
Trace-preserved invariance can mean the trace supplies sufficient information, not that visual memory is
deficient. The RL checkpoint is suggestive, but its narrow confidence interval cannot repair the missing
matched-training control.

You are testing a history-sampling distribution, not isolating non-contiguity. RANDOM changes frame age,
spacing, relevance, redundancy, and how often the target action is inferable from the supplied evidence. An
advantage could be ordinary train–test distribution alignment, robustness to stale screenshots, or compensation
for missing information. That is useful, but weaker than identifying a distinctive scattered-memory capability.

Your proposed diagnostics can improve while the agent gets worse. Oracle-minus-recency can increase entirely
because recency performance falls. Reduced context-insensitivity can mean increased distractibility.
Gold-action likelihood can improve without changing actions or task success. And a hindsight oracle selected
using the target action measures access to privileged selection—not deployable retrieval.

### (b) Confounds the paired design still leaves

Distinguish confounds from mechanism ambiguity. Frame age and relevance are components of your intervention,
not confounds to its total effect. They nevertheless prevent attributing that effect specifically to
non-contiguity. For that headline, prioritize an older-contiguous-history control, balancing temporal distance
and evidence relevance as far as feasible, over an ORACLE training arm.

The actual controls still matter: paired initialization and seeds, optimizer, training exposure,
checkpoint-selection budget, image preprocessing/token budget, frame ordering/timestamps, and text-trace
treatment. "Same image count" does not establish all of these. Also specify whether RANDOM is sampled once or
resampled each epoch; resampling adds an augmentation-diversity difference.

Your experimental replication is model-training runs, not 15,473 independently trained examples. Use multiple
paired seeds and uncertainty that respects task/trajectory clustering. A step-level interval for two
checkpoints establishes only checkpoint-conditional evidence.

Split task families before tuning selectors, oracles, or hyperparameters. Successful-trajectory SFT also leaves
a deployment gap: it teaches expert-prefix behavior, not recovery from the agent's own mistakes. Closed-loop
evaluation must use each agent's actual available history—not demonstration prefixes.

### (c) A better metric than exists-over-15-pairs

Use held-out selector value, not a per-state maximum. Fix a selector q using training tasks only; evaluate its
chosen histories on unseen templates. Keep that selector fixed across model arms to isolate policy adaptation.

Your primary training effect should be the interaction:
`I = [V_RANDOM-trained(q) − V_RANDOM-trained(recency)] − [V_RECENCY-trained(q) − V_RECENCY-trained(recency)]`

Here V should primarily measure action correctness, allowing multiple valid actions where possible, with
closed-loop task success as the deployment endpoint. Gold-action log probability is a secondary diagnostic.

Report all four absolute values. A positive interaction driven only by recency degradation is not evidence of a
better memory system.

Without a selector, use mean correctness across the same predeclared 15 pairs, rather than whether any pair
succeeds. For an informative-frame diagnostic, use independently annotated useful pairs, not pairs chosen by
maximizing the evaluated checkpoint's gold-action score.

Importantly, exists-over-15 is not inherently invalid: it is a fixed-search-budget hindsight coverage measure.
It becomes misleading when presented as achievable retrieval performance. Separate selector fitting from
evaluation; simply bootstrapping the same selected maxima does not remove selection optimism.

### (d) Is SFT the right instrument?

Yes. I would insist on matched SFT before insisting on RL. It provides the cleaner intervention: common states,
common action targets, controlled history exposure. RL introduces different state visitation and reward-driven
behavior changes, making the mechanism harder to isolate.

But I would not accept teacher-forced sensitivity alone. Require absolute gains on unseen templates and
closed-loop success with a feasible selector, alongside irrelevant-history controls to distinguish useful
dependence from distractibility.

On-policy distillation is the next instrument when expert-prefix gains fail to transfer. Preserve comparability
by training both arms on a shared mixture of rollout states with common teacher labels, rather than separate
arm-specific datasets. RL belongs in the subsequent end-to-end selector–policy experiment; it is not necessary
to establish training-dependent history use.

The defensible claim: "Matched history-sampling training changes the benefit agents obtain from retrieved
screenshots; frozen-agent comparisons therefore need not predict the ranking or performance of jointly adapted
systems." That is substantial—and much harder to dismiss than "training creates memory."

## 3. 采纳 / 不采纳(2026-09-04 12:50 PT,训练发射前预注册)

| 外审意见 | 处置 | 理由 |
|---|---|---|
| 主张过强,"训练造出能力"不成立——冻结模型本来就能用 oracle 对(Venus 无文本 +30.0pp、GUI-Owl +17.7pp) | **采纳,改判** | 它说得对,我把"能力是否存在"与"能力被用到多少"混为一谈了。主张改为其给出的版本:**配对的历史采样训练会改变 agent 从检索到的截图中获得的收益;因此冻结 agent 的比较不必然预测联合适配系统的排名与表现** |
| RANDOM 同时改了帧龄/间距/相关性/冗余,没有隔离"非连续"本身 | **采纳** | 加第三臂 **OLDER-CONTIGUOUS**(更靠前的两张**连续**帧),把时间距离与非连续性分开;优先级高于 ORACLE 训练臂 |
| 诊断量可以在 agent 变差时反而变好(oracle−recency 可能只是 recency 掉了;不敏感度下降可能是更容易被干扰) | **采纳** | 四个绝对值全部报,不只报差;加**无关历史对照**(喂与任务无关的历史帧)测干扰易感性 |
| 主指标改为**固定选择器 q 的交互效应**:I = [V_R训(q)−V_R训(recency)] − [V_rec训(q)−V_rec训(recency)] | **采纳为主指标** | q 只用训练任务拟合、跨臂固定,把"策略适配"与"选择器优劣"分开;∃-over-15 降级为固定预算的事后覆盖量,只作诊断且必须如此标注 |
| 统计单元是**训练 run** 不是 15,473 条样本;要多个配对种子 + 按任务/轨迹聚类的不确定度 | **采纳** | 每臂 ≥2 个配对种子;区间按轨迹聚类 bootstrap |
| 必须声明 RANDOM 是一次采样还是每 epoch 重采样 | **采纳** | 预注册为**一次采样、跨 epoch 固定**,与其他臂一致(避免增广多样性成为第二个变量) |
| 全部配平:初始化/种子/优化器/曝光量/checkpoint 选择预算/图像预处理与 token 预算/帧顺序与时间戳/文本历史处理 | **采纳** | 写进训练脚本的配置校验,发射前逐项 assert |
| 成功轨迹 SFT 只教专家前缀,不教从自身错误中恢复;闭环评测必须用 agent 自己的历史 | **采纳** | 闭环评测本来就走 agent 自身 rollout;limitation 写明 |
| 任务族切分要在调任何东西之前定死 | **采纳(已满足)** | `mw_split_v1.json` 2026-08-25 即固定,seed 与切分规则已入档 |
| SFT 是正确工具,应先于 RL;但只有 teacher-forced 敏感度不够,要未见模板上的绝对增益 + 可行选择器下的闭环成功 | **采纳** | 判据按此写;RL 留给后续端到端 selector–policy 实验 |
| OPD 作为专家前缀增益不迁移时的下一件工具,且两臂应训在共享的 rollout 状态混合上 | **记录,暂不做** | 依赖第二章结果 |

**预注册的判据(训练发射前写定)**:主结果 = 交互效应 I 在未见模板上显著为正,**且** RANDOM 臂在
未见模板上的绝对动作正确率不低于 RECENCY 臂(排除"只是 recency 掉了"),**且** 无关历史对照上
RANDOM 臂不显著更差(排除"只是更容易被干扰")。三条同时满足才进第三章;否则按其给出的更弱主张收敛或停线。
